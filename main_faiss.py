import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import jieba
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer
import faiss
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from utils.eval import load_json, evaluate_average_precision_at_1, calculate_category_wise_ap
from utils.reader import PDFReader

# FAISS檢索器
class FAISSRetriever:
    def __init__(self, embed_model='altaidevorg/bge-m3-distill-8l', device='cuda'):
        self.device = torch.device(device)
        if embed_model == 'BAAI/bge-m3':
            model_path = 'model/BAAI__bge-m3'
        elif embed_model == 'altaidevorg/bge-m3-distill-8l':
            model_path = 'model/altaidevorg__bge-m3-distill-8l'
        else:
            model_path = embed_model
            
        self.embed_model = SentenceTransformer(model_path, device=device)
        self.index = None
        self.doc_ids = None
        self.dimension = 1024  # BGE模型的向量維度

    def build_index(self, documents: List[str], doc_ids: List[int]):
        """建立FAISS索引"""
        # 將文檔轉換為向量
        embeddings = self.embed_model.encode(
            documents,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # 轉換為numpy數組
        embeddings = embeddings.cpu().numpy().astype('float32')
        
        # 創建FAISS索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 使用內積（餘弦相似度）
        self.index.add(embeddings)
        self.doc_ids = doc_ids

    def search(self, query: str, top_k: int = 1) -> List[int]:
        """使用FAISS進行檢索"""
        # 將查詢轉換為向量
        query_embedding = self.embed_model.encode(
            query,
            prompt="為這個句子生成表示，用以檢索相似的文章: ",
            convert_to_tensor=True
        )
        
        # 轉換為numpy數組並確保維度正確
        query_embedding = query_embedding.cpu().numpy().astype('float32')
        query_embedding = query_embedding.reshape(1, -1)  # 確保是 2D 數組
        
        # 使用FAISS進行檢索
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 返回文檔ID
        return [self.doc_ids[idx] for idx in indices[0]]

# 主要流程類別
class RetrieverPipeline:
    def __init__(self, question_path, source_path, output_path, ground_truth_path=None, use_markdown=False):
        self.question_path = question_path
        self.source_path = source_path
        self.output_path = output_path
        self.ground_truth_path = ground_truth_path
        self.answer_dict = {"answers": []}
        self.pdf_reader = PDFReader()
        self.use_markdown = use_markdown
        self.retrievers = {
            'finance': FAISSRetriever(embed_model='altaidevorg/bge-m3-distill-8l'),
            'insurance': FAISSRetriever(embed_model='altaidevorg/bge-m3-distill-8l'),
            'faq': FAISSRetriever(embed_model='altaidevorg/bge-m3-distill-8l')
        }
    
    def load_questions(self):
        with open(self.question_path, 'rb') as f:
            return json.load(f)
    
    def load_pid_map(self):
        with open(os.path.join(self.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)
            return {int(key): value for key, value in key_to_source_dict.items()}
    
    def load_corpus(self, category, source_ids):
        corpus_dict = {}
        
        if self.use_markdown:
            converted_dir = f"{category}_converted"
            source_path = os.path.join(self.source_path, converted_dir)
            
            for file_id in tqdm(source_ids, desc=f"Loading {category} corpus (Markdown)"):
                corpus_dict[file_id] = self.markdown_reader.read_markdown(source_path, file_id)
        else:
            source_path = os.path.join(self.source_path, category)
            for file in tqdm(os.listdir(source_path), desc=f"Loading {category} corpus (PDF)"):
                file_id = int(file.replace('.pdf', ''))
                if file_id in source_ids:
                    file_path = os.path.join(source_path, file)
                    corpus_dict[file_id] = self.pdf_reader.read_pdf(file_path)
        
        return corpus_dict
    
    def get_unique_source_ids(self, qs_ref, category):
        source_ids = set()
        for question in qs_ref["questions"]:
            if question["category"] == category:
                source_ids.update(question["source"])
        return sorted(source_ids)
    
    def process(self):
        qs_ref = self.load_questions()
        key_to_source_dict = self.load_pid_map()

        # 載入不同類別的corpus並建立FAISS索引
        corpus = {}
        for category in ['finance', 'insurance']:
            source_ids = self.get_unique_source_ids(qs_ref, category)
            corpus[category] = self.load_corpus(category, source_ids)
            
            # 建立FAISS索引
            documents = [corpus[category][doc_id] for doc_id in source_ids]
            self.retrievers[category].build_index(documents, source_ids)
        
        # 處理每個問題
        for q_dict in tqdm(qs_ref['questions'], desc='Processing questions'):
            category = q_dict['category']
            retriever = self.retrievers[category]
            
            if category == 'faq':
                corpus_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
                documents = [corpus_faq[doc_id] for doc_id in q_dict['source']]
                retriever.build_index(documents, q_dict['source'])
                retrieved = retriever.search(q_dict['query'])[0]
            else:
                retrieved = retriever.search(q_dict['query'])[0]
            
            self.answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        
        # 輸出結果
        with open(self.output_path, 'w', encoding='utf8') as f:
            json.dump(self.answer_dict, f, ensure_ascii=False, indent=4)
        
        # 如果提供了ground_truth_path，則進行評估
        if self.ground_truth_path:
            self.evaluate()
    
    def evaluate(self):
        if not self.ground_truth_path:
            print("未提供ground_truth_path，無法進行評估")
            return
        
        ground_truth = load_json(self.ground_truth_path)
        predictions = self.answer_dict
        
        average_precision_at_1 = evaluate_average_precision_at_1(ground_truth, predictions)
        print(f"\n整體評估結果:")
        print(f"Average Precision@1: {average_precision_at_1:.4f}")
        
        category_ap = calculate_category_wise_ap(ground_truth, predictions)
        print("\n各類別評估結果:")
        for category, ap in category_ap.items():
            print(f"{category}: {ap:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')
    parser.add_argument('--ground_truth_path', type=str, help='讀取ground truth路徑，用於評估結果')
    parser.add_argument('--use_markdown', action='store_true', help='是否使用Markdown格式讀取文件（預設：False）')

    args = parser.parse_args()

    pipeline = RetrieverPipeline(
        args.question_path, 
        args.source_path, 
        args.output_path,
        ground_truth_path=args.ground_truth_path,
        use_markdown=args.use_markdown
    )
    pipeline.process()

if __name__ == "__main__":
    main() 