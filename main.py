import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch
import gc
import asyncio
import torch.quantization
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
# Add concurrent.futures and partial imports after openai import
import concurrent.futures
from functools import partial


from utils.reader import PDFReader, MarkdownReader
from utils.eval import load_json, evaluate_average_precision_at_1, calculate_category_wise_ap
from utils.retriever import FAISSRetriever, get_shared_model



# 主要流程類別
class RetrieverPipeline:
    def __init__(self, question_path, source_path, output_path, ground_truth_path=None, use_markdown=False):
        self.question_path = question_path
        self.source_path = source_path
        self.output_path = output_path
        self.ground_truth_path = ground_truth_path
        self.answer_dict = {"answers": []}
        self.pdf_reader = PDFReader()
        self.markdown_reader = MarkdownReader()
        self.use_markdown = use_markdown  # 控制是否使用Markdown格式
        self.index_dir = os.path.join(self.source_path, 'faiss_indexes')
        os.makedirs(self.index_dir, exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shared_model = get_shared_model('altaidevorg/bge-m3-distill-8l', device)

        self.retrievers = {
            'finance': FAISSRetriever(embed_model='altaidevorg/bge-m3-distill-8l',
                                     index_path=os.path.join(self.index_dir, 'finance.index'),
                                     model=shared_model),
            'insurance': FAISSRetriever(embed_model='altaidevorg/bge-m3-distill-8l',
                                       index_path=os.path.join(self.index_dir, 'insurance.index'),
                                       model=shared_model),
            'faq': FAISSRetriever(embed_model='altaidevorg/bge-m3-distill-8l', model=shared_model)
        }
    
    def load_pid_map(self):
        with open(os.path.join(self.source_path, 'faq/pid_map_content.json'), 'r', encoding='utf-8') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            return {int(key): value for key, value in key_to_source_dict.items()}
    
    def load_corpus(self, category, source_ids):
        """
        載入指定類別的語料庫，支援PDF和Markdown兩種格式
        
        Args:
            category: 文件類別 ('finance' 或 'insurance')
            source_ids: 需要載入的文件ID列表
            
        Returns:
            dict: 文件ID到文件內容的映射
        """
        corpus_dict = {}
        
        if self.use_markdown:
            # 使用Markdown格式
            converted_dir = f"{category}_markdown"
            source_path = os.path.join(self.source_path, converted_dir)

            missing_tasks = []
            for file_id in source_ids:
                md_text = self.markdown_reader.read_markdown(
                    os.path.join(self.source_path, f"{category}_markdown"), file_id
                )
                if md_text == "":
                    original_pdf = os.path.join(self.source_path, category, f"{file_id}.pdf")
                    md_output = os.path.join(
                        self.source_path, f"{category}_markdown", str(file_id), f"{file_id}.md"
                    )
                    missing_tasks.append((original_pdf, md_output))
                else:
                    corpus_dict[file_id] = md_text

            # 先並行補轉缺失的 md
            bulk_pdf_to_markdown(missing_tasks, max_workers=4)

            # 轉檔完成後再次讀入缺失部分
            for pdf_path, md_path in missing_tasks:
                file_id = int(Path(pdf_path).stem)
                corpus_dict[file_id] = self.markdown_reader.read_markdown(
                    os.path.join(self.source_path, f"{category}_markdown"), file_id
                )
        else:
            # 使用PDF格式
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
        qs_ref = load_json(self.question_path)
        key_to_source_dict = self.load_pid_map()

        # 載入不同類別的corpus並建立索引
        corpus = {}
        for category in ['finance', 'insurance']:
            source_ids = self.get_unique_source_ids(qs_ref, category)
            corpus[category] = self.load_corpus(category, source_ids)
            index_file = self.retrievers[category].index_path
            if index_file and os.path.exists(index_file):
                self.retrievers[category].load_index(index_file)
            else:
                documents = [corpus[category][doc_id] for doc_id in source_ids]
                self.retrievers[category].build_index(documents, source_ids, save_path=index_file)
        
        # 處理每個問題
        for q_dict in tqdm(qs_ref['questions'], desc='Processing questions'):
            category = q_dict['category']
            retriever = self.retrievers[category]
            if retriever:
                if category == 'faq':
                    corpus_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
                    documents = [corpus_faq[doc_id] for doc_id in q_dict['source']]
                    retriever.build_index(documents, q_dict['source'])
                    retrieved = retriever.retrieve(q_dict['query'], q_dict['source'], corpus_faq)
                else:
                    retrieved = retriever.retrieve(q_dict['query'], q_dict['source'], corpus[category])
            else:
                raise ValueError("Invalid category")
            
            self.answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        
        # 輸出結果
        with open(self.output_path, 'w', encoding='utf8') as f:
            json.dump(self.answer_dict, f, ensure_ascii=False, indent=4)
        
        # 如果提供了ground_truth_path，則進行評估
        if self.ground_truth_path:
            self.evaluate()
    
    def evaluate(self):
        """評估預測結果的準確率"""
        if not self.ground_truth_path:
            print("未提供ground_truth_path，無法進行評估")
            return
        
        # 載入ground truth
        ground_truth = load_json(self.ground_truth_path)
        predictions = self.answer_dict
        
        # 計算整體Average Precision@1
        average_precision_at_1 = evaluate_average_precision_at_1(ground_truth, predictions)
        print(f"\n整體評估結果:")
        print(f"Average Precision@1: {average_precision_at_1:.4f}")
        
        # 計算各類別的Average Precision@1
        category_ap = calculate_category_wise_ap(ground_truth, predictions)
        print("\n各類別評估結果:")
        for category, ap in category_ap.items():
            print(f"{category}: {ap:.4f}")




def main():
    # 使用argparse解析命令列參數
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
