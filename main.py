import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
import gc
import faiss
import asyncio
import torch.quantization
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
# from FlagEmbedding import FlagModel 

def jina_retrieve(qs, source, corpus_dict):
    # 將文檔內容列出來以文字形式
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    #將文字內容轉為向量 embeddings
    # q_instruction = "為這個句子生成表示，用以檢索相似的文章"
    # d_instruction = "為這個文檔生成表示，用以被檢索"
    model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True ,device='cuda') # 使用BGE模型 m3 AP 0.77
    corpus_embeddings = model.encode(filtered_corpus, task="retrieval.passage")
    query_embedding = model.encode(qs, task="retrieval.query", prompt_name="retrieval.query" )[0]

    #計算相似度
    similarities = model.similarity(query_embedding, corpus_embeddings)
    max_sim_index = similarities.argmax().item()

    return source[max_sim_index]

def BERT_retrieve(qs, source, corpus_dict):
    '''
    由於Bert的上下文token限制只有512，因此還需要進行分段處理
    待辦事項：
        1. 處理長文本限制
    '''

    # 將文檔內容列出來以文字形式
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    #將文字內容轉為向量 embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('ckiplab/bert-base-chinese')
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach()
    
    corpus_embeddings = [get_embedding(doc) for doc in filtered_corpus]
    query_embedding = get_embedding(qs)

    def cosine_similarity(vec1, vec2):
        vec1 = vec1.numpy()
        vec2 = vec2.numpy()
        dot_product = np.dot(vec1, vec2.T)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    #計算相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in corpus_embeddings]
    max_sim_index = np.argmax(similarities)

    return source[max_sim_index]


# PDF讀取器類別
class PDFReader:
    def __init__(self):
        pass

    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容
            if text:
                pdf_text += text
        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本

# 檢索器基礎類別
class Retriever:
    def __init__(self, query, source, corpus_dict):
        raise NotImplementedError("定義好檢索器基礎類別後，應該繼承並實現具體的檢索器類別，最終輸出檔案名稱")

# BM25檢索器
class BM25Retriever(Retriever):
    def __init__(self):
        pass
    
    def retrieve(self, query, source, corpus_dict):
        filtered_corpus = [corpus_dict[int(file)] for file in source] # e.g. ['hello world', 'hello python',...]
        
        # [TODO] 可自行替換其他檢索方式，以提升效能
        tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
        bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = list(jieba.cut_for_search(query))  # 將查詢語句進行分詞
        ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        a = ans[0]
        # 找回與最佳匹配文本相對應的檔案名
        res = [key for key, value in corpus_dict.items() if value == a]

        return res[0]  # 回傳檔案名
    

class HybridRAGRetriever(Retriever):
    def __init__(self, embed_model='BAAI/bge-m3', device='cuda'):
        self.device = torch.device(device)
        self.embed_model = SentenceTransformer(embed_model, device=device)
        self.text_splitter = CharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=50,
            separator="。"  # 加入更多中文標點符號
        )

    def _get_bm25_scores(self, query, source, corpus_dict):
        """使用 BM25 進行檢索並返回分數"""
        filtered_corpus = [corpus_dict[int(file)] for file in source]
        tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = list(jieba.cut_for_search(query))
        
        # 獲取所有文檔的 BM25 分數
        doc_scores = bm25.get_scores(tokenized_query)
        
        # 將分數與文檔 ID 配對
        return [(score, src) for score, src in zip(doc_scores, source)]

    def _get_embedding_scores(self, query, source, corpus_dict):
        """使用 embedding 相似度進行檢索並返回分數"""
        query_embedding = self.embed_model.encode(
            query,
            prompt="為這個句子生成表示，用以檢索相似的文章: ",
            convert_to_tensor=True
        )
        
        scores = []
        for file_id in source:
            document = corpus_dict[int(file_id)]
            # 將文檔分成小塊
            chunks = self.text_splitter.split_text(document)
            chunk_scores = []
            
            for chunk in chunks:
                doc_embedding = self.embed_model.encode(chunk, convert_to_tensor=True)
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    doc_embedding.unsqueeze(0)
                ).item()
                chunk_scores.append(similarity)
            
            # 使用最高的塊分數作為文檔分數
            max_score = max(chunk_scores) if chunk_scores else 0
            scores.append((max_score, file_id))
        
        return scores

    def retrieve(self, query, source, corpus_dict):
        """結合 BM25 和 embedding 的混合檢索"""
        # 獲取 BM25 分數
        bm25_scores = self._get_bm25_scores(query, source, corpus_dict)
        
        # 獲取 embedding 相似度分數
        embedding_scores = self._get_embedding_scores(query, source, corpus_dict)
        
        # 正規化分數
        def normalize_scores(scores):
            if not scores:
                return []
            min_score = min(score for score, _ in scores)
            max_score = max(score for score, _ in scores)
            if max_score == min_score:
                return [(1.0, doc_id) for _, doc_id in scores]
            return [((score - min_score) / (max_score - min_score), doc_id) 
                   for score, doc_id in scores]
        
        bm25_scores_norm = normalize_scores(bm25_scores)
        embedding_scores_norm = normalize_scores(embedding_scores)
        
        # 將分數存入字典以便合併
        combined_scores = defaultdict(float)
        
        # 設定權重
        bm25_weight = 0.3
        embedding_weight = 0.7
        
        # 合併分數
        for score, doc_id in bm25_scores_norm:
            combined_scores[doc_id] += score * bm25_weight
        
        for score, doc_id in embedding_scores_norm:
            combined_scores[doc_id] += score * embedding_weight
        
        # 找出最高分數的文檔
        best_doc_id = max(combined_scores.items(), key=lambda x: x[1])[0]
        
        return best_doc_id

# BGE檢索器
class BGERetriever(Retriever):
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)

    def retrieve(self, query, source, corpus_dict):
        '''
        BGE模型檢索，使用BGE模型進行檢索，並返回最相似的文章

        '''
        # 將文檔內容以文字形式逐個處理
        max_similarity = -1.0
        most_similar_source = None

        # 對查詢進行編碼
        query_embedding = self.model.encode(
            query,
            prompt="為這個句子生成表示，用以檢索相似的文章: ",
            convert_to_tensor=True
        )

        for file_id in source:
            # 獲取單個文檔內容
            document = corpus_dict[int(file_id)]

            # 將單個文檔轉為嵌入
            document_embedding = self.model.encode(document, convert_to_tensor=True)

            # 計算相似度
            similarity = torch.nn.functional.cosine_similarity(query_embedding, document_embedding, dim=0).item()

            # 更新最大相似度和對應的文檔
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_source = file_id

        return most_similar_source
    
# BGE Rerank檢索器
class BGERetrieverWithRerank(Retriever):
    def __init__(self, embed_model='BAAI/bge-m3', rerank_model='BAAI/bge-reranker-large', device='cuda'):
        self.embed_model = SentenceTransformer(embed_model, device=device)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model)
        self.rerank_model.to(device)

    def retrieve(self, query, source, corpus_dict, top_k=2):
        '''
        BGE模型檢索，使用BGE模型進行檢索，並返回最相似的前兩篇文章
        優化後逐個處理文檔以節省記憶體
        '''
        top_k_similar = []  # 存儲(top_k)相似度和對應的文件ID的列表
        
        # 對查詢進行編碼
        query_embedding = self.embed_model.encode(
            query,
            prompt="為這個句子生成表示，用以檢索相似的文章",
            convert_to_tensor=True
        )
        
        for file_id in source:
            try:
                # 獲取單個文檔內容
                document = corpus_dict[int(file_id)]
            except KeyError:
                # 如果file_id不在corpus_dict中，跳過
                continue
            
            # 將單個文檔轉為嵌入
            document_embedding = self.embed_model.encode(document, convert_to_tensor=True)
            
            # 計算相似度，設置 dim=0 以適應1維張量，因為一次只轉一個文檔的嵌入
            similarity = torch.nn.functional.cosine_similarity(query_embedding, document_embedding, dim=0).item()
            
            # 插入到top_k_similar列表中，保持列表按相似度降序排列，最多保留top_k個元素
            if len(top_k_similar) < top_k:
                top_k_similar.append((similarity, file_id))
                top_k_similar.sort(key=lambda x: x[0], reverse=True)
            else:
                # 如果當前相似度大於列表中最小的相似度，則替換
                if similarity > top_k_similar[-1][0]:
                    top_k_similar[-1] = (similarity, file_id)
                    top_k_similar.sort(key=lambda x: x[0], reverse=True)

        
        # 提取Top-K的文件ID和文本
        top_k_sources = [file_id for _, file_id in top_k_similar]
        top_k_texts = [corpus_dict[int(file_id)] for file_id in top_k_sources]
        
        with torch.no_grad():
            pairs = [[query, text] for text in top_k_texts]
            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to('cuda')
            
            rerank_scores = self.rerank_model(**inputs).logits.view(-1,).float()
            
            # Get the single best document
            best_idx = rerank_scores.argmax().item()
            best_source = top_k_sources[best_idx]
            return best_source


    # def colbert_retrieve(self, query, source, corpus_dict):
    #     '''
    #     使用ColBERT模型進行檢索

    #     '''
    #     filtered_corpus_query_pair = [corpus_dict[int(file)] for file in source]

# 主要流程類別
class RetrieverPipeline:
    def __init__(self, question_path, source_path, output_path):
        self.question_path = question_path
        self.source_path = source_path
        self.output_path = output_path
        self.answer_dict = {"answers": []}
        self.pdf_reader = PDFReader()
        self.retrievers = {
            'finance': HybridRAGRetriever(),
            'insurance': HybridRAGRetriever(),
            'faq': HybridRAGRetriever()
        }
    
    def load_questions(self):
        with open(self.question_path, 'rb') as f:
            return json.load(f)
    
    def load_pid_map(self):
        with open(os.path.join(self.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            return {int(key): value for key, value in key_to_source_dict.items()}
    
    def load_corpus(self, category, source_ids):
        source_path = os.path.join(self.source_path, category)
        corpus_dict = {}
        for file in tqdm(os.listdir(source_path), desc=f"Loading {category} corpus"):
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

        # 載入不同類別的corpus
        corpus = {}
        for category in ['finance', 'insurance']:
            source_ids = self.get_unique_source_ids(qs_ref, category)
            corpus[category] = self.load_corpus(category, source_ids)
        
        # 處理每個問題
        for q_dict in tqdm(qs_ref['questions'], desc='Processing questions'):
            category = q_dict['category']
            retriever = self.retrievers[category]
            if retriever:
                if category == 'faq':
                    corpus_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
                    retrieved = retriever.retrieve(q_dict['query'], q_dict['source'], corpus_faq)
                else:
                    retrieved = retriever.retrieve(q_dict['query'], q_dict['source'], corpus[category])
            else:
                raise ValueError("Invalid category")
            
            self.answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        
        # 輸出結果
        with open(self.output_path, 'w', encoding='utf8') as f:
            json.dump(self.answer_dict, f, ensure_ascii=False, indent=4)


def main():
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    pipeline = RetrieverPipeline(args.question_path, args.source_path, args.output_path)
    pipeline.process()


if __name__ == "__main__":
    main()
