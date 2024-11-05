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

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
# def load_data(source_path):
#     '''
#     不使用了，浪費時間
#     '''
#     masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
#     corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
#     return corpus_dict


# 讀取單個PDF文件並返回其文本內容
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


# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source] # e.g. ['hello world', 'hello python',...]

    # [TODO] 可自行替換其他檢索方式，以提升效能
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == a]

    return res[0]  # 回傳檔案名

def Bge_retrieve(qs, source, corpus_dict):
    # 將文檔內容列出來以文字形式
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    #將文字內容轉為向量 embeddings
    model = SentenceTransformer('BAAI/bge-m3', device='cuda') # 使用BGE模型 m3 AP 0.77
    corpus_embeddings = model.encode(filtered_corpus, convert_to_tensor=True, device='cuda' )
    query_embedding = model.encode(qs,prompt="為這個句子生成表示，用以檢索相似的文章: " ,convert_to_tensor=True, device='cuda')

    # 計算相似度
    similarities = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
    max_sim_index = similarities.argmax().item()
    
    return source[max_sim_index]

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



def load_category_corpus(source_path, source_ids):
    """
    只讀取指定 source_ids 中的文件，以減少不必要的讀取。
    :param source_path: 資料夾路徑
    :param source_ids: 要讀取的文件 id 列表
    :return: 該類別對應的 corpus 字典
    """
    corpus_dict = {}
    for file in tqdm(os.listdir(source_path)):
        file_id = int(file.replace('.pdf', ''))
        if file_id in source_ids:  # 僅載入在 source_ids 中的文件
            file_path = os.path.join(source_path, file)
            corpus_dict[file_id] = read_pdf(file_path)
    return corpus_dict

def get_unique_source_ids(qs_ref, category):
    """
    根據指定的 category，獲取所有 unique source ids。
    :param qs_ref: 問題的 JSON 資料
    :param category: 類別名稱 (如 'insurance', 'finance')
    :return: 該類別下的 unique source ids 集合
    """
    source_ids = set()
    for question in qs_ref["questions"]:
        if question["category"] == category:
            source_ids.update(question["source"])
    return sorted(source_ids)


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案


    # 已經改成較有效率的方式讀取資料，原本光載入資料大概就要五分鐘
    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    insurance_source_ids = get_unique_source_ids(qs_ref, 'insurance')
    corpus_dict_insurance = load_category_corpus(source_path_insurance, insurance_source_ids)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    finance_source_ids = get_unique_source_ids(qs_ref, 'finance')
    corpus_dict_finance = load_category_corpus(source_path_finance, finance_source_ids)


    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = Bge_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = Bge_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = Bge_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
