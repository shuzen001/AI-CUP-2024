import jieba  # 用於中文文本分詞
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from langchain.text_splitter import CharacterTextSplitter
from collections import defaultdict


# embedding_model = 


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
    def __init__(self, embed_model='altaidevorg/bge-m3-distill-8l', device='cuda'):
        self.device = torch.device(device)
        if embed_model == 'BAAI/bge-m3':
            model_path = 'model/BAAI__bge-m3'
        elif embed_model == 'altaidevorg/bge-m3-distill-8l':
            model_path = 'model/altaidevorg__bge-m3-distill-8l'
        else:
            model_path = embed_model
        self.embed_model = SentenceTransformer(model_path, device=device)

        
        self.text_splitter = CharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=50,
            separator="。-",  # 加入更多中文標點符號
        )
        # Cache {file_id: (chunks, embeddings)} to avoid recomputing document embeddings
        self._doc_embedding_cache = {}

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

    def _get_cached_chunk_embeddings(self, file_id: int, document: str, batch_size: int = 32):
        """
        Return (chunks, embeddings) for a document, computing and caching if absent.
        Embeddings are generated in a single batch to minimize encode overhead.
        """
        if file_id in self._doc_embedding_cache:
            return self._doc_embedding_cache[file_id]

        chunks = self.text_splitter.split_text(document)
        embeddings = self.embed_model.encode(
            chunks,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        self._doc_embedding_cache[file_id] = (chunks, embeddings)
        return self._doc_embedding_cache[file_id]

    def _get_embedding_scores(self, query, source, corpus_dict):
        """
        使用 embedding 相似度進行檢索並返回分數
        改為一次批次計算並利用快取，可大幅降低延遲
        """
        query_embedding = self.embed_model.encode(
            query,
            prompt="為這個句子生成表示，用以檢索相似的文章: ",
            convert_to_tensor=True
        )

        scores = []
        for file_id in source:
            document = corpus_dict[int(file_id)]

            # 取得 (chunks, embeddings)；若已計算過則直接讀取快取
            chunks, doc_embeddings = self._get_cached_chunk_embeddings(file_id, document)

            # 對所有 chunk embeddings 計算與 query 的餘弦相似度
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), doc_embeddings, dim=1
            )

            max_score = similarities.max().item() if similarities.numel() > 0 else 0.0
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


class HybridRAGRetriever_rerank(HybridRAGRetriever):
    def __init__(self, 
                 embed_model='altaidevorg/bge-m3-distill-8l', 
                 rerank_model='BAAI/bge-reranker-v2-minicpm-layerwise', 
                 device='cuda'):
        super().__init__(embed_model=embed_model, device=device)
        # 使用 layerwise reranker 時，需用 AutoModelForCausalLM 來載入模型
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model, trust_remote_code=True)
        self.rerank_model = AutoModelForCausalLM.from_pretrained(
            rerank_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map= 'auto'
        )
        self.device = device

    def get_inputs(self, pairs, max_length=2048, prompt=None):
        """
        輔助函式：將查詢與段落配對轉換為模型所需的輸入格式。
        預設 prompt 為：
        "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        """
        if prompt is None:
            prompt = ("Given a query A and a passage B, determine whether the passage contains an answer "
                      "to the query by providing a prediction of either 'Yes' or 'No'.")
        sep = "\n"
        prompt_inputs = self.rerank_tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        sep_inputs = self.rerank_tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            query_inputs = self.rerank_tokenizer(
                f'A: {query}',
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True
            )
            passage_inputs = self.rerank_tokenizer(
                f'B: {passage}',
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True
            )
            item = self.rerank_tokenizer.prepare_for_model(
                [self.rerank_tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            # 將特殊符號加入後組成完整輸入
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return self.rerank_tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

    def retrieve(self, query, source, corpus_dict):
        """結合 BM25、embedding 的混合檢索，並加入 reranking"""
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
        
        # 合併兩部分分數
        combined_scores = defaultdict(float)
        bm25_weight = 0.2
        embedding_weight = 0.8
        for score, doc_id in bm25_scores_norm:
            combined_scores[doc_id] += score * bm25_weight
        for score, doc_id in embedding_scores_norm:
            combined_scores[doc_id] += score * embedding_weight
        
        # 選出前 top_k 個文檔進行 reranking
        top_k = 5  # 可根據需求調整
        top_k_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_ids = [doc_id for doc_id, _ in top_k_docs]
        
        # 從 corpus_dict 取出對應文檔內容
        top_k_texts = [corpus_dict[int(doc_id)] for doc_id in top_k_ids]
        
        # 執行 reranking：使用 layerwise reranker 模型
        with torch.no_grad():
            pairs = [[query, text] for text in top_k_texts]
            inputs = self.get_inputs(pairs, max_length=2048).to(self.device)
            # 設定 cutoff_layers，這裡以 [28] 為例
            outputs = self.rerank_model(**inputs, return_dict=True, cutoff_layers=[28])
            # outputs[0] 為所有指定層的 logits，取每個輸出中最後一個 token 的分數
            all_scores = [scores[:, -1].view(-1, ).float() for scores in outputs[0]]
            # 由於我們只設定一個 cutoff layer，取其第一個（也是唯一一個）分數
            rerank_scores = all_scores[0]
            best_idx = rerank_scores.argmax().item()
            best_source = top_k_ids[best_idx]
            
            return best_source

# BGE檢索器
class BGERetriever(Retriever):
    def __init__(self, embed_model='BAAI/bge-m3', device='cuda'):
        # 檢查是否使用本地模型
        if embed_model == 'BAAI/bge-m3':
            model_path = 'model/BAAI__bge-m3'
        elif embed_model == 'altaidevorg/bge-m3-distill-8l':
            model_path = 'model/altaidevorg__bge-m3-distill-8l'
        else:
            model_path = embed_model
            
        self.model = SentenceTransformer(model_path, device=device)

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
            convert_to_tensor=True,
            batch_size=32,
            max_length=1024
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
