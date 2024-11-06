# 初賽 : 找出最符合問題的文檔編號

## 格式說明

- **提交文件**: `pred.json`


## 預測結果 命令
```shell
python bm25_retrieve.py --question_path "競賽資料集/dataset/preliminary/questions_example.json" --source_path "競賽資料集/reference" --output_path "result/bge.json"
```

## 評估結果 命令
```shell
python eval.py --ground_truth_path "競賽資料集/dataset/preliminary/ground_truths_example.json" --output_path "XXXXXX.json"
```


## 分析類別準確率 命令
```shell
python analysis.py --ground_truth_path "競賽資料集/dataset/preliminary/ground_truths_example.json" --output_path "result/XXXXXX.json"
```

### 模型性能比較

| 模型名稱                     | Precision@1 | 保險      | 金融      | 常見問題  | 描述                                                                                           |
|------------------------------|-------------|-----------|-----------|-----------|------------------------------------------------------------------------------------------------|
| **BM25**                     | 0.7133      | 0.8000    | 0.4400    | 0.9000    | 基於 TF-IDF 原理的經典檢索模型，作為基線（Baseline）模型。由於其基礎原理，我們暫未深入優化此模型，轉而探索更先進的 dense vector 方法。 |
| **BAAI/bge-m3**              | 0.7733      | 0.6800    | 0.6800    | 0.9600    | 使用 BAAI/bge-m3 嵌入模型，根據相似度找出最佳文檔編號。bge-m3 的序列長度為 8192 tokens，能更好地處理長文本。試過加入 Prompt 效果都沒有比較好。                        |
| **BERT (bert-base-chinese)** | 0.4000      | 0.4400    | 0.1400    | 0.1800    | 使用 bert-base-chinese 模型進行預測，表現不如 BM25 和 bge-m3，主要原因可能是 BERT 的上下文窗口限制。                            |
| **BAAI/bge-m3 with Reranker + BM25** | **0.8467**      | 0.8000    | 0.7800    | 0.9600    | 使用 bert-base-chinese 模型進行預測，表現不如 BM25 和 bge-m3，主要原因可能是 BERT 的上下文窗口限制。                            |


## 待辦
- 用將pdf內容做分段，以方便檢索，否則模型的context size 不夠大，檢索準確度就會下降，就像是bert一樣
- 可以嘗試使用多模態的模型去分析pdf
- 用物件導向方式調用模型，避免在函式內一直呼叫模型
- 使用re-ranker模型嘗試
-
