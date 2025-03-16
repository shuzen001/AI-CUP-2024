# 初賽 : 找出最符合問題的文檔編號

## 格式說明

- **提交文件**: `pred.json`


## 預測結果與評估 命令
```shell
# 僅生成預測結果
python main.py --question_path "競賽資料集/dataset/preliminary/questions_example.json" --source_path "競賽資料集/reference" --output_path "result/bge.json"

# 生成預測結果並同時評估
python main.py --question_path "競賽資料集/dataset/preliminary/questions_example.json" --source_path "競賽資料集/reference" --output_path "result/bge.json" --ground_truth_path "競賽資料集/dataset/preliminary/ground_truths_example.json"
```


### 模型性能比較

| 模型名稱                     | Precision@1 | 保險      | 金融      | 常見問題  | 描述                                                                                           |
|------------------------------|-------------|-----------|-----------|-----------|------------------------------------------------------------------------------------------------|
| **BM25**                     | 0.7133      | 0.8000    | 0.4400    | 0.9000    | 基於TF-IDF的檢索模型，保險方面的表現特別好，值得深入探討 |
| **BAAI/bge-m3**              | 0.7733      | 0.6800    | 0.6800    | 0.9600    | bge-m3 嵌入模型，根據相似度找出最佳文檔編號。bge-m3 的序列長度為 8192 tokens，能更好地處理長文本。試過加入 Prompt 效果都沒有比較好。                        |
| **BERT (bert-base-chinese)** | 0.4000      | 0.4400    | 0.1400    | 0.1800    | Bert-base-chinese 模型進行預測，表現不如 BM25 和 bge-m3，推測原因為 BERT 的上下文窗口限制。                            |
| **BAAI/bge-m3 with Reranker + BM25** | **0.8467**      | 0.8000    | 0.7800    | 0.9600    | bge-m3搭配reranker在金融方面的表現不錯，在保險的方面採用BM25，這樣hybrid的方式大幅度增加了precision。                            |
| **HybridRAGRetriever-E5**             | 0.8067      | 0.9200    | 0.5600    | 0.9400    | 在保險和常見問題類別表現優異，但金融類別仍有改進空間。                            |
| **Hybrid Retrieval - BGE/M3**         | **0.8667**  | 0.9000    | 0.7400    | 0.9600    | 結合多種檢索策略的方法，在所有類別都達到了不錯的平衡，特別是在金融類別有明顯提升。                            |


## 待辦
- 用將pdf內容做分段，以方便檢索，否則模型的context size 不夠大，檢索準確度就會下降，就像是bert一樣
- 可以嘗試使用多模態的模型去分析pdf，OCR之類的方法
- 可以嘗試fine-tune model
- 可以嘗試做batch，使embedding 過程更快
- 可以嘗試做chunk
-
