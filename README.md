# 初賽 : 找出最符合問題的文檔編號

## 格式
input: questions.json
submit: pred.json 


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


## BＭ25 Precision@1 : 0.7133 (Baseline)
背後原理還是TF-IDF，所以沒有想繼續弄，比較想直接使用dense vector的作法

## BAAI/bge-m3 Precision@1 : 0.7733
用一個BAAI/bge-m3 這個embedding 模型，根據similarity找出最佳文檔編號
bge-m3 的 sequence length : 8192 token


## BERT Precision@1 : 0.4000
用bert-base-chinese結果


## 待辦
- 用將pdf內容做分段，以方便檢索，否則模型的context size 不夠大，檢索準確度就會下降，就像是bert一樣
- 可以嘗試使用多模態的模型去分析pdf
- 用物件導向方式調用模型，避免在函式內一直呼叫模型
- 使用re-ranker模型嘗試
-
