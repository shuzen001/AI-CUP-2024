# 初賽 : 找出最符合問題的文檔編號

## 預測結果 命令
```shell
python bm25_retrieve.py --question_path "競賽資料集/dataset/preliminary/questions_example.json" --source_path "競賽資料集/reference" --output_path "result/bge.json"
```

## 評估結果 命令
```shell
python eval.py --ground_truth_path "競賽資料集/dataset/preliminary/ground_truths_example.json" --output_path "XXXX.json"
```

## Baseline Precision@1 : 0.7133


## BAAI/bge-m3 Precision@1 : 0.7733
用一個BAAI/bge-m3 這個embedding 模型，根據similarity找出最佳文檔編號

## BERT Precision@1 : 0.4000
用bert-base-chinese結果

目前都還沒做retrieve以外的步驟