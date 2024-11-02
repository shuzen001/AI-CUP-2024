## 預測結果 命令
```shell
python bm25_retrieve.py --question_path "競賽資料集/dataset/preliminary/questions_example.json" --source_path "競賽資料集/reference" --output_path "result/檔名.json"
```

## 評估結果 命令
```shell
python eval.py --ground_truth_path "競賽資料集/dataset/preliminary/ground_truths_example.json" --output_path "XXXX.json"
```


## Baseline Precision@1 : 0.7133