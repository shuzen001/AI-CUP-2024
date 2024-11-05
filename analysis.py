import json
import argparse
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_category_wise_ap(ground_truth, predictions):
    # 按 category 分組 ground_truth 和 predictions
    ground_truth_by_category = defaultdict(list)
    predictions_by_category = defaultdict(list)
    
    for item in ground_truth["ground_truths"]:
        category = item["category"]
        ground_truth_by_category[category].append(item)
    
    for item in predictions["answers"]:
        # 假設每個 category 都有對應的 ground_truth，如果沒有則跳過
        if item["qid"] in {g["qid"] for g in ground_truth["ground_truths"]}:
            for category, gt_list in ground_truth_by_category.items():
                if any(g["qid"] == item["qid"] for g in gt_list):
                    predictions_by_category[category].append(item)
                    break

    # 計算每個 category 的 AP
    category_ap = {}
    for category, gt_list in ground_truth_by_category.items():
        correct = 0
        total = 0
        for gt in gt_list:
            qid = gt["qid"]
            gt_retrieve = gt["retrieve"]
            
            # 找到對應的預測
            pred = next((p for p in predictions_by_category[category] if p["qid"] == qid), None)
            if pred and pred["retrieve"] == gt_retrieve:
                correct += 1
            total += 1

        # 計算該 category 的 AP
        category_ap[category] = correct / total if total > 0 else 0

    return category_ap

def main(args):
    # 讀取資料
    ground_truth = load_json(args.ground_truth_path)
    predictions = load_json(args.output_path)
    
    # 計算每個 category 的 Average Precision@1
    category_ap = calculate_category_wise_ap(ground_truth, predictions)
    
    # 輸出結果
    print("Category-wise Average Precision@1:")
    for category, ap in category_ap.items():
        print(f"{category}: {ap:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the prediction JSON file.")
    args = parser.parse_args()
    
    main(args)
