import json
from collections import defaultdict


# 評估相關函數
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_average_precision_at_1(ground_truth, predictions):
    total = 0
    cumulative_precision = 0
    ground_truth_dict = {item["qid"]: item["retrieve"] for item in ground_truth["ground_truths"]}

    for pred in predictions["answers"]:
        qid = pred["qid"]
        predicted_retrieve = pred["retrieve"]
        
        if qid in ground_truth_dict and ground_truth_dict[qid] == predicted_retrieve:
            cumulative_precision += 1
        total += 1

    average_precision_at_1 = cumulative_precision / total if total > 0 else 0
    return average_precision_at_1

def calculate_category_wise_ap(ground_truth, predictions):
    ground_truth_by_category = defaultdict(list)
    predictions_by_category = defaultdict(list)
    
    for item in ground_truth["ground_truths"]:
        category = item["category"]
        ground_truth_by_category[category].append(item)
    
    for item in predictions["answers"]:
        if item["qid"] in {g["qid"] for g in ground_truth["ground_truths"]}:
            for category, gt_list in ground_truth_by_category.items():
                if any(g["qid"] == item["qid"] for g in gt_list):
                    predictions_by_category[category].append(item)
                    break

    category_ap = {}
    for category, gt_list in ground_truth_by_category.items():
        correct = 0
        total = 0
        for gt in gt_list:
            qid = gt["qid"]
            gt_retrieve = gt["retrieve"]
            
            pred = next((p for p in predictions_by_category[category] if p["qid"] == qid), None)
            if pred and pred["retrieve"] == gt_retrieve:
                correct += 1
            total += 1

        category_ap[category] = correct / total if total > 0 else 0

    return category_ap
