import json
import argparse

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_average_precision_at_1(ground_truth, predictions):
    total = 0
    cumulative_precision = 0

    # 建立字典來加速ground_truth的查找
    ground_truth_dict = {item["qid"]: item["retrieve"] for item in ground_truth["ground_truths"]}

    for pred in predictions["answers"]:
        qid = pred["qid"]
        predicted_retrieve = pred["retrieve"]
        
        # 如果 qid 存在於 ground_truth 且 retrieve 值相同，則計為正確
        if qid in ground_truth_dict and ground_truth_dict[qid] == predicted_retrieve:
            cumulative_precision += 1
        total += 1

    # 計算 Average Precision@1
    average_precision_at_1 = cumulative_precision / total if total > 0 else 0
    return average_precision_at_1

def main(args):
    # 讀取資料
    ground_truth = load_json(args.ground_truth_path)
    predictions = load_json(args.output_path)
    
    # 計算 Average Precision@1
    average_precision_at_1 = evaluate_average_precision_at_1(ground_truth, predictions)

    # 輸出結果
    print(f"Average Precision@1: {average_precision_at_1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the prediction JSON file.")
    args = parser.parse_args()
    
    main(args)
