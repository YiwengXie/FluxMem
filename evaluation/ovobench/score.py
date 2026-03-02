import os
import json
import argparse
from collections import defaultdict
from collections import OrderedDict

def score(results):
    def calculate_score_backward_realtime(results):
        def get_score(response, gt):
            if response is None:
                return 0
            return int(gt in response)
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores:
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
        def get_score_REC(response, gt):
            if response is None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)
        def get_score_SSR_CRR(response, gt):
            if response is None:
                return 0
            return int(gt in response)
        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            if result["task"] == "REC":
                for j, test_info_ in enumerate(result["test_info"]):
                    scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
            if result["task"] == "SSR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["SSR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
            if result["task"] == "CRR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["CRR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
        return results, scores

    evaluation_results = {
        "backward": {
            "tasks": {},
            "average": None
        },
        "realtime": {
            "tasks": {},
            "average": None
        },
        "forward": {
            "tasks": {},
            "average": None
        },
        "Overall Avg.": None,
    }
    backward_results = results["backward"]
    realtime_results = results["realtime"]
    forward_results = results["forward"]
    avg_scores = {
        "backward": [],
        "realtime": [],
        "forward": []
    }
    
    # Initialize scores
    backward_score = 0
    realtime_score = 0
    forward_score = 0

    if len(backward_results) > 0:
        print("Evaluate Backward Tracing...")
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        for k, v in backward_scores.items():
            print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            evaluation_results["backward"]["tasks"][k] = 100 * sum(v)/len(v)
            avg_scores["backward"].append(sum(v)/len(v))
        backward_score = 100 * sum(avg_scores['backward'])/len(avg_scores['backward'])
        print(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")
        evaluation_results["backward"]["average"] = 100 * sum(avg_scores['backward'])/len(avg_scores['backward'])
    else:
        pass
    if len(realtime_results) > 0:
        print("Evaluate Real-time Visual Perception...")
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        for k, v in realtime_scores.items():
            print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            evaluation_results["realtime"]["tasks"][k] = 100 * sum(v)/len(v)
            avg_scores["realtime"].append(sum(v)/len(v))
        realtime_score = 100 * sum(avg_scores['realtime'])/len(avg_scores['realtime'])
        print(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")
        evaluation_results["realtime"]["average"] = 100 * sum(avg_scores['realtime'])/len(avg_scores['realtime'])
    else:
        pass
    if len(forward_results) > 0:
        print("Evaluate Forward Active Responding...")
        forward_results, forward_scores = calculate_score_forward(forward_results)
        for k, v in forward_scores.items():
            print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            evaluation_results["forward"]["tasks"][k] = 100 * sum(v)/len(v)
            avg_scores["forward"].append(sum(v)/len(v))
        forward_score = 100 * sum(avg_scores['forward'])/len(avg_scores['forward'])
        print(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")
        evaluation_results["forward"]["average"] = 100 * sum(avg_scores['forward'])/len(avg_scores['forward'])
    else:
        pass
    print(f"Total Avg.: {(backward_score + realtime_score + forward_score) / 3:.2f}")
    evaluation_results["Overall Avg."] = (backward_score + realtime_score + forward_score) / 3
    return evaluation_results

def reorder_and_rename_score_dict(score_results):
    # 新结构和顺序
    mapping = [
        ("Real-Time Visual Perception", "realtime", ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]),
        ("Backward Tracing", "backward", ["EPM", "ASI", "HLD"]),
        ("Forward Active Responding", "forward", ["REC", "SSR", "CRR"]),
    ]
    from collections import OrderedDict
    new_score = OrderedDict()
    for new_main, old_main, sub_keys in mapping:
        if old_main in score_results:
            section = OrderedDict()
            tasks = score_results[old_main].get("tasks", {})
            for sub in sub_keys:
                if sub in tasks:
                    section[sub] = tasks[sub]
            # 平均值
            avg = score_results[old_main].get("average", None)
            section["Avg"] = avg
            new_score[new_main] = section
    # Overall
    overall = OrderedDict()
    overall["Avg"] = score_results.get("Overall Avg.", None)
    new_score["Overall"] = overall
    return new_score

def main(result_dir, run_name):
    merged = defaultdict(list)
    output_dir = os.path.join(result_dir, "outputs")
    for fname in os.listdir(output_dir):
        # 排除token相关的统计文件，只处理包含评估结果的文件
        if (fname.startswith(run_name) and fname.endswith('.jsonl') and 
            'token_drop_stats' not in fname and 'token_memory_stats' not in fname):
            with open(os.path.join(output_dir, fname), 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    item = json.loads(line)
                    # 检查是否包含task字段，如果不包含则跳过
                    if 'task' not in item:
                        continue
                    # 分类
                    if item['task'] in ['EPM', 'ASI', 'HLD']:
                        merged['backward'].append(item)
                    elif item['task'] in ['STU', 'OJR', 'ATR', 'ACR', 'OCR', 'FPD']:
                        merged['realtime'].append(item)
                    elif item['task'] in ['REC', 'SSR', 'CRR']:
                        merged['forward'].append(item)
    # 计算分数
    score_results = score(merged)
    # 顺序重排并重命名
    score_results = reorder_and_rename_score_dict(score_results)
    # 保存合并结果和分数
    if not os.path.exists(os.path.join(result_dir, "results")):
        os.makedirs(os.path.join(result_dir, "results"))
    merged_path = os.path.join(result_dir, "results", f"results_merged.json")
    with open(merged_path, 'w') as f:
        json.dump(merged, f, indent=4)
    score_path = os.path.join(result_dir, "results", f"score_merged.json")
    with open(score_path, 'w') as f:
        json.dump(score_results, f, indent=4)
    print(f"合并结果已保存到: {merged_path}")
    print(f"分数已保存到: {score_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    args = parser.parse_args()
    main(args.result_dir, args.run_name)