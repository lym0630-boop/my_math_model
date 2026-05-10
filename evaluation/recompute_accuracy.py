"""
对已有的 jsonl 推理结果重新提取答案并统计准确率，不需要重新推理
用法: python3 recompute_accuracy.py
"""
import json
import re
import os


def extract_pred_answer(pred_str):
    # 优先匹配 \boxed{...}
    match = re.search(r"\\boxed\{([\-\d\,\.]+)\}", pred_str)
    if match:
        return match.group(1).replace(",", "").strip()
    # 其次 ####
    match = re.search(r"####\s*([\-\d\,\.]+)", pred_str)
    if match:
        return match.group(1).replace(",", "").strip()
    # 最后取最后一个数字
    numbers = re.findall(r"[\-]?\d+(?:\.\d+)?", pred_str)
    if numbers:
        return numbers[-1].strip()
    return None


def recompute(jsonl_path):
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line.strip()))

    correct = 0
    for item in results:
        pred = extract_pred_answer(item["pred_text"])
        gold = item["gold_answer"]
        is_correct = (pred is not None) and (gold is not None) and (pred == gold)
        item["pred_answer"] = pred
        item["correct"] = is_correct
        if is_correct:
            correct += 1

    total = len(results)
    accuracy = correct / total * 100

    # 覆盖保存
    with open(jsonl_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 更新 summary
    summary_path = jsonl_path.replace(".jsonl", "_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    summary.update({"total": total, "correct": correct, "accuracy": round(accuracy, 4)})
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return total, correct, accuracy


BASE = "/cfs/cfs-esygraib/belvathliu/GLM-TTS/floders/pipline/gsm8k_results"

print("重新计算准确率...")
b_total, b_correct, b_acc = recompute(f"{BASE}/before_gsm8k.jsonl")
a_total, a_correct, a_acc = recompute(f"{BASE}/after_gsm8k.jsonl")

print()
print("========================================")
print(f"训练前 (Qwen2.5-Math-7B):  {b_acc:.2f}%  ({b_correct}/{b_total})")
print(f"训练后 (checkpoint-8538):   {a_acc:.2f}%  ({a_correct}/{a_total})")
print(f"变化:  {a_acc - b_acc:+.2f}%")
print("========================================")
