"""medical_05_stratified_sample.py - Stratified sampling for medical SFT tasks."""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from medical_conf import DERIVED_ALL, REPORT_DIR, SAMPLE, SAMPLE_REPORT, SAMPLED_TRAIN  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def complexity_bucket(item: dict) -> str:
    n_fields = item.get("n_fields", 0)
    input_len = item.get("input_len", 0)
    if n_fields <= 4 and input_len < 180:
        return "simple"
    if n_fields >= 9 or input_len >= 500:
        return "complex"
    return "medium"


def stratified_take(pool: list[dict], target: int) -> list[dict]:
    if target <= 0 or len(pool) <= target:
        return list(pool)

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in pool:
        key = (item.get("symptom_group", "未标注"), complexity_bucket(item))
        strata[key].append(item)

    selected = []
    keys = list(strata.keys())
    while len(selected) < target and keys:
        random.shuffle(keys)
        made_progress = False
        for key in list(keys):
            if strata[key] and len(selected) < target:
                selected.append(strata[key].pop())
                made_progress = True
            if not strata[key]:
                keys.remove(key)
        if not made_progress:
            break
    return selected


def main() -> None:
    print("=" * 60)
    print("medical_05_stratified_sample.py - 医疗问诊分层采样")
    print("=" * 60)

    random.seed(SAMPLE["random_seed"])
    data = load_jsonl(DERIVED_ALL)
    print(f"\n加载派生任务: {len(data)} 条")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_task[item["task_type"]].append(item)
    for pool in by_task.values():
        random.shuffle(pool)

    target_total = min(SAMPLE["candidate_target"], len(data))
    selected = []
    for task_type, ratio in SAMPLE["task_ratio"].items():
        target = int(target_total * ratio)
        selected.extend(stratified_take(by_task.get(task_type, []), target))

    # Fill any remainder from unused samples.
    selected_ids = {id(item) for item in selected}
    remainder = [item for item in data if id(item) not in selected_ids]
    random.shuffle(remainder)
    selected.extend(remainder[: max(0, target_total - len(selected))])
    random.shuffle(selected)

    save_jsonl(selected, SAMPLED_TRAIN)

    task_dist = Counter(item["task_type"] for item in selected)
    symptom_dist = Counter(item["symptom_group"] for item in selected)
    complexity_dist = Counter(complexity_bucket(item) for item in selected)
    report = {
        "step": "medical_stratified_sample",
        "created_at": datetime.now().isoformat(),
        "input_count": len(data),
        "target_total": target_total,
        "sampled_count": len(selected),
        "task_ratio_target": SAMPLE["task_ratio"],
        "task_distribution": dict(task_dist),
        "symptom_group_distribution": dict(symptom_dist),
        "complexity_distribution": dict(complexity_dist),
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(SAMPLE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n采样: {len(selected)} 条")
    print(f"任务分布: {dict(task_dist)}")
    print(f"保存: {SAMPLED_TRAIN}")
    print(f"报告: {SAMPLE_REPORT}")
    print("\n[medical_05_stratified_sample] 完成.")


if __name__ == "__main__":
    main()
