"""medical_06_to_chat_jsonl.py - Convert sampled medical tasks to chat-style JSONL."""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from medical_conf import (  # noqa: E402
    FINAL_METADATA,
    FINAL_TRAIN,
    FINAL_VALID,
    REPORT_DIR,
    ROOT_KEY,
    SAMPLE,
    SAMPLED_TRAIN,
    SYSTEM_PROMPT,
)


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


def to_chat_style(item: dict, idx: int) -> dict:
    return {
        "id": f"medical_{item['task_type']}_{idx:06d}",
        "source": "medical_consultation",
        "task_type": item["task_type"],
        "topic_schema": ROOT_KEY,
        "department": item.get("department", "未标注"),
        "symptom_group": item.get("symptom_group", "未标注"),
        "quality_tier": item.get("quality_tier", "unknown"),
        "messages": [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["output"]},
        ],
    }


def validate_outputs(data: list[dict]) -> tuple[int, int]:
    ok = 0
    fail = 0
    for item in data:
        try:
            parsed = json.loads(item["messages"][1]["content"])
            if ROOT_KEY not in parsed:
                fail += 1
            else:
                ok += 1
        except json.JSONDecodeError:
            fail += 1
    return ok, fail


def split_train_valid(data: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(data) <= 1:
        return data, []
    valid_size = max(1, int(round(len(data) * SAMPLE["internal_valid_ratio"])))
    valid_size = min(valid_size, len(data) - 1)
    return data[:-valid_size], data[-valid_size:]


def main() -> None:
    print("=" * 60)
    print("medical_06_to_chat_jsonl.py - 医疗问诊最终转写")
    print("=" * 60)

    random.seed(SAMPLE["random_seed"])
    data = load_jsonl(SAMPLED_TRAIN)
    print(f"\n加载采样任务: {len(data)} 条")

    chat_data = [to_chat_style(item, idx) for idx, item in enumerate(data)]
    ok, fail = validate_outputs(chat_data)
    if fail:
        raise ValueError(f"assistant JSON validation failed: ok={ok}, fail={fail}")

    random.shuffle(chat_data)
    train_split, valid_split = split_train_valid(chat_data)

    save_jsonl(train_split, FINAL_TRAIN)
    save_jsonl(valid_split, FINAL_VALID)

    spotcheck = []
    by_task = {}
    for item in train_split:
        by_task.setdefault(item["task_type"], []).append(item)
    for task_type in SAMPLE["task_ratio"]:
        pool = by_task.get(task_type, [])
        random.shuffle(pool)
        spotcheck.extend(pool[:3])
    spotcheck_path = os.path.join(REPORT_DIR, "medical_spotcheck_samples.jsonl")
    save_jsonl(spotcheck, spotcheck_path)

    train_task = Counter(item["task_type"] for item in train_split)
    valid_task = Counter(item["task_type"] for item in valid_split)
    metadata = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source": "de-identified medical consultation records or synthetic smoke data",
        "description": "医疗问诊结构化记录 SFT 数据集，输出根对象固定为 问诊记录",
        "system_prompt": SYSTEM_PROMPT,
        "format": "chat-style JSONL with messages field",
        "train_samples": len(train_split),
        "valid_samples": len(valid_split),
        "task_types": list(SAMPLE["task_ratio"].keys()),
        "task_ratio": SAMPLE["task_ratio"],
        "json_validation": {"ok": ok, "fail": fail},
        "train_task_distribution": dict(train_task),
        "valid_task_distribution": dict(valid_task),
        "pipeline_steps": [
            "medical_normalize (合并对话/统一角色/去标识化/生命体征标准化)",
            "medical_filter (隐私残留/长度/JSON/schema 硬过滤)",
            "medical_quality_tier (high/medium/low 质量分层)",
            "medical_derive_tasks (五类任务派生)",
            "medical_stratified_sample (任务/症状族/复杂度分层采样)",
            "medical_to_chat_jsonl (chat-style JSONL 转写)",
        ],
    }
    with open(FINAL_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(REPORT_DIR, "medical_sft_candidate_report.md")
    report_lines = [
        "# 医疗问诊结构化记录 SFT 候选集报告",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 执行摘要",
        f"- 输入采样任务: {len(data)}",
        f"- 最终 train: {len(train_split)}",
        f"- 最终 valid: {len(valid_split)}",
        f"- assistant JSON 合法性: {ok} 通过, {fail} 失败",
        "",
        "## 2. Train 任务分布",
    ]
    for task_type, count in train_task.most_common():
        report_lines.append(f"- {task_type}: {count}")
    report_lines.extend([
        "",
        "## 3. 安全边界",
        "- 该数据集目标是问诊记录结构化抽取，不训练诊断裁决。",
        "- 样本应来自已授权、已去标识化数据或明确标注为合成 smoke 数据。",
        "- 模型输出不得包含处方建议、治疗建议或原文未出现的医学事实。",
    ])
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n切分: train={len(train_split)}, valid={len(valid_split)}")
    print(f"JSON 合法性: {ok} 通过, {fail} 失败")
    print(f"保存: {FINAL_TRAIN}")
    print(f"保存: {FINAL_VALID}")
    print(f"保存: {FINAL_METADATA}")
    print(f"报告: {report_path}")
    print(f"Spotcheck: {spotcheck_path}")
    print("\n[medical_06_to_chat_jsonl] 完成.")


if __name__ == "__main__":
    main()
