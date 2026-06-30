"""medical_03_quality_tier.py - Assign high/medium/low quality tiers."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from medical_conf import (  # noqa: E402
    FILTERED_TRAIN,
    MEDICAL_SCHEMA_FIELDS,
    QUALITY_REPORT,
    REPORT_DIR,
    REQUIRED_CORE_FIELDS,
    ROOT_KEY,
    TIERED_TRAIN,
)


LIST_EVIDENCE_FIELDS = ["症状", "伴随症状", "否认症状", "既往史", "用药史", "过敏史", "风险信号"]


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


def values_as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [str(v) for v in value.values() if str(v).strip()]
    return [str(value)] if str(value).strip() else []


def evidence_ratio(record: dict, text: str) -> float:
    values = []
    for field in LIST_EVIDENCE_FIELDS:
        values.extend(values_as_list(record.get(field)))
    values = [v for v in values if v not in {"未提及", "不清楚", "无", "未用药"}]
    if not values:
        return 1.0
    matched = 0
    for value in values:
        if value in text or any(part and part in text for part in value.split("、")):
            matched += 1
    return matched / len(values)


def assign_tier(item: dict) -> tuple[str, dict]:
    record = item["gold_record"][ROOT_KEY]
    present_fields = [field for field in MEDICAL_SCHEMA_FIELDS if field in record]
    core_present = [field for field in REQUIRED_CORE_FIELDS if field in record]
    ev_ratio = evidence_ratio(record, item["input"])
    no_privacy = not item.get("privacy_residual")

    if no_privacy and len(core_present) == len(REQUIRED_CORE_FIELDS) and len(present_fields) >= 6 and ev_ratio >= 0.40:
        tier = "high"
    elif no_privacy and len(core_present) >= 2 and len(present_fields) >= 4:
        tier = "medium"
    else:
        tier = "low"

    metrics = {
        "present_fields": len(present_fields),
        "core_present": len(core_present),
        "evidence_ratio": round(ev_ratio, 4),
        "no_privacy_residual": no_privacy,
    }
    return tier, metrics


def main() -> None:
    print("=" * 60)
    print("medical_03_quality_tier.py - 医疗问诊质量分层")
    print("=" * 60)

    data = load_jsonl(FILTERED_TRAIN)
    print(f"\n加载过滤后数据: {len(data)} 条")

    tiered = []
    tier_counter = Counter()
    field_counter = Counter()
    for item in data:
        tier, metrics = assign_tier(item)
        item = dict(item)
        item["quality_tier"] = tier
        item["quality_metrics"] = metrics
        tiered.append(item)
        tier_counter[tier] += 1
        field_counter.update(item["gold_record"][ROOT_KEY].keys())

    save_jsonl(tiered, TIERED_TRAIN)

    report = {
        "step": "medical_quality_tier",
        "created_at": datetime.now().isoformat(),
        "input_count": len(data),
        "tier_distribution": dict(tier_counter),
        "field_distribution": dict(field_counter),
        "tier_rules": {
            "high": "no privacy residual + all core fields + >=6 schema fields + evidence_ratio >= 0.40",
            "medium": "no privacy residual + >=2 core fields + >=4 schema fields",
            "low": "remaining samples",
        },
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n质量分布: {dict(tier_counter)}")
    print(f"保存: {TIERED_TRAIN}")
    print(f"报告: {QUALITY_REPORT}")
    print("\n[medical_03_quality_tier] 完成.")


if __name__ == "__main__":
    main()
