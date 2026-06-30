"""medical_02_filter.py - Hard-filter normalized medical consultation samples."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from medical_conf import (  # noqa: E402
    FILTER_REPORT,
    FILTERED_TRAIN,
    HARD_FILTER,
    MEDICAL_SCHEMA_FIELDS,
    NORM_TRAIN,
    REPORT_DIR,
    ROOT_KEY,
    TRIAGE_ENUM,
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


def validate_record(record: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not isinstance(record, dict) or ROOT_KEY not in record or not isinstance(record.get(ROOT_KEY), dict):
        return False, ["missing_root_record"]

    body = record[ROOT_KEY]
    if len(body) < HARD_FILTER["min_record_fields"]:
        reasons.append("too_few_record_fields")

    unknown_fields = [field for field in body if field not in MEDICAL_SCHEMA_FIELDS]
    if unknown_fields:
        reasons.append("unknown_schema_fields")

    triage = body.get("初步分诊")
    if triage is not None and triage not in TRIAGE_ENUM:
        reasons.append("invalid_triage_enum")

    try:
        json.loads(json.dumps(record, ensure_ascii=False))
    except json.JSONDecodeError:
        reasons.append("invalid_gold_json")

    return len(reasons) == 0, reasons


def filter_item(item: dict) -> tuple[bool, list[str]]:
    reasons = []
    text = item.get("input", "")
    if len(text) < HARD_FILTER["min_input_len"]:
        reasons.append("too_short_input")
    if len(text) > HARD_FILTER["max_input_len"]:
        reasons.append("too_long_input")
    if item.get("privacy_residual"):
        reasons.append("privacy_residual")

    record_ok, record_reasons = validate_record(item.get("gold_record", {}))
    if not record_ok:
        reasons.extend(record_reasons)

    return len(reasons) == 0, reasons


def main() -> None:
    print("=" * 60)
    print("medical_02_filter.py - 医疗问诊硬过滤")
    print("=" * 60)

    data = load_jsonl(NORM_TRAIN)
    print(f"\n加载标准化数据: {len(data)} 条")

    kept = []
    rejected = []
    reason_counter = Counter()
    for item in data:
        ok, reasons = filter_item(item)
        if ok:
            kept.append(item)
        else:
            rejected.append({"id": item.get("id"), "reasons": reasons})
            reason_counter.update(reasons)

    save_jsonl(kept, FILTERED_TRAIN)

    report = {
        "step": "medical_filter",
        "created_at": datetime.now().isoformat(),
        "input_count": len(data),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "reject_reasons": dict(reason_counter),
        "rejected_samples": rejected[:50],
        "thresholds": HARD_FILTER,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(FILTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n保留: {len(kept)} | 剔除: {len(rejected)}")
    print(f"保存: {FILTERED_TRAIN}")
    print(f"报告: {FILTER_REPORT}")
    print("\n[medical_02_filter] 完成.")


if __name__ == "__main__":
    main()
