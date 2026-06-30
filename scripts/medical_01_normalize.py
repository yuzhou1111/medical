"""medical_01_normalize.py - Normalize de-identified medical consultation data.

Input JSONL supports either:
  - {"dialogue": [{"role": "doctor", "text": "..."}, ...], "gold_record": {...}}
  - {"text"|"input"|"consultation": "...", "gold_record": {...}}

Output is a normalized JSONL with unified role labels, de-identified text, standardized
vital-sign strings, and a root-key-normalized gold record.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from medical_conf import (  # noqa: E402
    MEDICAL_SCHEMA_FIELDS,
    NORMALIZE_REPORT,
    NORM_TRAIN,
    RAW_TRAIN,
    REPORT_DIR,
    ROLE_MAP,
    ROOT_KEY,
)


PRIVACY_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "medical_id": re.compile(r"(?:病历号|就诊号|住院号|门诊号)[:：]?\s*[A-Za-z0-9-]{4,}"),
    "name": re.compile(r"(?:姓名|患者姓名)[:：]\s*[\u4e00-\u9fa5A-Za-z]{2,8}"),
    "address": re.compile(r"(?:住址|地址)[:：]\s*[^，。,；;\n]{4,}"),
}


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def deidentify_text(text: str) -> tuple[str, dict[str, int]]:
    hits: dict[str, int] = {}
    result = text
    replacements = {
        "phone": "[PHONE]",
        "id_card": "[ID_CARD]",
        "medical_id": "[MEDICAL_ID]",
        "name": "[NAME]",
        "address": "[ADDRESS]",
    }
    for name, pattern in PRIVACY_PATTERNS.items():
        result, count = pattern.subn(replacements[name], result)
        if count:
            hits[name] = count
    return result, hits


def has_privacy_residual(text: str) -> bool:
    return any(pattern.search(text) for pattern in PRIVACY_PATTERNS.values())


def normalize_vitals(text: str) -> str:
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:度|摄氏度|℃)", r"\1℃", text)
    text = re.sub(r"血压\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:毫米汞柱|mmhg|mmHg)?", r"血压\1/\2mmHg", text)
    text = re.sub(r"心率\s*(\d{2,3})\s*(?:次每分|次/分钟|次/分)?", r"心率\1次/分", text)
    return text


def normalize_dialogue(item: dict) -> str:
    if isinstance(item.get("dialogue"), list):
        lines = []
        for turn in item["dialogue"]:
            role = ROLE_MAP.get(str(turn.get("role", "")).strip(), str(turn.get("role", "")).strip() or "未知")
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if text:
                lines.append(f"{role}：{text}")
        return "\n".join(lines)
    return str(item.get("consultation") or item.get("input") or item.get("text") or "").strip()


def normalize_gold_record(item: dict) -> dict:
    raw = item.get("gold_record") or item.get("gold_output") or item.get("record") or {}
    if not isinstance(raw, dict):
        return {ROOT_KEY: {}}
    if ROOT_KEY in raw and isinstance(raw[ROOT_KEY], dict):
        record = raw[ROOT_KEY]
    else:
        record = raw
    normalized = {field: record[field] for field in MEDICAL_SCHEMA_FIELDS if field in record}
    return {ROOT_KEY: normalized}


def normalize_item(item: dict, idx: int) -> dict:
    raw_text = normalize_dialogue(item)
    raw_text = normalize_vitals(raw_text)
    text, privacy_hits = deidentify_text(raw_text)
    gold_record = normalize_gold_record(item)
    gold_text, gold_privacy_hits = deidentify_text(json.dumps(gold_record, ensure_ascii=False))
    gold_record = json.loads(gold_text)

    return {
        "id": str(item.get("id") or f"medical_raw_{idx:06d}"),
        "source": item.get("source", "medical_raw"),
        "department": item.get("department", "未标注"),
        "symptom_group": item.get("symptom_group", "未标注"),
        "input": text,
        "gold_record": gold_record,
        "privacy_hits": privacy_hits,
        "gold_privacy_hits": gold_privacy_hits,
        "privacy_residual": has_privacy_residual(text) or has_privacy_residual(gold_text),
    }


def main() -> None:
    print("=" * 60)
    print("medical_01_normalize.py - 医疗问诊标准化与去标识化")
    print("=" * 60)

    data = load_jsonl(RAW_TRAIN)
    print(f"\n加载原始数据: {len(data)} 条")

    normalized = [normalize_item(item, idx) for idx, item in enumerate(data)]
    save_jsonl(normalized, NORM_TRAIN)

    dept_dist = Counter(item["department"] for item in normalized)
    symptom_dist = Counter(item["symptom_group"] for item in normalized)
    privacy_total = Counter()
    for item in normalized:
        privacy_total.update(item["privacy_hits"])
        privacy_total.update({f"gold_{k}": v for k, v in item["gold_privacy_hits"].items()})

    report = {
        "step": "medical_normalize",
        "created_at": datetime.now().isoformat(),
        "input_path": RAW_TRAIN,
        "output_path": NORM_TRAIN,
        "input_count": len(data),
        "output_count": len(normalized),
        "department_distribution": dict(dept_dist),
        "symptom_group_distribution": dict(symptom_dist),
        "privacy_replacements": dict(privacy_total),
        "privacy_residual_count": sum(1 for item in normalized if item["privacy_residual"]),
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(NORMALIZE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n保存: {NORM_TRAIN}")
    print(f"报告: {NORMALIZE_REPORT}")
    print("\n[medical_01_normalize] 完成.")


if __name__ == "__main__":
    main()
