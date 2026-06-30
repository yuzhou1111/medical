"""medical_04_derive_tasks.py - Derive SFT tasks for medical structured records."""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from medical_conf import (  # noqa: E402
    DERIVE_REPORT,
    DERIVED_ALL,
    MEDICAL_SCHEMA_FIELDS,
    REPORT_DIR,
    ROOT_KEY,
    SAMPLE,
    SYSTEM_PROMPT,
    TIERED_TRAIN,
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


def dump_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def filter_record(record: dict, fields: list[str]) -> dict:
    body = record[ROOT_KEY]
    return {ROOT_KEY: {field: body[field] for field in fields if field in body}}


def base_metadata(item: dict, task_type: str, output: dict, prompt: str) -> dict:
    return {
        "task_type": task_type,
        "original_id": item["id"],
        "source": item.get("source", "medical_raw"),
        "cate": "问诊记录",
        "department": item.get("department", "未标注"),
        "symptom_group": item.get("symptom_group", "未标注"),
        "quality_tier": item["quality_tier"],
        "input_len": len(item["input"]),
        "n_fields": len(output.get(ROOT_KEY, {})),
        "prompt": prompt,
        "output": dump_json(output),
    }


def make_medical_record_extraction(item: dict) -> dict:
    schema_str = json.dumps(MEDICAL_SCHEMA_FIELDS, ensure_ascii=False)
    prompt = (
        "请根据给定 schema，将下面的问诊内容整理为结构化 JSON。\n"
        f"Schema: {schema_str}\n"
        "要求: 根对象固定为 \"问诊记录\"；只记录原文明确提到的信息；"
        "不要输出诊断、处方或治疗建议；只输出合法 JSON。\n"
        f"问诊内容:\n{item['input']}"
    )
    return base_metadata(item, "medical_record_extraction", item["gold_record"], prompt)


def make_schema_constraint(item: dict) -> dict:
    fields = ["主诉", "症状", "伴随症状", "否认症状", "既往史", "用药史", "过敏史", "初步分诊", "建议就诊科室"]
    output = filter_record(item["gold_record"], fields)
    schema_str = json.dumps(fields, ensure_ascii=False)
    prompt = (
        "请严格按照 schema 抽取问诊记录。只能输出 schema 中定义的字段，缺失信息不要编造。\n"
        f"Schema: {schema_str}\n"
        f"初步分诊只能从 {json.dumps(TRIAGE_ENUM, ensure_ascii=False)} 中选择。\n"
        "只输出合法 JSON。\n"
        f"问诊内容:\n{item['input']}"
    )
    return base_metadata(item, "schema_constraint", output, prompt)


def make_format_following(item: dict) -> dict:
    constraints = [
        "只输出合法 JSON，不要附加任何解释文字，不要使用 Markdown 代码块。",
        "禁止输出 JSON 以外的内容；字段名必须使用 schema 中的中文字段名。",
        "输出必须是一个 JSON 对象，根对象固定为 \"问诊记录\"。",
    ]
    prompt = (
        f"{random.choice(constraints)}\n"
        f"Schema: {json.dumps(MEDICAL_SCHEMA_FIELDS, ensure_ascii=False)}\n"
        f"问诊内容:\n{item['input']}"
    )
    return base_metadata(item, "format_following", item["gold_record"], prompt)


def make_negation_uncertainty(item: dict) -> dict:
    fields = ["主诉", "现病史", "症状", "伴随症状", "否认症状", "风险信号", "随访问题", "初步分诊"]
    output = filter_record(item["gold_record"], fields)
    prompt = (
        "请从问诊内容中抽取结构化记录，特别注意区分阳性症状、明确否认症状和未提及信息。\n"
        "规则: 患者明确说没有/否认/不伴有的信息才进入 \"否认症状\"；"
        "不确定信息写入现病史或随访问题；未提及的信息不要补充。\n"
        f"Schema: {json.dumps(fields, ensure_ascii=False)}\n"
        "只输出合法 JSON。\n"
        f"问诊内容:\n{item['input']}"
    )
    return base_metadata(item, "negation_uncertainty", output, prompt)


def perturb_record(record: dict) -> tuple[dict, str]:
    perturbed = copy.deepcopy(record)
    body = perturbed[ROOT_KEY]
    fields = list(body.keys())

    if "症状" in body:
        body["临床表现"] = body.pop("症状")
        return perturbed, "字段名 '症状' 被错误写成 '临床表现'"
    if fields:
        removed = fields[0]
        body.pop(removed, None)
        return perturbed, f"缺少字段 '{removed}'"
    body["诊断"] = "不应编造的诊断"
    return perturbed, "添加了 schema 外字段 '诊断'"


def make_schema_repair(item: dict) -> dict:
    broken, desc = perturb_record(item["gold_record"])
    prompt = (
        "下面是一份问诊结构化 JSON 草稿，可能存在字段名错误、字段类型错误、额外字段或 JSON 格式问题。\n"
        "请按目标 schema 修复，保留原文可支持的信息，删除 schema 外字段，不要补充原文没有的信息。\n"
        f"目标 schema: {json.dumps(MEDICAL_SCHEMA_FIELDS, ensure_ascii=False)}\n"
        f"错误类型: {desc}\n"
        f"错误 JSON 草稿: {dump_json(broken)}\n"
        f"原始问诊内容:\n{item['input']}\n"
        "只输出修复后的合法 JSON。"
    )
    task = base_metadata(item, "schema_repair", item["gold_record"], prompt)
    task["perturbation"] = desc
    return task


def main() -> None:
    print("=" * 60)
    print("medical_04_derive_tasks.py - 医疗问诊任务派生")
    print("=" * 60)

    random.seed(SAMPLE["random_seed"])
    data = load_jsonl(TIERED_TRAIN)
    candidates = [item for item in data if item.get("quality_tier") in {"high", "medium"}]
    print(f"\n加载分层数据: {len(data)} 条 | 可派生: {len(candidates)} 条")

    derived = []
    makers = [
        make_medical_record_extraction,
        make_schema_constraint,
        make_format_following,
        make_negation_uncertainty,
        make_schema_repair,
    ]
    for item in candidates:
        for maker in makers:
            derived.append(maker(item))

    save_jsonl(derived, DERIVED_ALL)
    task_dist = Counter(item["task_type"] for item in derived)
    symptom_dist = Counter(item["symptom_group"] for item in derived)

    report = {
        "step": "medical_derive_tasks",
        "created_at": datetime.now().isoformat(),
        "system_prompt": SYSTEM_PROMPT,
        "input_count": len(data),
        "candidate_count": len(candidates),
        "derived_count": len(derived),
        "task_distribution": dict(task_dist),
        "symptom_group_distribution": dict(symptom_dist),
        "task_types": list(SAMPLE["task_ratio"].keys()),
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(DERIVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n任务分布: {dict(task_dist)}")
    print(f"保存: {DERIVED_ALL}")
    print(f"报告: {DERIVE_REPORT}")
    print("\n[medical_04_derive_tasks] 完成.")


if __name__ == "__main__":
    main()
