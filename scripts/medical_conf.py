"""Shared configuration for the medical consultation structured-record pipeline."""

from __future__ import annotations

import os


# ── Paths ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "medical_raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "medical_processed")
CAND_DIR = os.path.join(BASE_DIR, "data", "medical_sft_candidate")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "medical")

RAW_TRAIN = os.path.join(RAW_DIR, "smoke.jsonl")

NORM_TRAIN = os.path.join(PROC_DIR, "normalized.jsonl")
FILTERED_TRAIN = os.path.join(PROC_DIR, "filtered.jsonl")
TIERED_TRAIN = os.path.join(PROC_DIR, "tiered.jsonl")
DERIVED_ALL = os.path.join(PROC_DIR, "derived_tasks.jsonl")
SAMPLED_TRAIN = os.path.join(PROC_DIR, "sampled_tasks.jsonl")

NORMALIZE_REPORT = os.path.join(REPORT_DIR, "normalize_report.json")
FILTER_REPORT = os.path.join(REPORT_DIR, "filter_report.json")
QUALITY_REPORT = os.path.join(REPORT_DIR, "quality_report.json")
DERIVE_REPORT = os.path.join(REPORT_DIR, "derive_report.json")
SAMPLE_REPORT = os.path.join(REPORT_DIR, "sample_report.json")

FINAL_TRAIN = os.path.join(CAND_DIR, "train.jsonl")
FINAL_VALID = os.path.join(CAND_DIR, "valid.jsonl")
FINAL_METADATA = os.path.join(CAND_DIR, "metadata.json")


# ── Schema ───────────────────────────────────────────────────────────────

ROOT_KEY = "问诊记录"

MEDICAL_SCHEMA_FIELDS = [
    "主诉",
    "现病史",
    "起病时间",
    "症状",
    "伴随症状",
    "否认症状",
    "既往史",
    "用药史",
    "过敏史",
    "生命体征",
    "风险信号",
    "初步分诊",
    "建议就诊科室",
    "随访问题",
]

REQUIRED_CORE_FIELDS = ["主诉", "症状", "初步分诊"]

TRIAGE_ENUM = ["急诊", "普通门诊", "随访观察", "信息不足"]

SYSTEM_PROMPT = (
    "你是一个医疗问诊记录结构化助手。请严格按照 schema 将问诊内容整理为合法 JSON，"
    "只记录原文明确提到的信息，不要诊断、不要编造、不要输出治疗或处方建议。"
)


# ── Normalization / filtering ────────────────────────────────────────────

ROLE_MAP = {
    "doctor": "医生",
    "physician": "医生",
    "医生": "医生",
    "assistant": "医生",
    "patient": "患者",
    "患者": "患者",
    "user": "患者",
    "guardian": "家属",
    "家属": "家属",
}

HARD_FILTER = {
    "min_input_len": 20,
    "max_input_len": 1200,
    "min_record_fields": 3,
}


# ── Sampling ─────────────────────────────────────────────────────────────

SAMPLE = {
    "candidate_target": 5000,
    "internal_valid_ratio": 0.20,
    "random_seed": 42,
    "task_ratio": {
        "medical_record_extraction": 0.45,
        "schema_constraint": 0.20,
        "format_following": 0.15,
        "negation_uncertainty": 0.10,
        "schema_repair": 0.10,
    },
}
