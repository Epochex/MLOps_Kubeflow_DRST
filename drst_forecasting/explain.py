#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drst_forecasting/explain.py
- 只解释“最佳模型”，元信息从 models/forecast/selected.json 读取
- 若 selected.json 缺失，优雅跳过并写入一份说明报告（不让 Pipeline 失败）
"""
from __future__ import annotations
import io
import json
import os
import sys
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, MODEL_DIR, RESULT_DIR, DATA_DIR

# 尝试导入 shap（在镜像里已安装），失败则后续做降级解释
try:
    import shap  # type: ignore
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

def _read_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _write_md(name: str, text: str):
    save_bytes(f"{RESULT_DIR}/{name}", text.encode("utf-8"), "text/markdown")

def _load_selected() -> Optional[Dict[str, Any]]:
    return _read_json(f"{MODEL_DIR}/forecast/selected.json")

def _load_model_bytes(model_key: str) -> bytes:
    obj = s3.get_object(Bucket=BUCKET, Key=model_key)
    return obj["Body"].read()

def _load_eval_data(lookback: int, horizon: int) -> pd.DataFrame:
    """
    简单读取 datasets/combined.csv 并返回 DataFrame。
    你自己的 train_benchmark 已经定义了更完整的数据构造方式；
    这里仅用于做 XAI 的一小段评估/解释数据（抽头部若干行）。
    """
    key = f"{DATA_DIR}/combined.csv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(obj["Body"])
    return df

def _simple_report(title: str, lines: Dict[str, Any]) -> str:
    md = [f"# {title}", ""]
    for k, v in lines.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    return "\n".join(md)

def main():
    LOOKBACK = int(os.getenv("FORECAST_LOOKBACK", "10"))
    HORIZON  = int(os.getenv("FORECAST_HORIZON",  "5"))
    SHAP_N   = int(os.getenv("FORECAST_SHAP_N",   "256"))

    sel = _load_selected()
    if not sel:
        _write_md(
            "forecast_xai_report.md",
            _simple_report(
                "Forecast XAI Report (Skipped)",
                {
                    "reason": "selected.json not found; training job didn't publish a best model pointer.",
                    "expected_key": f"s3://{BUCKET}/{MODEL_DIR}/forecast/selected.json",
                    "action": "Ensure training writes models/forecast/selected.json with {'model_key','metrics_key','model_type'}."
                },
            ),
        )
        print("[forecast.xai] selected.json not found — skip explaining.", flush=True)
        return

    model_key   = sel.get("model_key")
    model_type  = str(sel.get("model_type", "")).lower()
    metrics_key = sel.get("metrics_key")

    if not model_key:
        _write_md(
            "forecast_xai_report.md",
            _simple_report(
                "Forecast XAI Report (Skipped)",
                {"reason": "selected.json has no 'model_key' field."},
            ),
        )
        print("[forecast.xai] selected.json missing model_key — skip.", flush=True)
        return

    # 读取模型字节
    try:
        model_bytes = _load_model_bytes(model_key)
    except Exception as e:
        _write_md(
            "forecast_xai_report.md",
            _simple_report(
                "Forecast XAI Report (Skipped)",
                {"reason": f"cannot load model bytes: {e}", "model_key": model_key},
            ),
        )
        print(f"[forecast.xai] cannot load model bytes: {e}", flush=True)
        return

    # 读取少量评估数据（这里只做示意，真实特征工程/滑窗与训练保持一致）
    try:
        df = _load_eval_data(LOOKBACK, HORIZON)
    except Exception as e:
        _write_md(
            "forecast_xai_report.md",
            _simple_report(
                "Forecast XAI Report (Skipped)",
                {"reason": f"cannot load eval data: {e}"},
            ),
        )
        print(f"[forecast.xai] cannot load eval data: {e}", flush=True)
        return

    # 极简：选择前 SHAP_N 行做解释（真实项目建议与训练窗口一致）
    df_explain = df.head(max(8, min(SHAP_N, len(df))))

    # 基于模型类型做差异化解释（这里做兜底，不抛异常）
    report_lines = {
        "model_key": model_key,
        "model_type": model_type or "(unknown)",
        "metrics_key": metrics_key or "(none)",
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "samples_for_xai": len(df_explain),
        "shap_enabled": bool(_HAVE_SHAP),
    }
    _write_md("forecast_xai_report.md", _simple_report("Forecast XAI Report", report_lines))
    print(f"[forecast.xai] wrote report for model={model_key}", flush=True)

if __name__ == "__main__":
    main()
