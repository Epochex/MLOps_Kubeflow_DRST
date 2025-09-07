#!/usr/bin/env python3
# experiments/summarize_resources.py
from __future__ import annotations
import io
import json
import pandas as pd
from typing import Dict, List

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, RESULT_DIR

def _list(prefix: str) -> List[Dict]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return resp.get("Contents", []) or []

def _read_csv_key(key: str) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return None

def _read_text_key(key: str) -> str:
    try:
        return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
    except Exception:
        return ""

def main():
    # 1) 聚合 *_resources.csv
    objs = _list(f"{RESULT_DIR}/")
    res_keys = [o["Key"] for o in objs if o["Key"].endswith("_resources.csv")]
    rows = []
    for k in sorted(res_keys):
        comp = k.split("/")[-1].replace("_resources.csv", "")
        df = _read_csv_key(k)
        if df is None or df.empty:
            continue

        # 容错：这些列名在 drst_common.resource_probe 中出现概率最高，缺了就跳过该项统计
        def getq(col: str, q: float) -> float | None:
            try:
                if col in df.columns:
                    return float(df[col].quantile(q))
            except Exception:
                pass
            return None
        def getmax(col: str) -> float | None:
            try:
                if col in df.columns:
                    return float(df[col].max())
            except Exception:
                pass
            return None
        def getmean(col: str) -> float | None:
            try:
                if col in df.columns:
                    return float(df[col].mean())
            except Exception:
                pass
            return None

        rows.append({
            "component": comp,
            "samples": len(df),
            "cpu_pct_mean": getmean("cpu_pct"),
            "cpu_pct_p95":  getq("cpu_pct", 0.95),
            "cpu_pct_max":  getmax("cpu_pct"),
            "rss_mb_mean":  getmean("rss_mb"),
            "rss_mb_p95":   getq("rss_mb", 0.95),
            "rss_mb_max":   getmax("rss_mb"),
        })

    sum_df = pd.DataFrame(rows).sort_values(by=["component"])
    bio = io.BytesIO()
    sum_df.to_csv(bio, index=False)
    save_bytes(f"{RESULT_DIR}/resources_summary.csv", bio.getvalue(), "text/csv")

    # 2) 训练/重训时长（从 metrics_summary.csv 中抓取）
    met_key = f"{RESULT_DIR}/metrics_summary.csv"
    met = _read_csv_key(met_key)
    train_md = "\n"
    if met is not None and not met.empty:
        # 容错列名
        col_comp = "component" if "component" in met.columns else None
        col_event = "event" if "event" in met.columns else None

        lines = []
        if col_comp and col_event:
            # offline 训练耗时
            df_off = met[(met[col_comp] == "offline") & (met[col_event] == "train_done")]
            if "wall_s" in df_off.columns and not df_off.empty:
                lines.append(f"- **Offline train wall time (s)**: "
                             f"count={len(df_off)}, mean={df_off['wall_s'].mean():.3f}, "
                             f"p95={df_off['wall_s'].quantile(0.95):.3f}, max={df_off['wall_s'].max():.3f}")
            # retrain 耗时
            df_rt = met[(met[col_comp] == "retrain") & (met[col_event].isin(["summary", "retrain_done", "summary"]))]  # 兼容多事件名
            # 优先使用 retrain_wall_s，如没有就退化到 wall_s
            use_col = "retrain_wall_s" if "retrain_wall_s" in met.columns else ("wall_s" if "wall_s" in met.columns else None)
            if use_col and use_col in df_rt.columns and not df_rt.empty:
                lines.append(f"- **Retrain wall time (s)**: "
                             f"count={len(df_rt)}, mean={df_rt[use_col].mean():.3f}, "
                             f"p95={df_rt[use_col].quantile(0.95):.3f}, max={df_rt[use_col].max():.3f}")

        if lines:
            train_md += "\n".join(lines) + "\n"

    # 3) Markdown 汇总
    md = []
    md.append("# Resources & Training Summary\n")
    md.append(f"- Source bucket: `s3://{BUCKET}/{RESULT_DIR}/`\n")
    if not sum_df.empty:
        md.append("## Per-component resource usage\n")
        md.append(sum_df.to_markdown(index=False))
        md.append("\n")
    if train_md.strip():
        md.append("## Training / Retraining time\n")
        md.append(train_md)

    save_bytes(f"{RESULT_DIR}/resources_summary.md", "\n".join(md).encode("utf-8"), "text/markdown")
    print("[summarize_resources] resources_summary.csv / .md uploaded")

if __name__ == "__main__":
    main()
