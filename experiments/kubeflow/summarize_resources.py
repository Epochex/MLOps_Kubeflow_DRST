#!/usr/bin/env python3
# experiments/kubeflow/summarize_resources.py
from __future__ import annotations
import io
import json
from typing import Dict, List, Optional

import pandas as pd

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, RESULT_DIR

# ---------- S3 helpers ----------
def _list(prefix: str) -> List[Dict]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return resp.get("Contents", []) or []

def _read_csv_key(key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return None

def _put_csv(key: str, df: pd.DataFrame) -> None:
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

# ---------- Normalization ----------
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名，保证有 ts、cpu_pct、vcpu、rss_mb 三列"""
    df = df.copy()
    # ts
    if "ts" not in df.columns:
        # 兼容 timestamp/epoch_ms 等
        if "timestamp" in df.columns:
            df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
        elif "epoch_ms" in df.columns:
            df["ts"] = pd.to_numeric(df["epoch_ms"], errors="coerce") / 1000.0
        else:
            raise ValueError("resources csv missing 'ts' column")
    else:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    # cpu_pct
    if "cpu_pct" not in df.columns and "cpu_percent" in df.columns:
        df["cpu_pct"] = pd.to_numeric(df["cpu_percent"], errors="coerce")
    elif "cpu_pct" in df.columns:
        df["cpu_pct"] = pd.to_numeric(df["cpu_pct"], errors="coerce")
    else:
        df["cpu_pct"] = pd.NA

    # vcpu（优先使用 vcpu / cpu_vcpu；否则用 cpu_pct * ncpu/100 推算）
    vcpu_col = None
    for c in df.columns:
        if c.lower() in ("vcpu", "cpu_vcpu"):
            vcpu_col = c
            break
    if vcpu_col:
        df["vcpu"] = pd.to_numeric(df[vcpu_col], errors="coerce")
    else:
        ncpu_col = None
        for c in df.columns:
            if c.lower() in ("host_cpus", "ncpu", "n_cpus", "cpu_count"):
                ncpu_col = c
                break
        if ncpu_col is not None and "cpu_pct" in df.columns:
            df["vcpu"] = pd.to_numeric(df["cpu_pct"], errors="coerce") * pd.to_numeric(df[ncpu_col], errors="coerce") / 100.0
        else:
            df["vcpu"] = pd.NA

    # rss_mb（优先 rss_mb；否则从 rss_kb / rss_bytes 推算）
    if "rss_mb" in df.columns:
        df["rss_mb"] = pd.to_numeric(df["rss_mb"], errors="coerce")
    elif "rss_kb" in df.columns:
        df["rss_mb"] = pd.to_numeric(df["rss_kb"], errors="coerce") / 1024.0
    elif "rss_bytes" in df.columns:
        df["rss_mb"] = pd.to_numeric(df["rss_bytes"], errors="coerce") / (1024.0 * 1024.0)
    else:
        df["rss_mb"] = pd.NA

    return df[["ts", "cpu_pct", "vcpu", "rss_mb"]]

def _agg_quantiles(df: pd.DataFrame, comp: str) -> Dict[str, Optional[float]]:
    """抽取用于 summary 的聚合统计"""
    def getq(col: str, q: float) -> Optional[float]:
        try:
            if col in df.columns:
                return float(df[col].quantile(q))
        except Exception:
            pass
        return None

    def getmax(col: str) -> Optional[float]:
        try:
            if col in df.columns:
                return float(df[col].max())
        except Exception:
            pass
        return None

    def getmean(col: str) -> Optional[float]:
        try:
            if col in df.columns:
                return float(df[col].mean())
        except Exception:
            pass
        return None

    return {
        "component": comp,
        "samples": len(df),
        "cpu_pct_mean": getmean("cpu_pct"),
        "cpu_pct_p95":  getq("cpu_pct", 0.95),
        "cpu_pct_max":  getmax("cpu_pct"),
        "rss_mb_mean":  getmean("rss_mb"),
        "rss_mb_p95":   getq("rss_mb", 0.95),
        "rss_mb_max":   getmax("rss_mb"),
    }

# ---------- Infer total builder ----------
def _build_infer_total(prefix: str = f"{RESULT_DIR}/") -> Optional[str]:
    """把 infer1/2/3 的采样按 0.5s 对齐后求和，写出 infer_total_resources.csv"""
    objs = _list(prefix)
    keys = [o["Key"] for o in objs if o["Key"].endswith("_resources.csv")]

    # 尽量匹配出三个推理文件
    infer_keys = []
    for k in keys:
        base = k.split("/")[-1]
        if base.startswith("infer1_") or base == "infer1_resources.csv":
            infer_keys.append(k)
        elif base.startswith("infer2_") or base == "infer2_resources.csv":
            infer_keys.append(k)
        elif base.startswith("infer3_") or base == "infer3_resources.csv":
            infer_keys.append(k)
    # 退一步：任何以 infer 开头的资源文件都算
    if not infer_keys:
        infer_keys = [k for k in keys if k.split("/")[-1].startswith("infer")]

    dfs: List[pd.DataFrame] = []
    for k in sorted(infer_keys):
        df = _read_csv_key(k)
        if df is None or df.empty:
            continue
        try:
            df = _normalize_df(df)
        except Exception:
            continue
        # 0.5s 对齐（与 probe 默认采样间隔 500ms 对齐）
        bin_ts = (df["ts"].astype(float) * 2.0).round() / 2.0
        df = df.assign(bin_ts=bin_ts)
        # 同一个进程自己的重叠行先聚合（避免重复）
        df = df.groupby("bin_ts", as_index=False)[["vcpu", "cpu_pct", "rss_mb"]].sum()
        dfs.append(df)

    if not dfs:
        return None

    # 不同 infer 进程跨表 outer-join 后再汇总
    merged = pd.concat(dfs, ignore_index=True)
    total = merged.groupby("bin_ts", as_index=False)[["vcpu", "cpu_pct", "rss_mb"]].sum()
    # 排序与保留精度
    total = total.sort_values("bin_ts").rename(columns={"bin_ts": "ts"})
    total["vcpu"]   = total["vcpu"].astype(float).round(4)
    total["cpu_pct"] = total["cpu_pct"].astype(float).round(4)
    total["rss_mb"] = total["rss_mb"].astype(float).round(3)

    out_key = f"{RESULT_DIR}/infer_total_resources.csv"
    _put_csv(out_key, total)
    return out_key

# ---------- Main ----------
def main():
    # 先生成合并版 infer_total
    infer_total_key = _build_infer_total(f"{RESULT_DIR}/")
    if infer_total_key:
        print(f"[summarize_resources] wrote {infer_total_key}")

    # 重新枚举所有 *_resources.csv 进入 summary
    objs = _list(f"{RESULT_DIR}/")
    res_keys = [o["Key"] for o in objs if o["Key"].endswith("_resources.csv")]
    rows = []

    for k in sorted(res_keys):
        comp = k.split("/")[-1].replace("_resources.csv", "")
        df = _read_csv_key(k)
        if df is None or df.empty:
            continue
        try:
            df_n = _normalize_df(df)
        except Exception:
            # 老格式至少也试着做 cpu_pct / rss_mb 的均值
            df_n = df.copy()
            for c in ("cpu_pct", "rss_mb"):
                if c in df_n.columns:
                    df_n[c] = pd.to_numeric(df_n[c], errors="coerce")
            if "ts" not in df_n.columns:
                df_n["ts"] = pd.NA

        rows.append(_agg_quantiles(df_n, comp))

    sum_df = pd.DataFrame(rows)
    if not sum_df.empty:
        sum_df = sum_df.sort_values(by=["component"])

    # 写 CSV
    bio = io.BytesIO()
    sum_df.to_csv(bio, index=False)
    save_bytes(f"{RESULT_DIR}/resources_summary.csv", bio.getvalue(), "text/csv")

    # 训练/重训时长（沿用旧逻辑）
    met_key = f"{RESULT_DIR}/metrics_summary.csv"
    met = _read_csv_key(met_key)
    train_lines: List[str] = []
    if met is not None and not met.empty:
        if "component" in met.columns and "event" in met.columns:
            df_off = met[(met["component"] == "offline") & (met["event"] == "train_done")]
            if "wall_s" in df_off.columns and not df_off.empty:
                train_lines.append(
                    f"- **Offline train wall time (s)**: "
                    f"count={len(df_off)}, mean={df_off['wall_s'].mean():.3f}, "
                    f"p95={df_off['wall_s'].quantile(0.95):.3f}, max={df_off['wall_s'].max():.3f}"
                )
            df_rt = met[(met["component"] == "retrain") & (met["event"].isin(["summary", "retrain_done"]))]
            use_col = "retrain_wall_s" if "retrain_wall_s" in met.columns else ("wall_s" if "wall_s" in met.columns else None)
            if use_col and use_col in df_rt.columns and not df_rt.empty:
                train_lines.append(
                    f"- **Retrain wall time (s)**: "
                    f"count={len(df_rt)}, mean={df_rt[use_col].mean():.3f}, "
                    f"p95={df_rt[use_col].quantile(0.95):.3f}, max={df_rt[use_col].max():.3f}"
                )

    md = []
    md.append("# Resources & Training Summary\n")
    md.append(f"- Source bucket: `s3://{BUCKET}/{RESULT_DIR}/`\n")
    if not sum_df.empty:
        md.append("## Per-component resource usage\n")
        md.append(sum_df.to_markdown(index=False))
        md.append("")
    if train_lines:
        md.append("## Training / Retraining time\n")
        md.extend(train_lines)

    save_bytes(f"{RESULT_DIR}/resources_summary.md", "\n".join(md).encode("utf-8"), "text/markdown")
    print("[summarize_resources] resources_summary.csv / .md uploaded")

if __name__ == "__main__":
    main()
