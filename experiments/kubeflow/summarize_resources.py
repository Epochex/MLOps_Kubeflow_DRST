#!/usr/bin/env python3
# experiments/kubeflow/summarize_resources.py
from __future__ import annotations
import io
from typing import Dict, List, Optional

import pandas as pd

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, RESULT_DIR

BIN_SEC = 0.5  # 聚合对齐到 500ms 桶（probe 也为 500ms）

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
    """
    统一列名，保证有 ts、cpu_pct、vcpu、rss_mb、host_cpus
    """
    df = df.copy()
    # ts
    if "ts" not in df.columns:
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

    # vcpu
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

    # rss_mb
    if "rss_mb" in df.columns:
        df["rss_mb"] = pd.to_numeric(df["rss_mb"], errors="coerce")
    elif "rss_kb" in df.columns:
        df["rss_mb"] = pd.to_numeric(df["rss_kb"], errors="coerce") / 1024.0
    elif "rss_bytes" in df.columns:
        df["rss_mb"] = pd.to_numeric(df["rss_bytes"], errors="coerce") / (1024.0 * 1024.0)
    else:
        df["rss_mb"] = pd.NA

    # host_cpus
    if "host_cpus" in df.columns:
        df["host_cpus"] = pd.to_numeric(df["host_cpus"], errors="coerce")
    else:
        df["host_cpus"] = 1  # 容错：缺失时给默认值

    return df[["ts", "cpu_pct", "vcpu", "rss_mb", "host_cpus"]]

# ---------- Infer total builder ----------
def _build_infer_total(prefix: str = f"{RESULT_DIR}/") -> Optional[str]:
    """
    汇总三个 inference pod 到 500ms 对齐的总曲线：
      - 先对每个 pod 按 0.5s 分箱（vcpu/cpu_pct 取 mean，rss_mb 取 max，host_cpus 取 max）
      - 再跨 pod 在同一 0.5s 桶上相加：vcpu_sum、rss_mb_sum、host_cpus=max
      - 最后计算 cpu_pct_total = vcpu_sum / host_cpus * 100
    不会落任何按 pod 的 500ms 文件，只写 infer_total_resources.csv
    """
    objs = _list(prefix)
    keys = [o["Key"] for o in objs if o["Key"].endswith("_resources.csv")]

    # 捕捉 infer1/2/3
    infer_keys = []
    for k in keys:
        base = k.split("/")[-1]
        if base.startswith("infer1_") or base == "infer1_resources.csv":
            infer_keys.append(k)
        elif base.startswith("infer2_") or base == "infer2_resources.csv":
            infer_keys.append(k)
        elif base.startswith("infer3_") or base == "infer3_resources.csv":
            infer_keys.append(k)
    if not infer_keys:
        # 容错：任何 infer 开头的资源文件
        infer_keys = [k for k in keys if k.split("/")[-1].startswith("infer")]

    if not infer_keys:
        return None

    dfs = []
    for k in sorted(infer_keys):
        df_raw = _read_csv_key(k)
        if df_raw is None or df_raw.empty:
            continue
        try:
            df = _normalize_df(df_raw)
        except Exception:
            continue

        # 对齐到 0.5s 桶（四舍五入），防止各 pod 采样相位略有偏移
        bin_ts = (df["ts"].astype(float) * (1.0 / BIN_SEC * 1.0)).round() / (1.0 / BIN_SEC)
        # 更稳妥写法等价于： (df["ts"] * 2).round() / 2
        df = df.assign(bin_ts=bin_ts)

        df_b = (
            df.groupby("bin_ts", as_index=False)
              .agg(
                  vcpu=("vcpu", "mean"),        # 负载（500ms 采样下 mean≈点值）
                  cpu_pct=("cpu_pct", "mean"),
                  rss_mb=("rss_mb", "max"),     # 水位
                  host_cpus=("host_cpus", "max")
              )
        )
        dfs.append(df_b)

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    total = (
        merged.groupby("bin_ts", as_index=False)
              .agg(
                  vcpu=("vcpu", "sum"),
                  rss_mb=("rss_mb", "sum"),
                  host_cpus=("host_cpus", "max"),
              )
              .sort_values("bin_ts")
              .rename(columns={"bin_ts": "ts"})
              .reset_index(drop=True)
    )

    with pd.option_context("mode.use_inf_as_na", True):
        total["cpu_pct"] = (total["vcpu"] / total["host_cpus"]) * 100.0

    # 精度
    total["vcpu"]    = total["vcpu"].astype(float).round(4)
    total["cpu_pct"] = total["cpu_pct"].astype(float).round(4)
    total["rss_mb"]  = total["rss_mb"].astype(float).round(3)

    out_key = f"{RESULT_DIR}/infer_total_resources.csv"
    _put_csv(out_key, total)
    return out_key

# ---------- Summary helpers ----------
def _agg_quantiles(df: pd.DataFrame, comp: str) -> Dict[str, Optional[float]]:
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

# ---------- Main ----------
def main():
    infer_total_key = _build_infer_total(f"{RESULT_DIR}/")
    if infer_total_key:
        print(f"[summarize_resources] wrote {infer_total_key}")

    # 汇总所有 *_resources.csv（现在这些就是 500ms 的原始采样）
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

    # （可选）生成 Markdown 摘要，沿用旧逻辑
    md = []
    md.append("# Resources Summary\n")
    md.append(f"- Source bucket: `s3://{BUCKET}/{RESULT_DIR}/`\n")
    if not sum_df.empty:
        md.append("## Per-component resource usage (raw 500ms samples)\n")
        md.append(sum_df.to_markdown(index=False))
        md.append("")
    save_bytes(f"{RESULT_DIR}/resources_summary.md", "\n".join(md).encode("utf-8"), "text/markdown")
    print("[summarize_resources] resources_summary.csv / .md uploaded")

if __name__ == "__main__":
    main()
