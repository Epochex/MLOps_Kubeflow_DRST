# drst_preprocess/perf/preprocess_perf.py
# -*- coding: utf-8 -*-
"""
读取 MinIO 上的原始 perf 目录，预处理为特征 CSV（63 列，顺序与“标准答案”一致），
并写回 MinIO 到 datasets/perf/ 下。

输入（MinIO 路径，仅作为 key 前缀，非本地路径）：
- raw/random_rates/exp-*/{tx_stats.csv, rx_stats.csv, latency*.csv, firewall.csv, nf_router.csv, ndpi_stats.csv, payload_scan.csv, bridge.csv}
- raw/resource_stimulus/exp-*/...
- raw/intervention/exp-*/...

输出（MinIO）：
- datasets/perf/random_rates_exp-*.csv
- datasets/perf/resource_stimulus_exp-*.csv
- datasets/perf/intervention_exp-*.csv
"""

from __future__ import annotations
import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

# 直接复用你的 MinIO 小工具
from drst_common.minio_helper import s3, BUCKET

# ------------ 常量：标准列顺序（与“标准答案”一致）------------
VNF_ORDER = ["firewall", "nf_router", "ndpi_stats", "payload_scan", "bridge"]
PERF_FEATURES = [
    "instructions", "branches", "branch-misses", "branch-load-misses",
    "cache-misses", "cache-references", "cycles",
    "L1-dcache-load-misses", "L1-dcache-loads",
    "LLC-load-misses", "LLC-stores", "LLC-loads",
]
STANDARD_COLS = (
    ["input_rate", "output_rate", "latency"]
    + [f"{vnf}_{feat}" for feat in PERF_FEATURES for vnf in VNF_ORDER]
)

# S3 路径前缀
RAW_PREFIX = "raw"
OUT_PREFIX = "datasets/perf"

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# ------------ S3 基础 ------------
def _s3_list(prefix: str) -> List[str]:
    """列出某个前缀下的全部对象 key。"""
    keys: List[str] = []
    token = None
    while True:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, ContinuationToken=token) if token else \
               s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.endswith("/"):
                continue
            keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def _s3_read_text(key: str) -> List[str]:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    try:
        txt = data.decode("utf-8")
    except Exception:
        txt = data.decode("latin-1", errors="ignore")
    return txt.splitlines()

def _s3_read_csv_df(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def _s3_write_csv_df(key: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")

# ------------ 解析小工具 ------------
def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

def _parse_first_number(s: str) -> Optional[float]:
    m = NUM_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _numbers_in_tokens(tokens: Iterable[str]) -> List[float]:
    vals: List[float] = []
    for t in tokens:
        m = NUM_RE.search(t)
        if m:
            try:
                v = float(m.group(0))
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
    return vals

def _read_tx_rx_series(key: str, prefer_cols=("Mbit", "PacketRate"), fallback_index=5) -> List[float]:
    """优先读表头列；否则退回逐行取某列或正则提数。"""
    # 1) DataFrame 尝试
    try:
        df = _s3_read_csv_df(key)
        for c in prefer_cols:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce").tolist()
                return [float(v) if v == v else math.nan for v in col]
        # 任取一个像样的数值列
        for c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").tolist()
            if np.isfinite(np.nanmean(col)):
                return [float(v) if v == v else math.nan for v in col]
    except Exception:
        pass
    # 2) 逐行兜底
    vals: List[float] = []
    lines = _s3_read_text(key)
    if not lines:
        return vals
    # 跳过表头
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) > fallback_index:
            v = _safe_float(parts[fallback_index])
            if v is None:
                v = _parse_first_number(parts[fallback_index])
            vals.append(v if v is not None else math.nan)
        else:
            vals.append(math.nan)
    return vals

def _read_latency_series(key: str, unit: str = "auto", is_fixed: bool = False) -> List[float]:
    """默认按 µs→ms（/1000）；fixed_rate 常见前几行非数据跳过。"""
    lines = _s3_read_text(key)
    if is_fixed and len(lines) >= 4:
        lines = lines[4:]
    raw: List[float] = []
    for ln in lines:
        v = _parse_first_number(ln)
        if v is not None:
            raw.append(v)
    if not raw:
        return []
    if unit == "ms":
        return raw
    if unit == "us":
        return [v / 1000.0 for v in raw]
    med = np.nanmedian([abs(v) for v in raw]) if raw else 0
    return [v / 1000.0 for v in raw] if med > 1000 else raw

def _parse_perf_stat_linewise(key: str, feature: str) -> List[float]:
    """
    稳健解析：一行里“任一列”匹配事件名（大小写不敏感），
    再在该行所有数字里选“幅值最大”的那个作为计数（避开 scale≈1.x）。
    """
    vals: List[float] = []
    feat_lower = feature.lower()
    for ln in _s3_read_text(key):
        tokens = [t.strip().strip('"').strip("'") for t in ln.split(",")]
        if not tokens:
            continue
        if not any(t.lower() == feat_lower for t in tokens):
            continue
        nums = _numbers_in_tokens(tokens)
        if not nums:
            continue
        bigs = [x for x in nums if not (0.1 <= abs(x) <= 10.0)]
        v = max(bigs, key=abs) if bigs else max(nums, key=abs)
        vals.append(v)
    return vals

# ------------ 实验扫描与构建 ------------
@dataclass
class Exp:
    scenario: str       # "random_rates" / "resource_stimulus" / "intervention"
    name: str           # exp-1 / exp-2 / ...
    base: str           # S3 目录前缀，如 raw/random_rates/exp-1/

def _find_experiments() -> List[Exp]:
    exps: List[Exp] = []
    for scenario in ("random_rates", "resource_stimulus", "intervention"):
        pref = f"{RAW_PREFIX}/{scenario}/"
        keys = _s3_list(pref)
        # 从 key 推断出 exp-* 目录
        seen = set()
        for k in keys:
            # 形如 raw/<scenario>/exp-7/foo.csv
            parts = k.split("/")
            if len(parts) >= 3 and parts[-1].endswith(".csv"):
                exp = parts[-2]
                if exp.startswith("exp-") and (scenario, exp) not in seen:
                    seen.add((scenario, exp))
                    exps.append(Exp(scenario=scenario, name=exp, base=f"{pref}{exp}/"))
    # 自然排序：按数字
    def natkey(e: Exp):
        import re
        m = re.search(r"(\d+)", e.name)
        return (e.scenario, int(m.group(1)) if m else 1_000_000)
    exps.sort(key=natkey)
    return exps

def _build_one_df(exp: Exp) -> pd.DataFrame:
    # 必备文件
    vnf_files = {v: f"{exp.base}{v}.csv" for v in VNF_ORDER}
    tx_key = f"{exp.base}tx_stats.csv"
    rx_key = f"{exp.base}rx_stats.csv"

    # latency 可能叫 latency.csv / latency_old.csv
    lat_key = None
    for cand in ("latency.csv", "latency_old.csv", "latency-old.csv"):
        k = f"{exp.base}{cand}"
        try:
            s3.head_object(Bucket=BUCKET, Key=k)
            lat_key = k
            break
        except Exception:
            continue

    # 读 KPI
    tx = _read_tx_rx_series(tx_key)
    rx = _read_tx_rx_series(rx_key)
    lat = _read_latency_series(lat_key, unit="auto", is_fixed=False) if lat_key else []

    # 用 firewall_instructions 决定序列长度
    fw_instr = _parse_perf_stat_linewise(vnf_files["firewall"], "instructions")
    cands = [arr for arr in (fw_instr, tx, rx, lat) if arr]
    seq_len = min([len(a) for a in cands]) if cands else 0
    if seq_len <= 0 and tx and rx:
        seq_len = min(len(tx), len(rx))
    if seq_len <= 0:
        raise RuntimeError(f"{exp.base}: 无有效数据列，无法构建序列")

    dic: Dict[str, List[float]] = {}

    # VNF × FEATURE
    for vnf in VNF_ORDER:
        for feat in PERF_FEATURES:
            arr = _parse_perf_stat_linewise(vnf_files[vnf], feat)
            if len(arr) < seq_len:
                arr = list(arr) + [math.nan] * (seq_len - len(arr))
            dic[f"{vnf}_{feat}"] = arr[:seq_len]

    # KPI
    dic["input_rate"]  = tx[:seq_len] if tx else [math.nan] * seq_len
    dic["output_rate"] = rx[:seq_len] if rx else [math.nan] * seq_len
    dic["latency"]     = lat[:seq_len] if lat else [math.nan] * seq_len

    df = pd.DataFrame(dic)

    # 列顺序：标准在前，其它在后
    std_set = set(STANDARD_COLS)
    cols = [c for c in STANDARD_COLS if c in df.columns] + [c for c in df.columns if c not in std_set]
    df = df[cols]
    return df

# ------------ 主入口 ------------
def main():
    print("=" * 88, flush=True)
    print(f"[START] Perf preprocess (MinIO) — bucket={BUCKET} raw_prefix={RAW_PREFIX} out_prefix={OUT_PREFIX}", flush=True)

    exps = _find_experiments()
    print(f"[INFO] 发现 {len(exps)} 个实验：", flush=True)
    for e in exps:
        print(f"   - {e.scenario} :: {e.name} -> {e.base}", flush=True)

    ok = err = 0
    for e in exps:
        try:
            df = _build_one_df(e)
            out_key = f"{OUT_PREFIX}/{e.scenario}_{e.name}.csv"
            _s3_write_csv_df(out_key, df)
            print(f"[OK] write -> s3://{BUCKET}/{out_key} (rows={len(df)}, cols={len(df.columns)})", flush=True)

            expected_cols = 3 + len(PERF_FEATURES) * len(VNF_ORDER)
            if len(df.columns) != expected_cols:
                miss = [c for c in STANDARD_COLS if c not in df.columns]
                extra = [c for c in df.columns if c not in set(STANDARD_COLS)]
                print(f"[WARN] 列数 {len(df.columns)} != 期望 {expected_cols}. 缺失={len(miss)} 多余={len(extra)}", flush=True)
            ok += 1
        except Exception as ex:
            err += 1
            print(f"[ERR] {e.scenario} :: {e.name} -> {ex}", flush=True)

    print(f"[DONE] 成功 {ok} 个，失败 {err} 个。", flush=True)
    print("=" * 88, flush=True)

if __name__ == "__main__":
    main()
