# drst_preprocess/pcm/preprocess_pcm.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import csv
import re
from typing import List, Optional
import numpy as np
import pandas as pd

from drst_common.minio_helper import s3, BUCKET, save_bytes, load_csv

RAW_PREFIX      = "raw/pcm/const_input"
OUT_PREFIX      = "datasets/pcm"  # 产物：pcm_part_<folder>.csv
DEFAULT_FOLDERS = [
    "const-1gbps","const-2gbps","const-3gbps","const-4gbps",
    "const-5gbps","const-6gbps","const-7gbps","const-8gbps",
    "const-9gbps","const-10gbps-1","const-10gbps-2","const-10gbps-3","const-10gbps-4"
]
VNFS = ["bridge","firewall","ndpi_stats","nf_router","payload_scan","rx","tx"]

_float_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def _to_num(obj):
    s = "" if obj is None else str(obj)
    m = _float_re.search(s)
    return float(m.group(1)) if m else np.nan

def list_keys(prefix: str) -> List[str]:
    keys, token = [], None
    while True:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, ContinuationToken=token) \
               if token else s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def simple_read_csv_bytes(key: str) -> List[List[str]]:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    raw = obj["Body"].read().decode("utf-8", errors="ignore").splitlines()
    return list(csv.reader(raw))

def _dedup_cols(cols: List[str]) -> List[str]:
    """重复列名加后缀 __dupN，确保 df[col] 返回 Series。"""
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        n = seen.get(c, 0)
        if n == 0:
            out.append(c)
        else:
            out.append(f"{c}__dup{n}")
        seen[c] = n + 1
    return out

# ------------ 数值化工具（元素级，更稳健） ------------
def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.applymap(_to_num)

# ------------ 你的清洗逻辑（改为元素级数值化 & 去重） ------------
def combine_csv_headers_rows(rows: List[List[str]]) -> pd.DataFrame:
    if len(rows) < 2:
        return pd.DataFrame()
    first, second = rows[0], rows[1]
    header = [f"{a}-{b}" for a, b in zip(first, second)]
    body = rows[2:]
    df = pd.DataFrame(body, columns=header)
    df.columns = _dedup_cols(list(df.columns))
    return _coerce_numeric_df(df)

def reshape_pcie_rows(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    original_header = rows[0]
    new_header = (
        [f'skt-0_{h}-total' for h in original_header] +
        [f'skt-0_{h}-miss'  for h in original_header] +
        [f'skt-0_{h}-hit'   for h in original_header] +
        [f'skt-1_{h}-total' for h in original_header] +
        [f'skt-1_{h}-miss'  for h in original_header] +
        [f'skt-1_{h}-hit'   for h in original_header]
    )
    reshaped = []
    for i in range(1, len(rows), 7):
        take = []
        for k in range(6):
            if i + k < len(rows):
                take += rows[i + k]
        if take:
            reshaped.append(take)

    drop_names = {'skt-0_Skt-total','skt-1_Skt-total','skt-0_Skt-miss','skt-1_Skt-miss','skt-0_Skt-hit','skt-1_Skt-hit'}
    keep_idx = [i for i,h in enumerate(new_header) if h not in drop_names]
    kept_header = [new_header[i] for i in keep_idx]
    kept_rows   = [[row[i] if i < len(row) else '' for i in keep_idx] for row in reshaped]

    df = pd.DataFrame(kept_rows, columns=kept_header)
    df.columns = _dedup_cols(list(df.columns))
    return _coerce_numeric_df(df)

def memory_clean_rows(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    body   = rows[1:]

    def is_empty_col(idx: int) -> bool:
        for r in body:
            if idx < len(r) and r[idx] != '':
                return False
        return True

    drop_idx = [i for i,h in enumerate(header)
                if str(h).strip().lower() in ('-date','-time') or is_empty_col(i)]
    keep_idx = [i for i in range(len(header)) if i not in set(drop_idx)]

    new_header = [header[i] for i in keep_idx]
    new_rows   = [[r[i] if i < len(r) else '' for i in keep_idx] for r in body]

    df = pd.DataFrame(new_rows, columns=new_header)
    df.columns = _dedup_cols(list(df.columns))   # ★ 关键：先去重列名
    return _coerce_numeric_df(df)                # ★ 关键：元素级数值化

def latency_to_series(key: str, delimiter: str = ' ') -> pd.Series:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(obj["Body"], delimiter=delimiter, header=None)
    if df.shape[1] >= 2:
        df = df.drop(columns=df.columns[0])
    df.columns = ["latency"]
    return pd.to_numeric(df["latency"], errors="coerce")

def _key(folder: str, name: str) -> str:
    return f"{RAW_PREFIX}/{folder}/{name}"

def _exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False

def build_one_folder(folder: str) -> Optional[pd.DataFrame]:
    print(f"[PCM] folder={folder}", flush=True)
    pieces: List[pd.DataFrame] = []

    # vnf-* : 双行表头 -> 数值化
    for v in VNFS:
        k = _key(folder, f"{v}-pcm.csv")
        if _exists(k):
            rows = simple_read_csv_bytes(k)
            df = combine_csv_headers_rows(rows)
            if not df.empty:
                df = df.add_prefix(f"{v}-")
                pieces.append(df)
                print(f"  [+] {k} -> {df.shape}", flush=True)

    # pcie
    k_pcie = _key(folder, "pcm-pcie.csv")
    if _exists(k_pcie):
        rows = simple_read_csv_bytes(k_pcie)
        df = reshape_pcie_rows(rows)
        if not df.empty:
            df = df.add_prefix("pcie-")
            pieces.append(df)
            print(f"  [+] {k_pcie} -> {df.shape}", flush=True)

    # memory
    k_mem = _key(folder, "pcm-memory.csv")
    if _exists(k_mem):
        rows = simple_read_csv_bytes(k_mem)
        df = memory_clean_rows(rows)
        if not df.empty:
            df = df.add_prefix("mem-")
            pieces.append(df)
            print(f"  [+] {k_mem} -> {df.shape}", flush=True)

    # nf_out 展宽
    k_nfo = _key(folder, "nf_out.csv")
    if _exists(k_nfo):
        df_raw = load_csv(k_nfo)
        NF_HEADER = ["tag","instance_id","service_id","thread_info.core",
                     "rx_pps","tx_pps","rx","tx","act_out","act_tonf","act_drop","thread_info.parent",
                     "state","rte_atomic16_read","rx_drop_rate","tx_drop_rate","rx_drop","tx_drop",
                     "act_next","act_buffer","act_returned"]
        if list(df_raw.columns) != NF_HEADER:
            df_raw.columns = (NF_HEADER[:df_raw.shape[1]] + [f"col{i}" for i in range(max(0, df_raw.shape[1]-len(NF_HEADER)))])[:df_raw.shape[1]]
        tags = ['ndpi_stat','router','payload_scan','bridge','firewall']
        features = ["thread_info.core","rx_pps","tx_pps","rx","tx","act_out","act_tonf","act_drop",
                    "thread_info.parent","state","rte_atomic16_read","rx_drop_rate","tx_drop_rate",
                    "rx_drop","tx_drop","act_next","act_buffer","act_returned"]
        new_cols = [f"{t}-{f}" for t in tags for f in features]
        new_df = pd.DataFrame(columns=new_cols)
        for t in tags:
            part = df_raw[df_raw['tag']==t].reset_index(drop=True)
            for f in features:
                new_df[f"{t}-{f}"] = pd.to_numeric(part[f], errors="coerce") if f in part.columns else np.nan
        if not new_df.empty:
            pieces.append(new_df)
            print(f"  [+] {k_nfo} -> {new_df.shape}", flush=True)

    # latency
    lat_key = _key(folder, "latency.csv")
    lat = latency_to_series(lat_key) if _exists(lat_key) else None
    if lat is not None and len(lat) > 0:
        lat_df = pd.DataFrame({"latency": pd.to_numeric(lat, errors="coerce")})
        pieces.append(lat_df)
        print(f"  [+] {lat_key} -> {(len(lat_df), 1)}", flush=True)

    if not pieces:
        print(f"  [!] {folder} no usable files; skip.", flush=True)
        return None

    min_len = min(len(p) for p in pieces if len(p) > 0)
    if min_len <= 0:
        print(f"  [!] {folder} min_len<=0; skip.", flush=True)
        return None

    aligned = [p.iloc[:min_len].reset_index(drop=True) for p in pieces]
    df = pd.concat(aligned, axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def main():
    print(f"[PCM-preprocess] bucket={BUCKET} raw-prefix={RAW_PREFIX}", flush=True)
    any_done = 0
    for folder in DEFAULT_FOLDERS:
        df = build_one_folder(folder)
        if df is None or df.empty:
            continue
        out_key = f"{OUT_PREFIX}/pcm_part_{folder}.csv"
        bio = io.StringIO(); df.to_csv(bio, index=False)
        save_bytes(out_key, bio.getvalue().encode("utf-8"), "text/csv")
        print(f"[SAVE] s3://{BUCKET}/{out_key} rows={len(df)} cols={len(df.columns)}", flush=True)
        any_done += 1
    if any_done == 0:
        print("[PCM-preprocess] no folder produced output; check raw/pcm layout.", flush=True)
    else:
        print(f"[PCM-preprocess] done. parts={any_done}", flush=True)

if __name__ == "__main__":
    main()
