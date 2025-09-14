# drst_preprocess/perf/extract_perf.py
# -*- coding: utf-8 -*-
"""
把 datasets/perf/ 下的 per-exp CSV（由 preprocess_perf.py 生成）合并为场景总表，
并生成你需要的 stage 别名文件。

输入（MinIO）：
- datasets/perf/random_rates_exp-*.csv
- datasets/perf/resource_stimulus_exp-*.csv
- datasets/perf/intervention_exp-*.csv

输出（MinIO）：
- datasets/perf/stage1_random_rates.csv
- datasets/perf/resource_stimulus_global.csv
- datasets/perf/stage2_resource_stimulus_global_A-B-C_modified.csv  # 与上同内容，重命名别名
- datasets/perf/intervention_global.csv
- datasets/perf/stage3_intervention_global.csv                       # 与上同内容，重命名别名
"""

from __future__ import annotations
import io
import re
from typing import List

import pandas as pd

from drst_common.minio_helper import s3, BUCKET

OUT_PREFIX = "datasets/perf"

def _s3_list(prefix: str) -> List[str]:
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

def _s3_read_csv_df(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def _s3_write_csv_df(key: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")

def _naturalsort_key(s: str):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def _merge(prefix_glob: str, out_key: str) -> pd.DataFrame:
    files = sorted([k for k in _s3_list(OUT_PREFIX + "/") if k.startswith(f"{OUT_PREFIX}/{prefix_glob}")],
                   key=_naturalsort_key)
    print(f"[INFO] 匹配 {prefix_glob} -> {len(files)} 个文件", flush=True)
    dfs = []
    for f in files:
        try:
            df = _s3_read_csv_df(f)
            dfs.append(df)
            print(f"[ADD] {f.split('/')[-1]} (rows={len(df)}, cols={len(df.columns)})", flush=True)
        except Exception as e:
            print(f"[SKIP] {f} -> {e}", flush=True)
    if not dfs:
        raise RuntimeError(f"{prefix_glob} 没有可读 CSV")
    combined = pd.concat(dfs, ignore_index=True)
    _s3_write_csv_df(out_key, combined)
    print(f"[OK] combined -> s3://{BUCKET}/{out_key} (rows={len(combined)}, cols={len(combined.columns)})", flush=True)
    return combined

def main():
    print("=" * 88, flush=True)
    print(f"[START] Perf extract (MinIO) — bucket={BUCKET} out_prefix={OUT_PREFIX}", flush=True)

    # stage1：random_rates 合并（全量；Producer 内只取 head(3000)）
    _merge("random_rates_exp-", f"{OUT_PREFIX}/stage1_random_rates.csv")

    # stage2：resource_stimulus 合并 + 别名
    rs = _merge("resource_stimulus_exp-", f"{OUT_PREFIX}/resource_stimulus_global.csv")
    _s3_write_csv_df(f"{OUT_PREFIX}/stage2_resource_stimulus_global_A-B-C_modified.csv", rs)
    print(f"[OK] alias  -> s3://{BUCKET}/{OUT_PREFIX}/stage2_resource_stimulus_global_A-B-C_modified.csv", flush=True)

    # stage3：intervention 合并 + 别名
    iv = _merge("intervention_exp-", f"{OUT_PREFIX}/intervention_global.csv")
    _s3_write_csv_df(f"{OUT_PREFIX}/stage3_intervention_global.csv", iv)
    print(f"[OK] alias  -> s3://{BUCKET}/{OUT_PREFIX}/stage3_intervention_global.csv", flush=True)

    print("[DONE] 合并完成。", flush=True)
    print("=" * 88, flush=True)

if __name__ == "__main__":
    main()
