# drst_preprocess/pcm/extract_pcm.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import io
from typing import List
import pandas as pd

from drst_common.minio_helper import s3, BUCKET, load_csv, save_bytes

PREFIX_IN  = "datasets/pcm/"
OUT_FULL   = "datasets/pcm/pcm_global.csv"
OUT_SMALL  = "datasets/pcm/pcm_global_head10k.csv"

def _list_keys(prefix: str) -> List[str]:
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

def _is_part(key: str) -> bool:
    name = key.split("/")[-1].lower()
    if name in {"pcm_global.csv", "pcm_global_head10k.csv"}:
        return False
    return name.startswith("pcm_part_") and name.endswith(".csv")

def main():
    print(f"[PCM-extract] bucket={BUCKET} prefix={PREFIX_IN}", flush=True)
    keys = [k for k in _list_keys(PREFIX_IN) if _is_part(k)]
    print(f"[PCM-extract] found {len(keys)} csv parts", flush=True)
    if not keys:
        print("[PCM-extract] no parts, nothing to do.", flush=True)
        return

    dfs: List[pd.DataFrame] = []
    for k in sorted(keys):
        try:
            df = load_csv(k)
            if not df.empty:
                dfs.append(df)
                print(f"[ADD] {k} rows={len(df)} cols={len(df.columns)}", flush=True)
            else:
                print(f"[SKIP] {k} empty", flush=True)
        except Exception as e:
            print(f"[SKIP] {k} -> {e}", flush=True)

    if not dfs:
        print("[PCM-extract] all parts empty/failed; stop.", flush=True)
        return

    # 统一列集：出现次数最多的列集合
    from collections import Counter
    sigs = [tuple(df.columns) for df in dfs]
    most_cols = Counter(sigs).most_common(1)[0][0]

    # 一次性对齐并补 0，避免 fragmented warning
    norm = []
    for df in dfs:
        sub = df.reindex(columns=list(most_cols))
        sub = sub.fillna(0.0)
        norm.append(sub)

    full = pd.concat(norm, ignore_index=True)
    print(f"[OK] merged rows={len(full)} cols={len(full.columns)}", flush=True)

    buf = io.StringIO(); full.to_csv(buf, index=False)
    save_bytes(OUT_FULL, buf.getvalue().encode("utf-8"), "text/csv")
    print(f"[SAVE] s3://{BUCKET}/{OUT_FULL}", flush=True)

    if len(full) > 10000:
        buf2 = io.StringIO(); full.head(10000).to_csv(buf2, index=False)
        save_bytes(OUT_SMALL, buf2.getvalue().encode("utf-8"), "text/csv")
        print(f"[SAVE] s3://{BUCKET}/{OUT_SMALL}", flush=True)

if __name__ == "__main__":
    main()
