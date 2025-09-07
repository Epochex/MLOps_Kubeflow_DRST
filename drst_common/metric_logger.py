#!/usr/bin/env python3
# drst_common/metric_logger.py
from __future__ import annotations
import os, json, time, threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .minio_helper import s3, save_bytes
from .config import RESULT_DIR, BUCKET

# 可控行为（环境变量）：
_METRICS_STREAM_STDOUT   = os.getenv("METRICS_STREAM_STDOUT", "0").strip().lower() in ("1","true","yes","on")
_METRICS_FLUSH_EVERY     = int(os.getenv("METRICS_FLUSH_EVERY", "0"))
_METRICS_FLUSH_INTERVALS = int(os.getenv("METRICS_FLUSH_INTERVAL_S", "0"))

_SHARD_PREFIX = os.getenv("METRICS_SHARD_PREFIX", f"{RESULT_DIR}/metrics")
_SUMMARY_KEY  = os.getenv("METRICS_SUMMARY_KEY",  f"{RESULT_DIR}/metrics_summary.csv")

CSV_HEADER = "ts,component,event,pod,kv\n"

_BUF: List[Dict[str, Any]] = []
_LOCK = threading.Lock()
_LAST_FLUSH = time.time()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _pod() -> str:
    return os.getenv("HOSTNAME", "pod")

def _row_line(r: Dict[str, Any]) -> str:
    kv = json.dumps(r.get("kv", {}), ensure_ascii=False, separators=(",", ":"))
    def safe(x): return str(x).replace("\n", " ").replace("\r", " ")
    return f"{safe(r['ts'])},{safe(r['component'])},{safe(r['event'])},{safe(r['pod'])},{kv}"

def _append_lines(key: str, lines: List[str]) -> None:
    try:
        old = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8", "ignore")
    except Exception:
        old = ""
    if not old:
        payload = CSV_HEADER + "\n".join(lines) + "\n"
    else:
        if not old.splitlines()[0].startswith("ts,component,event,pod,kv"):
            old = CSV_HEADER + old
        payload = old + ("" if old.endswith("\n") else "\n") + "\n".join(lines) + "\n"
    save_bytes(key, payload.encode("utf-8"), "text/csv")

def _flush_locked():
    global _BUF, _LAST_FLUSH
    if not _BUF:
        return
    buckets: Dict[str, List[str]] = {}
    for r in _BUF:
        comp = r.get("component", "misc")
        buckets.setdefault(comp, []).append(_row_line(r))
    pod = _pod()
    for comp, lines in buckets.items():
        shard = f"{_SHARD_PREFIX}/{comp}-{pod}.csv"
        _append_lines(shard, lines)
    _BUF.clear()
    _LAST_FLUSH = time.time()

def log_metric(*, component: str, event: str, **kwargs) -> None:
    row = {
        "ts": _now_iso(),
        "component": component,
        "event": event,
        "pod": _pod(),
        "kv": kwargs or {},
    }
    if _METRICS_STREAM_STDOUT:
        kv_pairs = " ".join(f"{k}={v}" for k, v in (kwargs or {}).items())
        print(f"[metrics:{row['component']}/{row['pod']}] {row['event']} | {kv_pairs}", flush=True)

    do_flush = False
    with _LOCK:
        _BUF.append(row)
        if _METRICS_FLUSH_EVERY and len(_BUF) >= _METRICS_FLUSH_EVERY:
            do_flush = True
        elif _METRICS_FLUSH_INTERVALS and (time.time() - _LAST_FLUSH) >= _METRICS_FLUSH_INTERVALS:
            do_flush = True
    if do_flush:
        with _LOCK:
            _flush_locked()

def _list_shards(prefix: str) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, ContinuationToken=token) if token \
            else s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        for o in (resp.get("Contents") or []):
            if o["Key"].endswith(".csv"):
                keys.append(o["Key"])
        if not resp.get("IsTruncated"): break
        token = resp.get("NextContinuationToken")
    return keys

def _merge_to_summary():
    shards = _list_shards(_SHARD_PREFIX.rstrip("/") + "/")
    if not shards:
        return
    parts: List[str] = [CSV_HEADER.rstrip("\n")]
    for k in shards:
        try:
            txt = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode("utf-8", "ignore")
        except Exception:
            continue
        if not txt: continue
        lines = txt.splitlines()
        if lines and lines[0].startswith("ts,component,event,pod,kv"):
            lines = lines[1:]
        parts.extend(lines)
    payload = "\n".join([ln for ln in parts if ln.strip()]) + "\n"
    save_bytes(_SUMMARY_KEY, payload.encode("utf-8"), "text/csv")

def sync_all_metrics_to_minio() -> None:
    with _LOCK:
        _flush_locked()
    try:
        _merge_to_summary()
    except Exception as e:
        print(f"[metric_logger] merge shards failed: {e}")
