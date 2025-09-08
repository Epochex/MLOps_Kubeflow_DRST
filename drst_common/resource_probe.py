#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的资源采样器：
- 进程级（当前组件容器内的主进程）：CPU、内存
- 整机级（host，总 CPU/内存，注意在容器内读取到的是节点视角的 /proc，基本可当“宿主机整体”）
- 在 stop() 时：
  - 上传本组件的进程资源 CSV：   results/<component>_resources.csv
  - 上传本组件的整机资源 CSV：   results/_host_<component>.csv（中间产物）
  - 自动聚合所有 infer_* 的总消耗：results/infer_total_resources.csv（按秒求和 vcpu + mem_mb）
  - 自动聚合所有 _host_*.csv 为： results/host_resources.csv（按秒聚合整机 CPU/内存；多 Pod 采样取均值）

列与精度（满足你的要求）：
- 进程资源：
  ts_iso, epoch_s, component, pod, cpu_pct(4位小数), cpu_ratio(4位小数), vcpu(4位小数), rss_kb(整数), mem_mb(3位小数), host_ncpu
- 整机资源（每条样本一条）：
  ts_iso, epoch_s, pod, host_cpu_pct(4位小数), host_vcpu(4位小数), host_mem_used_mb(3位小数), host_mem_total_mb(3位小数), host_mem_pct(4位小数), host_ncpu

可调环境变量（都有默认）：
- RESOURCE_INTERVAL_MS      采样周期，默认 500 ms
- RESOURCE_FLUSH_EVERY      内存中累计多少“行”后允许中间 flush（默认 0=只在 stop 时写一次）
- RESOURCE_AGGREGATE_ON_STOP stop 时是否做聚合，默认 1
- HOST_NCPU                 若从 CSV/系统无法识别，回退的 CPU 逻辑核数，默认 12
"""

from __future__ import annotations
import os
import io
import time
import math
import threading
from typing import Callable, List, Dict, Optional

import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from .minio_helper import s3, save_bytes
from .config import BUCKET, RESULT_DIR

# ====== 可配置项（环境变量） ======
_DEF_INTERVAL_MS  = int(os.getenv("RESOURCE_INTERVAL_MS", "500") or 500)
_DEF_FLUSH        = int(os.getenv("RESOURCE_FLUSH_EVERY", "0") or 0)    # 0 = 仅在 stop() 时写一次
_DO_AGG_ON_STOP   = (os.getenv("RESOURCE_AGGREGATE_ON_STOP", "1") in ("1", "true", "True"))
_FALLBACK_NCPU    = int(os.getenv("HOST_NCPU", "12") or 12)

# ====== 工具函数 ======
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _round(x: float, nd: int) -> float:
    try:
        return round(float(x), nd)
    except Exception:
        return float("nan")

def _to_epoch_s_from_any(x) -> float:
    """把可接受的时间（epoch/ISO）转成秒(float)；本模块内部只用现在时间，所以很少用到。"""
    if isinstance(x, (int, float)):
        return float(x)
    try:
        dt = pd.to_datetime(x, errors="coerce", utc=True)
        if pd.isna(dt):
            return float("nan")
        return float(dt.value) / 1e9
    except Exception:
        return float("nan")

def _list(prefix: str) -> List[Dict]:
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        return resp.get("Contents", []) or []
    except Exception:
        return []

def _read_csv_key(key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception:
        return None

def _write_csv_s3(key: str, df: pd.DataFrame):
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

# ====== 采样线程 ======
class _ProbeThread(threading.Thread):
    def __init__(self, component: str, interval_ms: int = _DEF_INTERVAL_MS, flush_every: int = _DEF_FLUSH):
        super().__init__(daemon=True)
        self.component = component
        self.interval_ms = max(50, int(interval_ms))
        self.flush_every = max(0, int(flush_every))
        self._stop = threading.Event()

        self.pod = os.getenv("HOSTNAME", component)
        self.proc = psutil.Process(os.getpid())

        # 逻辑核数
        self.host_ncpu = psutil.cpu_count(logical=True) or _FALLBACK_NCPU

        # 采样缓存
        self.rows_proc: List[Dict] = []
        self.rows_host: List[Dict] = []

        # 预热：让 cpu_percent 有基准
        try:
            self.proc.cpu_percent(None)
        except Exception:
            pass
        try:
            psutil.cpu_percent(None)
        except Exception:
            pass

    def run(self):
        last_flush = 0
        while not self._stop.is_set():
            ts_iso = _now_iso()
            epoch_s = time.time()

            # ---- 进程级 ----
            try:
                cpu_pct = float(self.proc.cpu_percent(None))  # 0~100*逻辑核（psutil 折算为全核百分比）
            except Exception:
                cpu_pct = float("nan")
            cpu_ratio = cpu_pct / 100.0
            vcpu = cpu_ratio * float(self.host_ncpu)

            try:
                rss_b = float(self.proc.memory_info().rss)  # 进程 RSS bytes
            except Exception:
                rss_b = float("nan")

            rss_kb = int(rss_b / 1024.0) if math.isfinite(rss_b) else 0
            mem_mb = (rss_b / (1024.0 * 1024.0)) if math.isfinite(rss_b) else float("nan")

            self.rows_proc.append({
                "ts_iso": ts_iso,
                "epoch_s": _round(epoch_s, 6),
                "component": self.component,
                "pod": self.pod,
                "cpu_pct": _round(cpu_pct, 4),
                "cpu_ratio": _round(cpu_ratio, 4),
                "vcpu": _round(vcpu, 4),
                "rss_kb": int(rss_kb),
                "mem_mb": _round(mem_mb, 3),
                "host_ncpu": int(self.host_ncpu),
            })

            # ---- 整机级 ----
            try:
                host_cpu_pct = float(psutil.cpu_percent(None))  # 全机 CPU 利用率（0~100）
            except Exception:
                host_cpu_pct = float("nan")
            host_vcpu = host_cpu_pct / 100.0 * float(self.host_ncpu)

            try:
                vm = psutil.virtual_memory()
                host_mem_total_mb = float(vm.total) / (1024.0 * 1024.0)
                host_mem_used_mb  = float(vm.used)  / (1024.0 * 1024.0)
                host_mem_pct = float(vm.percent)  # 0~100
            except Exception:
                host_mem_total_mb = host_mem_used_mb = host_mem_pct = float("nan")

            self.rows_host.append({
                "ts_iso": ts_iso,
                "epoch_s": _round(epoch_s, 6),
                "pod": self.pod,
                "host_cpu_pct": _round(host_cpu_pct, 4),
                "host_vcpu": _round(host_vcpu, 4),
                "host_mem_used_mb": _round(host_mem_used_mb, 3),
                "host_mem_total_mb": _round(host_mem_total_mb, 3),
                "host_mem_pct": _round(host_mem_pct, 4),
                "host_ncpu": int(self.host_ncpu),
            })

            # 可选中间 flush（默认关）
            if self.flush_every > 0 and (len(self.rows_proc) - last_flush) >= self.flush_every:
                try:
                    self._flush_partial()
                    last_flush = len(self.rows_proc)
                except Exception:
                    pass

            # sleep
            time.sleep(self.interval_ms / 1000.0)

    def stop(self):
        self._stop.set()

    # 中途 flush：将缓存写到临时对象（用于崩溃容错；不影响最终产物）
    def _flush_partial(self):
        if not self.rows_proc:
            return
        dfp = pd.DataFrame(self.rows_proc)
        dfh = pd.DataFrame(self.rows_host)
        _write_csv_s3(f"{RESULT_DIR}/._{self.component}_resources.partial.csv", dfp)
        _write_csv_s3(f"{RESULT_DIR}/._host_{self.component}.partial.csv", dfh)

    # 结束时写最终 CSV
    def dump_final(self):
        dfp = pd.DataFrame(self.rows_proc)
        dfh = pd.DataFrame(self.rows_host)
        if not dfp.empty:
            _write_csv_s3(f"{RESULT_DIR}/{self.component}_resources.csv", dfp)
        if not dfh.empty:
            _write_csv_s3(f"{RESULT_DIR}/_host_{self.component}.csv", dfh)


# ====== 聚合：infer 总和 + host 全局 ======
def _infer_vcpu(df: pd.DataFrame) -> pd.Series:
    if "vcpu" in df.columns:
        return pd.to_numeric(df["vcpu"], errors="coerce").astype(float)
    # 回退：根据 cpu_ratio 或 cpu_pct 推断（理论上不会走到，因为我们已经写了 vcpu）
    host_ncpu = None
    for c in ("host_ncpu", "ncpu", "num_cpu", "logical_cpus"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if not v.empty:
                host_ncpu = int(v.iloc[0]); break
    if not host_ncpu or host_ncpu <= 0:
        host_ncpu = _FALLBACK_NCPU
    if "cpu_ratio" in df.columns:
        return pd.to_numeric(df["cpu_ratio"], errors="coerce").astype(float) * float(host_ncpu)
    if "cpu_pct" in df.columns:
        return pd.to_numeric(df["cpu_pct"], errors="coerce").astype(float) / 100.0 * float(host_ncpu)
    return pd.Series(np.nan, index=df.index, dtype=float)

def _infer_mem_mb(df: pd.DataFrame) -> pd.Series:
    if "mem_mb" in df.columns:
        return pd.to_numeric(df["mem_mb"], errors="coerce").astype(float)
    if "rss_kb" in df.columns:
        kb = pd.to_numeric(df["rss_kb"], errors="coerce")
        return (kb / 1024.0).astype(float)
    if "rss_bytes" in df.columns:
        b = pd.to_numeric(df["rss_bytes"], errors="coerce")
        return (b / (1024.0 * 1024.0)).astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)

def _to_epoch_series(df: pd.DataFrame) -> pd.Series:
    if "epoch_s" in df.columns:
        return pd.to_numeric(df["epoch_s"], errors="coerce").astype(float)
    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        return (dt.view("int64") / 1e9)
    if "ts_iso" in df.columns:
        dt = pd.to_datetime(df["ts_iso"], errors="coerce", utc=True)
        return (dt.view("int64") / 1e9)
    return pd.Series(np.nan, index=df.index, dtype=float)

def _aggregate_infer_total():
    """合并所有 infer_*_resources.csv，按秒对齐并求和 vCPU + 内存(MB)。"""
    objs = _list(f"{RESULT_DIR}/")
    keys = [o["Key"] for o in objs if o["Key"].endswith("_resources.csv") and o["Key"].split("/")[-1].startswith("infer_")]
    if not keys:
        return
    frames = []
    for k in keys:
        df = _read_csv_key(k)
        if df is None or df.empty:
            continue
        one = pd.DataFrame()
        one["epoch_s"] = _to_epoch_series(df)
        one["vcpu"]    = _infer_vcpu(df)
        one["mem_mb"]  = _infer_mem_mb(df)
        one = one.dropna(subset=["epoch_s"])
        frames.append(one)
    if not frames:
        return

    big = pd.concat(frames, ignore_index=True)
    big["t_floor"] = np.floor(big["epoch_s"].astype(float)).astype("int64")

    agg = big.groupby("t_floor", as_index=False).agg(
        vcpu_sum=("vcpu", "sum"),
        mem_mb_sum=("mem_mb", "sum"),
    )
    # 精度
    agg["vcpu_sum"]   = agg["vcpu_sum"].astype(float).round(4)
    agg["mem_mb_sum"] = agg["mem_mb_sum"].astype(float).round(3)

    agg["ts_iso"] = pd.to_datetime(agg["t_floor"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agg.rename(columns={"t_floor": "epoch_s"}, inplace=True)
    agg.insert(2, "component", "infer_total")

    agg = agg[["ts_iso", "epoch_s", "component", "vcpu_sum", "mem_mb_sum"]]
    _write_csv_s3(f"{RESULT_DIR}/infer_total_resources.csv", agg)

def _aggregate_host_total():
    """合并所有 _host_*.csv，按秒对齐并对重复采样取均值，得到全局整机资源时间序列。"""
    objs = _list(f"{RESULT_DIR}/")
    keys = [o["Key"] for o in objs if o["Key"].split("/")[-1].startswith("_host_") and o["Key"].endswith(".csv")]
    if not keys:
        return

    frames = []
    for k in keys:
        df = _read_csv_key(k)
        if df is None or df.empty:
            continue
        one = pd.DataFrame()
        one["epoch_s"] = _to_epoch_series(df)
        for col in ["host_cpu_pct", "host_vcpu", "host_mem_used_mb", "host_mem_total_mb", "host_mem_pct", "host_ncpu"]:
            if col in df.columns:
                one[col] = pd.to_numeric(df[col], errors="coerce")
        one = one.dropna(subset=["epoch_s"])
        frames.append(one)
    if not frames:
        return

    big = pd.concat(frames, ignore_index=True)
    big["t_floor"] = np.floor(big["epoch_s"].astype(float)).astype("int64")

    agg = big.groupby("t_floor", as_index=False).agg({
        "host_cpu_pct": "mean",
        "host_vcpu": "mean",
        "host_mem_used_mb": "mean",
        "host_mem_total_mb": "max",   # 总内存基本固定，用 max/mean 均可
        "host_mem_pct": "mean",
        "epoch_s": "mean",            # 只是占位，不使用
        "host_ncpu": "max",
    })
    # 精度
    for c, nd in [("host_cpu_pct", 4), ("host_vcpu", 4), ("host_mem_used_mb", 3), ("host_mem_total_mb", 3), ("host_mem_pct", 4)]:
        agg[c] = agg[c].astype(float).round(nd)

    agg["ts_iso"] = pd.to_datetime(agg["t_floor"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agg.rename(columns={"t_floor": "epoch_s"}, inplace=True)
    agg = agg[["ts_iso", "epoch_s", "host_cpu_pct", "host_vcpu", "host_mem_used_mb", "host_mem_total_mb", "host_mem_pct", "host_ncpu"]]
    _write_csv_s3(f"{RESULT_DIR}/host_resources.csv", agg)

# ====== 外部接口 ======
def start(component: str, interval_ms: int = _DEF_INTERVAL_MS, flush_every: int = _DEF_FLUSH) -> Callable[[], None]:
    """
    启动资源采样线程，返回 stop() 回调。
      - component: 组件名（例如 'offline'、'monitor'、'retrain'、'infer_<pod>'）
      - interval_ms: 采样周期（毫秒）
      - flush_every: 缓存多少行做一次“中途备份 flush 到 S3”（0 = 只在 stop 时写最终 CSV）
    """
    th = _ProbeThread(component=component, interval_ms=interval_ms, flush_every=flush_every)
    th.start()

    def _stop():
        try:
            th.stop()
            th.join(timeout=2.0)
        except Exception:
            pass
        # 写最终 CSV
        try:
            th.dump_final()
        except Exception:
            pass

        # 聚合（一次即可，谁先/后执行都无所谓，最后一个写入者覆盖即可）
        if _DO_AGG_ON_STOP:
            try:
                _aggregate_infer_total()
            except Exception:
                pass
            try:
                _aggregate_host_total()
            except Exception:
                pass

    return _stop
