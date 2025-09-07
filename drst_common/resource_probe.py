#!/usr/bin/env python3
# drst_common/resource_probe.py
from __future__ import annotations
import time, threading, psutil
from typing import Callable
from .minio_helper import save_bytes
from .config import RESULT_DIR

def start(component: str, interval: float = 1.0) -> Callable[[], None]:
    """
    后台采样当前进程 CPU% 与内存 RSS/VMS，每秒一条（默认）。
    结束时写入 MinIO:  s3://<bucket>/{RESULT_DIR}/{component}_resources.csv

    用法:
        stop = start("retrain")
        try:
            ... 主逻辑 ...
        finally:
            stop()
    """
    proc = psutil.Process()
    stop_evt = threading.Event()
    rows = ["ts_epoch,cpu_percent,mem_rss_mb,mem_vms_mb\n"]

    def loop():
        proc.cpu_percent(None)  # prime
        while not stop_evt.wait(interval):
            cpu = proc.cpu_percent(None)
            mi = proc.memory_info()
            rows.append(
                f"{time.time():.3f},{cpu:.1f},{mi.rss/1048576:.1f},{mi.vms/1048576:.1f}\n"
            )

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def stop():
        try:
            stop_evt.set()
            th.join(timeout=2)
        finally:
            payload = "".join(rows).encode("utf-8")
            key = f"{RESULT_DIR}/{component}_resources.csv"
            save_bytes(key, payload, "text/csv")
    return stop
