#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量进程/主机资源采样器（改进版）：
- 每个组件进程内启动 1 个后台线程，按固定周期采样；
- 进程 vCPU 计算：使用“进程 CPU 时间(usr+sys)增量 / 墙钟时间增量”，避免 cpu_percent 的取整误差；
- 写入 S3:
    * 组件级：results/{component}_resources.csv
    * （仅 monitor 或显式开启）整机级：results/host_resources.csv
- 字段（组件级）：ts, elapsed_s, cpu_pct, vcpu, rss_mb, js
  * cpu_pct：进程 CPU 百分比 = vcpu * 100（保留 6 位小数）
  * vcpu   ：等价 vCPU（保留 6 位小数）
  * rss_mb ：常驻内存 MB（保留 3 位小数）
  * js     ：当前 JS 值（仅 monitor 更新；其他组件为空）
- 字段（整机级）：ts, elapsed_s, cpu_pct, vcpu, rss_mb
  * vcpu   ：整机 vCPU 等价 = (总 CPU 利用率% / 100) * 逻辑核数（保留 6 位小数）
"""
from __future__ import annotations

import io
import os
import time
import threading
from typing import Dict, Optional

import psutil

from .minio_helper import save_bytes
from .config import RESULT_DIR, BUCKET

# 采样周期 & 批量 flush 行数（可用环境变量覆盖）
_SAMPLING_MS = int(os.getenv("RESOURCE_SAMPLING_MS", "500") or 500)
_FLUSH_EVERY = int(os.getenv("RESOURCE_FLUSH_EVERY", "100") or 100)

# 仅 monitor 组件记录整机资源（也可用 RESOURCE_HOST_GLOBAL=1 强制启用）
_FORCE_HOST_GLOBAL = os.getenv("RESOURCE_HOST_GLOBAL", "").lower() in ("1", "true", "yes")

# —— 进程级全局状态（单进程仅一个探针）——
_probe_thread: Optional[threading.Thread] = None
_probe_stop = threading.Event()
_probe_lock = threading.RLock()
_started_at: float = 0.0
_component_name: str = "component"

# 写入缓存（本地文件路径；统一覆盖上传）
_local_proc_csv: Optional[str] = None
_local_host_csv: Optional[str] = None

# 采样计数
_rows_since_flush = 0

# monitor 用：外部可随时更新的“额外字段”（例如 js）
_extra_lock = threading.Lock()
_extra_fields: Dict[str, float] = {}   # e.g. {"js": 0.00321}

# 进程 CPU 时间差分基线
_proc_obj: Optional[psutil.Process] = None
_prev_proc_cpu_s: Optional[float] = None  # user+system（秒）
_prev_wall_s: Optional[float] = None      # time.time()（秒）

# CPU 预热（host）
_psutil_host_primer_done = False


def update_extra(**kwargs):
    """由组件在运行时更新额外字段（例如 monitor 在计算 JS 后调用 update_extra(js=...））"""
    with _extra_lock:
        for k, v in kwargs.items():
            try:
                _extra_fields[str(k)] = float(v)
            except Exception:
                # 忽略不可转为 float 的值
                pass


# 兼容旧名
set_extra = update_extra


def _ensure_local(path: str, header: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")


def _upload_file(local_path: str, key: str) -> None:
    with open(local_path, "rb") as f:
        save_bytes(key, f.read(), "text/csv")


def _read_proc_cpu_times_s(p: psutil.Process) -> float:
    """返回进程 usr+sys 的累计 CPU 秒数（忽略 children，以免把子进程算进来）"""
    try:
        t = p.cpu_times()
        # 某些平台无 iowait 字段；这里只取 user+system
        return float(getattr(t, "user", 0.0) + getattr(t, "system", 0.0))
    except Exception:
        return 0.0


def _proc_row() -> str:
    """采集当前进程资源（差分法），返回一行 CSV 字符串（不含换行）"""
    global _prev_proc_cpu_s, _prev_wall_s

    now_wall = time.time()
    rss_mb = 0.0
    vcpu = 0.0

    try:
        p = _proc_obj or psutil.Process()
        rss_mb = float(p.memory_info().rss) / (1024.0 * 1024.0)
        cur_cpu = _read_proc_cpu_times_s(p)
    except Exception:
        cur_cpu = 0.0

    if _prev_proc_cpu_s is None or _prev_wall_s is None:
        # 首次：建立基线，不产出 CPU 使用率（保持 vcpu=0）
        _prev_proc_cpu_s = cur_cpu
        _prev_wall_s = now_wall
    else:
        dt = now_wall - _prev_wall_s
        dcpu = cur_cpu - _prev_proc_cpu_s
        if dt > 0 and dcpu >= 0:
            vcpu = dcpu / dt  # “核”的占用（可能 >1，多线程时）
        _prev_proc_cpu_s = cur_cpu
        _prev_wall_s = now_wall

    cpu_pct = vcpu * 100.0

    with _extra_lock:
        js = _extra_fields.get("js", None)

    elapsed = now_wall - _started_at
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_wall))
    js_str = (f"{js:.6f}" if js is not None else "")
    # 提高精度到 6 位小数（cpu_pct / vcpu），rss_mb 保持 3 位
    return f"{ts},{elapsed:.3f},{cpu_pct:.6f},{vcpu:.6f},{rss_mb:.3f},{js_str}"


def _host_row() -> str:
    """整机资源（用于 monitor 组件）；vcpu 为“总逻辑核等价”：n_cpu * avg_util% / 100"""
    global _psutil_host_primer_done

    # 让 psutil 内部先有一个基线，避免第一次 0.0
    if not _psutil_host_primer_done:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        _psutil_host_primer_done = True

    try:
        cpu_pct_host = float(psutil.cpu_percent(interval=None))
    except Exception:
        cpu_pct_host = 0.0

    ncpu = psutil.cpu_count(logical=True) or 1
    vcpu_host = (cpu_pct_host / 100.0) * ncpu
    mem = psutil.virtual_memory()
    used_mb = float(mem.used) / (1024.0 * 1024.0)

    now = time.time()
    elapsed = now - _started_at
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    # host 的 cpu 相关也升到 6 位小数
    return f"{ts},{elapsed:.3f},{cpu_pct_host:.6f},{vcpu_host:.6f},{used_mb:.3f}"


def _sampler_loop(record_host: bool) -> None:
    global _rows_since_flush
    proc_header = "ts,elapsed_s,cpu_pct,vcpu,rss_mb,js"
    host_header = "ts,elapsed_s,cpu_pct,vcpu,rss_mb"

    # 建立初始基线
    try:
        p = _proc_obj or psutil.Process()
        _ = p.memory_info()  # 也触发一下，确保存在
        # 初始化“进程 CPU 秒数 / 墙钟秒数”的基线
        cpu0 = _read_proc_cpu_times_s(p)
    except Exception:
        cpu0 = 0.0

    # 记录首次基线
    global _prev_proc_cpu_s, _prev_wall_s
    if _prev_proc_cpu_s is None:
        _prev_proc_cpu_s = cpu0
    if _prev_wall_s is None:
        _prev_wall_s = time.time()

    # host primer（为避免第一次 0）
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass

    while not _probe_stop.is_set():
        try:
            # 进程行
            row = _proc_row()
            with open(_local_proc_csv, "a", encoding="utf-8") as f:
                f.write(row + "\n")

            # 主机行（仅 monitor 或显式启用）
            if record_host and _local_host_csv:
                row_h = _host_row()
                with open(_local_host_csv, "a", encoding="utf-8") as f:
                    f.write(row_h + "\n")

            _rows_since_flush += 1
            if _rows_since_flush >= _FLUSH_EVERY:
                _rows_since_flush = 0
                # 上传至 S3
                _upload_file(_local_proc_csv, f"{RESULT_DIR}/{_component_name}_resources.csv")
                if record_host and _local_host_csv:
                    _upload_file(_local_host_csv, f"{RESULT_DIR}/host_resources.csv")
        except Exception as e:
            # 降级为日志；不中断采样
            print(f"[resource_probe:{_component_name}] sample/flush error: {e}", flush=True)

        time.sleep(max(0.001, _SAMPLING_MS / 1000.0))

    # 线程退出前做一次最终 flush
    try:
        _upload_file(_local_proc_csv, f"{RESULT_DIR}/{_component_name}_resources.csv")
        if record_host and _local_host_csv:
            _upload_file(_local_host_csv, f"{RESULT_DIR}/host_resources.csv")
    except Exception as e:
        print(f"[resource_probe:{_component_name}] final flush error: {e}", flush=True)


def start(component_name: str) -> callable:
    """
    启动采样线程。
    - component_name：用于生成 S3 路径的友好名（例如 'offline' / 'monitor' / 'producer' / 'retrain' / 'infer1' / 'infer2' / 'infer3'）
    返回：stop() 可调用，停止采样并做最终 flush。
    """
    global _probe_thread, _started_at, _component_name, _local_proc_csv, _local_host_csv, _proc_obj
    with _probe_lock:
        if _probe_thread is not None:
            # 已经启动过，返回 no-op
            def _noop(): 
                pass
            return _noop

        _started_at = time.time()
        _component_name = component_name.strip().replace("/", "_")
        _proc_obj = psutil.Process()

        # 本地临时文件（写完再整体上传）
        _local_proc_csv = f"/tmp/{_component_name}_resources.csv"
        _ensure_local(_local_proc_csv, "ts,elapsed_s,cpu_pct,vcpu,rss_mb,js")

        # 是否记录 host：默认仅 monitor；也可通过 RESOURCE_HOST_GLOBAL=1 强制
        record_host = _FORCE_HOST_GLOBAL or (_component_name.lower() == "monitor")
        if record_host:
            _local_host_csv = "/tmp/host_resources.csv"
            _ensure_local(_local_host_csv, "ts,elapsed_s,cpu_pct,vcpu,rss_mb")
        else:
            _local_host_csv = None

        _probe_stop.clear()
        _probe_thread = threading.Thread(target=_sampler_loop, args=(record_host,), daemon=True)
        _probe_thread.start()

        def _stop():
            global _probe_thread
            with _probe_lock:
                if _probe_thread is None:
                    return
                _probe_stop.set()
                try:
                    _probe_thread.join(timeout=5)
                except Exception:
                    pass
                _probe_thread = None

        return _stop
