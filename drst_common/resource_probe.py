# -*- coding: utf-8 -*-
# drst_common/resource_probe.py
from __future__ import annotations

import atexit
import io
import os
import time
import threading
from typing import Dict, List, Optional, Tuple

import psutil

from .config import RESULT_DIR
from .minio_helper import save_bytes

# Global Mutable State (Thread-Safe)

_LOCK = threading.RLock()
_EXTRA: Dict[str, float] = {}             
_HOST_THREAD_STARTED = False               
_HOST_THREAD = None                        
_THREADS: Dict[str, "ProbeHandle"] = {}    
_ATEXIT_REGISTERED = False


SAMPLING_MS = int(os.getenv("RESOURCE_SAMPLING_MS", "500") or 500)    
FLUSH_EVERY = int(os.getenv("RESOURCE_FLUSH_EVERY", "200") or 100)     
ROUND_CPU = 6
ROUND_MEM = 3

# Should subprocesses be counted? (Recommended: Enabled)
PROBE_CHILDREN = os.getenv("PROBE_CHILDREN", "1").lower() in ("1", "true", "yes")

COLUMNS = ["ts", "cpu_pct", "vcpu", "rss_mb", "host_cpus", "js"]


def _now_ts() -> float:
    return time.time()

def update_extra(**kwargs) -> None:
    with _LOCK:
        for k, v in kwargs.items():
            if v is None:
                _EXTRA.pop(k, None)
            else:
                try:
                    _EXTRA[k] = float(v)
                except Exception:
                    pass

class _CsvBuffer:

    def __init__(self, s3_key: str):
        self.s3_key = s3_key
        base = os.path.basename(s3_key)
        self.local_path = os.path.join("/tmp", base)
        self._wrote_header = False
        self._rows_since_upload = 0
        try:
            if os.path.exists(self.local_path):
                os.remove(self.local_path)
        except Exception:
            pass

    def _open_and_write(self, line: str, header_if_needed: bool = False):
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        mode = "a" 
        with open(self.local_path, mode, encoding="utf-8") as f:
            if header_if_needed and not self._wrote_header:
                f.write(",".join(COLUMNS) + "\n")
                self._wrote_header = True
            f.write(line)

    @staticmethod
    def _fmt_row(row: Dict) -> str:
        vals = []
        for c in COLUMNS:
            v = row.get(c, "")
            if v is None:
                vals.append("")
            else:
                vals.append(str(v))
        return ",".join(vals) + "\n"

    def append(self, row: Dict):
        line = self._fmt_row(row)
        self._open_and_write(line, header_if_needed=True)
        self._rows_since_upload += 1
        if self._rows_since_upload >= FLUSH_EVERY:
            self._upload_all()
            self._rows_since_upload = 0

    def _upload_all(self):
        try:
            with open(self.local_path, "rb") as f:
                save_bytes(self.s3_key, f.read(), "text/csv")
        except FileNotFoundError:
            pass

    def flush(self):
        self._upload_all()


def _proc_cpu_mem_totals(proc: psutil.Process, include_children: bool) -> Tuple[float, int]:
    total_cpu = 0.0
    total_rss = 0
    try:
        with proc.oneshot():
            ct = proc.cpu_times()
            total_cpu += float(getattr(ct, "user", 0.0) + getattr(ct, "system", 0.0))
            total_rss += int(proc.memory_info().rss)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return total_cpu, total_rss
    except Exception:
        pass

    if include_children:
        try:
            for c in proc.children(recursive=True):
                try:
                    with c.oneshot():
                        ct = c.cpu_times()
                        total_cpu += float(getattr(ct, "user", 0.0) + getattr(ct, "system", 0.0))
                        total_rss += int(c.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except Exception:
                    continue
        except Exception:
            pass

    return total_cpu, total_rss

class ProbeHandle:
    def __init__(self, component: str, pid: Optional[int] = None):
        self.component = component
        self.pid = pid or os.getpid()
        self.proc = psutil.Process(self.pid)
        self.stop_evt = threading.Event()
        self.buf = _CsvBuffer(f"{RESULT_DIR}/{component}_resources.csv")
        self.thread = threading.Thread(target=self._run, name=f"probe-{component}", daemon=True)

        # Differential Baseline
        self._last_ts: Optional[float] = None
        self._last_cpu_s: Optional[float] = None

    def _prime(self):
        cpu_s, _ = _proc_cpu_mem_totals(self.proc, include_children=PROBE_CHILDREN)
        self._last_cpu_s = cpu_s
        self._last_ts = _now_ts()

    def _tick(self) -> Optional[Dict]:
        now = _now_ts()
        cpu_s, rss_bytes = _proc_cpu_mem_totals(self.proc, include_children=PROBE_CHILDREN)
        if self._last_cpu_s is None or self._last_ts is None:
            self._last_cpu_s = cpu_s
            self._last_ts = now
            return None

        dt = max(0.0, now - self._last_ts)
        d_cpu = max(0.0, cpu_s - self._last_cpu_s)

        self._last_cpu_s = cpu_s
        self._last_ts = now

        if dt <= 0.0:
            return None

        vcpu = d_cpu / dt                  
        cpu_pct = vcpu * 100.0             
        rss_mb = rss_bytes / (1024.0 * 1024.0)
        ncpu = max(1, psutil.cpu_count(logical=True) or 1)

        with _LOCK:
            extra = dict(_EXTRA)

        row = {
            "ts": round(now, 3),
            "cpu_pct": round(cpu_pct, ROUND_CPU),
            "vcpu": round(vcpu, ROUND_CPU),
            "rss_mb": round(rss_mb, ROUND_MEM),
            "host_cpus": ncpu,
            "js": (round(float(extra["js"]), 6) if "js" in extra else None),
        }
        return row

    def _run(self):
        self._prime()
        period_s = max(0.01, float(SAMPLING_MS) / 1000.0)
        while not self.stop_evt.is_set():
            time.sleep(period_s)
            try:
                row = self._tick()
                if row is not None:
                    self.buf.append(row)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break
            except Exception:
                continue
        self.buf.flush()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_evt.set()
        try:
            self.thread.join(timeout=2.5)
        except Exception:
            pass
        self.buf.flush()

    def set_extra(self, **kwargs):
        update_extra(**kwargs)

def _ensure_host_thread():
    global _HOST_THREAD_STARTED, _HOST_THREAD
    if _HOST_THREAD_STARTED:
        return

    class _HostThread(threading.Thread):
        def __init__(self):
            super().__init__(name="probe-host", daemon=True)
            self.stop_evt = threading.Event()
            self.buf = _CsvBuffer(f"{RESULT_DIR}/host_resources.csv")

        def run(self):
            try:
                psutil.cpu_percent(None)
            except Exception:
                pass

            period_s = max(0.01, float(SAMPLING_MS) / 1000.0)
            ncpu = max(1, psutil.cpu_count(logical=True) or 1)
            while not self.stop_evt.is_set():
                time.sleep(period_s)
                ts = _now_ts()
                try:
                    cpu_pct = float(psutil.cpu_percent(None))
                    vm = psutil.virtual_memory()
                    rss_mb = float(vm.used) / (1024.0 * 1024.0)
                    vcpu = cpu_pct * ncpu / 100.0
                except Exception:
                    cpu_pct, rss_mb, vcpu = 0.0, 0.0, 0.0

                with _LOCK:
                    extra = dict(_EXTRA)

                row = {
                    "ts": round(ts, 3),
                    "cpu_pct": round(cpu_pct, ROUND_CPU),
                    "vcpu": round(vcpu, ROUND_CPU),
                    "rss_mb": round(rss_mb, ROUND_MEM),
                    "host_cpus": ncpu,
                    "js": (round(float(extra["js"]), 6) if "js" in extra else None),
                }
                self.buf.append(row)
            self.buf.flush()

    _HOST_THREAD = _HostThread()
    _HOST_THREAD.start()
    _HOST_THREAD_STARTED = True

def _stop_all():
    global _HOST_THREAD
    with _LOCK:
        threads = list(_THREADS.items())
        _THREADS.clear()
    for _, h in threads:
        try: h.stop()
        except Exception: pass
    if _HOST_THREAD is not None:
        try:
            _HOST_THREAD.stop_evt.set()
            _HOST_THREAD.join(timeout=2.5)
            _HOST_THREAD.buf.flush()
        except Exception:
            pass


def start(component: str):

    global _ATEXIT_REGISTERED
    if not component or not isinstance(component, str):
        component = "proc"

    _ensure_host_thread()

    with _LOCK:
        if component in _THREADS:
            handle = _THREADS[component]
            def _stop(): handle.stop()
            return _stop

        handle = ProbeHandle(component).start()
        _THREADS[component] = handle

        if not _ATEXIT_REGISTERED:
            atexit.register(_stop_all)
            _ATEXIT_REGISTERED = True

    def _stop():
        with _LOCK:
            h = _THREADS.pop(component, None)
        if h:
            h.stop()

    return _stop
