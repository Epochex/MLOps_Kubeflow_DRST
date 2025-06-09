# shared/profiler.py
# ------------------------------------------------------------
# 轻量级计时器 / 资源统计，用 with Timer("name"): …
# ------------------------------------------------------------
import time, psutil
from contextlib import ContextDecorator
from .metric_logger import log_metric

_proc = psutil.Process()

class Timer(ContextDecorator):
    def __init__(self, name: str, component: str | None = None):
        self.name = name
        self.component = component or name.lower().split()[0]

    def __enter__(self):
        self.t0_wall = time.perf_counter()
        self.t0_cpu  = _proc.cpu_times()
        return self

    def __exit__(self, *exc):
        t1_wall = time.perf_counter()
        t1_cpu  = _proc.cpu_times()

        wall_ms = (t1_wall - self.t0_wall) * 1000
        cpu_ms  = ((t1_cpu.user + t1_cpu.system) -
                   (self.t0_cpu.user + self.t0_cpu.system)) * 1000
        log_metric(component=self.component,
                   event=f"{self.name}_runtime",
                   runtime_ms=round(wall_ms, 3),
                   cpu_time_ms=round(cpu_ms, 3))
        return False        # 不吞异常
