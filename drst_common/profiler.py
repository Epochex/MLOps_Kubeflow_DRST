#!/usr/bin/env python3
# drst_common/profiler.py
from __future__ import annotations
import time
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any

from .metric_logger import log_metric

@contextmanager
def time_block(component: str, event: str, extra: Optional[Dict[str, Any]] = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = round((time.perf_counter() - t0) * 1000, 3)
        payload = {"runtime_ms": dt_ms}
        if extra: payload.update(extra)
        log_metric(component=component, event=event, **payload)
