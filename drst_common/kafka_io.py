#!/usr/bin/env python3
# drst_common/kafka_io.py
from __future__ import annotations
import os
import json
from typing import Iterable, Optional

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable  # 供外部引用

# ---- 尝试读取 config，但不强依赖其字段 ----
try:
    from . import config as _cfg
except Exception:
    _cfg = None

def _get(name: str, *alts, default=None):
    """
    读取优先级：
      1) drst_common.config 中的属性（按 name, *alts 顺序）
      2) 同名环境变量（按 name, *alts 顺序）
      3) default
    """
    for k in (name,) + tuple(alts):
        if _cfg is not None and hasattr(_cfg, k):
            return getattr(_cfg, k)
    for k in (name,) + tuple(alts):
        v = os.getenv(k)
        if v is not None:
            return v
    return default

# ---- 基础连接参数（默认值可直接用；也可被 config 或环境变量覆盖）----
BOOTSTRAP = _get(
    "KAFKA_SERVERS", "KAFKA_BOOTSTRAP_SERVERS", "BOOTSTRAP_SERVERS",
    default="kafka.default.svc.cluster.local:9092",
)
SECURITY_PROTOCOL = str(_get("KAFKA_SECURITY_PROTOCOL", default="PLAINTEXT") or "PLAINTEXT").upper()
SASL_MECHANISM = _get("KAFKA_SASL_MECHANISM", default=None)
SASL_USERNAME  = _get("KAFKA_SASL_USERNAME",  default=None)
SASL_PASSWORD  = _get("KAFKA_SASL_PASSWORD",  default=None)
SSL_CAFILE     = _get("KAFKA_SSL_CAFILE",     default=None)

# 这些是旧代码里可能被 import 的名字，这里给默认值，避免 ImportError
AUTO_OFFSET_RESET = str(_get("KAFKA_AUTO_OFFSET_RESET", "AUTO_OFFSET_RESET", default="earliest") or "earliest")
ACKS        = _get("KAFKA_ACKS", "ACKS", default="1")
LINGER_MS   = int(_get("KAFKA_LINGER_MS", "LINGER_MS", default=0) or 0)
RETRIES     = int(_get("KAFKA_RETRIES", "RETRIES", default=5) or 5)
MAX_IN_FLIGHT = int(_get("KAFKA_MAX_IN_FLIGHT", "MAX_IN_FLIGHT", default=1) or 1)

def _common_kwargs():
    kw = {
        "bootstrap_servers": [s.strip() for s in str(BOOTSTRAP).split(",") if s.strip()],
        "security_protocol": SECURITY_PROTOCOL,
    }
    if SECURITY_PROTOCOL.startswith("SASL"):
        if SASL_MECHANISM:
            kw["sasl_mechanism"] = SASL_MECHANISM
        if SASL_USERNAME is not None:
            kw["sasl_plain_username"] = SASL_USERNAME
        if SASL_PASSWORD is not None:
            kw["sasl_plain_password"] = SASL_PASSWORD
    if SECURITY_PROTOCOL.endswith("SSL") and SSL_CAFILE:
        kw["ssl_cafile"] = SSL_CAFILE
    return kw

def create_producer(**overrides) -> KafkaProducer:
    """
    创建 KafkaProducer，默认 JSON 序列化。
    """
    kwargs = _common_kwargs()
    kwargs.update({
        "acks": ACKS,
        "linger_ms": LINGER_MS,
        "retries": RETRIES,
        "max_in_flight_requests_per_connection": MAX_IN_FLIGHT,
        "value_serializer": lambda d: json.dumps(d).encode("utf-8"),
    })
    kwargs.update(overrides)
    return KafkaProducer(**kwargs)

def create_consumer(topic: Optional[str] = None,
                    group_id: Optional[str] = None,
                    **overrides) -> KafkaConsumer:
    """
    创建 KafkaConsumer，默认 auto_offset_reset=earliest，JSON 反序列化。
    """
    kwargs = _common_kwargs()
    kwargs.update({
        "group_id": group_id,
        "auto_offset_reset": AUTO_OFFSET_RESET,  # "earliest"/"latest"
        "enable_auto_commit": True,
        "value_deserializer": lambda b: json.loads(b.decode("utf-8")) if isinstance(b, (bytes, bytearray)) else b,
    })
    kwargs.update(overrides)
    c = KafkaConsumer(**kwargs)
    if topic:
        c.subscribe([topic])
    return c

def broadcast_sentinel(producer: KafkaProducer, topic: str, payload: dict,
                       partitions: Optional[Iterable[int]] = None) -> None:
    """
    给每个分区各发一个“收尾标记”；若未提供 partitions，就发一条。
    """
    if partitions:
        for p in partitions:
            producer.send(topic, payload, partition=int(p))
    else:
        producer.send(topic, payload)
    producer.flush()

def partitions_count(consumer: KafkaConsumer, topic: str) -> int:
    """
    返回 topic 的分区数量；取不到则返回 0。
    """
    try:
        parts = consumer.partitions_for_topic(topic) or set()
        return len(parts)
    except Exception:
        return 0
