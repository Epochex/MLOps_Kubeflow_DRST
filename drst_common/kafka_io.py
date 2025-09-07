#!/usr/bin/env python3
# drst_common/kafka_io.py
from __future__ import annotations
import os, json, atexit, time
from typing import List, Optional, Dict

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# ---------------------------
# Defaults (可被 env 覆盖)
# ---------------------------
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")  # or SASL_PLAINTEXT / SASL_SSL
SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
SSL_CAFILE    = os.getenv("KAFKA_SSL_CAFILE", "")

AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")
GROUP_ID_DEFAULT  = os.getenv("KAFKA_GROUP_ID", "drst-consumers")
CLIENT_ID_PREFIX  = os.getenv("KAFKA_CLIENT_ID_PREFIX", "drst")

# Sentinel headers
SENTINEL_HEADER_KEY = "drst-sentinel"
RUN_ID_HEADER_KEY   = "drst-run-id"
SENTINEL_HEADER_VAL = b"1"
SENTINEL_FALLBACK_VALUE = b"__DRST_EOF__"   # 兼容旧值

def _security_kwargs() -> Dict:
    kw = {"bootstrap_servers": BOOTSTRAP}
    if SECURITY_PROTOCOL != "PLAINTEXT":
        kw.update({
            "security_protocol": SECURITY_PROTOCOL,
            "sasl_mechanism": SASL_MECHANISM,
            "sasl_plain_username": SASL_USERNAME,
            "sasl_plain_password": SASL_PASSWORD,
        })
        if SECURITY_PROTOCOL.endswith("SSL") and SSL_CAFILE:
            kw["ssl_cafile"] = SSL_CAFILE
    return kw

def create_producer(client_id: Optional[str] = None) -> KafkaProducer:
    kw = _security_kwargs()
    kw.update({
        "client_id": client_id or f"{CLIENT_ID_PREFIX}-producer",
        "acks": "all",
        "linger_ms": int(os.getenv("KAFKA_LINGER_MS", "10")),
        "retries": int(os.getenv("KAFKA_RETRIES", "3")),
        "max_in_flight_requests_per_connection": 1,
        "value_serializer": lambda d: json.dumps(d).encode("utf-8") if not isinstance(d, (bytes, bytearray)) else d,
    })
    producer = KafkaProducer(**kw)

    def _close():
        try:
            producer.flush(int(os.getenv("KAFKA_CLOSE_FLUSH_SECS", "10")))
            producer.close(int(os.getenv("KAFKA_CLOSE_TIMEOUT_SECS", "10")))
        except Exception:
            pass
    atexit.register(_close)
    return producer

def create_consumer(
    topic: str,
    group_id: Optional[str] = None,
    client_id: Optional[str] = None,
    start_from_end: bool = True,
    enable_auto_commit: bool = True,
) -> KafkaConsumer:
    kw = _security_kwargs()
    kw.update({
        "group_id": group_id or GROUP_ID_DEFAULT,
        "client_id": client_id or f"{CLIENT_ID_PREFIX}-consumer",
        "auto_offset_reset": AUTO_OFFSET_RESET,
        "enable_auto_commit": enable_auto_commit,
        "value_deserializer": lambda b: json.loads(b) if (b and b != SENTINEL_FALLBACK_VALUE) else {},
        "consumer_timeout_ms": 1000,
        "max_poll_records": int(os.getenv("KAFKA_MAX_POLL_RECORDS", "500")),
    })
    consumer = KafkaConsumer(**kw)
    consumer.subscribe([topic])

    # 等待分区分配
    t0 = time.time()
    while not consumer.assignment():
        consumer.poll(200)
        if time.time() - t0 > 10:
            break

    # 从当前末尾开始，避免吃历史 backlog
    if start_from_end and consumer.assignment():
        for tp in list(consumer.assignment()):
            consumer.seek_to_end(tp)
    return consumer

def partitions_for_topic(topic: str) -> List[int]:
    tmp = KafkaConsumer(**_security_kwargs())
    try:
        parts = tmp.partitions_for_topic(topic) or set()
        return sorted(list(parts))
    finally:
        try:
            tmp.close()
        except Exception:
            pass

# 兼容老代码：接受 consumer 或 topic 字符串
def partitions_count(consumer_or_topic, topic: Optional[str] = None) -> int:
    if isinstance(consumer_or_topic, KafkaConsumer):
        t = topic or next(iter(consumer_or_topic.subscription() or []), None)
        if not t:
            return 0
        ps = consumer_or_topic.partitions_for_topic(t) or set()
        return len(ps)
    return len(partitions_for_topic(str(consumer_or_topic)))

def is_sentinel(record, run_id: Optional[str] = None) -> bool:
    try:
        headers = dict(record.headers or [])
        if headers.get(SENTINEL_HEADER_KEY, b"") == SENTINEL_HEADER_VAL:
            if run_id is None:
                return True
            return headers.get(RUN_ID_HEADER_KEY, b"").decode("utf-8", "ignore") == run_id
    except Exception:
        pass
    try:
        if record.value == SENTINEL_FALLBACK_VALUE:
            return True
    except Exception:
        pass
    try:
        obj = json.loads(record.value)  # 如果 value 是 JSON
        if obj.get("_sentinel", False):
            return (run_id is None) or (obj.get("run_id") == run_id)
    except Exception:
        pass
    return False

def broadcast_sentinel(producer: KafkaProducer, topic: str, run_id: Optional[str] = None, partitions: Optional[List[int]] = None) -> int:
    parts = partitions or partitions_for_topic(topic)
    sent = 0
    headers = [(SENTINEL_HEADER_KEY, SENTINEL_HEADER_VAL)]
    if run_id:
        headers.append((RUN_ID_HEADER_KEY, run_id.encode("utf-8")))
    for p in parts:
        producer.send(topic, value=SENTINEL_FALLBACK_VALUE, partition=p, headers=headers)
        sent += 1
    producer.flush(int(os.getenv("KAFKA_CLOSE_FLUSH_SECS", "10")))
    return sent
