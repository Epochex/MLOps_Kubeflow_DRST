#!/usr/bin/env python3
# drst_common/kafka_io.py
# Unified Kafka connection/retry/producer/partition/sentinel broadcast utilities
from __future__ import annotations
import json
import time
from typing import Optional, Iterable, Dict, Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from .config import (
    KAFKA_SERVERS, AUTO_OFFSET_RESET, ENABLE_AUTO_COMMIT,
)

def create_consumer(topic: str, group_id: str,
                    max_retries: int = 10, backoff_s: float = 5.0,
                    value_deserializer=None) -> KafkaConsumer:
    if value_deserializer is None:
        value_deserializer = lambda m: json.loads(m.decode())
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            cons = KafkaConsumer(
                topic,
                bootstrap_servers=",".join(KAFKA_SERVERS),
                group_id=group_id,
                auto_offset_reset=AUTO_OFFSET_RESET,
                enable_auto_commit=ENABLE_AUTO_COMMIT,
                value_deserializer=value_deserializer,
                api_version_auto_timeout_ms=10000,
            )
            return cons
        except NoBrokersAvailable as e:
            last_err = e
            print(f"[kafka_io] brokers unavailable ({attempt}/{max_retries}), retry in {backoff_s}s …")
            time.sleep(backoff_s)
    raise RuntimeError(f"[kafka_io] Kafka unreachable after {max_retries} retries") from last_err

def create_producer(value_serializer=None) -> KafkaProducer:
    if value_serializer is None:
        value_serializer = lambda m: json.dumps(m).encode()
    return KafkaProducer(
        bootstrap_servers=",".join(KAFKA_SERVERS),
        value_serializer=value_serializer,
    )

def partitions_count(consumer: KafkaConsumer, topic: str) -> int:
    # Must join group before calling this (waiting ~1s after creation is safer)
    parts = consumer.partitions_for_topic(topic) or set()
    return len(parts)

def broadcast_sentinel(producer: KafkaProducer, topic: str,
                       payload: Dict[str, Any] | None = None,
                       partitions: Optional[Iterable[int]] = None) -> int:
    """Send a termination message to every partition of the topic (default {"producer_done": True}). Returns number of partitions sent to."""
    if payload is None:
        payload = {"producer_done": True}
    if partitions is None:
        partitions = create_consumer(topic, group_id="tmp_list_partitions").partitions_for_topic(topic) or []
    cnt = 0
    for p in partitions:
        producer.send(topic, partition=p, value=payload)
        cnt += 1
    producer.flush()
    return cnt
