#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理 MinIO：删除 BUCKET 中除 KEEP_PREFIXES 下的对象外的所有对象。
为绕过 MinIO 在 HTTP 网关下对 DeleteObjects(批量删除) 的 Content-MD5 强制要求，
改为逐个 delete_object，稳定可靠。
"""

import sys
import time
import requests
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

# ======== 与上传脚本保持一致的显式配置 ========
MINIO_S3_ENDPOINT = "http://s3.45.149.207.13.nip.io:30080"
ACCESS_KEY = "minio"
SECRET_KEY = "minio123"
BUCKET     = "onvm-demo2"
KEEP_PREFIXES = ["datasets/"]     # 需要保留的前缀（目录语义以 / 结尾）
# ============================================

def check_health(endpoint: str) -> None:
    url = endpoint.rstrip("/") + "/minio/health/ready"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            print(f"[WARN] 健康检查 {url} 返回 {r.status_code}，继续尝试连接 S3 ...")
    except Exception as e:
        print(f"[WARN] 健康检查失败：{e}，继续尝试连接 S3 ...")

def make_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_S3_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=3,
            read_timeout=30,
        ),
    )

def should_keep(key: str) -> bool:
    for p in KEEP_PREFIXES:
        if p.endswith("/"):
            if key.startswith(p):
                return True
        else:
            if key == p or key.startswith(p + "/"):
                return True
    return False

def main():
    print(f"[INFO] 连接 MinIO：{MINIO_S3_ENDPOINT}")
    check_health(MINIO_S3_ENDPOINT)

    try:
        s3 = make_client()

        # 桶存在性检查
        try:
            s3.head_bucket(Bucket=BUCKET)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            print(f"[ERR] head_bucket 失败（桶不存在或无权访问）：{code}")
            return 3

        kept = deleted = scanned = 0
        token = None
        t0 = time.time()

        while True:
            if token:
                resp = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=1000, ContinuationToken=token)
            else:
                resp = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=1000)

            objs = resp.get("Contents", []) or []
            if not objs:
                break

            for o in objs:
                key = o["Key"]
                scanned += 1
                if should_keep(key):
                    kept += 1
                    continue
                # 单个删除，避免 DeleteObjects 的 Content-MD5 限制
                try:
                    s3.delete_object(Bucket=BUCKET, Key=key)
                    deleted += 1
                except ClientError as e:
                    print(f"[WARN] 删除失败 {key}: {e}")

                if deleted and deleted % 200 == 0:
                    dt = time.time() - t0
                    print(f"[clean] progress: scanned={scanned} deleted={deleted} kept={kept} ({deleted/max(1,dt):.1f} del/s)")

            token = resp.get("NextContinuationToken")
            if not resp.get("IsTruncated"):
                break

        print(f"[clean] completed: scanned={scanned}, deleted={deleted}, kept={kept}")
        return 0

    except EndpointConnectionError as e:
        print(f"[连接失败] 无法访问 MinIO S3 端点：{MINIO_S3_ENDPOINT}\n{e}")
        return 2
    except Exception as e:
        print(f"[ERR] 未处理异常：{e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
