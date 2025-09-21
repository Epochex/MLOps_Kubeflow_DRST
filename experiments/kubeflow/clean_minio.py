#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean up MinIO: delete all objects in BUCKET except those under KEEP_PREFIXES.
To avoid the Content-MD5 requirement that MinIO enforces for DeleteObjects
(batch delete) when accessed via an HTTP gateway, we delete objects one by one
with delete_object, which is stable and reliable.

This script preserves objects under these “directory prefixes”:
- datasets/
- raw/
- datasets_pcm
"""

import sys
import time
import requests
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

# ======== Explicit config kept consistent with the upload script ========
MINIO_S3_ENDPOINT = "http://s3.45.149.207.13.nip.io:30080"
ACCESS_KEY = "minio"
SECRET_KEY = "minio123"
BUCKET     = "onvm-demo2"

# Prefixes to keep (directory semantics end with /). Any object whose key starts
# with one of these prefixes will be skipped. Non-slash-terminated entries are
# treated as exact keys and as parent prefixes for “key/...” children.
KEEP_PREFIXES = ["datasets/", "raw/", "datasets_pcm/"]
# =======================================================================

def check_health(endpoint: str) -> None:
    url = endpoint.rstrip("/") + "/minio/health/ready"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            print(f"[WARN] Health check {url} returned {r.status_code}; continuing to try S3 ...")
    except Exception as e:
        print(f"[WARN] Health check failed: {e}; continuing to try S3 ...")

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
    """
    Keep the object if its key starts with any element in KEEP_PREFIXES
    (common case: “directory” ending with /). For entries without a trailing /,
    keep both the exact key and anything under “key/”.
    """
    for p in KEEP_PREFIXES:
        if p.endswith("/"):
            if key.startswith(p):
                return True
        else:
            if key == p or key.startswith(p + "/"):
                return True
    return False

def main():
    print(f"[INFO] Connecting to MinIO: {MINIO_S3_ENDPOINT}")
    check_health(MINIO_S3_ENDPOINT)

    try:
        s3 = make_client()

        # Check bucket existence
        try:
            s3.head_bucket(Bucket=BUCKET)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            print(f"[ERR] head_bucket failed (bucket missing or not authorized): {code}")
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
                # Delete individually to avoid DeleteObjects Content-MD5 constraint
                try:
                    s3.delete_object(Bucket=BUCKET, Key=key)
                    deleted += 1
                except ClientError as e:
                    print(f"[WARN] Delete failed {key}: {e}")

                if deleted and deleted % 200 == 0:
                    dt = time.time() - t0
                    print(f"[clean] progress: scanned={scanned} deleted={deleted} kept={kept} ({deleted/max(1,dt):.1f} del/s)")

            token = resp.get("NextContinuationToken")
            if not resp.get("IsTruncated"):
                break

        print(f"[clean] completed: scanned={scanned}, deleted={deleted}, kept={kept}")
        return 0

    except EndpointConnectionError as e:
        print(f"[CONNECT FAIL] Cannot reach MinIO S3 endpoint: {MINIO_S3_ENDPOINT}\n{e}")
        return 2
    except Exception as e:
        print(f"[ERR] Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
