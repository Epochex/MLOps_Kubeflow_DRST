#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import requests
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

# ======== 可修改的配置（尽量写死，避免环境注入）========
# # 方案A：本机直连 NodePort（推荐，最省事）
# MINIO_S3_ENDPOINT = "http://127.0.0.1:30900"   # 或 "http://45.149.207.13:30900"

# 如需经 Istio 域名（你之后想换回域名），把上面一行改成：
MINIO_S3_ENDPOINT = "http://s3.45.149.207.13.nip.io:30080"

ACCESS_KEY = "minio"
SECRET_KEY = "minio123"
BUCKET     = "onvm-demo2"
# 上传到 bucket 下的前缀
PREFIX     = "datasets/"

# 要上传的本地文件 => 目标对象键
FILES = {
    "datasets/combined.csv":                          PREFIX + "combined.csv",
    "datasets/random_rates.csv":                      PREFIX + "random_rates.csv",
    "datasets/resource_stimulus_global_A-B-C_modified.csv": PREFIX + "resource_stimulus_global_A-B-C_modified.csv",
    "datasets/intervention_global.csv":               PREFIX + "intervention_global.csv",
}
# =====================================================

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
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

def ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[OK] bucket 已存在：{bucket}")
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == 404 or e.response.get("Error", {}).get("Code") in ("404", "NoSuchBucket", "NotFound"):
            print(f"[INFO] 创建 bucket：{bucket}")
            s3.create_bucket(Bucket=bucket)
        else:
            raise

def upload_one(s3, src: str, key: str):
    if not os.path.exists(src):
        print(f"[SKIP] 找不到文件：{src}")
        return False
    print(f"[UP] {src}  ->  s3://{BUCKET}/{key}")
    extra = {"ContentType": "text/csv"}
    s3.upload_file(src, BUCKET, key, ExtraArgs=extra)
    return True

def main():
    print(f"[INFO] 连接 MinIO：{MINIO_S3_ENDPOINT}")
    check_health(MINIO_S3_ENDPOINT)

    try:
        s3 = make_client()
        # 简单探活：列 bucket（MinIO 可能返回 403/500，此处只要能连通即可继续）
        try:
            s3.list_buckets()
        except ClientError as e:
            # 某些老版本偶发 500，这里不阻断后续流程
            print(f"[WARN] list_buckets 返回异常但端口可达：{e}")

        ensure_bucket(s3, BUCKET)

        ok_cnt = 0
        for src, key in FILES.items():
            if upload_one(s3, src, key):
                ok_cnt += 1

        print(f"\n✅ 完成：上传 {ok_cnt}/{len(FILES)} 个文件到 s3://{BUCKET}/{PREFIX}")
        print("   读取路径保持与旧项目一致，例如：")
        for src, key in FILES.items():
            print(f"   - s3://{BUCKET}/{key}")

    except EndpointConnectionError as e:
        print(f"\n[连接失败] 无法访问 MinIO S3 端点： {MINIO_S3_ENDPOINT}\n{e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERR] 未处理异常：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
