#!/usr/bin/env python3
# compile -> YAML, then submit via v1 run_pipeline

import os
import time
import importlib.util
import types
from kfp import Client, compiler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE_FILE = os.path.join(ROOT, "experiments", "kubeflow", "pipeline.py")
PKG = os.getenv("KFP_PACKAGE", os.path.join(ROOT, "experiments", "kubeflow", "drift_stream_v2.yaml"))

KFP_HOST = os.getenv("KFP_HOST", "").strip()  # e.g. http://<IP>:30080/pipeline
NS       = os.getenv("KFP_NAMESPACE", "kubeflow-user-example-com")
EXPN     = os.getenv("KFP_EXPERIMENT", "drift-stream-v2-exp")
RUN_NAME = f"DRST-FluxSys-v2-{int(time.time())}"

def _load_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("drst_pipeline_mod", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _pick_pipeline_func(mod: types.ModuleType):
    for name in ("drift_stream_v2_pipeline", "drift_stream"):
        fn = getattr(mod, name, None)
        if fn:
            return fn
    raise RuntimeError("找不到 pipeline 函数（期待 drift_stream_v2_pipeline 或 drift_stream）。")

# 1) 编译为 YAML
mod = _load_module(PIPELINE_FILE)
fn  = _pick_pipeline_func(mod)
compiler.Compiler().compile(pipeline_func=fn, package_path=PKG)
print(f" compiled -> {PKG}")

# 2) 提交 YAML（v1 客户端 run_pipeline），不传 params，全部走 config.py 默认
client = Client(host=KFP_HOST) if KFP_HOST else Client()
exp = client.create_experiment(name=EXPN, namespace=NS)
exp_id = getattr(exp, "experiment_id", None) or getattr(exp, "id", None)

res = client.run_pipeline(
    experiment_id=exp_id,
    job_name=RUN_NAME,
    pipeline_package_path=PKG,
    enable_caching=False,
)
run_id = getattr(res, "run_id", None) or getattr(res, "id", None)
base = KFP_HOST or (client._get_url_prefix() if hasattr(client, "_get_url_prefix") else "")
print(" submitted.")
print(f"   experiment: {EXPN}")
print(f"   run_id    : {run_id}")
if base:
    print(f"   run url   : {base}/#/runs/details/{run_id}")
