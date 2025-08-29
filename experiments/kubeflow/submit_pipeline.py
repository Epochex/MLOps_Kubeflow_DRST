#!/usr/bin/env python3
from __future__ import annotations
import os, time, importlib.util, types
from kfp import Client, compiler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE_FILE = os.path.join(ROOT, "experiments", "kubeflow", "pipeline.py")
PKG = os.getenv("KFP_PACKAGE", os.path.join(ROOT, "experiments", "kubeflow", "drift_demo_v2.yaml"))

def _load_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("drst_pipeline_mod", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

mod = _load_module(PIPELINE_FILE)
compiler.Compiler().compile(mod.drift_stream, package_path=PKG)
print(f" compiled -> {PKG}")

host = os.getenv("KFP_HOST", "").strip()  #  http://<IP>:30080/pipeline
ns   = os.getenv("KFP_NAMESPACE", "kubeflow-user-example-com")
expn = os.getenv("KFP_EXPERIMENT", "drift-stream-v2-exp")
run  = f"drift-stream-v2-{int(time.time())}"

client = Client(host=host) if host else Client()

exp = client.create_experiment(name=expn, namespace=ns)
exp_id = getattr(exp, "experiment_id", None) or getattr(exp, "id", None)

res = client.run_pipeline(
    experiment_id=exp_id,
    job_name=run,
    pipeline_package_path=PKG,
    enable_caching=False,
)
run_id = getattr(res, "run_id", None) or getattr(res, "id", None)
print(" submitted.")
base = client._get_url_prefix() if hasattr(client, "_get_url_prefix") else (host or "")
print("Run URL:", f"{base}/#/runs/details/{run_id}")
