#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drst_forecasting/explain.py

目标：
- 解释“当前线上使用的最新模型”（重训或先验基线都可）。
- 在 watch 模式下，持续监听“最新模型指针（MODEL_DIR/latest.json）变化”
  或“重训完成 flag（RESULT_DIR/retrain_done.flag）”——任一变化即触发解释。

输出（MinIO）：
- results/xai/xai_report_{ts}.md
- results/xai/perm_importance_{ts}.csv
- 出错/跳过则写 results/xai/xai_*.md 说明

依赖：
- drst_common.artefacts: read_latest / load_model_by_key / load_scaler / load_selected_feats
- monitor 写入的 results/latest_batch.npy (+ .columns.json) （若无则回退 datasets/combined.csv）

环境变量（可选）：
- FORECAST_XAI_WATCH=1/0
- POLL_INTERVAL_S=2
- FORECAST_LOOKBACK/HORIZON/SHAP_N/HIDDEN/LAYERS（兼容保留，不强制使用）
"""

from __future__ import annotations
import io
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, MODEL_DIR, RESULT_DIR, DATA_DIR, TARGET_COL
from drst_common.artefacts import (
    read_latest, load_model_by_key, load_scaler, load_selected_feats
)

# 可选 shap
try:
    import shap  # noqa: F401
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False


# ---------- S3 I/O ----------
def _read_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _obj_mtime(key: str) -> float:
    try:
        return s3.head_object(Bucket=BUCKET, Key=key)["LastModified"].timestamp()
    except Exception:
        return 0.0

def _write_md(name: str, text: str):
    key = f"{RESULT_DIR}/xai/{name}"
    save_bytes(key, text.encode("utf-8"), "text/markdown")

def _write_csv(name: str, df: pd.DataFrame):
    key = f"{RESULT_DIR}/xai/{name}"
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")


# ---------- 数据 ----------
def _load_latest_batch() -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    arr_key = f"{RESULT_DIR}/latest_batch.npy"
    col_key = f"{RESULT_DIR}/latest_batch.columns.json"
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=arr_key)["Body"].read()
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        cols = _read_json(col_key) or []
        cols = [str(c) for c in cols] if isinstance(cols, list) else []
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and len(cols) == arr.shape[1]:
            if TARGET_COL in cols:
                y_idx = cols.index(TARGET_COL)
                X = arr[:, [i for i,c in enumerate(cols) if c != TARGET_COL]].astype(np.float32)
                y = arr[:, y_idx].astype(np.float32)
                feat_cols = [c for c in cols if c != TARGET_COL]
                return X, y, feat_cols
        return None
    except Exception:
        return None

def _load_combined_selected() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    key = f"{DATA_DIR}/combined.csv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    df = pd.read_csv(io.BytesIO(obj))
    feats = load_selected_feats()
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"TARGET_COL '{TARGET_COL}' not found in {key}")
    df = df[feats + [TARGET_COL]].copy()
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    X = df[feats].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y, feats


# ---------- 简易解释：置换重要性 ----------
def _mae(y, p) -> float:
    y = np.asarray(y, np.float32); p = np.asarray(p, np.float32)
    return float(np.mean(np.abs(p - y)))

def _predict_torch(mdl, X: np.ndarray) -> np.ndarray:
    import torch
    mdl.eval()
    with torch.no_grad():
        out = mdl(torch.from_numpy(X).float()).cpu().numpy().ravel()
    return out

def _perm_importance(mdl, X: np.ndarray, y: np.ndarray, feat_names: List[str], repeat: int = 3) -> pd.DataFrame:
    base = _mae(y, _predict_torch(mdl, X))
    rows = []
    rng = np.random.default_rng(0)
    for j, name in enumerate(feat_names):
        incs = []
        for _ in range(max(1, int(repeat))):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            incs.append(_mae(y, _predict_torch(mdl, Xp)) - base)
        rows.append({"feature": name, "mae_increase": float(np.mean(incs))})
    df = pd.DataFrame(rows).sort_values("mae_increase", ascending=False).reset_index(drop=True)
    return df


# ---------- 生成报告 ----------
def _simple_report(title: str, lines: Dict[str, Any]) -> str:
    md = [f"# {title}", ""]
    for k, v in lines.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    return "\n".join(md)

def _align_to_model_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    d = X.shape[1]
    if d == in_dim: return X
    if d > in_dim:  return X[:, :in_dim]
    import numpy as np
    pad = np.zeros((X.shape[0], in_dim - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

def _explain_once_latest() -> bool:
    latest = read_latest()
    if not latest:
        _write_md("xai_latest_skipped.md", _simple_report("XAI Latest (Skipped)", {"reason": "no latest model"}))
        print("[xai] latest: no latest model; skip.", flush=True)
        return False

    model_key, metrics_key, _ts = latest
    try:
        mdl, _raw = load_model_by_key(model_key)
    except Exception as e:
        _write_md("xai_error_load.md", _simple_report("XAI Latest (Error)", {"reason": f"load {model_key} failed: {e}"}))
        print(f"[xai] load latest error: {e}", flush=True)
        return False

    batch = _load_latest_batch()
    if batch is not None:
        Xraw, y, feat_cols = batch
    else:
        Xraw, y, feat_cols = _load_combined_selected()

    sc = load_scaler()
    Xs = sc.transform(Xraw.astype(np.float32))
    in_dim = getattr(getattr(mdl, "net", [None])[0], "in_features", Xs.shape[1])
    Xs = _align_to_model_dim(Xs, in_dim)

    try:
        df_imp = _perm_importance(mdl, Xs, y, feat_cols, repeat=3)
    except Exception as e:
        df_imp = pd.DataFrame([{"feature": "N/A", "mae_increase": np.nan}])
        _write_md("xai_warn_perm.md", _simple_report("XAI Latest (Warning)", {"perm_importance": f"failed: {e}"}))

    mets = _read_json(metrics_key) or {}
    base = float(mets.get("baseline_mae", np.nan))
    newm = float(mets.get("mae", np.nan))
    gain = (float(mets.get("acc@0.15", 0.0)) - float(mets.get("baseline_acc@0.15", 0.0)))
    ts = int(time.time())

    _write_csv(f"perm_importance_{ts}.csv", df_imp)
    report = {
        "model_key": model_key,
        "metrics_key": metrics_key,
        "samples_for_xai": int(Xs.shape[0]),
        "features_used": len(feat_cols),
        "baseline_mae": base,
        "current_mae": newm,
        "acc_gain_pp@0.15": round(gain, 6),
        "top5_features_by_perm": ", ".join(df_imp.head(5)["feature"].tolist()) if not df_imp.empty else "(none)",
    }
    _write_md(f"xai_report_{ts}.md", _simple_report("XAI Report (Latest Model)", report))
    print(f"[xai] wrote report for latest model -> ts={ts}", flush=True)
    return True

def _startup_once_from_selected():
    sel_key = f"{MODEL_DIR}/forecast/selected.json"
    sel = _read_json(sel_key)
    if not sel:
        _write_md("xai_startup_skipped.md", _simple_report("XAI Startup (Skipped)", {
            "reason": "selected.json not found",
            "expected": f"s3://{BUCKET}/{sel_key}",
        }))
        print("[xai] startup: no selected.json, skip once.", flush=True)
        return
    info = {
        "selected_kind": sel.get("selected_kind"),
        "lookback": sel.get("lookback"),
        "horizon": sel.get("horizon"),
        "meta_key": sel.get("meta_key"),
        "artifact": sel.get("artifact"),
        "note": "Startup report for pre-selected model (before any retrain).",
    }
    ts = int(time.time())
    _write_md(f"xai_startup_{ts}.md", _simple_report("XAI Startup (Pre-selected Model)", info))
    print(f"[xai] startup explained pre-selected model -> ts={ts}", flush=True)


def main():
    WATCH  = os.getenv("FORECAST_XAI_WATCH", "1") in ("1","true","True","TRUE")
    POLL_S = int(os.getenv("POLL_INTERVAL_S", "2") or 2)

    # 启动兜底：若有先验 selected.json，先写一份“启动解释”
    _startup_once_from_selected()

    if not WATCH:
        _ = _explain_once_latest()
        return

    print(f"[xai] watcher start (poll={POLL_S}s). Follow latest.json / retrain flag.", flush=True)
    latest_ptr = f"{MODEL_DIR}/latest.json"
    retrain_flag = f"{RESULT_DIR}/retrain_done.flag"

    last_m = _obj_mtime(latest_ptr)
    last_f = _obj_mtime(retrain_flag)

    while True:
        try:
            m = _obj_mtime(latest_ptr)
            f = _obj_mtime(retrain_flag)
            if m > last_m or f > last_f:
                last_m, last_f = m, f
                _ = _explain_once_latest()
        except KeyboardInterrupt:
            break
        except Exception as e:
            _write_md("xai_runtime_error.md", _simple_report("XAI Watcher (Error)", {"reason": str(e)}))
            print(f"[xai] watcher error: {e}", flush=True)
        time.sleep(max(1, POLL_S))


if __name__ == "__main__":
    main()
