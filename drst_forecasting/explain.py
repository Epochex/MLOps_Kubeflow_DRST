#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import io
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import s3, save_bytes
from drst_common.config import BUCKET, MODEL_DIR, RESULT_DIR, DATA_DIR, TARGET_COL
from drst_common.artefacts import read_latest, load_model_by_key, load_scaler, load_selected_feats


# -----------------------
# Utilities: paths and S3 I/O
# -----------------------

def _abs_key(key: str) -> str:
    """Automatically add the models/ prefix for relative keys."""
    k = str(key)
    return k if ("/" in k) else f"{MODEL_DIR}/{k}"

def _read_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=_abs_key(key))["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _write_md(name: str, text: str):
    key = f"{RESULT_DIR}/xai/{name}"
    save_bytes(key, text.encode("utf-8"), "text/markdown")

def _write_csv(name: str, df: pd.DataFrame):
    key = f"{RESULT_DIR}/xai/{name}"
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

def _simple_report(title: str, lines: Dict[str, Any]) -> str:
    md = [f"# {title}", ""]
    for k, v in lines.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    return "\n".join(md)


# -----------------------
# Data loading
# -----------------------

def _load_latest_batch() -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Prefer monitor outputs: latest_batch.npy + latest_batch.columns.json."""
    arr_key = f"{RESULT_DIR}/latest_batch.npy"
    col_key = f"{RESULT_DIR}/latest_batch.columns.json"
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=arr_key)["Body"].read()
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        cols = _read_json(col_key) or []
        cols = [str(c) for c in cols] if isinstance(cols, list) else []
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and len(cols) == arr.shape[1] and TARGET_COL in cols:
            y_idx = cols.index(TARGET_COL)
            X = arr[:, [i for i, c in enumerate(cols) if c != TARGET_COL]].astype(np.float32)
            y = arr[:, y_idx].astype(np.float32)
            feat_cols = [c for c in cols if c != TARGET_COL]
            return X, y, feat_cols
    except Exception:
        pass
    return None

def _load_combined_selected() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fallback: datasets/combined.csv + models/selected_feats.json."""
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


# -----------------------
# Model loading (PyTorch 2.6+ compatible)
# -----------------------

def _safe_torch_load_from_s3(model_key: str):
    """
    PyTorch 2.6+ compatibility: try weights_only=False first; if that fails, fallback with allowlist;
    finally try legacy torch load. Only use if the source of weights is trusted.
    """
    import io as _io
    import torch as _torch
    raw = s3.get_object(Bucket=BUCKET, Key=_abs_key(model_key))["Body"].read()
    bio = _io.BytesIO(raw)

    try:
        return _torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        bio.seek(0)
        return _torch.load(bio, map_location="cpu")
    except Exception:
        pass

    try:
        from drst_inference.offline.model import MLPRegressor
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([MLPRegressor])
        except Exception:
            pass
    except Exception:
        pass

    try:
        bio.seek(0)
        return _torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        bio.seek(0)
        return _torch.load(bio, map_location="cpu")


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
    return pd.DataFrame(rows).sort_values("mae_increase", ascending=False).reset_index(drop=True)


# -----------------------
# Main logic (single run)
# -----------------------

def _align_to_model_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    d = X.shape[1]
    if d == in_dim:
        return X
    if d > in_dim:
        return X[:, :in_dim]
    pad = np.zeros((X.shape[0], in_dim - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

def main():
    # 1) Read latest model pointer
    latest = read_latest()
    if not latest:
        _write_md("xai_latest_skipped.md", _simple_report("XAI Latest (Skipped)", {"reason": "no latest model"}))
        print("[xai] latest: no latest model; exit.", flush=True)
        return

    model_key, metrics_key, _ts = latest
    mkey = _abs_key(model_key)
    metkey = _abs_key(metrics_key)

    # 2) Load model (try general interface, fallback to safe torch load)
    try:
        mdl, _raw = load_model_by_key(mkey)
    except Exception as e1:
        print(f"[xai] load_model_by_key failed: {e1} — trying safe torch load ...", flush=True)
        try:
            mdl = _safe_torch_load_from_s3(mkey)
        except Exception as e2:
            _write_md(
                "xai_error_load.md",
                _simple_report("XAI Latest (Error)", {
                    "reason": f"load {mkey} failed",
                    "primary": str(e1),
                    "fallback": str(e2),
                    "hint": "If saved via torch.save(model), PyTorch 2.6+ needs weights_only=False or allowlisting."
                }),
            )
            print(f"[xai] fallback load error: {e2}", flush=True)
            return

    # 3) Load data for explanation
    batch = _load_latest_batch()
    if batch is not None:
        Xraw, y, feat_cols = batch
    else:
        Xraw, y, feat_cols = _load_combined_selected()

    # 4) Standardize and align dimensions
    sc = load_scaler()
    Xs = sc.transform(Xraw.astype(np.float32))
    in_dim = getattr(getattr(mdl, "net", [None])[0], "in_features", Xs.shape[1])
    in_dim = int(in_dim) if isinstance(in_dim, (int, np.integer)) else Xs.shape[1]
    Xs = _align_to_model_dim(Xs, in_dim)

    # 5) Permutation importance (non-fatal; write warning on failure)
    try:
        df_imp = _perm_importance(mdl, Xs, y, feat_cols, repeat=3)
    except Exception as e:
        df_imp = pd.DataFrame([{"feature": "N/A", "mae_increase": np.nan}])
        _write_md("xai_warn_perm.md", _simple_report("XAI Latest (Warning)", {"perm_importance": f"failed: {e}"}))

    # 6) Summarize metrics and write report
    mets = _read_json(metkey) or {}
    base = float(mets.get("baseline_mae", np.nan))
    newm = float(mets.get("mae", np.nan))
    gain = (float(mets.get("acc@0.15", 0.0)) - float(mets.get("baseline_acc@0.15", 0.0)))
    ts = int(time.time())

    _write_csv(f"perm_importance_{ts}.csv", df_imp)
    report = {
        "model_key": mkey,
        "metrics_key": metkey,
        "samples_for_xai": int(Xs.shape[0]),
        "features_used": len(feat_cols),
        "baseline_mae": base,
        "current_mae": newm,
        "acc_gain_pp@0.15": round(gain, 6),
        "top5_features_by_perm": ", ".join(df_imp.head(5)["feature"].tolist()) if not df_imp.empty else "(none)",
    }
    _write_md(f"xai_report_{ts}.md", _simple_report("XAI Report (Latest Model)", report))
    print(f"[xai] report written, ts={ts}. Exit.", flush=True)


if __name__ == "__main__":
    main()
