#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drst_forecasting/explain.py  —— 单次运行版

用途：
- 由 pipeline 在 retrain 结束后直接触发本脚本；
- 读取 MODEL_DIR/latest.json 指向的最新模型与指标；
- 选用 results/latest_batch.npy（若有）或 datasets/combined.csv + models/selected_feats.json 作为解释数据；
- 计算简易置换重要性并写入 results/xai/ 下的报告；
- 完成后立即退出（不再 watch）。

兼容性修复：
- 统一对 model/metrics key 做相对键补全（不带斜杠时自动加 "models/" 前缀），避免 NoSuchKey；
- 兼容 PyTorch 2.6+（weights_only 默认 True）导致的 torch.load 失败：先试 weights_only=False，再允许特定类，最后兼容旧版本。
"""

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
# 工具：路径与 S3 I/O
# -----------------------

def _abs_key(key: str) -> str:
    """相对键自动补上 models/ 前缀。"""
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
# 数据加载
# -----------------------

def _load_latest_batch() -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """优先使用监控侧产物：latest_batch.npy + latest_batch.columns.json。"""
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
    """回退：datasets/combined.csv + models/selected_feats.json。"""
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
# 模型加载（兼容 torch 2.6+）
# -----------------------

def _safe_torch_load_from_s3(model_key: str):
    """
    兼容 PyTorch 2.6+：优先 weights_only=False；若失败再 allowlist；最后兼容旧 torch。
    仅在你信任权重来源时使用。
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
        from drst_inference.offline.model import MLPRegressor  # 若项目中存在该类
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


# -----------------------
# 置换重要性（简易）
# -----------------------

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
# 主逻辑（单次运行）
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
    # 1) 读取最新指针
    latest = read_latest()
    if not latest:
        _write_md("xai_latest_skipped.md", _simple_report("XAI Latest (Skipped)", {"reason": "no latest model"}))
        print("[xai] latest: no latest model; exit.", flush=True)
        return

    model_key, metrics_key, _ts = latest
    mkey = _abs_key(model_key)
    metkey = _abs_key(metrics_key)

    # 2) 加载模型（先通用接口，失败则安全加载）
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

    # 3) 取解释用数据
    batch = _load_latest_batch()
    if batch is not None:
        Xraw, y, feat_cols = batch
    else:
        Xraw, y, feat_cols = _load_combined_selected()

    # 4) 标准化、维度对齐
    sc = load_scaler()
    Xs = sc.transform(Xraw.astype(np.float32))
    in_dim = getattr(getattr(mdl, "net", [None])[0], "in_features", Xs.shape[1])
    in_dim = int(in_dim) if isinstance(in_dim, (int, np.integer)) else Xs.shape[1]
    Xs = _align_to_model_dim(Xs, in_dim)

    # 5) 置换重要性（失败不终止，只写 warning）
    try:
        df_imp = _perm_importance(mdl, Xs, y, feat_cols, repeat=3)
    except Exception as e:
        df_imp = pd.DataFrame([{"feature": "N/A", "mae_increase": np.nan}])
        _write_md("xai_warn_perm.md", _simple_report("XAI Latest (Warning)", {"perm_importance": f"failed: {e}"}))

    # 6) 汇总指标并写报告
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
