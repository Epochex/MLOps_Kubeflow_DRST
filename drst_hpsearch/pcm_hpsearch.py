#!/usr/bin/env python3
# /data/mlops/DRST-SoftwarizedNetworks/drst_hpsearch/pcm_hpsearch.py
from __future__ import annotations
import io, os, json, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import save_bytes, s3, BUCKET
from drst_common.config import MODEL_DIR, RESULT_DIR
from drst_forecasting.dataset import build_sliding_window
from drst_forecasting.models import build_model, evaluate_metrics, benchmark_latency, publish_best_selection

# ---- 全局配置（可被环境变量覆盖）----
LOOKBACK  = int(os.getenv("FORECAST_LOOKBACK", "10"))
HORIZON   = int(os.getenv("FORECAST_HORIZON",  "5"))
EPOCHS    = int(os.getenv("FORECAST_EPOCHS",   "200"))
PATIENCE  = int(os.getenv("FORECAST_PATIENCE", "10"))
# 只取末尾 N 条样本窗口来加速；<=0 表示用全量
TAKE_LAST = int(os.getenv("FORECAST_TAKE_LAST", "0"))

def _canon(name: str) -> str | None:
    n = str(name).strip().lower()
    if n in ("ridge",): return "ridge"
    if n in ("xgboost","xgb","xgbregressor"): return "xgboost"
    if n in ("transformerlight","transformer","transformerlight1d"): return "transformerlight"
    if n in ("randomforest","rf","gradientboosting","gbrt","gbdt"): return "xgboost"
    return None

def _load_candidates_for_pcm() -> List[str] | None:
    """
    优先读： models/forecast/model_candidates.json
    兼容旧： models/forecasting/pcm_select_suggestion.json
    返回规范化模型名（子集于：ridge/xgboost/transformerlight）
    """
    key1 = f"{MODEL_DIR}/forecast/model_candidates.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key1)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        if str(data.get("task")) == "pcm" and data.get("candidates"):
            cand = []
            for x in data["candidates"]:
                c = _canon(x)
                if c and c not in cand:
                    cand.append(c)
            if cand:
                print(f"[pcm.hpsearch] candidates from selection: {cand}", flush=True)
                return cand
    except Exception:
        pass

    key2 = f"{MODEL_DIR}/forecasting/pcm_select_suggestion.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key2)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        items = data.get("recommend_topk") or []
        cand = []
        for rec in items:
            c = _canon(rec.get("model",""))
            if c and c not in cand:
                cand.append(c)
        if cand:
            print(f"[pcm.hpsearch] candidates from suggestion: {cand}", flush=True)
            return cand
    except Exception:
        pass

    return None

def _grid() -> Dict[str, List[Dict]]:
    """
    全量网格（若上游已跑过 pcm 的 model_selection，会按候选集合裁剪）
    """
    full = {
        "ridge": [
            {"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        ],
        "xgboost": [
            {
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            }
            for n in [200, 400]
            for d in [4, 6, 8]
            for lr in [0.1, 0.05]
        ],
        "lstm": [
            {
                "hidden": h, "layers": L, "dropout": dr,
                "lr": lr, "batch": bs, "epochs": EPOCHS, "patience": PATIENCE
            }
            for h in [64, 128, 256]
            for L in [1, 2]
            for dr in [0.1, 0.5]
            for lr in [1e-3, 5e-4]
            for bs in [32, 64, 128]
        ],
        "directlstm": [
            {
                "hidden": h, "layers": L, "dropout": dr,
                "lr": lr, "batch": bs, "epochs": EPOCHS, "patience": PATIENCE
            }
            for h in [64, 128, 256]
            for L in [1, 2]
            for dr in [0.1, 0.5]
            for lr in [1e-3, 5e-4]
            for bs in [16, 32, 64]
        ],
        "transformerlight": [
            {
                "d_model": dm, "heads": hd, "dropout": dr,
                "lr": lr, "batch": bs, "epochs": EPOCHS, "patience": PATIENCE
            }
            for dm in [128, 256]
            for hd in [4, 8]
            for dr in [0.1, 0.3]
            for lr in [1e-3, 1e-4]
            for bs in [32, 64]
        ],
    }

    cand = _load_candidates_for_pcm()
    if cand:
        keep = set(cand)
        full = {k: v for k, v in full.items() if k in keep}
        if not full:
            full = {"transformerlight": full.get("transformerlight", [{"d_model":128,"heads":4,"dropout":0.1,"lr":1e-3,"batch":64,"epochs":EPOCHS,"patience":PATIENCE}])}
        print(f"[pcm.hpsearch] grid pruned by selection -> {list(full.keys())}", flush=True)
    return full

def _train_val_split(X: np.ndarray, Y: np.ndarray, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X); v = max(1, int(n * val_frac))
    return X[:-v], X[-v:], Y[:-v], Y[-v:]

def _save_csv(key: str, df: pd.DataFrame):
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

def main():
    print(f"[pcm.hpsearch] start lookback={LOOKBACK} horizon={HORIZON} take_last={TAKE_LAST}", flush=True)

    try:
        X, Y, feats = build_sliding_window(
            LOOKBACK, HORIZON,
            take_last_n=(None if TAKE_LAST <= 0 else TAKE_LAST),
            multi_output=True   # 与深度模型输出对齐
        )
    except Exception as e:
        msg = str(e)
        if "不足以构成任何一个滑动窗口" in msg or "not enough" in msg.lower():
            save_bytes(f"{RESULT_DIR}/forecast_train_skipped.flag", b"not_enough_data\n", "text/plain")
            print("[pcm.hpsearch] not enough data; SKIP.", flush=True)
            return
        raise

    if X is None or len(X) < 32:
        save_bytes(f"{RESULT_DIR}/forecast_train_skipped.flag", b"too_few_samples\n", "text/plain")
        print("[pcm.hpsearch] too few samples; SKIP.", flush=True)
        return

    Xtr, Xva, Ytr, Yva = _train_val_split(X, Y, 0.2)
    grids = _grid()
    rows = []
    best = None  # (mae, latency, record)
    prefix = f"{MODEL_DIR}/forecast"

    for kind, plist in grids.items():
        print(f"[pcm.hpsearch] model={kind} | candidates={len(plist)}", flush=True)
        for i, p in enumerate(plist, 1):
            try:
                mdl = build_model(kind, LOOKBACK, HORIZON, feats, p)
                t0 = time.perf_counter()
                mdl.fit(Xtr, Ytr)
                fit_s = time.perf_counter() - t0

                yp = mdl.predict(Xva)
                mets = evaluate_metrics(Yva, yp)  # {'r2','mae','acc@0.15'}
                lat = benchmark_latency(lambda Z: mdl.predict(Z), Xva, repeat=1)

                rec = {
                    "kind": kind,
                    "params": json.dumps(p, ensure_ascii=False),
                    "r2": round(mets["r2"], 6),
                    "mae": round(mets["mae"], 6),
                    "acc@0.15": round(mets["acc@0.15"], 6),
                    "fit_seconds": round(fit_s, 6),
                    "latency_per_sample_s": round(lat, 6),
                }
                rows.append(rec)

                # 选型：先比 MAE，平手比 latency
                key = (rec["mae"], rec["latency_per_sample_s"])
                if (best is None) or (key < (best[0], best[1])):
                    best = (rec["mae"], rec["latency_per_sample_s"], (kind, p, mdl, rec))
                if i % 10 == 0:
                    print(f"  - {kind} {i}/{len(plist)} best_mae={best[0]:.6f}", flush=True)
            except Exception as ex:
                rows.append({
                    "kind": kind,
                    "params": json.dumps(p, ensure_ascii=False),
                    "r2": float("nan"),
                    "mae": float("inf"),
                    "acc@0.15": float("nan"),
                    "fit_seconds": float("nan"),
                    "latency_per_sample_s": float("inf"),
                    "error": str(ex)[:200],
                })

    df = pd.DataFrame(rows)
    _save_csv(f"{prefix}/grid_results.csv", df)
    print(f"[pcm.hpsearch] grid done — {len(df)} trials", flush=True)

    assert best is not None, "no model successfully trained"
    _, _, (best_kind, best_params, best_mdl, best_rec) = best

    info = best_mdl.save_to_s3(prefix)  # 写 best_model + best_meta.json
    selection = {
        "selected_kind": best_kind,
        "selected_params": best_params,
        "metrics": best_rec,
        "meta_key": info["meta_key"],      # API 会先读 selected.json → meta_key → artifact
        "artifact": info["model_key"],
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "features": best_mdl.cfg.features if hasattr(best_mdl, "cfg") else [],
    }
    publish_best_selection(prefix, selection)
    print("[pcm.hpsearch] BEST:", json.dumps(selection, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
