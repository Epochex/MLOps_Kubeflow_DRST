#!/usr/bin/env python3
# drst_forecasting/train_benchmark.py
from __future__ import annotations
import io, os, json, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import save_bytes
from drst_common.config import MODEL_DIR, RESULT_DIR
from .dataset import build_sliding_window
from .models import build_model, evaluate_metrics, benchmark_latency, publish_best_selection

LOOKBACK  = int(os.getenv("FORECAST_LOOKBACK", "10"))
HORIZON   = int(os.getenv("FORECAST_HORIZON",  "5"))
EPOCHS    = int(os.getenv("FORECAST_EPOCHS",   "200"))
PATIENCE  = int(os.getenv("FORECAST_PATIENCE", "10"))

def _grid() -> Dict[str, List[Dict]]:
    return {
        "ridge": [
            {"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        ],
        "xgboost": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0}
            for n in [200, 400]
            for d in [4, 6, 8]
            for lr in [0.1, 0.05]
        ],
        "lstm": [
            {"hidden": h, "layers": L, "dropout": dr, "lr": lr, "batch": bs, "epochs": EPOCHS, "patience": PATIENCE}
            for h in [64, 128, 256]
            for L in [1, 2]
            for dr in [0.1, 0.5]
            for lr in [1e-3, 5e-4]
            for bs in [32, 64, 128]
        ],
        "directlstm": [
            {"hidden": h, "layers": L, "dropout": dr, "lr": lr, "batch": bs, "epochs": EPOCHS, "patience": PATIENCE}
            for h in [64, 128, 256]
            for L in [1, 2]
            for dr in [0.1, 0.5]
            for lr in [1e-3, 5e-4]
            for bs in [16, 32, 64]
        ],
        "transformerlight": [
            {"d_model": dm, "heads": hd, "dropout": dr, "lr": lr, "batch": bs, "epochs": EPOCHS, "patience": PATIENCE}
            for dm in [128, 256]
            for hd in [4, 8]
            for dr in [0.1, 0.3]
            for lr in [1e-3, 1e-4]
            for bs in [32, 64]
        ],
    }

def _train_val_split(X: np.ndarray, Y: np.ndarray, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X); v = max(1, int(n * val_frac))
    return X[:-v], X[-v:], Y[:-v], Y[-v:]

def _save_csv(key: str, df: pd.DataFrame):
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

def main():
    print(f"[forecast.train] start lookback={LOOKBACK} horizon={HORIZON}", flush=True)
    try:
        X, Y, feats = build_sliding_window(LOOKBACK, HORIZON, take_last_n=None)
    except Exception as e:
        msg = str(e)
        if "不足以构成任何一个滑动窗口" in msg or "not enough" in msg.lower():
            save_bytes(f"{RESULT_DIR}/forecast_train_skipped.flag", b"not_enough_data\n", "text/plain")
            print("[forecast.train] not enough data; SKIP.", flush=True)
            return
        raise

    if X is None or len(X) < 32:
        save_bytes(f"{RESULT_DIR}/forecast_train_skipped.flag", b"too_few_samples\n", "text/plain")
        print("[forecast.train] too few samples; SKIP.", flush=True)
        return

    Xtr, Xva, Ytr, Yva = _train_val_split(X, Y, 0.2)

    grids = _grid()
    rows = []
    best = None  # (mae, latency, record)
    prefix = f"{MODEL_DIR}/forecast"

    for kind, plist in grids.items():
        print(f"[forecast.train] model={kind} | candidates={len(plist)}", flush=True)
        for i, p in enumerate(plist, 1):
            try:
                mdl = build_model(kind, LOOKBACK, HORIZON, feats, p)
                t0 = time.perf_counter()
                mdl.fit(Xtr, Ytr)
                fit_s = time.perf_counter() - t0

                yp = mdl.predict(Xva)
                mets = evaluate_metrics(Yva, yp)
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
                    "kind": kind, "params": json.dumps(p, ensure_ascii=False),
                    "r2": float("nan"), "mae": float("inf"), "acc@0.15": float("nan"),
                    "fit_seconds": float("nan"), "latency_per_sample_s": float("inf"),
                    "error": str(ex)[:200],
                })

    df = pd.DataFrame(rows)
    _save_csv(f"{prefix}/grid_results.csv", df)
    print(f"[forecast.train] grid done — {len(df)} trials", flush=True)

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
        "features": best_mdl.cfg.features if hasattr(best_mdl, "cfg") else feats
    }
    publish_best_selection(prefix, selection)
    print("[forecast.train] BEST:", json.dumps(selection, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
