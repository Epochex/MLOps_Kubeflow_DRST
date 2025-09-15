#!/usr/bin/env python3
# drst_forecasting/train_benchmark.py
from __future__ import annotations
import io, os, json, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from drst_common.minio_helper import save_bytes, s3, BUCKET
from drst_common.config import MODEL_DIR, RESULT_DIR
from .dataset import build_sliding_window
from .models import build_model, evaluate_metrics, benchmark_latency, publish_best_selection

# ---- 全局配置（可被环境变量覆盖）----
LOOKBACK  = int(os.getenv("FORECAST_LOOKBACK", "10"))
HORIZON   = int(os.getenv("FORECAST_HORIZON",  "5"))
EPOCHS    = int(os.getenv("FORECAST_EPOCHS",   "200"))
PATIENCE  = int(os.getenv("FORECAST_PATIENCE", "10"))
# 只取末尾 N 条样本窗口来加速；<=0 表示用全量（建议在管道里给个 3000~8000）
TAKE_LAST = int(os.getenv("FORECAST_TAKE_LAST", "0"))

def _load_candidates_for_pcm() -> List[str] | None:
    """
    从 model_selection 写入的候选清单里裁剪模型集合：
      s3://{BUCKET}/{MODEL_DIR}/forecast/model_candidates.json
    格式:
      {"task":"pcm", "topk":3, "candidates":["transformerlight","xgboost", ...]}
    """
    key = f"{MODEL_DIR}/forecast/model_candidates.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        if str(data.get("task")) == "pcm" and data.get("candidates"):
            # 去重保序
            uniq = list(dict.fromkeys([str(x).lower() for x in data["candidates"]]))
            print(f"[forecast.train] candidates from selection: {uniq}", flush=True)
            return uniq
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
            # 容错：候选里若全是没实现的名字，退回一个最稳健集合
            full = {"transformerlight": full.get("transformerlight", [{"d_model":128,"heads":4,"dropout":0.1,"lr":1e-3,"batch":64,"epochs":EPOCHS,"patience":PATIENCE}])}
        print(f"[forecast.train] grid pruned by selection -> {list(full.keys())}", flush=True)
    return full

def _train_val_split(X: np.ndarray, Y: np.ndarray, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X); v = max(1, int(n * val_frac))
    return X[:-v], X[-v:], Y[:-v], Y[-v:]

def _save_csv(key: str, df: pd.DataFrame):
    bio = io.BytesIO(); df.to_csv(bio, index=False)
    save_bytes(key, bio.getvalue(), "text/csv")

def main():
    print(f"[forecast.train] start lookback={LOOKBACK} horizon={HORIZON} take_last={TAKE_LAST}", flush=True)

    try:
        X, Y, feats = build_sliding_window(
            LOOKBACK, HORIZON,
            take_last_n=(None if TAKE_LAST <= 0 else TAKE_LAST)
        )
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
        "features": best_mdl.cfg.features if hasattr(best_mdl, "cfg") else [],
    }
    publish_best_selection(prefix, selection)
    print("[forecast.train] BEST:", json.dumps(selection, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
