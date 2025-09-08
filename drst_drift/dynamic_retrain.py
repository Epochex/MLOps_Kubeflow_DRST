# ===== 文件: drst_drift/dynamic_retrain.py =====
#!/usr/bin/env python3
from __future__ import annotations
import io, os, time, json
from itertools import product
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from drst_common import config as _cfg
from drst_common.config import (
    RESULT_DIR, TARGET_COL, abc_grids, BUCKET,
    RETRAIN_WARM_EPOCHS, RETRAIN_EPOCHS_FULL, RETRAIN_EARLY_PATIENCE,
    RETRAIN_MODE, RETRAIN_FREEZE_N, RETRAIN_VAL_FRAC,
)
from drst_common.minio_helper import load_np, save_bytes, s3
from drst_common.artefacts import load_scaler, load_selected_feats, read_latest, load_model_by_key, write_latest
from drst_common.utils import calculate_accuracy_within_threshold
from drst_common.metric_logger import log_metric, sync_all_metrics_to_minio
from drst_common.runtime import touch_ready, write_kfp_metadata
from drst_common.resource_probe import start as start_probe
from drst_inference.offline.model import MLPRegressor

ACC_THR  = float(getattr(_cfg, "ACC_THR", 0.25))
_thr_str = ("%.2f" % ACC_THR).rstrip("0").rstrip(".")

device   = "cuda" if torch.cuda.is_available() else "cpu"
pod_name = os.getenv("HOSTNAME", "retrain")

LOCK_KEY = f"{RESULT_DIR}/retrain_lock.flag"
DONE_KEY = f"{RESULT_DIR}/retrain_done.flag"
WATCH    = os.getenv("RETRAIN_WATCH", "0") in ("1", "true", "True")
POLL_S   = int(os.getenv("POLL_INTERVAL_S", "2") or 2)

def _read_grid_flag() -> str:
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=f"{RESULT_DIR}/retrain_grid.flag")["Body"].read()
        s = raw.decode().strip().upper()
        return s if s in ("A","B","C") else "B"
    except Exception:
        return "B"

def _split_xy(arr: np.ndarray, cols: Optional[List[str]], feat_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d = len(feat_names)
    if cols and TARGET_COL in cols:
        y_idx = cols.index(TARGET_COL)
        X = arr[:, [cols.index(c) for c in feat_names if c in cols]].astype(np.float32)
        y = arr[:, y_idx].astype(np.float32)
        return X, y
    if arr.shape[1] == d + 1:
        return arr[:, :d].astype(np.float32), arr[:, -1].astype(np.float32)
    return arr[:, :d].astype(np.float32), None

def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    err = np.abs(y_pred - y_true)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    acc_thr = calculate_accuracy_within_threshold(y_true, y_pred, ACC_THR)
    acc15   = calculate_accuracy_within_threshold(y_true, y_pred, 0.15)
    return mae, rmse, acc_thr, acc15

def _loss_fn(name: str):
    name = name.lower()
    if name == "mse": return torch.nn.MSELoss()
    if name in ("huber","smoothl1","smooth_l1"): return torch.nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"unsupported loss: {name}")

def _freeze_n_linear(model: torch.nn.Module, n: int):
    if n <= 0: return
    taken = 0
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            for p in m.parameters():
                p.requires_grad = False
            taken += 1
            if taken >= n: break

def _hidden_of(model: torch.nn.Module) -> List[int]:
    dims = []
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            dims.append(m.out_features)
    return dims[:-1] if dims else []

def _can_finetune(current_model, hidden: Tuple[int, ...], act: str) -> bool:
    try:
        cfg = getattr(current_model, "config", None)
        if cfg and list(hidden) == list(cfg.get("hidden", [])) and act.lower() == str(cfg.get("act","relu")).lower():
            return True
    except Exception:
        pass
    return list(hidden) == _hidden_of(current_model)

def _align_to_dim(X: np.ndarray, in_dim: int) -> np.ndarray:
    d = X.shape[1]
    if d == in_dim: return X
    if d > in_dim:  return X[:, :in_dim]
    pad = np.zeros((X.shape[0], in_dim - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)

def _obj_mtime(key: str) -> float | None:
    try:
        return s3.head_object(Bucket=BUCKET, Key=key)["LastModified"].timestamp()
    except Exception:
        return None

def _exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key); return True
    except Exception:
        return False

def _train_one(Xtr, Ytr, Xva, Yva, in_dim: int, hidden: Tuple[int, ...], act: str,
               lr: float, batch: int, loss_name: str, weight_decay: float,
               mode: str, maybe_base_model) -> Tuple[torch.nn.Module, Dict[str, float]]:
    if mode == "finetune" and maybe_base_model is not None:
        model = maybe_base_model
    else:
        model = MLPRegressor(in_dim, hidden=hidden, act=act, dropout=0.0)
    if mode == "finetune" and _cfg.RETRAIN_FREEZE_N > 0:
        _freeze_n_linear(model, _cfg.RETRAIN_FREEZE_N)

    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float().view(-1,1))
    dl = DataLoader(ds, batch_size=int(batch), shuffle=True, drop_last=False)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = _loss_fn(loss_name)
    model = model.to(device)

    best_rmse = float("inf"); best_state: Optional[bytes] = None
    bad = 0
    for _ in range(max(1, RETRAIN_EPOCHS_FULL)):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
        _, rmse, _, _ = _eval_metrics(Yva, pv)
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse; bad = 0
            bio = io.BytesIO(); torch.save(model.to("cpu"), bio); best_state = bio.getvalue()
            model.to(device)
        else:
            bad += 1
            if bad > RETRAIN_EARLY_PATIENCE: break

    if best_state is not None:
        model = torch.load(io.BytesIO(best_state), map_location=device)
    with torch.no_grad():
        pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
    mae, rmse, acc_thr, acc15 = _eval_metrics(Yva, pv)
    return model, {"mae": mae, "rmse": rmse, f"acc@{_thr_str}": acc_thr, "acc@0.15": acc15}

def _warm_score(Xtr, Ytr, Xva, Yva, in_dim: int, hidden: Tuple[int, ...], act: str,
                lr: float, batch: int, loss_name: str, weight_decay: float,
                mode: str, maybe_base_model) -> float:
    if mode == "finetune" and maybe_base_model is not None:
        model = maybe_base_model
    else:
        model = MLPRegressor(in_dim, hidden=hidden, act=act, dropout=0.0)
    if mode == "finetune" and RETRAIN_FREEZE_N > 0:
        _freeze_n_linear(model, RETRAIN_FREEZE_N)

    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float().view(-1,1))
    dl = DataLoader(ds, batch_size=int(batch), shuffle=True, drop_last=False)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = _loss_fn(loss_name)
    model = model.to(device)

    for _ in range(max(1, RETRAIN_WARM_EPOCHS)):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pv = model(torch.from_numpy(Xva).float().to(device)).cpu().numpy().ravel()
    _, rmse, _, _ = _eval_metrics(Yva, pv)
    return rmse

def _run_once() -> bool:
    t0_wall = time.time()
    try:
        try:
            arr = load_np(f"{RESULT_DIR}/latest_batch.npy")
        except Exception as e:
            print(f"[retrain] latest_batch.npy not found: {e}")
            return False

        grid_letter = _read_grid_flag()
        feat_names = load_selected_feats()
        scaler = load_scaler()
        try:
            raw = s3.get_object(Bucket=BUCKET, Key=f"{RESULT_DIR}/latest_batch.columns.json")["Body"].read()
            cols = (raw and json.loads(raw.decode())) or None
        except Exception:
            cols = None

        X, y = _split_xy(arr, cols, feat_names)
        if y is None:
            save_bytes(f"{RESULT_DIR}/retrain_skipped_no_labels.flag", b"", "text/plain")
            log_metric(component="retrain", event="skipped", train_rows=int(X.shape[0]))
            sync_all_metrics_to_minio(); write_kfp_metadata()
            return False

        Xs = scaler.transform(X.astype(np.float32))
        n = len(Xs)
        idx = np.random.permutation(n)
        val_n = max(1, int(n * float(RETRAIN_VAL_FRAC)))
        va_idx, tr_idx = idx[:val_n], idx[val_n:]
        Xtr, Ytr = Xs[tr_idx], y[tr_idx]
        Xva, Yva = Xs[va_idx], y[va_idx]
        in_dim = Xs.shape[1]

        baseline_model, _ = load_model_by_key("baseline_model.pt")
        baseline_model.eval().to(device)
        baseline_in_dim = baseline_model.net[0].in_features
        Xva_b = _align_to_dim(Xva, baseline_in_dim)
        with torch.no_grad():
            pb = baseline_model(torch.from_numpy(Xva_b).float().to(device)).cpu().numpy().ravel()
        base_mae, base_rmse, base_acc_thr, base_acc15 = _eval_metrics(Yva, pb)

        latest = read_latest()
        current_model_for_ft = None
        current_hidden = None
        if latest:
            model_key, _, _ = latest
            try:
                m, _ = load_model_by_key(model_key)
                current_model_for_ft = m
                current_hidden = _hidden_of(m)
            except Exception:
                pass

        grids = abc_grids(current_hidden)
        G = grids.get(grid_letter, grids["B"])

        combos = list(product(G["learning_rate"], G["batch_size"], G["hidden_layers"],
                              G["activation"], G["loss"], G["wd"]))
        print(f"[retrain] grid={grid_letter} candidates={len(combos)}")

        scores = []
        for lr, bs, hid, act, loss_name, wd in combos:
            hid_t = tuple(hid)
            if RETRAIN_MODE == "scratch":
                mode = "scratch"; base_for_ft = None
            elif RETRAIN_MODE == "finetune":
                mode = "finetune" if (current_model_for_ft and _can_finetune(current_model_for_ft, hid_t, act)) else "scratch"
                base_for_ft = current_model_for_ft if mode == "finetune" else None
            else:
                mode = "finetune" if (current_model_for_ft and _can_finetune(current_model_for_ft, hid_t, act)) else "scratch"
                base_for_ft = current_model_for_ft if mode == "finetune" else None
            rmse = _warm_score(Xtr, Ytr, Xva, Yva, in_dim, hid_t, act, float(lr), int(bs), str(loss_name), float(wd), mode, base_for_ft)
            scores.append(((lr, bs, hid_t, act, loss_name, wd, mode), rmse))

        scores.sort(key=lambda x: x[1])
        topk = int(G.get("topk", 2))
        finalists = scores[:topk]

        best_model = None; best_metrics = None; best_cfg = None
        for (lr, bs, hid, act, loss_name, wd, mode), _ in finalists:
            base_for_ft = current_model_for_ft if (mode == "finetune") else None
            mdl, mets = _train_one(Xtr, Ytr, Xva, Yva, in_dim, tuple(hid), act, float(lr), int(bs), str(loss_name), float(wd), mode, base_for_ft)
            if (best_metrics is None) or (mets["rmse"] < best_metrics["rmse"] - 1e-9):
                best_model, best_metrics, best_cfg = mdl, mets, {
                    "lr": lr, "batch_size": bs, "hidden_layers": list(hid),
                    "activation": act, "loss": loss_name, "wd": wd, "mode": mode,
                }

        buf = io.BytesIO(); torch.save(best_model.to("cpu"), buf)
        model_bytes = buf.getvalue()
        model_mb = round(len(model_bytes) / (1024*1024), 4)
        ts = int(time.time())

        train_secs = round(time.time() - t0_wall, 3)

        metrics = {
            f"acc@{_thr_str}": best_metrics[f"acc@{_thr_str}"],
            "acc@0.15": best_metrics["acc@0.15"],
            f"baseline_acc@{_thr_str}": base_acc_thr,
            "baseline_acc@0.15": base_acc15,
            "mae": best_metrics["mae"], "rmse": best_metrics["rmse"],
            "baseline_mae": base_mae, "baseline_rmse": base_rmse,
            "train_rows": int(len(tr_idx)), "val_rows": int(len(va_idx)),
            "model_size_mb": model_mb,
            "grid": grid_letter, "retrain_wall_s": train_secs,
            **best_cfg,
        }

        model_key   = f"model_{ts}.pt"
        metrics_key = f"metrics_{ts}.json"
        write_latest(model_bytes, metrics, model_key=model_key, metrics_key=metrics_key)

        done_info = {
            "ts": ts, "grid": grid_letter,
            "best_rmse": metrics["rmse"], "best_mae": metrics["mae"],
            "model_key": model_key, "metrics_key": metrics_key
        }
        save_bytes(f"{RESULT_DIR}/retrain_done.flag", json.dumps(done_info).encode("utf-8"), "application/json")

        log_metric(component="retrain", event="summary",
                   train_rows=int(len(tr_idx)), mae=metrics["mae"], rmse=metrics["rmse"],
                   **{f"accuracy@{_thr_str}": metrics[f"acc@{_thr_str}"]},
                   model_size_mb=model_mb, grid=grid_letter, retrain_wall_s=train_secs,
                   lr=best_cfg["lr"], batch_size=best_cfg["batch_size"],
                   loss=best_cfg["loss"], activation=best_cfg["activation"])

        sync_all_metrics_to_minio(); write_kfp_metadata()
        print(f"[retrain] done. grid={grid_letter} mode={best_cfg['mode']} "
              f"rmse={metrics['rmse']:.4f} acc@{_thr_str}={metrics[f'acc@{_thr_str}']:.4f} "
              f"wall={train_secs:.3f}s")
        return True
    except Exception as ex:
        print(f"[retrain] ERROR during run_once: {ex}")
        return False

def main():
    touch_ready("retrain", pod_name)
    stop_probe = start_probe("retrain")
    try:
        if not WATCH:
            _ = _run_once()
            return

        max_secs = int(_cfg.get_max_wall_secs())
        print(f"[retrain] watcher started: poll={POLL_S}s, max_watch={max_secs} (<=0 means infinite)", flush=True)

        start_ts = time.time()
        last_done_ts = _obj_mtime(DONE_KEY) or 0.0
        last_lock_attempt_ts: float | None = None

        def _still_in_window() -> bool:
            return (max_secs <= 0) or ((time.time() - start_ts) < max_secs)

        while _still_in_window():
            lock_ts = _obj_mtime(LOCK_KEY)

            if lock_ts and lock_ts > last_done_ts and lock_ts != last_lock_attempt_ts:
                last_lock_attempt_ts = lock_ts
                print(f"[retrain] lock detected (ts={lock_ts}); start one retrain...", flush=True)
                _ = _run_once()

                for _wait in range(20):
                    new_done = _obj_mtime(DONE_KEY) or last_done_ts
                    if new_done > last_done_ts:
                        last_done_ts = new_done
                        break
                    time.sleep(0.5)

            time.sleep(POLL_S)

        print("[retrain] watcher timeout reached; exit.", flush=True)
    finally:
        stop_probe()

if __name__ == "__main__":
    main()
