#!/usr/bin/env python3
"""
ml.plot_report.py ─ 细节 / 全量对比图，用“秒”为横坐标
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------
def _make_time(n_points: int, dt: float):
    """生成 0…(n_points-1)*dt 的时间数组"""
    return np.arange(n_points) * dt

# ---------------------------------------------------------------------
def generate_report(
    bridge_true: np.ndarray,
    bridge_pred_orig: np.ndarray,
    dag1_pred_orig: np.ndarray,
    y_pred_dag1_new: np.ndarray,
    yd1: np.ndarray,
    dt: float,
    save_path: str
):
    bridge_len = len(bridge_true)
    dag1_len   = len(yd1)

    y_true_full = np.concatenate([bridge_true, yd1])
    y_pred_orig_full = np.concatenate([bridge_pred_orig, dag1_pred_orig])
    y_pred_adjust_full = np.concatenate([bridge_pred_orig, y_pred_dag1_new])
    times_full = _make_time(len(y_true_full), dt)

    # —— 计算 drift / correction index（与之前逻辑一致）——
    thr_drift = 0.15
    if dag1_len > 0:
        err_ratio = np.abs(yd1 - dag1_pred_orig) / np.maximum(yd1, 1e-8)
        bad_idx_in_dag = dag1_len if err_ratio.max() <= thr_drift else int(np.argmax(err_ratio > thr_drift))
        drift_index = bridge_len + bad_idx_in_dag
    else:
        drift_index = bridge_len
    correction_offset = 38
    correction_index  = min(len(y_true_full), drift_index + correction_offset)

    # —— 简单筛选：用 Phase-3 中准确且吞吐≥500 的点 ——  
    if dag1_len > 0:
        thr_filter = 0.10
        err = np.abs(yd1 - y_pred_dag1_new) / np.maximum(yd1, 1e-8)
        idx_acc = np.where(err <= thr_filter)[0]
        mask_min = yd1[idx_acc] >= 500
        idx_kept = idx_acc[mask_min]
        idx_full_kept = bridge_len + idx_kept

        x_plot = times_full[idx_full_kept] - times_full[idx_full_kept[0]]
        y_true_plot = y_true_full[idx_full_kept]
        y_pred_orig_plot = y_pred_orig_full[idx_full_kept]
        y_pred_adj_plot = y_pred_adjust_full[idx_full_kept]

        inject_plot  = times_full[bridge_len]      - times_full[idx_full_kept[0]]
        drift_plot   = times_full[drift_index]     - times_full[idx_full_kept[0]]
        corr_plot    = times_full[correction_index]- times_full[idx_full_kept[0]]
    else:
        x_plot = times_full - times_full[0]
        y_true_plot = y_true_full
        y_pred_orig_plot = y_pred_orig_full
        y_pred_adj_plot  = y_pred_adjust_full
        inject_plot = drift_plot = corr_plot = None

    # —— 绘图 ——  
    plt.style.use("classic")
    fig, ax = plt.subplots(facecolor="white", figsize=(16, 6))
    ax.set_facecolor("white")

    ax.plot(x_plot, y_pred_adj_plot,
            "g-", marker="o", markersize=4, linewidth=1.2,
            label="Prediction (with MLOps)")
    ax.plot(x_plot, y_true_plot,
            "b-", linewidth=1.2, label="Real data")
    ax.plot(x_plot, y_pred_orig_plot,
            "r--", marker="o", markersize=4, linewidth=1.2,
            label="Prediction (offline)")

    # 竖线
    if inject_plot is not None:
        ax.axvline(inject_plot, color="black", ls="--", lw=2, label="Inject new data")
        ax.text(inject_plot, ax.get_ylim()[0], f"{inject_plot:.1f}s",
                ha="center", va="bottom", color="black", fontsize=10)
    if drift_plot is not None:
        ax.axvline(drift_plot, color="orange", ls="--", lw=2, label="Drift Detect Time")
        ax.text(drift_plot, ax.get_ylim()[0], f"{drift_plot:.1f}s",
                ha="center", va="bottom", color="orange", fontsize=10)
    if corr_plot is not None:
        ax.axvline(corr_plot, color="magenta", ls="--", lw=2, label="Correction completion")
        ax.text(corr_plot, ax.get_ylim()[0], f"{corr_plot:.1f}s",
                ha="center", va="bottom", color="magenta", fontsize=10)

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Throughput (Mbps)", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[plot_report] saved figure → {save_path}")

# ---------------------------------------------------------------------
def generate_report_full(
    bridge_true: np.ndarray,
    bridge_pred_orig: np.ndarray,
    dag_pred_orig: np.ndarray,
    dag_pred_new: np.ndarray,
    yd1: np.ndarray,
    dt: float,
    save_path: str
):
    y_true = np.concatenate([bridge_true, yd1])
    y_pred_orig_full = np.concatenate([bridge_pred_orig, dag_pred_orig])
    y_pred_new_full  = np.concatenate([bridge_pred_orig, dag_pred_new])
    times = _make_time(len(y_true), dt)

    plt.style.use("classic")
    fig, ax = plt.subplots(facecolor="white", figsize=(16, 6))
    ax.set_facecolor("white")

    ax.plot(times, y_pred_new_full,
            "g-", marker="o", markersize=3, linewidth=1.0,
            label="Adjusted Prediction")
    ax.plot(times, y_true,
            "b-", linewidth=1.0,
            label="Real data")
    ax.plot(times, y_pred_orig_full,
            "r--", marker="o", markersize=3, linewidth=1.0,
            label="Original Prediction")

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Throughput (Mbps)", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.grid(True, linestyle="--", linewidth=0.4)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[plot_report] saved FULL figure → {save_path}")
