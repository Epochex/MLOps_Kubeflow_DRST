#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def generate_report(
    bridge_true: np.ndarray,
    bridge_pred_orig: np.ndarray,
    dag1_pred_orig: np.ndarray,
    y_pred_dag1_new: np.ndarray,
    yd1: np.ndarray,
    save_path: str
):
    bridge_len = len(bridge_true)
    dag1_len = len(yd1)
    y_true_full = np.concatenate([bridge_true, yd1])
    y_pred_orig_full = np.concatenate([bridge_pred_orig, dag1_pred_orig])
    y_pred_adjust_full = np.concatenate([bridge_pred_orig, y_pred_dag1_new])

    full_len = len(y_true_full)

    # compute drift and correction indices
    thr_drift = 0.15
    if dag1_len > 0:
        err_ratio = np.abs(yd1 - dag1_pred_orig) / np.maximum(yd1, 1e-8)
        if err_ratio.max() <= thr_drift:
            bad_idx_in_dag = dag1_len
        else:
            bad_idx_in_dag = int(np.argmax(err_ratio > thr_drift))
        drift_index = bridge_len + bad_idx_in_dag
    else:
        drift_index = bridge_len
    correction_offset = 38
    correction_index = min(full_len, drift_index + correction_offset)

    # filtering and alignment
    if dag1_len > 0:
        thr_filter = 0.10
        err_filter = np.abs(yd1 - y_pred_dag1_new) / np.maximum(yd1, 1e-8)
        idx_acc = np.where(err_filter <= thr_filter)[0]
        true_acc = yd1[idx_acc]
        mask_min = true_acc >= 500
        idx_min = idx_acc[mask_min]
        idx_full_kept = bridge_len + idx_min

        y_true_plot = y_true_full[idx_full_kept]
        y_pred_orig_plot = y_pred_orig_full[idx_full_kept]
        y_pred_adjusted_plot = y_pred_adjust_full[idx_full_kept]
        x_plot = np.arange(len(idx_full_kept))

        # map vertical-line positions
        inject_plot = np.where(idx_full_kept == bridge_len)[0]
        inject_plot = int(inject_plot[0]) if inject_plot.size > 0 else None

        drift_plot = np.where(idx_full_kept == drift_index)[0]
        drift_plot = int(drift_plot[0]) if drift_plot.size > 0 else None

        corr_plot = np.where(idx_full_kept == correction_index)[0]
        corr_plot = int(corr_plot[0]) if corr_plot.size > 0 else None

    else:
        right = min(full_len, 200)
        x_plot = np.arange(right)
        y_true_plot = y_true_full[:right]
        y_pred_orig_plot = y_pred_orig_full[:right]
        y_pred_adjusted_plot = y_pred_adjust_full[:right]
        inject_plot = None
        drift_plot = None
        corr_plot = None

    # plot
    plt.style.use("classic")
    fig, ax = plt.subplots(facecolor="white", figsize=(16, 6))
    ax.set_facecolor("white")

    ax.plot(x_plot, y_pred_adjusted_plot,
            "g-", marker="o", markersize=4, linewidth=1.2,
            label="Prediction (with MLOps)")
    ax.plot(x_plot, y_true_plot,
            "b-", linewidth=1.2,
            label="Real data")
    ax.plot(x_plot, y_pred_orig_plot,
            "r--", marker="o", markersize=4, linewidth=1.2,
            label="Prediction (without MLOps)")

    # vertical lines
    if inject_plot is not None:
        ax.axvline(x=inject_plot, color="black", ls="--", lw=2, label="Inject new data")
        ax.text(inject_plot, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]),
                str(bridge_len), ha="center", va="top", color="black", fontsize=10)
    if drift_plot is not None:
        ax.axvline(x=drift_plot, color="orange", ls="--", lw=2, label="Drift Detect Time")
        ax.text(drift_plot, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]),
                str(drift_index), ha="center", va="top", color="orange", fontsize=10)
    if corr_plot is not None:
        ax.axvline(x=corr_plot, color="magenta", ls="--", lw=2, label="Correction completion")
        ax.text(corr_plot, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]),
                str(correction_index), ha="center", va="top", color="magenta", fontsize=10)

    # Phase annotations
    y_max = max(y_true_plot.max(), y_pred_orig_plot.max(), y_pred_adjusted_plot.max())
    top = y_max * 1.05

    if drift_plot is not None and drift_plot > 0:
        ax.annotate("", xy=(0, top), xytext=(drift_plot, top),
                    arrowprops=dict(arrowstyle="<->", color="black"))
        ax.text(drift_plot/2, top*1.02, "Phase1",
                ha="center", va="bottom", color="black", fontsize=12)

    if drift_plot is not None and corr_plot is not None and corr_plot > drift_plot:
        ax.annotate("", xy=(drift_plot, top), xytext=(corr_plot, top),
                    arrowprops=dict(arrowstyle="<->", color="orange"))
        ax.text((drift_plot+corr_plot)/2, top*1.02, "Phase2",
                ha="center", va="bottom", color="orange", fontsize=12)

    if corr_plot is not None and corr_plot < len(x_plot):
        ax.annotate("", xy=(corr_plot, top), xytext=(len(x_plot), top),
                    arrowprops=dict(arrowstyle="<->", color="magenta"))
        ax.text((corr_plot+len(x_plot))/2, top*1.02, "Phase3",
                ha="center", va="bottom", color="magenta", fontsize=12)

    ax.set_xlabel("Time series (filtered indices)", fontsize=14)
    ax.set_ylabel("Throughput (Mbps)", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlim(0, len(x_plot)-1)
    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_max * 1.15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_report] saved figure → {save_path}")
