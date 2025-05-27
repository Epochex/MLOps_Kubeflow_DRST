"""
生成三曲线 + 三阶段标注图
------------------------------------------------------------
调用示例
------------------------------------------------------------
from ml.plot_report import generate_report
generate_report(
    bridge_true=yb,
    bridge_pred_orig=bridge_pred_orig,
    dag1_pred_orig=dag1_pred_orig,
    y_pred_dag1_new=y_pred_stream_new,
    yd1=yd1_global,
    correction_offset=38,
    save_path="results/report.png"
)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from shared.utils import filter_predictions, filter_min

plt.style.use("classic")

def _dynamic_join(bridge_pred_orig,
                  dag1_pred_orig,
                  y_pred_dag1_new,
                  y_true_dag1,
                  bridge_true,
                  correction_offset=38,
                  thr=.10):
    """
    返回：x_values, y_pred_selected_adjusted, y_pred_selected, y_selected_adjusted,
         inject_plot, drift_plot, correction_plot
    """
    # ① 过滤 Dag-1 中误差≤thr 的点
    pred_good, true_good, _ = filter_predictions(
        y_true_dag1, y_pred_dag1_new, threshold=thr, accurate=True)

    # ② 拼接
    corr_off = min(correction_offset, len(dag1_pred_orig))
    y_pred_adjusted_full = np.concatenate([
        bridge_pred_orig,
        np.concatenate([dag1_pred_orig[:corr_off], pred_good])
    ])
    y_true_adjusted_full = np.concatenate([bridge_true, true_good])
    y_pred_original_full = np.concatenate([bridge_pred_orig, dag1_pred_orig])

    join_index = len(bridge_true)
    tail_len   = len(dag1_pred_orig)
    index_start= max(0, join_index-70)
    index_end  = min(len(y_true_adjusted_full), join_index+130)
    idx_range  = np.arange(index_start, index_end)

    y_pred_sel          = y_pred_original_full [idx_range]
    y_pred_sel_adj      = y_pred_adjusted_full [idx_range]
    y_sel_adj           = y_true_adjusted_full[idx_range]

    # ③ throughput <500 过滤
    y_pred_sel_adj, y_sel_adj, ad_ind = filter_min(
        y_sel_adj, y_pred_sel_adj, min_value=500)
    y_pred_sel = y_pred_sel[ad_ind]
    x_values   = np.arange(len(y_pred_sel_adj))

    # ④ 自动漂移 / 修正点
    thr_drift  = 0.15
    err_ratio  = np.abs(y_true_dag1 - y_pred_dag1_new) / y_true_dag1
    first_bad  = int(np.argmax(err_ratio > thr_drift))
    drift_glb  = join_index + first_bad
    plot_shift = index_start          # 左移 offset
    inject_pt  = drift_glb - plot_shift
    correct_pt = inject_pt + corr_off

    return (x_values, y_pred_sel_adj, y_pred_sel, y_sel_adj,
            inject_pt, drift_glb-plot_shift, correct_pt)

def generate_report(*,                # 强制关键字参数 ✅
        bridge_true: np.ndarray,
        bridge_pred_orig: np.ndarray,
        dag1_pred_orig: np.ndarray,
        y_pred_dag1_new: np.ndarray,
        yd1: np.ndarray,              # 流式真实值
        correction_offset: int = 38,
        save_path: str | None = None,
    ):
    """
    生成并（可选）保存绘图。所有 ndarray 形状均为 (N,)。
    """
    (x_vals, y_adj, y_orig, y_real,
     inject_pt, drift_pt, corr_pt) = _dynamic_join(
         bridge_pred_orig, dag1_pred_orig,
         y_pred_dag1_new, yd1,
         bridge_true, correction_offset)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    ax.set_facecolor("white")

    # --- 三条曲线 ---
    ax.plot(x_vals, y_adj,  "g-", marker="o", ms=4, lw=1, label="Adjusted")
    ax.plot(x_vals, y_orig, "r--",marker="o", ms=4, lw=1, label="Original")
    ax.plot(x_vals, y_real, "b-",            lw=1,          label="Real")

    # --- 垂直线 ---
    for x, c, lab in [(inject_pt,"black","Inject new data"),
                      (drift_pt,"orange","Drift detect time"),
                      (corr_pt, "magenta","Correction completion")]:
        ax.axvline(x, color=c, ls="--", lw=2, label=lab)
        ax.text(x, ax.get_ylim()[0]-100, str(x), ha="center",
                va="top", color=c, fontsize=9)

    # --- Phase 注释 ---
    y_max   = max(y_adj.max(), y_orig.max(), y_real.max())
    ax.annotate("", xy=(0,y_max), xytext=(drift_pt,y_max),
                arrowprops=dict(arrowstyle="<->",color="black"))
    ax.annotate("", xy=(drift_pt,y_max), xytext=(corr_pt,y_max),
                arrowprops=dict(arrowstyle="<->",color="orange"))
    ax.annotate("", xy=(corr_pt,y_max), xytext=(x_vals[-1],y_max),
                arrowprops=dict(arrowstyle="<->",color="magenta"))
    ax.text(drift_pt/2,             y_max*1.02,"Phase 1",ha="center")
    ax.text((drift_pt+corr_pt)/2,   y_max*1.02,"Phase 2",ha="center")
    ax.text((corr_pt+x_vals[-1])/2, y_max*1.02,"Phase 3",ha="center")

    # --- 轴 & 样式 ---
    fig.text(0.5,0.02,"Time Series Index",ha="center",fontsize=14)
    fig.text(0.04,0.5,"Throughput(Mbps)",va="center",rotation="vertical",fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.grid(ls="--",lw=.5); plt.legend(loc="lower right",fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot] saved ↗ {save_path}")
    return fig, ax     # 方便 Jupyter 直接显示
