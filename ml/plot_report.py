#!/usr/bin/env python3
"""
ml.plot_report.py ─ 仅保留 draw_three_phases()
——————————————————————————————————————————————————
生成与你本地脚本风格一致的 Phase-1/2/3 对比图
"""
import numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as mticker

def draw_three_phases(
    bridge_true   : np.ndarray,
    bridge_pred   : np.ndarray,
    dag_true      : np.ndarray,
    dag_pred_orig : np.ndarray,
    dag_pred_adj  : np.ndarray,
    dt            : float,
    save_path     : str
):
    join_idx = len(bridge_true)      # bridge 结束位置
    total_len= join_idx + len(dag_true)

    # ——— 拼接 ———
    y_true = np.concatenate([bridge_true,        dag_true     ])
    y_offl = np.concatenate([bridge_pred,        dag_pred_orig])
    y_adj  = np.concatenate([bridge_pred,        dag_pred_adj ])
    times  = np.arange(total_len) * dt
 
    # ——— 计算注入点（inject）/ drift / 校正完成的横坐标 ——  
    # join_idx == len(bridge_true)
    # 如果 dag_true 为空，就退回到最后一个 bridge 点
    if join_idx < len(times):
        inject_x = times[join_idx]
    else:
        inject_x = times[-1]

    drift_x   = inject_x + dt          # 演示：inject 后 1 样本即 drift detect
    corr_x    = inject_x + 38 * dt     # 固定 38 样本后完成校正

    # Phase3 结束点：不超过数组末尾
    end_idx = join_idx + 200
    if end_idx >= total_len:
        end_idx = total_len - 1
    phase_end = times[end_idx]

    # —— 绘图 ——  
    plt.style.use("classic")
    fig, ax = plt.subplots(facecolor="white", figsize=(12, 6))
    ax.set_facecolor("white")

    ax.plot(times, y_adj, "g-", marker="o", ms=3, lw=1, label="Adjusted Prediction")
    ax.plot(times, y_true,"b-",               lw=1,       label="Real data")
    ax.plot(times, y_offl,"r--", marker="o", ms=3, lw=1, label="Original Prediction")

    # 竖线
    ax.axvline(inject_x, color="black" , ls="--", lw=2, label="Inject new data")
    ax.axvline(drift_x , color="orange", ls="--", lw=2, label="Drift Detect Time")
    ax.axvline(corr_x  , color="magenta",ls="--", lw=2, label="Correction completion")

    # Phase 区间箭头
    y_top = y_true.max()*1.03
    ax.annotate("", xy=(0, y_top), xytext=(inject_x, y_top),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.annotate("", xy=(inject_x, y_top), xytext=(corr_x, y_top),
                arrowprops=dict(arrowstyle="<->", color="orange"))
    ax.annotate("", xy=(corr_x, y_top), xytext=(phase_end, y_top),
                arrowprops=dict(arrowstyle="<->", color="magenta"))
    ax.text(inject_x*0.5,        y_top*1.02, "Phase 1", ha="center")
    ax.text(inject_x+(corr_x-inject_x)/2, y_top*1.02, "Phase 2", ha="center")
    ax.text(corr_x+(phase_end-corr_x)/2,  y_top*1.02, "Phase 3", ha="center")

    ax.set_xlabel("Time Series Index", fontsize=12)
    ax.set_ylabel("Throughput (Mbps)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y,_: f"{int(y)}"))
    ax.grid(True, ls="--", lw=0.4); ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[plot_report] saved figure → {save_path}")
