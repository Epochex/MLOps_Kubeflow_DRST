#!/usr/bin/env python3
"""
ml.plot_report.py ─ Phase-1 / 2 / 3 对比图（精简 + 高可读版）
──────────────────────────────────────────────────────────────
• 只暴露 draw_three_phases()
• 按 “注入点” 前后裁剪，可在 config 中调窗口宽度
• 纵向三条曲线：Baseline / Adaptive / Real
• 额外三条垂直线：Inject、Drift-Detect、Correction-Done
• 顶部三段双向箭头 + Phase 标签
"""

from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ───── 可调超参 ────────────────────────────────────────────────
# 如果想改窗口大小，只需 export 两个环境变量即可
_PLOT_WIN_BEFORE = int(os.getenv("PLOT_WINDOW_BEFORE", 70))   # 注入点前保留多少个样本
_PLOT_WIN_AFTER  = int(os.getenv("PLOT_WINDOW_AFTER",  130))  # 注入点后保留多少个样本
_CORR_OFFSET     = int(os.getenv("CORRECTION_OFFSET",  38))   # 校正完成点 = inject + 38
_DRIFT_OFFSET    = 1                                          # 演示：inject 后 1 样本即 drift detect

# ───── 主绘图函数 ─────────────────────────────────────────────
def draw_three_phases(
    bridge_true   : np.ndarray,
    bridge_pred   : np.ndarray,
    dag_true      : np.ndarray,
    dag_pred_orig : np.ndarray,
    dag_pred_adj  : np.ndarray,
    dt            : float,          # ← 依旧保留，备用；本图用 sample index
    save_path     : str
) -> None:
    """
    Parameters
    ----------
    bridge_true / bridge_pred : Bridge 阶段（Phase-1）真值 / Baseline 预测
    dag_true / dag_pred_*     : DAG 数据流（Phase-2/3）真值 / Baseline / Adaptive
    dt                        : 下游仍传入，但本图横轴改为 “样本序号”
    save_path                 : PNG 输出路径
    """
    # ── 拼接完整序列 ──────────────────────────────────────────
    inject_idx = len(bridge_true)                # 注入点 = Bridge 结束下标
    y_true_all = np.concatenate([bridge_true,      dag_true     ])
    y_pred_bls = np.concatenate([bridge_pred,      dag_pred_orig])
    y_pred_adp = np.concatenate([bridge_pred,      dag_pred_adj ])

    total_len  = len(y_true_all)

    # ── 裁剪窗口（前后可调）────────────────────────────────────
    lo = max(0, inject_idx - _PLOT_WIN_BEFORE)
    hi = min(total_len, inject_idx + _PLOT_WIN_AFTER)
    x  = np.arange(lo, hi) - lo      # 横轴用“样本序号”，从 0 开始

    y_true = y_true_all [lo:hi]
    y_bls  = y_pred_bls[lo:hi]
    y_adp  = y_pred_adp[lo:hi]

    # 注入 / drift / 校正完成在线段内的位置
    inj_x   = inject_idx   - lo
    drift_x = inj_x + _DRIFT_OFFSET
    corr_x  = inj_x + _CORR_OFFSET
    phase3_end = len(x) - 1

    # ── 绘图 ─────────────────────────────────────────────────
    plt.style.use("classic")
    fig, ax = plt.subplots(facecolor="white", figsize=(12, 8))
    ax.set_facecolor("white")

    ax.plot(x, y_adp, "g-", marker="o", ms=4, lw=1, label="Adjusted Prediction")
    ax.plot(x, y_bls, "r--", marker="o", ms=4, lw=1, label="Original Prediction")
    ax.plot(x, y_true,"b-",               lw=1,        label="Real data")

    # 垂直参考线
    ax.axvline(inj_x  , color="black" , ls="--", lw=2, label="Inject new data")
    ax.axvline(drift_x, color="orange", ls="--", lw=2, label="Drift Detect Time")
    ax.axvline(corr_x , color="magenta",ls="--", lw=2, label="Correction completion")

    # ── 顶部三段 Phase 箭头 ──────────────────────────────────
    y_top = y_true.max() * 1.03
    ax.annotate("", xy=(0,          y_top), xytext=(inj_x,   y_top),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.annotate("", xy=(inj_x,      y_top), xytext=(corr_x,  y_top),
                arrowprops=dict(arrowstyle="<->", color="orange"))
    ax.annotate("", xy=(corr_x,     y_top), xytext=(phase3_end, y_top),
                arrowprops=dict(arrowstyle="<->", color="magenta"))

    ax.text(inj_x/2,                y_top*1.02, "Phase 1", ha="center", fontsize=14)
    ax.text((inj_x+corr_x)/2,       y_top*1.02, "Phase 2", ha="center", color="orange", fontsize=14)
    ax.text((corr_x+phase3_end)/2,  y_top*1.02, "Phase 3", ha="center", color="magenta", fontsize=14)

    # 纵轴 / 网格 / 图例
    ax.set_xlabel("Time Series Index", fontsize=14)
    ax.set_ylabel("Throughput (Mbps)", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.grid(True, ls="--", lw=0.4)
    ax.legend(loc="lower right", fontsize=12)

    # 顶部留白
    ax.set_ylim(top=y_top * 1.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_report] saved figure → {save_path}")
