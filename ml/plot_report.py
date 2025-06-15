#!/usr/bin/env python3
"""
ml.plot_report.py ─ Phase-1 / 2 / 3 对比图（定制版）
──────────────────────────────────────────────────────────────
• draw_three_phases():
  - 注入点固定为样本序号 400
  - 保留注入前 100 点和注入后 700 点，共 800 点窗口
  - 丢弃真实值 output < 500 的样本
  - 绘制 Baseline / Adaptive / Real 三条曲线及垂直参考线与 Phase 标签
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def draw_three_phases(
    bridge_true: np.ndarray,
    bridge_pred: np.ndarray,
    dag_true: np.ndarray,
    dag_pred_orig: np.ndarray,
    dag_pred_adj: np.ndarray,
    dt: float,
    save_path: str
) -> None:
    """
    Parameters
    ----------
    bridge_true : np.ndarray
        Phase-1 真实值序列
    bridge_pred : np.ndarray
        Phase-1 Baseline 预测序列
    dag_true : np.ndarray
        Phase-2/3 真实值序列
    dag_pred_orig : np.ndarray
        Phase-2/3 Baseline 预测序列
    dag_pred_adj : np.ndarray
        Phase-2/3 Adaptive 预测序列
    dt : float
        保留输入兼容性（未在本函数中使用）
    save_path : str
        输出 PNG 文件路径

    Behavior
    --------
    - 注入点设为样本索引 400
    - 切片窗口为 [inject_idx-100 : inject_idx+700]
    - 过滤掉 y_true < 500 的样本
    - 绘制三条曲线，添加垂直参考线及 Phase 标签
    """
    # 拼接完整数据序列
    y_true_all = np.concatenate([bridge_true, dag_true])
    y_pred_bls = np.concatenate([bridge_pred, dag_pred_orig])
    y_pred_adp = np.concatenate([bridge_pred, dag_pred_adj])

    # 定义注入点和窗口范围
    inject_idx = 400
    lo = inject_idx - 100
    hi = inject_idx + 700

    # 窗口切片
    y_true = y_true_all[lo:hi]
    y_bls = y_pred_bls[lo:hi]
    y_adp = y_pred_adp[lo:hi]

    # 丢弃真实值小于 500 的样本
    mask = (y_true >= 500)
    y_true = y_true[mask]
    y_bls = y_bls[mask]
    y_adp = y_adp[mask]
    x_all = np.arange(len(mask))[mask]

    # 参考线位置
    inj_x = 100
    drift_offset = 1
    corr_offset = 38
    drift_x = inj_x + drift_offset
    corr_x = inj_x + corr_offset
    phase3_end = x_all.max() if x_all.size else inj_x

    # 计算 y 轴上限
    y_top = max(
        y_true.max() if y_true.size else 0,
        y_bls.max() if y_bls.size else 0,
        y_adp.max() if y_adp.size else 0
    ) * 1.03

    # 开始绘图
    fig, ax = plt.subplots(facecolor="white", figsize=(12, 8))
    ax.set_facecolor("white")

    ax.plot(x_all, y_adp, "g-", marker="o", ms=4, lw=1, label="Adjusted Prediction")
    ax.plot(x_all, y_bls, "r--", marker="o", ms=4, lw=1, label="Original Prediction")
    ax.plot(x_all, y_true, "b-", lw=1, label="Real data")

    # 垂直参考线
    ax.axvline(inj_x, ls="--", lw=2, label="Inject new data")
    ax.axvline(drift_x, ls="--", lw=2, label="Drift Detect Time")
    ax.axvline(corr_x, ls="--", lw=2, label="Correction completion")

    # Phase 标签
    ax.text(inj_x / 2, y_top, "Phase 1", ha="center", fontsize=14)
    ax.text((inj_x + corr_x) / 2, y_top, "Phase 2", ha="center", fontsize=14)
    ax.text((corr_x + phase3_end) / 2, y_top, "Phase 3", ha="center", fontsize=14)

    # 坐标轴与网格
    ax.set_xlabel("Sample Index (windowed)", fontsize=14)
    ax.set_ylabel("Throughput ≥500 only", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.grid(True, ls="--", lw=0.4)
    ax.legend(loc="lower right", fontsize=12)

    ax.set_ylim(top=y_top * 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot_report] saved figure → {save_path}")
