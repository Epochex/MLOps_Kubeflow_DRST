# kafka_streaming/plot_final.py
import numpy as np
from ml.plot_report import generate_report
from shared.minio_helper import load_np, save_bytes
from shared.config import RESULT_DIR, CORRECTION_OFFSET

def load_with_fallback(key: str) -> np.ndarray:
    print(f"[read] loading from MinIO: {key}")
    try:
        return load_np(key)
    except Exception as e:
        print(f"[missing] {key} 不存在，使用空数组代替: {e}")
        return np.empty((0,), dtype=np.float32)

bridge_true    = load_with_fallback(f"{RESULT_DIR}/bridge_true.npy")
bridge_pred    = load_with_fallback(f"{RESULT_DIR}/bridge_pred.npy")
dag1_pred_orig = load_with_fallback(f"{RESULT_DIR}/dag1_pred_orig.npy")
dag1_true      = load_with_fallback(f"{RESULT_DIR}/dag1_true.npy")
idx            = load_with_fallback(f"{RESULT_DIR}/idx.npy")

if any(arr.size == 0 for arr in [bridge_true, bridge_pred, dag1_pred_orig, dag1_true, idx]):
    print("[skip] 缺少必要数据，跳过绘图")
else:
    tmp_path = "report_tmp.png"
    print(f"[plot] generating report to {tmp_path}")
    try:
        generate_report(
            bridge_true=bridge_true,
            bridge_pred_orig=bridge_pred,
            dag1_pred_orig=dag1_pred_orig,
            y_pred_dag1_new=dag1_pred_orig,
            yd1=dag1_true,
            correction_offset=CORRECTION_OFFSET,
            save_path=tmp_path
        )
    except Exception as e:
        print(f"[error] generate_report 失败: {e}")
    else:
        key = f"{RESULT_DIR}/report_final.png"
        print(f"[write] uploading to MinIO: {key}")
        try:
            with open(tmp_path, "rb") as f:
                save_bytes(key, f.read(), content_type="image/png")
        except Exception as e:
            print(f"[error] failed to write {key}: {e}")
        else:
            print("[plot_final] report_final.png uploaded")

print("[plot_final] DONE")
