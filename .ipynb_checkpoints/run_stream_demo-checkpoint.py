#!/usr/bin/env python
"""
One-click demo launcher

流程：
1. 先执行离线初训 (ml.train_offline) —— 生成 scaler / pca / mlp / base_acc
2. 再前台启动 consumer（持续监听 Kafka，空闲 30 s 自动结束）
3. consumer 启动后，后台启动 producer（把 CSV 推流进 Kafka）
4. Ctrl-C 时同时优雅终止 producer + consumer

运行：
    python run_stream_demo.py
"""
import subprocess, signal, sys, os, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parent      # repo 根
PY   = sys.executable                               # 当前 python 解释器

os.environ.setdefault("PYTHONPATH", str(ROOT))      # 让 “ml.” “shared.” 可 import

CMD_OFFLINE = [PY, "-m", "ml.train_offline"]
CMD_CONS    = [PY, "-m", "kafka_streaming.consumer"]
CMD_PROD    = [PY, "-m", "kafka_streaming.producer"]

def start(cmd, **kw):
    """wrap Popen，加统一打印"""
    print(f"[launcher] exec: {' '.join(cmd)}")
    return subprocess.Popen(cmd, **kw)

def main():
    #  离线初训（若 artefacts 已存在会覆写）
    subprocess.check_call(CMD_OFFLINE)

    # 前台 consumer（实时看日志）
    cons = start(CMD_CONS)

    # 后台 producer
    prod = start(CMD_PROD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        cons.wait()           # 阻塞直到 consumer idle-timeout 自行退出
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl-C pressed, stopping…")

    # 统一收尾
    for p in (prod, cons):
        if p.poll() is None:          # 还在运行
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    # 如果想查看 producer backlog，可解注释下两行
    # print(prod.stdout.read().decode())
    # print(prod.stderr.read().decode())

    print("[launcher] DONE.  see results/report_final.png")

if __name__ == "__main__":
    main()
