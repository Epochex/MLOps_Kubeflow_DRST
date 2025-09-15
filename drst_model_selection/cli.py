# drst_model_selection/cli.py
from __future__ import annotations
import argparse

# 这里改成新文件名
from .pcm_select import run_pcm_selection
from .perf_select import run_perf_selection

def main():
    p = argparse.ArgumentParser("DRST model selection")
    p.add_argument("--task", choices=["pcm", "perf"], required=True)

    # PCM 参数
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--take_last", type=int, default=4000)
    p.add_argument("--topk", type=int, default=3)

    # PERF 参数
    p.add_argument("--perf_key", type=str, default="datasets/perf/stage1_random_rates.csv")
    p.add_argument("--include_svr", type=int, default=0)
    p.add_argument("--include_dt", type=int, default=0)

    args = p.parse_args()

    if args.task == "pcm":
        run_pcm_selection(
            lookback=args.lookback,
            horizon=args.horizon,
            take_last=args.take_last,
            topk=args.topk,
        )
    else:
        run_perf_selection(
            perf_key=args.perf_key,
            topk=args.topk,
            include_svr=bool(args.include_svr),
            include_dt=bool(args.include_dt),
        )

if __name__ == "__main__":
    main()
