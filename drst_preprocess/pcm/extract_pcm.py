# DRST-SoftwarizedNetworks/drst_preprocess/pcm/extract_pcm.py
# -*- coding: utf-8 -*-
"""
抽取 PCM 预处理后的特征，生成训练用的汇总表。
修复点：
1) System-Date / System-Time 保留为文本，不再被强制转 float 导致空白。
2) 列名前缀与“正确答案”保持一致：带 *_processed_ 前缀。
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd

DEFAULT_FOLDERS = [
    'const-1gbps','const-2gbps','const-3gbps','const-4gbps',
    'const-5gbps','const-6gbps','const-7gbps','const-8gbps',
    'const-9gbps','const-10gbps-1','const-10gbps-2','const-10gbps-3','const-10gbps-4'
]

# 明确保留为文本的列名（来自 ndpi_stats/pcm 工具）
TEXT_COL_SUFFIXES = ('System-Date', 'System-Time')


def read_headers(csv_path: Path):
    with csv_path.open(newline='') as f:
        reader = csv.reader(f)
        return next(reader)


def parse_column(csv_path: Path, col_name: str):
    """
    对指定列进行解析：
    - System-Date / System-Time 原样保留字符串
    - 其他列尽量转为 float，失败则 NaN
    """
    vals = []
    with csv_path.open(newline='') as f:
        r = csv.reader(f)
        header = next(r)
        if col_name not in header:
            return []
        idx = header.index(col_name)
        is_text = any(col_name.endswith(suf) for suf in TEXT_COL_SUFFIXES)
        for row in r:
            raw = row[idx] if idx < len(row) else ''
            if is_text:
                vals.append(raw)
            else:
                try:
                    vals.append(float(raw))
                except (ValueError, TypeError):
                    vals.append(np.nan)
    return vals


def min_common_len(arrs):
    return min(len(a) for a in arrs if a is not None and len(a) > 0)


def classify_packetrate(rate: float) -> int:
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):  # NaN
        return 1514
    if rate > 10.5: return 64
    if 7   <= rate <= 10.5: return 128
    if 3   <= rate < 7:     return 256
    if 1.8 <= rate < 3:     return 512
    if 1   <= rate < 1.8:   return 1024
    return 1514


def make_one_df(base: Path) -> pd.DataFrame:
    # 基础三列
    tx_csv = base / 'tx_stats.csv'
    rx_csv = base / 'rx_stats.csv'
    lat_csv = base / 'latency_processed.csv'

    # 优先用 Mbit；没有就用 PacketRate
    tx_series = parse_column(tx_csv, 'Mbit') if tx_csv.exists() else []
    if not tx_series and tx_csv.exists():
        tx_series = parse_column(tx_csv, 'PacketRate')

    rx_series = parse_column(rx_csv, 'Mbit') if rx_csv.exists() else []
    if not rx_series and rx_csv.exists():
        rx_series = parse_column(rx_csv, 'PacketRate')

    latency = []
    if lat_csv.exists():
        # latency_processed.csv 只有一列 latency；根据你们采集，有的单位是 μs，有的是 ms
        # 这里按原先实现：尝试 /1000（μs→ms）；如确认已是 ms，可把 /1000.0 去掉。
        with lat_csv.open() as f:
            next(f, None)  # skip header
            for line in f:
                cell = line.strip().split(',')[0]
                try:
                    latency.append(float(cell) / 1000.0)
                except ValueError:
                    latency.append(np.nan)

    # 目标长度
    seq_len = min_common_len([tx_series, rx_series, latency]) if latency else min_common_len([tx_series, rx_series])

    data = {
        'input_rate':  tx_series[:seq_len],
        'output_rate': rx_series[:seq_len],
    }
    if latency:
        data['latency'] = latency[:seq_len]

    # ndpi_stats-pcm_processed.csv
    ndpi_csv = base / 'ndpi_stats-pcm_processed.csv'
    if ndpi_csv.exists():
        ndpi_hdr = read_headers(ndpi_csv)
        for h in ndpi_hdr:
            col = parse_column(ndpi_csv, h)[:seq_len]
            # 补齐
            if len(col) < seq_len:
                if any(h.endswith(suf) for suf in TEXT_COL_SUFFIXES):
                    col = list(col) + [''] * (seq_len - len(col))
                else:
                    col = list(col) + [np.nan] * (seq_len - len(col))
            data[f'ndpi_stats-pcm_processed_{h.replace(",","")}'] = col

    # pcm-pcie_processed.csv
    pcie_csv = base / 'pcm-pcie_processed.csv'
    if pcie_csv.exists():
        pcie_hdr = read_headers(pcie_csv)
        for h in pcie_hdr:
            col = parse_column(pcie_csv, h)[:seq_len]
            if len(col) < seq_len:
                col = list(col) + [np.nan]*(seq_len-len(col))
            data[f'pcm-pcie_processed_{h.replace(",","")}'] = col

    # pcm-memory_processed.csv
    mem_csv = base / 'pcm-memory_processed.csv'
    if mem_csv.exists():
        mem_hdr = read_headers(mem_csv)
        for h in mem_hdr:
            col = parse_column(mem_csv, h)[:seq_len]
            if len(col) < seq_len:
                col = list(col) + [np.nan]*(seq_len-len(col))
            data[f'pcm-memory_processed_{h.replace(",","")}'] = col

    # tx_stats 一些额外列
    for h in ['PacketRate','Mbit','MbitWithFraming','TotalPackets','TotalBytes']:
        col = parse_column(tx_csv, h) if tx_csv.exists() else []
        if col:
            col = col[:seq_len]
            if len(col) < seq_len:
                col = list(col) + [np.nan]*(seq_len-len(col))
            data[f'tx_stats_{h}'] = col

    df = pd.DataFrame(data)

    # 派生 packetsize
    if 'tx_stats_PacketRate' in df.columns:
        df['packetsize'] = df['tx_stats_PacketRate'].apply(classify_packetrate)
    elif 'tx_stats_Mbit' in df.columns:
        df['packetsize'] = df['tx_stats_Mbit'].apply(classify_packetrate)
    else:
        df['packetsize'] = 1514

    return df


def extract_all(input_root: Path, exp_root: str, folders, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        base = input_root / 'pcm' / exp_root / folder
        if not base.exists():
            print(f'[WARN] {base} not found, skip.')
            continue
        df = make_one_df(base)
        out_path = out_dir / f'{exp_root}_{folder}.csv'
        df.to_csv(out_path, index=False)
        print(f'[OK] write -> {out_path} (rows={len(df)})')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_root', type=str, default='../data')
    ap.add_argument('--exp_root', type=str, default='const_input')
    ap.add_argument('--folders', type=str, nargs='*', default=DEFAULT_FOLDERS)
    ap.add_argument('--out_dir', type=str, default='../datasets_pcm')
    args = ap.parse_args()
    extract_all(Path(args.input_root), args.exp_root, args.folders, Path(args.out_dir))


if __name__ == '__main__':
    main()
