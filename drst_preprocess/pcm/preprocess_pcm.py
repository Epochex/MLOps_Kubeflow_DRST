# DRST-SoftwarizedNetworks/drst_preprocess/pcm/preprocess_pcm.py
# -*- coding: utf-8 -*-
import re
import csv
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

VNFS = ['bridge','firewall','ndpi_stats','nf_router','payload_scan','rx','tx']
DEFAULT_FOLDERS = [
    'const-1gbps','const-2gbps','const-3gbps','const-4gbps',
    'const-5gbps','const-6gbps','const-7gbps','const-8gbps',
    'const-9gbps','const-10gbps-1','const-10gbps-2','const-10gbps-3','const-10gbps-4'
]

def combine_csv_headers(input_file: Path, output_file: Path) -> None:
    with input_file.open(newline='') as f_in:
        reader = csv.reader(f_in)
        first = next(reader)
        second = next(reader)
        combined = [f"{a}-{b}" for a, b in zip(first, second)]

    with output_file.open('w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow(combined)
        with input_file.open(newline='') as f_in2:
            reader = csv.reader(f_in2)
            next(reader); next(reader)  # skip first two header lines
            for row in reader:
                w.writerow(row)

def reshape_pcie_and_clean(input_file: Path, output_file: Path) -> None:
    with input_file.open('r') as f:
        rows = list(csv.reader(f))
    if not rows:
        output_file.write_text('')
        return

    original_header = rows[0]
    new_header = (
        [f'skt-0_{h}-total' for h in original_header] +
        [f'skt-0_{h}-miss'  for h in original_header] +
        [f'skt-0_{h}-hit'   for h in original_header] +
        [f'skt-1_{h}-total' for h in original_header] +
        [f'skt-1_{h}-miss'  for h in original_header] +
        [f'skt-1_{h}-hit'   for h in original_header]
    )

    reshaped = []
    for i in range(1, len(rows), 7):
        take = []
        for k in range(6):  # rows i .. i+5
            if i + k < len(rows):
                take += rows[i + k]
        if take:
            reshaped.append(take)

    # drop useless “Skt-*” summary cols if present
    to_drop = {}
    for name in [
        'skt-0_Skt-total','skt-1_Skt-total',
        'skt-0_Skt-miss','skt-1_Skt-miss',
        'skt-0_Skt-hit','skt-1_Skt-hit'
    ]:
        if name in new_header:
            to_drop[name] = new_header.index(name)

    keep_idx = [i for i,_ in enumerate(new_header) if i not in set(to_drop.values())]
    kept_header = [new_header[i] for i in keep_idx]
    kept_rows   = [[row[i] if i < len(row) else '' for i in keep_idx] for row in reshaped]

    # numeric cleaning with robust float regex
    float_re = re.compile(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    df = pd.DataFrame(kept_rows, columns=kept_header)
    for c in df.columns:
        df[c] = (
            df[c].astype(str)
                 .apply(lambda s: float(float_re.search(s).group(1)) if float_re.search(s) else pd.NA)
        )
    df.to_csv(output_file, index=False)

def remove_date_time_empty_cols(input_file: Path, output_file: Path) -> None:
    with input_file.open('r') as f:
        rows = list(csv.reader(f))
    if not rows:
        output_file.write_text('')
        return
    header = rows[0]
    def is_empty_col(idx: int) -> bool:
        for r in rows[1:]:
            if idx < len(r) and r[idx] != '':
                return False
        return True
    drop_idx = [i for i,h in enumerate(header)
                if h.strip().lower() in ('-date','-time') or is_empty_col(i)]
    keep_idx = [i for i in range(len(header)) if i not in set(drop_idx)]
    new_header = [header[i] for i in keep_idx]
    new_rows = [[r[i] if i < len(r) else '' for i in keep_idx] for r in rows[1:]]
    with output_file.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(new_header); w.writerows(new_rows)

def latency_pre(input_csv: Path, output_csv: Path, delimiter: str = ' ') -> None:
    df = pd.read_csv(input_csv, delimiter=delimiter, header=None)
    if df.shape[1] >= 2:   # 丢弃第一列时间戳/编号，只留数值列
        df.drop(columns=df.columns[0], inplace=True)
    df.columns = ['latency']
    df.to_csv(output_csv, index=False)

NF_HEADER = ["tag","instance_id","service_id","thread_info.core",
             "rx_pps","tx_pps","rx","tx","act_out","act_tonf","act_drop","thread_info.parent",
             "state","rte_atomic16_read","rx_drop_rate","tx_drop_rate","rx_drop","tx_drop",
             "act_next","act_buffer","act_returned"]

def nf_out_pre(input_file: Path, output_file: Path) -> None:
    with input_file.open(newline='') as f:
        data = list(csv.reader(f))
    with output_file.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(NF_HEADER); w.writerows(data)

def nf_out_widen_inplace(file_path: Path) -> None:
    df = pd.read_csv(file_path)
    if {'tag','instance_id','service_id'}.issubset(df.columns):
        df.drop(['instance_id','service_id'], axis=1, inplace=True, errors='ignore')
    tags = ['ndpi_stat','router','payload_scan','bridge','firewall']
    features = ["thread_info.core","rx_pps","tx_pps","rx","tx","act_out","act_tonf","act_drop",
                "thread_info.parent","state","rte_atomic16_read","rx_drop_rate","tx_drop_rate",
                "rx_drop","tx_drop","act_next","act_buffer","act_returned"]
    new_cols = [f"{t}-{f}" for t in tags for f in features]
    new_df = pd.DataFrame(columns=new_cols)
    for t in tags:
        part = df[df['tag']==t].reset_index(drop=True)
        for f in features:
            new_df[f"{t}-{f}"] = part[f] if f in part.columns else pd.NA
    new_df.to_csv(file_path, index=False)

def preprocess_all(root: Path, exp_root: str = 'const_input', folders: Optional[List[str]] = None) -> None:
    folders = folders or DEFAULT_FOLDERS
    for folder in folders:
        base = root / 'pcm' / exp_root / folder
        base.mkdir(parents=True, exist_ok=True)

        # 1) vnf -pcm: 合并双行表头
        for v in VNFS:
            src = base / f'{v}-pcm.csv'
            dst = base / f'{v}-pcm_processed.csv'
            if src.exists():
                combine_csv_headers(src, dst)

        # 2) pcie：reshape + clean
        pcie_src = base / 'pcm-pcie.csv'
        pcie_out = base / 'pcm-pcie_processed.csv'
        if pcie_src.exists():
            reshape_pcie_and_clean(pcie_src, pcie_out)

        # 3) memory：合并双行表头 -> 删日期/空列
        mem_src = base / 'pcm-memory.csv'
        mem_mid = base / 'pcm-memory_processed.csv'
        if mem_src.exists():
            combine_csv_headers(mem_src, mem_mid)
            remove_date_time_empty_cols(mem_mid, mem_mid)

        # 4) latency：去掉第一列，保留一列 latency
        lat_src = base / 'latency.csv'
        lat_out = base / 'latency_processed.csv'
        if lat_src.exists():
            latency_pre(lat_src, lat_out)

        # 5) nf_out：补表头 -> 展宽
        nfo_src = base / 'nf_out.csv'
        nfo_out = base / 'nf_out_processed.csv'
        if nfo_src.exists():
            nf_out_pre(nfo_src, nfo_out)
            nf_out_widen_inplace(nfo_out)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_root', type=str, default='../data', help='包含 pcm/<const_input> 的根目录')
    ap.add_argument('--exp_root', type=str, default='const_input')
    ap.add_argument('--folders', type=str, nargs='*', default=DEFAULT_FOLDERS)
    args = ap.parse_args()
    preprocess_all(Path(args.input_root), args.exp_root, args.folders)
    print('[OK] PCM preprocess done.')

if __name__ == '__main__':
    sys.exit(main())
