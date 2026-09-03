"""DCC .npz → POMO+PIP TSPTW .pkl 参考桥接（单车辆 TSPTW 松弛）。

⚠️ 仅用于「验证 PIP 代码可运行」，结果**不可比** —— 见本目录 README.md。

把一个 DCC 实例退化成单车辆 TSPTW：
  - 丢弃容量 / 多车辆 / 需求（单车辆访问所有客户一次）；
  - service 置 0（PIP 硬 TSPTW 训练分布无 service）；
  - 坐标与时间窗整体 ×100 对齐（DCC coords∈[0,1]² → [0,100]²，
    TW∈[0,24] → [0,2400]，速度=1，可行性不变）。

输出格式与 `POMO+PIP/TSPTWEnv.load_dataset` 一致：
  .pkl = list(zip(node_xy, service_time, tw_start, tw_end))，每项为 (N,2)/(N,)/(N,)/(N,)
  的 python list；node_xy[0] 为 depot。

用法（服务器, pip conda env）:
    python dcc_vrp/dcc_data_tsptw.py --data <dcc.npz> --out <out.pkl>
"""
import argparse
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DCC .npz 文件")
    ap.add_argument("--out", required=True, help="输出 TSPTW .pkl 路径")
    opts = ap.parse_args()

    d = np.load(opts.data, allow_pickle=True)
    coords = d["coords"].astype(np.float32)       # (B, N+1, 2) depot index 0, [0,1]
    tw_start = d["tw_start"].astype(np.float32)   # (B, N+1) [0,24]
    tw_end = d["tw_end"].astype(np.float32)

    LOC_FACTOR = 100.0                            # 对齐 POMO+PIP 的 loc_factor
    B = coords.shape[0]
    rows = []
    for i in range(B):
        node_xy = (coords[i] * LOC_FACTOR).tolist()
        service = [0.0] * coords.shape[1]         # 单车辆 TSPTW：service 置 0
        tws = (tw_start[i] * LOC_FACTOR).tolist()
        twe = (tw_end[i] * LOC_FACTOR).tolist()
        rows.append((node_xy, service, tws, twe))

    # POMO+PIP 的 generate_dataset 用 list(zip(*dataset))，等价于按字段转置
    with open(opts.out, "wb") as f:
        pickle.dump(list(zip(*rows)), f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {B} TSPTW instances (single-vehicle relaxation) to {opts.out}")


if __name__ == "__main__":
    main()
