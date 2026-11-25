# -*- coding: utf-8 -*-
"""R101 witness 修复实验：Split 序列变体可行性对比（56 例全查）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standard_vrp.core.standard_evaluator import (VRPTWInstance, evaluate_vrptw, split_vrptw)
from standard_vrp.solver.standard_pool_search import nn_witness

d = HERE / "data/solomon/instances"
bad_a, bad_b, bad_c = [], [], []
kmax = []
for f in sorted(d.glob("*.vrp")):
    inst = VRPTWInstance.parse_solomon(f)
    n = inst.dimension
    # A: 现有 NN + 车队限额分布
    wa = nn_witness(inst)
    oka, ka, da, _, _ = evaluate_vrptw(inst, wa)
    # B: 全客户 NN 序（忽略窗/容量）+ Split
    seq, cur, left = [], 0, set(range(1, n + 1))
    while left:
        nxt = min(left, key=lambda i: inst.dist[cur][i])
        seq.append(nxt)
        left.remove(nxt)
        cur = nxt
    out_b = split_vrptw(inst, seq)
    okb = False
    if out_b:
        okb, kb, db, _, _ = evaluate_vrptw(inst, out_b["routes"])
    # C: 窗口开始时间排序 + Split
    seq_c = sorted(range(1, n + 1), key=lambda i: inst.windows[i][0])
    out_c = split_vrptw(inst, seq_c)
    okc = False
    if out_c:
        okc, kc, dc, _, _ = evaluate_vrptw(inst, out_c["routes"])
    if not oka: bad_a.append(f.stem)
    if not okb: bad_b.append(f.stem)
    if not okc: bad_c.append(f.stem)
    if okb and (not oka or f.stem == "R101"):
        print("%-6s B: K=%d D=%.2f (A %s K=%d D=%.2f; C %s)" % (f.stem, kb, db, oka, ka, da, okc))
print("A infeasible:", bad_a)
print("B infeasible:", bad_b)
print("C infeasible:", bad_c)
