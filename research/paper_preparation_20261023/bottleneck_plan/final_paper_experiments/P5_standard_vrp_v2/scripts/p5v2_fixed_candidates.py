# -*- coding: utf-8 -*-
"""固定候选数量诊断：相同初解/掩码/候选集/插入顺序，比较 4 种排序的插位质量。

先于运行登记（EXPERIMENT_PROTOCOL_V2 §3）。要点：
- 初解 = nn_witness（与 60 秒面板同一构造式见证）；
- 掩码 = 40 客户（锁版池容量），按"锚点空间最近"确定性选取；
- 候选集 = 全部容量/时窗可行插位 + 新增单车候选（每个客户相同）；
- 插入顺序 = 种子固定洗牌（与排序无关，四种排序共用同一顺序）；
- 排序变体：
    distance         dist 增量最小
    heat_raw         预测器原输出（v2_fix 特征）
    heat_calibrated  锁版校准 spatial + 0.25·residual（v2_fix 特征）
    heat_norm        锁版校准（v2_norm 特征：需求/容量、坐标中位对齐）
- 逐客户记录：候选数、oracle 最优 key、所选 key、后悔值、oracle 在该排序中的名次；
  汇总：后悔值总和、recall@1/@3/@5、候选评价数、最终解目标。

用法（远端 MASKCO_env）：
  JAX_PLATFORMS=cpu python p5v2_fixed_candidates.py <inst_dir> <manifest> <case> <seed> <out.json>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from standard_vrp.adapters.objective import adapter_for
from standard_vrp.core.calibration import calibrated_heat
from standard_vrp.core.standard_evaluator import CVRPInstance, VRPTWInstance
from standard_vrp.core.standard_features import StandardFeatureBuilder
from standard_vrp.solver.pool_search import nn_witness

PROJECT = Path("/home/hzeng/project/MASKCO-Main/_maskco_cross_dataset_20261119")
MASK_SIZE = 40


def load_instance(path, objective):
    p = Path(path)
    if objective == "cvrp":
        return CVRPInstance.parse_vrp(p.read_text(encoding="utf-8"))
    return VRPTWInstance.parse_solomon(p)


def make_predictor():
    sys.path.insert(0, str(PROJECT))
    import compare
    access, core, source, locked = compare.setup("SLOT")
    policy, scope = compare.load("SLOT", access, core, source, locked, "c205")
    from profit_neural import predict

    def forward(batch):
        with access.restricted_scope(scope):
            logits = predict(policy, batch)
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    return forward


def select_mask(inst, routes, seed):
    rng = np.random.default_rng(seed)
    customers = sorted(c for r in routes for c in r)
    anchor = customers[int(rng.integers(len(customers)))]
    ax, ay = inst.coords[anchor]
    ranked = sorted(customers, key=lambda c: (math.hypot(inst.coords[c][0] - ax,
                                                         inst.coords[c][1] - ay), c))
    return ranked[: min(MASK_SIZE, len(ranked))]


def candidates_for(adapter, routes, cust):
    """[(kind, route_idx, pos, key, new_route, successor)]。"""
    out = []
    for ri, r in enumerate(routes):
        for pos in range(len(r) + 1):
            if not adapter.feasible_insert(r, pos, cust):
                continue
            new_route = list(r[:pos]) + [cust] + list(r[pos:])
            succ = new_route[pos + 1] if pos + 1 < len(new_route) else 0
            out.append(("insert", ri, pos,
                        adapter.key(new_route, adapter.insert_delta(r, pos, cust)),
                        new_route, succ))
    out.append(("new", len(routes), 0,
                adapter.key([cust], adapter.singleton_delta(cust)), [cust], 0))
    return out


def scalar(key):
    return key[1] if isinstance(key, tuple) else key


def pick_distance(adapter, cands):
    best = cands[0]
    for c in cands[1:]:
        if adapter.is_better(c[3], best[3]):
            best = c
    return best


def rank_of(adapter, cands, chosen):
    return sum(1 for c in cands if adapter.is_better(c[3], chosen[3]))


def main(inst_dir, manifest_path, case, seed, dest):
    dev = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    row = next(r for r in dev["instances"] if r["name"] == case)
    objective = "cvrp" if row["kind"] == "cvrp" else "vrptw"
    inst = load_instance(Path(inst_dir) / row["file"], objective)
    adapter = adapter_for(inst)
    seed = int(seed)
    init = [tuple(r) for r in nn_witness(inst)]
    ok0, val0, k0 = adapter.evaluate(init)
    if not ok0:
        raise SystemExit("initial witness infeasible")
    mask = select_mask(inst, init, seed)
    mset = set(mask)
    kept = [tuple(c for c in r if c not in mset) for r in init]
    kept = [r for r in kept if r]
    # 掩码状态是"部分解"（缺 40 客户），不能用完整解评价（覆盖性必然不足）；
    # 只需保证每条剩余路线自身可行（容量/时窗），记录其路线代价之和。
    if any((not r) or (not adapter.route_ok(r)) for r in kept):
        raise SystemExit("state after mask removal infeasible")
    valk = sum(adapter.route_proxy(r) for r in kept)
    kk = len(kept)

    forward = make_predictor()
    fb_fix = StandardFeatureBuilder(inst, objective, mode="v2_fix")
    fb_norm = StandardFeatureBuilder(inst, objective, mode="v2_norm")
    t0 = time.perf_counter()
    logits_fix = np.array(forward(fb_fix.build(kept, mset)), dtype=float)
    t_fix = time.perf_counter() - t0
    t0 = time.perf_counter()
    logits_norm = np.array(forward(fb_norm.build(kept, mset)), dtype=float)
    t_norm = time.perf_counter() - t0
    D = np.array(inst.dist, dtype=float)
    hmap = {"heat_raw": logits_fix,
            "heat_calibrated": calibrated_heat(D, logits_fix),
            "heat_norm": calibrated_heat(D, logits_norm)}

    # 固定插入顺序（所有排序共用）
    rng = np.random.default_rng(seed + 1)
    order = [int(x) for x in rng.permutation(np.array(mask, dtype=int))]

    results = {}
    for variant in ("distance", "heat_raw", "heat_calibrated", "heat_norm"):
        routes = [tuple(r) for r in kept]
        steps = []
        regret_sum = 0.0
        r1 = r3 = r5 = 0
        n_evals = 0
        for cust in order:
            cands = candidates_for(adapter, routes, cust)
            n_evals += len(cands)
            oracle = pick_distance(adapter, cands)
            if variant == "distance":
                chosen = oracle
                at = 0
            else:
                hm = hmap[variant]
                def hscore(c, cust=cust, hm=hm):
                    return (-float(hm[cust][c[5]]), c[1], c[2])
                chosen = min(cands, key=hscore)
                at = rank_of(adapter, cands, chosen)
            regret = scalar(oracle[3]) - scalar(chosen[3])
            regret_sum += max(0.0, regret)
            r1 += (at == 0)
            r3 += (at < 3)
            r5 += (at < 5)
            steps.append(dict(cust=int(cust), n_cands=len(cands),
                              oracle=scalar(oracle[3]), chosen=scalar(chosen[3]),
                              regret=max(0.0, regret), oracle_rank=at,
                              kind=chosen[0]))
            routes = ([tuple(x) for x in routes] + [tuple(chosen[4])] if chosen[0] == "new"
                      else [tuple(chosen[4]) if i == chosen[1] else r
                            for i, r in enumerate(routes)])
        ok, val, k = adapter.evaluate(routes)
        n = len(order)
        results[variant] = dict(final=dict(feasible=ok, distance=val, vehicles=k),
                                regret_sum=regret_sum,
                                mean_regret=regret_sum / n,
                                recall_at_1=r1 / n, recall_at_3=r3 / n, recall_at_5=r5 / n,
                                candidate_evals=n_evals, steps=steps)

    payload = dict(case=case, seed=seed, objective=objective, dimension=inst.dimension,
                   mask=mask, insertion_order=[int(x) for x in order],
                   initial=dict(distance=val0, vehicles=k0),
                   initial_routes=[[int(x) for x in r] for r in init],
                   initial_U=dict(distance=val0, vehicles=k0),
                   after_removal=dict(distance=valk, vehicles=kk,
                                      routes=[list(r) for r in kept]),
                   forward_s=dict(v2_fix=t_fix, v2_norm=t_norm),
                   variants=results,
                   registration=dict(mask_size=MASK_SIZE,
                                     variants=["distance", "heat_raw",
                                               "heat_calibrated", "heat_norm"]))
    Path(dest).write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                     default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                          encoding="utf-8")
    print(json.dumps({v: dict(final=results[v]["final"],
                              mean_regret=results[v]["mean_regret"],
                              recall_at_1=results[v]["recall_at_1"])
                      for v in results}, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:6])
