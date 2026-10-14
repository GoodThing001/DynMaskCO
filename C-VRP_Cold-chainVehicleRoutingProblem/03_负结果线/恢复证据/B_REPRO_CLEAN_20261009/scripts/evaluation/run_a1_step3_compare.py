"""A-v1 步骤 3 比较驱动：MaskCO 场景生成器 vs 条件历史采样强基线（共用 SAA 下游，同 K/时限/口径）。

锁定（步骤 3 协议）：主比较 = maskco − cond_hist 按天配对，均值 ≥ δ₃=20 且 CI 下界>0；
开发集 = seed 20260926（已查看，标开发证据）；论文测试 = seed 20260927/28（--final-test 才跑）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from flax import nnx

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coldchain_evaluator_a1 import add_v2_initial_quality
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, build_history, build_history_times_classes,
                          make_c0_contract_v2)
from maskco_scenario import (MaskCOScenarioModel, MaskCOScenarioSampler, HistoryPools,
                             load_model, M_MAX)
from run_a1_step2_gate import (run_arm, summarize, paired_all_days, PENALTIES,
                               ARM_SEEDS, MAIN_PENALTY)

DELTA_3 = 20.0
TRAIN_SEED = 20260925
DEV_SEED = 20260926
FINAL_SEEDS = (20260927, 20260928)


def _build_sampler(spec):
    kind = spec["kind"]
    if kind == "cond_hist":
        return CondHistoricalSampler(spec["history"], n_neighbors=spec.get("k_neighbors", 10))
    if kind == "explicit_feat":
        return ExplicitFeatureSampler(spec["history"])
    if kind == "uncond_hist":
        return UncondHistoricalSampler(spec["history"])
    if kind == "maskco":
        cvrp = None
        if spec.get("cvrp_ckpt"):
            from mpre import load_cvrp_model
            cvrp, _cfg, _step = load_cvrp_model(spec["cvrp_ckpt"])
        model = MaskCOScenarioModel(dim=spec["dim"], arm=spec["arm"],
                                    cvrp_model=cvrp, rngs=nnx.Rngs(0))
        model = load_model(model, spec["model_bin"])
        return MaskCOScenarioSampler(model, spec["pools"], m_max=spec.get("m_max", M_MAX))
    raise ValueError("unknown sampler kind " + kind)


def _worker(task):
    (pname, name, gate_ds, contract, B, capacity, num_vehicles, cooling_share,
     spec, p_c, K, time_limit, arm_seed) = task
    sampler = _build_sampler(spec)
    rows = run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
                   sampler, p_c, K, time_limit, arm_seed)
    return pname, name, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bin", required=True, help="训练好的 maskco 模型 model.bin")
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained"], required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=M_MAX)
    ap.add_argument("--cvrp-ckpt", default=None)
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--dev-instances", type=int, default=40)
    ap.add_argument("--final-test", action="store_true",
                    help="在论文测试日（20260927/28）上跑；默认用开发集 20260926")
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--k-neighbors", type=int, default=10)
    ap.add_argument("--penalty", default=None)
    ap.add_argument("--main-only", action="store_true",
                    help="只跑主 p_c=(5,10,15)（首轮门判定用）")
    ap.add_argument("--only", default=None,
                    help="只跑指定臂（如 maskco），配合 --baseline-from 复用基线")
    ap.add_argument("--baseline-from", default=None,
                    help="从既有 gate.json 读基线臂（cond/explicit/uncond）逐日数据，省重跑")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    contract = make_c0_contract_v2()
    train_ds = generate_dataset(args.train_instances, args.n_orders, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, TRAIN_SEED, args.rho, hist_tc,
        k=args.k_neighbors)
    pools = HistoryPools(hist)

    seeds = FINAL_SEEDS if args.final_test else (DEV_SEED,)
    spec_maskco = {"kind": "maskco", "model_bin": os.path.abspath(args.model_bin),
                   "arm": args.arm, "dim": args.dim, "m_max": args.m_max,
                   "cvrp_ckpt": args.cvrp_ckpt, "pools": pools}
    specs = {"maskco": spec_maskco,
             "cond_hist": {"kind": "cond_hist", "history": hist,
                           "k_neighbors": args.k_neighbors},
             "explicit_feat": {"kind": "explicit_feat", "history": hist},
             "uncond_hist": {"kind": "uncond_hist", "history": hist}}
    if args.only:
        specs = {args.only: specs[args.only]}

    baseline_rows = {}
    if args.baseline_from:
        base = json.load(open(args.baseline_from, encoding="utf-8"))
        for seed_s, blocks in base["results"].items():
            for pname, blk in blocks.items():
                baseline_rows[(int(seed_s), pname)] = blk["per_day"]

    penalty_items = ([(args.penalty, PENALTIES[args.penalty])] if args.penalty
                     else ([(MAIN_PENALTY, PENALTIES[MAIN_PENALTY])] if args.main_only
                           else list(PENALTIES.items())))
    per_day, results = {}, {}
    for gate_seed in seeds:
        gate_ds = add_v2_initial_quality(
            generate_dataset(args.dev_instances, args.n_orders, gate_seed), contract)
        tasks = []
        for pname, p_c in penalty_items:
            for name, spec in specs.items():
                tasks.append((pname, name, gate_ds, contract, B, args.capacity,
                              args.num_vehicles, cooling_share, spec, p_c,
                              args.K, args.time_limit, ARM_SEEDS.get(name, 7009)))
        if args.workers > 1:
            with Pool(args.workers) as pool:
                task_results = pool.map(_worker, tasks)
        else:
            task_results = [_worker(t) for t in tasks]
        for pname, name, rows in task_results:
            blk = results.setdefault(gate_seed, {}).setdefault(pname, {})
            blk.setdefault("arms", {})[name] = summarize(rows, gate_seed)
            per_day.setdefault(gate_seed, {}).setdefault(name, {})[pname] = rows
        # 从基线文件补上未重跑的基线臂
        for pname in list(results[gate_seed]):
            if (gate_seed, pname) in baseline_rows:
                for arm_name in ("cond_hist", "explicit_feat", "uncond_hist"):
                    if arm_name in results[gate_seed][pname]["arms"]:
                        continue
                    b_rows = baseline_rows[(gate_seed, pname)].get(arm_name)
                    if b_rows is None:
                        continue
                    per_day[gate_seed].setdefault(arm_name, {})[pname] = b_rows
                    results[gate_seed][pname]["arms"][arm_name] = summarize(b_rows, gate_seed)
        for pname in results[gate_seed]:
            m = results[gate_seed][pname]["arms"]
            st = paired_all_days(per_day[gate_seed]["maskco"][pname],
                                 per_day[gate_seed]["cond_hist"][pname], gate_seed)
            both_ok = (m["maskco"]["hard_feasible_rate"] == 1.0
                       and m["cond_hist"]["hard_feasible_rate"] == 1.0)
            passed = bool(both_ok and np.isfinite(st["mean"])
                          and st["mean"] >= DELTA_3 and st["ci_lo"] > 0)
            results[gate_seed][pname]["main_comparison"] = {
                "maskco_minus_cond_hist": st, "delta_3": DELTA_3,
                "criterion": "mean>=delta_3 AND ci_lo>0 AND both hard_feasible_rate==1.0",
                "passed": passed, "violations_present": not both_ok}
            results[gate_seed][pname]["maskco_minus_explicit_feat"] = paired_all_days(
                per_day[gate_seed]["maskco"][pname],
                per_day[gate_seed]["explicit_feat"][pname], gate_seed)
            results[gate_seed][pname]["maskco_minus_uncond_hist"] = paired_all_days(
                per_day[gate_seed]["maskco"][pname],
                per_day[gate_seed]["uncond_hist"][pname], gate_seed)
            results[gate_seed][pname]["per_day"] = {
                name: per_day[gate_seed][name][pname] for name in per_day[gate_seed]}

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED,
                  "final_test_reserved": list(FINAL_SEEDS),
                  "ran": list(seeds)},
        "budget": {"B": float(B), "cooling_share": float(cooling_share)},
        "delta_3": DELTA_3,
        "results": {str(s): results[s] for s in seeds},
        "note": "A-v1 步骤 3：MaskCO 场景生成器 vs 条件历史采样强基线（共用 SAA 下游、同 K/时限、不筛天）。"
                "主比较 = maskco − cond_hist；只赢无条件/显式特征不算 MaskCO 贡献。",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
