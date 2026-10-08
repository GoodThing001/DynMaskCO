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
                             load_model, check_model_generation, M_MAX)
from run_a1_step2_gate import (run_arm, summarize, paired_all_days, PENALTIES,
                               ARM_SEEDS, MAIN_PENALTY)
from run_identity import (SHARED_SOURCE_FILES, source_seal, seal_end,
                          contract_identity, dataset_identity,
                          dataset_meta_sha256, pairing_identity_check)

# 步骤 3 封存清单（本驱动内联定义：run_identity.py 被 09:09 步骤 2 批次启动封存，
# 批次落地前不得改动——待批次结束后再把这行迁回 run_identity.py 统一维护）。
# 与步骤 2 的差异 = 本驱动 + maskco 模型/训练脚本（训练脚本是 .bin 的谱系来源）。
SOURCE_FILES_STEP3 = [
    'scripts/evaluation/run_a1_step3_compare.py',
    'scripts/evaluation/run_a1_step2_gate.py',
    'scripts/models/maskco_scenario.py',
    'scripts/training/train_maskco_scenario.py',
    'scripts/evaluation/run_identity.py',
] + SHARED_SOURCE_FILES

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
        if spec.get("cvrp_ckpt") and spec["arm"] in ("pretrained", "random_cvrp"):
            from mpre import load_cvrp_model
            cvrp, cfg, _step = load_cvrp_model(spec["cvrp_ckpt"])
            if spec["arm"] == "random_cvrp":
                # 路线图 09-30：同架构随机初始化消融（与训练时构造一致；2026-09-30 修复：
                # 经 config.construct_model + dataclasses.replace，不可直接 import CVRPModel）。
                from dataclasses import replace
                cvrp = replace(cfg, rngs=int(spec.get("model_seed", 42))).construct_model()
        model = MaskCOScenarioModel(dim=spec["dim"], arm=spec["arm"],
                                    cvrp_model=cvrp, rngs=nnx.Rngs(0))
        model = load_model(model, spec["model_bin"])
        return MaskCOScenarioSampler(model, spec["pools"], m_max=spec.get("m_max", M_MAX),
                                     iterative=spec.get("iterative", False),
                                     remask_frac=spec.get("remask_frac", 0.5),
                                     rounds=spec.get("rounds", 1))
    raise ValueError("unknown sampler kind " + kind)


def _worker(task):
    (pname, name, gate_ds, contract, B, capacity, num_vehicles, cooling_share,
     spec, p_c, K, time_limit, arm_seed, shadow_mode, energy_pricing,
     future_policy, standby_orders) = task
    sampler = _build_sampler(spec)
    rows = run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
                   sampler, p_c, K, time_limit, arm_seed, shadow_mode,
                   energy_pricing, future_policy, standby_orders)
    return pname, name, rows


def _sha256_path(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _run_identity(args, contract, gate_ds, gate_seed, model_sha, B, cooling_share,
                  src_seal):
    """Q-04 严格版：启动封存 + 结束复查源码；合同用完整 contract_hash；
    共享行为配置（time_limit/capacity/num_vehicles/n_orders）与步骤 2 同键同构，
    供跨运行 pairing_identity_check 逐项核对。"""
    return {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "shared_config": {"time_limit": args.time_limit,
                          "capacity": args.capacity,
                          "num_vehicles": args.num_vehicles,
                          "n_orders": args.n_orders},
        "budget_B": float(B),
        "cooling_share": float(cooling_share),
        "source_sha256": src_seal['startup_sha256'],
        "source_end_sha256": src_seal['end_sha256'],
        "source_stable": bool(src_seal['stable']),
        "shared_source_sha256": {k: src_seal['startup_sha256'][k]
                                 for k in SHARED_SOURCE_FILES},
        "contract_sha256": contract_identity(contract),
        "data_sha256": dataset_identity(gate_ds),
        "data_meta_sha256": dataset_meta_sha256(gate_ds),
        "seeds": {"train": TRAIN_SEED, "gate": int(gate_seed),
                  "final_test_reserved": list(FINAL_SEEDS)},
        "model_bin_sha256": model_sha,
        "model_bin": os.path.abspath(args.model_bin),
    }


def _baseline_identity_for(base, gate_seed):
    """兼容两种基线文件：步骤 2 gate.json（顶层 identity）与步骤 3 gate.json
    （results[seed]["_identity"]）。两者皆无 → None（旧批次，不升级正式裁决）。"""
    if base.get("identity"):
        return base["identity"]
    blk = base.get("results", {}).get(str(gate_seed), {})
    return blk.get("_identity")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bin", required=True, help="训练好的 maskco 模型 model.bin")
    ap.add_argument("--arm", choices=["random", "explicit", "pretrained", "random_cvrp"],
                    required=True,
                    help="random_cvrp=同 CVRP encoder 架构随机权重（同架构消融，须配 --cvrp-ckpt）")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m-max", type=int, default=M_MAX)
    ap.add_argument("--iterative", action="store_true",
                    help="I2：迭代重构模式（保留高置信槽 + 重掩码再生成）")
    ap.add_argument("--remask-frac", type=float, default=0.5,
                    help="I2：每轮重掩码比例（1-remask_frac 保留）")
    ap.add_argument("--rounds", type=int, default=1,
                    help="I2：重掩码轮数（1=单轮迭代；与 --iterative 同用时生效）")
    ap.add_argument("--cvrp-ckpt", default=None)
    ap.add_argument("--train-instances", type=int, default=None,
                    help="训练历史天数（默认读 model.bin 同目录 config.json 的 "
                         "train_instances，与训练谱系一致；读不到则 200）")
    ap.add_argument("--dev-instances", type=int, default=40)
    ap.add_argument("--final-test", action="store_true",
                    help="在论文测试日（20260927/28）上跑；默认用开发集 20260926")
    ap.add_argument("--gate-seed", type=int, default=None,
                    help="覆盖开发判定日 seed（默认 20260926；复验用 20260930/20261001，"
                         "与 A3 在线比较预声明三种子口径一致）")
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
    ap.add_argument("--shadow-mode", default="greedy", choices=["greedy", "ls"],
                    help="锁定下游透传（与步骤 2 同口径）")
    ap.add_argument("--energy-pricing", default="amortized", choices=["amortized", "marginal"],
                    help="锁定下游透传（与步骤 2 同口径）")
    ap.add_argument("--future-policy", default="reveal", choices=["reveal", "density"],
                    help="锁定下游透传（与步骤 2 同口径）")
    ap.add_argument("--standby-orders", type=float, default=None,
                    help="锁定下游透传（与步骤 2 同口径）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-code-mismatch", action="store_true",
                    help="放行模型代码世代不匹配（显式标注，仅诊断用途）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    contract = make_c0_contract_v2()
    src_seal = source_seal(SOURCE_FILES_STEP3)   # Q-04：启动封存源码 hash（结束复查见报告前）
    model_sha = _sha256_path(args.model_bin)
    model_gen = check_model_generation(args.model_bin,
                                       allow_mismatch=args.allow_code_mismatch)
    print("model generation check:", model_gen)
    os.makedirs(args.out, exist_ok=True)   # 先建目录，防运行中途任何写入失败
    # 160/200 历史身份一致性（路线图 09-30）：权重、解码池、预算必须用同一训练日集合——
    # 默认从模型 config.json 读训练谱系；显式传 --train-instances 可覆盖（须自知含义）。
    if args.train_instances is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.model_bin)),
                                "config.json")
        args.train_instances = 200
        if os.path.exists(cfg_path):
            _mcfg = json.load(open(cfg_path, encoding="utf-8"))
            _ti = _mcfg.get("args", {}).get("train_instances")
            if _ti:
                args.train_instances = int(_ti)
            if _mcfg.get("args", {}).get("dim"):
                args.dim = int(_mcfg["args"]["dim"])
            if _mcfg.get("args", {}).get("m_max"):
                args.m_max = int(_mcfg["args"]["m_max"])
            if _mcfg.get("args", {}).get("arm"):
                args.arm = _mcfg["args"]["arm"]
            print("model config.json: train_instances=%d dim=%d m_max=%d arm=%s"
                  % (args.train_instances, args.dim, args.m_max, args.arm))
    train_ds = generate_dataset(args.train_instances, args.n_orders, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, _ = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, TRAIN_SEED, args.rho, hist_tc,
        k=args.k_neighbors)
    pools = HistoryPools(hist)

    if args.final_test:
        seeds = FINAL_SEEDS
    elif args.gate_seed is not None:
        seeds = (int(args.gate_seed),)
    else:
        seeds = (DEV_SEED,)
    spec_maskco = {"kind": "maskco", "model_bin": os.path.abspath(args.model_bin),
                   "arm": args.arm, "dim": args.dim, "m_max": args.m_max,
                   "cvrp_ckpt": args.cvrp_ckpt, "pools": pools,
                   "iterative": args.iterative, "remask_frac": args.remask_frac,
                   "rounds": args.rounds}
    specs = {"maskco": spec_maskco,
             "cond_hist": {"kind": "cond_hist", "history": hist,
                           "k_neighbors": args.k_neighbors},
             "explicit_feat": {"kind": "explicit_feat", "history": hist},
             "uncond_hist": {"kind": "uncond_hist", "history": hist}}
    if args.only:
        specs = {args.only: specs[args.only]}

    baseline_rows, baseline_doc = {}, None
    if args.baseline_from:
        baseline_doc = json.load(open(args.baseline_from, encoding="utf-8"))
        # 兼容两种结构：步骤 2（gate[pname]，单种子）与步骤 3（results[seed][pname]）
        if "gate" in baseline_doc:
            b_seed = int((baseline_doc.get("seeds") or {}).get("gate")
                         or baseline_doc.get("identity", {}).get("seeds", {}).get("gate"))
            for pname, blk in baseline_doc["gate"].items():
                if isinstance(blk, dict) and "per_day" in blk:
                    baseline_rows[(b_seed, pname)] = blk["per_day"]
        else:
            for seed_s, blocks in baseline_doc.get("results", {}).items():
                for pname, blk in blocks.items():
                    if isinstance(blk, dict) and "per_day" in blk:
                        baseline_rows[(int(seed_s), pname)] = blk["per_day"]

    penalty_items = ([(args.penalty, PENALTIES[args.penalty])] if args.penalty
                     else ([(MAIN_PENALTY, PENALTIES[MAIN_PENALTY])] if args.main_only
                           else list(PENALTIES.items())))
    per_day, results, gate_ds_by_seed = {}, {}, {}
    for gate_seed in seeds:
        gate_ds = add_v2_initial_quality(
            generate_dataset(args.dev_instances, args.n_orders, gate_seed), contract)
        gate_ds_by_seed[gate_seed] = gate_ds
        tasks = []
        for pname, p_c in penalty_items:
            for name, spec in specs.items():
                tasks.append((pname, name, gate_ds, contract, B, args.capacity,
                              args.num_vehicles, cooling_share, spec, p_c,
                              args.K, args.time_limit, ARM_SEEDS.get(name, 7009),
                              args.shadow_mode, args.energy_pricing,
                              args.future_policy, args.standby_orders))
        if args.workers > 1:
            with Pool(args.workers) as pool:
                task_results = pool.map(_worker, tasks)
        else:
            task_results = [_worker(t) for t in tasks]
        for pname, name, rows in task_results:
            blk = results.setdefault(gate_seed, {}).setdefault(pname, {})
            blk.setdefault("arms", {})[name] = summarize(rows, gate_seed, args.dev_instances)
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
                    results[gate_seed][pname]["arms"][arm_name] = summarize(
                        b_rows, gate_seed, args.dev_instances)
        for pname in results[gate_seed]:
            m = results[gate_seed][pname]["arms"]
            st = paired_all_days(per_day[gate_seed]["maskco"][pname],
                                 per_day[gate_seed]["cond_hist"][pname],
                                 gate_seed, args.dev_instances)
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
                per_day[gate_seed]["explicit_feat"][pname], gate_seed, args.dev_instances)
            results[gate_seed][pname]["maskco_minus_uncond_hist"] = paired_all_days(
                per_day[gate_seed]["maskco"][pname],
                per_day[gate_seed]["uncond_hist"][pname], gate_seed, args.dev_instances)
            results[gate_seed][pname]["per_day"] = {
                name: per_day[gate_seed][name][pname] for name in per_day[gate_seed]}

    # Q-04：运行结束复查源码，再逐种子封存身份并做跨运行配对核实
    seal_end(src_seal)
    for gate_seed in seeds:
        ident = _run_identity(args, contract, gate_ds_by_seed[gate_seed], gate_seed,
                              model_sha, B, cooling_share, src_seal)
        results[gate_seed]["_identity"] = ident
        if args.baseline_from:
            b_id = _baseline_identity_for(baseline_doc, gate_seed)
            pairing = pairing_identity_check(ident, {"identity": b_id} if b_id else {})
            verified = bool(pairing["identity_verified"])
            pairing_note = "跨运行配对（cond_hist 复用自 %s）" % args.baseline_from
        else:
            # 同运行内 maskco 与 cond_hist 同进程/同源码/同数据：身份核实以封存为准
            pairing = {"identity_verified": bool(ident["source_stable"]),
                       "problems": ([] if ident["source_stable"]
                                    else ["current_source_not_stable"])}
            verified = bool(ident["source_stable"])
            pairing_note = ("maskco 与 cond_hist 同进程同源码同数据（同运行内配对），"
                            "身份核实以 source_stable + 数据身份为准")
        results[gate_seed]["_pairing_identity"] = pairing
        for pname in results[gate_seed]:
            if not isinstance(results[gate_seed][pname], dict):
                continue
            blk = results[gate_seed][pname].get("main_comparison")
            if blk is None:
                continue
            blk["identity_verified"] = verified
            # `passed` is retained as the historical numeric-only alias.
            # A formal result also requires all expected days and both hard-feasible arms.
            tier = results[gate_seed][pname]
            arms = tier.get("arms") or {}
            rows_ok = bool((blk.get("maskco_minus_cond_hist") or {}).get("adjudicable"))
            arms_ok = all((arms.get(a) or {}).get("adjudicable") is True and
                          (arms.get(a) or {}).get("hard_feasible_rate") == 1.0
                          for a in ("maskco", "cond_hist"))
            blk["numeric_passed"] = bool(blk["passed"])
            blk["formal_adjudication"] = bool(verified and rows_ok and arms_ok)
            blk["formal_passed"] = bool(blk["numeric_passed"] and
                                         blk["formal_adjudication"])
            blk["identity_problems"] = pairing.get("problems", [])
            blk["pairing_note"] = pairing_note

    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED,
                  "final_test_reserved": list(FINAL_SEEDS),
                  "ran": list(seeds)},
        "budget": {"B": float(B), "cooling_share": float(cooling_share)},
        "model_generation_check": model_gen,
        "delta_3": DELTA_3,
        # 路线图 09-30：主比较预声明登记——当前开发锁定最强条件非学习基线 =
        # 步骤 2 同口径 cond_hist(k=10)；若后续 S3-6 等开发锁定更强条件基线，
        # 正式 A3 运行前必须先改此处登记并换对照，不能拿旧对照结果称协议主门。
        "main_comparison_predeclaration": {
            "maskco_minus": "cond_hist(k=%d)" % args.k_neighbors,
            "criterion": "mean>=delta_3 AND ci_lo>0 AND both hard_feasible_rate==1.0",
            "delta_3": DELTA_3,
            "registration": ("2026-09-30：cond_hist 为当前最强条件非学习基线（步骤 2 "
                             "同口径）；正式运行前若开发锁定更强条件基线须改此登记。"),
        },
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
