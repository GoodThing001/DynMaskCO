"""A-v1 步骤 2 无训练条件信息门驱动（三臂共用 SAA 决策闭环的按天配对比较）。

锁定（协议 §2.6）：
  - 臂 1 uncond_hist / 臂 2 cond_hist / 臂 3 explicit_feat，共用 SaaReplanner（同下游）；
  - train seed 20260925（200 天历史）、gate seed 20260926（40 天，全新未看）；
  - K=10 场景/决策、每决策 10s、超时统一降级（保计划+拒单）；
  - 主 p_c=(5,10,15)；p_c=0 与 (10,20,30) 为独立决策运行；
  - 不筛天：逐日保存全部结果与违规类型；存在硬违规 = 不宣称过门；
  - 主比较 = cond_hist − uncond_hist：均值 ≥ δ*=20 且 bootstrap 95% CI 下界 > 0。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'coldchain'),
           os.path.join(_SCRIPTS, 'simulation'), os.path.join(_SCRIPTS, 'evaluation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from a1_strong_controls import ConsensusReplanner, RolloutVReplanner
from coldchain_evaluator_a1 import add_v2_initial_quality, evaluate_trace_a1
from run_exp_encoder_v3 import compute_budget_and_dwell
from run_exp_reserve import generate_dataset, _stat, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from run_identity import (SOURCE_FILES_STEP2, SHARED_SOURCE_FILES,
                          source_seal, seal_end, contract_identity,
                          dataset_identity, dataset_meta_sha256)
from scenario_saa import (SaaReplanner, UncondHistoricalSampler, CondHistoricalSampler,
                          ExplicitFeatureSampler, SoftKNNHistoricalSampler,
                          build_history, build_history_times_classes,
                          make_c0_contract_v2)
from strict_online_env import StrictOnlineEnv

PENALTIES = {
    "p_c=0": {0: 0.0, 1: 0.0, 2: 0.0},
    "p_c=(5,10,15)": {0: 5.0, 1: 10.0, 2: 15.0},
    "p_c=(10,20,30)": {0: 10.0, 1: 20.0, 2: 30.0},
}
MAIN_PENALTY = "p_c=(5,10,15)"
DELTA_STAR = 20.0
TRAIN_SEED = 20260925
GATE_SEED = 20260926
FINAL_TEST_SEEDS = (20260927, 20260928)  # 只声明，不跑
ARM_SEEDS = {"uncond_hist": 7001, "cond_hist": 7002, "explicit_feat": 7003,
             "a1_consensus": 7004, "a1_rollout": 7005, "soft_knn": 7006}
# A1 强动态对照（2026-10-03，设计件《A1强动态对照_助手实施设计》）：arm → (sampler, replanner,
# extra_kwargs)。legacy 三臂保持默认集合不变；A1 臂经 --arms 显式加入。
A1_ARM_NAMES = ("a1_consensus", "a1_rollout")


def _single(ds, i):
    return {k: v[i] for k, v in ds.items()}


def run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
            sampler, p_c, K, time_limit, arm_seed, shadow_mode='greedy',
            energy_pricing='amortized', future_policy='reveal', standby_orders=None,
            overage_price=None, incr_eval=False, anytime_vote=False,
            replanner_cls=SaaReplanner, extra_kwargs=None, progress_file=None):
    kw = dict(budget=B, capacity=capacity, booking_horizon=BOOKING_HORIZON,
              contract=contract, cooling_share=cooling_share, sampler=sampler,
              reject_penalty=p_c, K=K, time_limit=time_limit, arm_seed=arm_seed,
              shadow_mode=shadow_mode, energy_pricing=energy_pricing,
              future_policy=future_policy, standby_orders=standby_orders,
              overage_price=overage_price, incr_eval=incr_eval,
              anytime_vote=anytime_vote)
    if extra_kwargs:
        kw.update(extra_kwargs)
    rp = replanner_cls(**kw)
    env = StrictOnlineEnv(gate_ds, capacity=capacity, num_vehicles=num_vehicles,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract, booking_horizon=BOOKING_HORIZON)
    rows = []
    for i in range(gate_ds['coords'].shape[0]):
        t0 = time.time()
        try:
            traces, _ = env.run(i)
            acc, rej = set(rp._accepted), set(rp._rejected)
            ev = evaluate_trace_a1(traces, _single(gate_ds, i), contract, acc, rej, B, p_c)
        except Exception as exc:   # 2026-09-28 A-03：执行期异常按缺失日记为「不可裁决」，不伪造全 0
            rows.append(dict(i=int(i), utility=None, hard_feasible=False,
                             failures=['execution_crash'], error=repr(exc),
                             n_served=None, n_rejected=None, n_fish_unsalable=None,
                             energy_kwh=None, budget_violated=None,
                             revenue=None, fuel_cost=None, reject_loss=None,
                             distance_km=None, service_rate=None, reject_rate=None,
                             salable_rate=None, tomato_mean_quality=None,
                             timeouts=int(rp.timeouts),
                             partial_commits=int(rp.partial_commits),
                             early_stops=int(rp.early_stops),
                             saa_completion_rate=getattr(rp, 'saa_completion_rate', None),
                             n_decisions=int(getattr(rp, 'n_decisions', 0)),
                             elapsed_s=float(time.time() - t0)))
            if progress_file:
                with open(progress_file, 'a', encoding='utf-8') as pf:
                    pf.write('%d %.1f crash\n' % (i, time.time() - t0))
            continue
        if progress_file:
            with open(progress_file, 'a', encoding='utf-8') as pf:
                pf.write('%d %.1f ok\n' % (i, time.time() - t0))
        # 2026-09-26 审计：保留评价器全字段（收入/燃油/拒绝损失/分温区接拒/可售/品质），
        # 用于证明效用提升来自有效预算配置而非单纯多接单。
        rows.append(dict(
            i=int(i),
            utility=(float(ev['utility']) if np.isfinite(ev['utility']) else None),
            hard_feasible=bool(ev['hard_feasible']),
            failures=sorted(set(ev['failures'])),
            n_served=int(ev['n_served']), n_rejected=int(ev['n_rejected']),
            n_fish_unsalable=int(ev['n_fish_unsalable']),
            energy_kwh=float(ev['energy_kwh']),
            budget_violated=bool(ev['budget_violated']),
            revenue=float(ev['revenue']), fuel_cost=float(ev['fuel_cost']),
            reject_loss=float(ev['reject_loss']), distance_km=float(ev['distance_km']),
            service_rate=ev['service_rate'], reject_rate=ev['reject_rate'],
            salable_rate=ev['salable_rate'],
            tomato_mean_quality=ev['tomato_mean_quality'],
            timeouts=int(rp.timeouts),
            partial_commits=int(rp.partial_commits),
            early_stops=int(rp.early_stops),
            saa_completion_rate=(float(rp.saa_completion_rate)
                                 if getattr(rp, 'saa_completion_rate', None) is not None
                                 else None),
            n_decisions=int(getattr(rp, 'n_decisions', 0)),
            elapsed_s=float(time.time() - t0)))
    return rows


def _worker(task):
    """(pname, name, gate_ds, contract, B, capacity, num_vehicles, cooling_share,
        sampler, p_c, K, time_limit, arm_seed, shadow_mode, energy_pricing,
        future_policy, standby_orders, overage_price, incr_eval, anytime_vote,
        replanner_cls, extra_kwargs, out)
        -> (pname, name, rows)"""
    (pname, name, gate_ds, contract, B, capacity, num_vehicles, cooling_share,
     sampler, p_c, K, time_limit, arm_seed, shadow_mode, energy_pricing,
     future_policy, standby_orders, overage_price, incr_eval, anytime_vote,
     replanner_cls, extra_kwargs, out) = task
    progress_file = os.path.join(out, 'progress_%s_%s.txt' % (pname, name))
    rows = run_arm(gate_ds, contract, B, capacity, num_vehicles, cooling_share,
                   sampler, p_c, K, time_limit, arm_seed, shadow_mode,
                   energy_pricing, future_policy, standby_orders, overage_price,
                   incr_eval=incr_eval, anytime_vote=anytime_vote,
                   replanner_cls=replanner_cls, extra_kwargs=extra_kwargs,
                   progress_file=progress_file)
    return pname, name, rows


def summarize(rows, seed, expected=None):
    """A-03（+2026-09-30 完整性修复）：区分完整可评价效用与缺失。
    可裁决 = 实际天数 == 预期天数 且 day_id 恰为 0..N-1 且无 utility=None。
    缺日（无论单侧或双侧共缺）只作诊断，不得判可裁决。违规日仍全部保留。"""
    n_expected = int(expected) if expected is not None else len(rows)
    ok = np.array([r['hard_feasible'] for r in rows])
    u = np.array([r['utility'] if r['utility'] is not None else float('nan')
                  for r in rows], float)
    finite = np.isfinite(u)
    # 2026-09-30：NaN/Inf 与 None 一样记缺失（旧基线文件可能含 NaN 而非 None）
    missing_ids = [r['i'] for r in rows
                   if r['utility'] is None or not np.isfinite(r['utility'])]
    day_ids = sorted(int(r['i']) for r in rows)
    days_complete = (len(rows) == n_expected and day_ids == list(range(n_expected)))
    adjudicable = bool(days_complete and len(missing_ids) == 0)
    out = {
        "n_days": int(len(rows)),
        "expected_n": n_expected,
        "observed_n": int(len(rows)),
        "finite_n": int(finite.sum()),
        "missing_day_ids": [int(x) for x in missing_ids],
        "day_ids_complete": bool(days_complete),
        "adjudicable": adjudicable,
        "hard_feasible_rate": (float(ok.mean()) if adjudicable else None),
        "utility_all_days": (_stat(u, seed) if adjudicable
                             else dict(adjudicable=False, reason='missing_days')),
        "utility_finite_diagnostic_only": (_stat(u[finite], seed)
                                           if finite.sum() > 0 else None),
        "fail_counts": {k: int(v) for k, v in Counter(
            f for r in rows for f in r['failures']).items()},
        "mean_timeouts": float(np.mean([r['timeouts'] for r in rows])),
        "timeout_days": int(sum(1 for r in rows if r['timeouts'] > 0)),
        "mean_partial_commits": float(np.mean([r.get('partial_commits', 0) for r in rows])),
        "partial_commit_days": int(sum(1 for r in rows if r.get('partial_commits', 0) > 0)),
        "mean_early_stops": float(np.mean([r.get('early_stops', 0) for r in rows])),
        "mean_saa_completion_rate": (
            float(np.mean([r['saa_completion_rate'] for r in rows
                           if r.get('saa_completion_rate') is not None]))
            if any(r.get('saa_completion_rate') is not None for r in rows) else None),
        "elapsed_total_s": float(np.sum([r['elapsed_s'] for r in rows])),
    }
    return out


def paired_all_days(a_rows, b_rows, seed, expected=None):
    """A-03（+2026-09-30 完整性修复）：按 day_id 显式配对。
    可裁决 = 双侧天数都等于预期且 day_id 均为 0..N-1 且无缺日/None。
    双侧共缺同一天（如 39/39）也判不可裁决——以预期 40 个唯一 day ID 验收。"""
    n_expected = int(expected) if expected is not None else len(a_rows)
    b_by_i = {r['i']: r['utility'] for r in b_rows}
    pa, pb = [], []
    missing = []
    for r in a_rows:
        if r['i'] not in b_by_i:
            missing.append(('b_missing_day', int(r['i'])))
            continue
        ua, ub = r['utility'], b_by_i[r['i']]
        if ua is None or ub is None or not np.isfinite(ua) or not np.isfinite(ub):
            missing.append(('missing_utility', int(r['i'])))
            continue
        pa.append(ua)
        pb.append(ub)
    if not pa:
        return {'mean': None, 'ci_lo': None, 'ci_hi': None, 'n': 0,
                'adjudicable': False, 'missing': missing}
    st = _stat(np.array(pa, float) - np.array(pb, float), seed)
    a_ids = sorted(r['i'] for r in a_rows)
    b_ids = sorted(r['i'] for r in b_rows)
    ids_complete = (a_ids == b_ids == list(range(n_expected)))
    st['adjudicable'] = bool(len(missing) == 0 and ids_complete
                             and len(a_rows) == n_expected and len(b_rows) == n_expected)
    st['n_paired'] = len(pa)
    st['missing'] = missing
    return st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--gate-instances", type=int, default=40)
    ap.add_argument("--n-orders", type=int, default=200)
    ap.add_argument("--capacity", type=float, default=50.0)
    ap.add_argument("--num-vehicles", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--k-neighbors", type=int, default=10)
    ap.add_argument("--penalty", default=None,
                    help="只跑一档 p_c（烟测用）；默认全部三档")
    ap.add_argument("--gate-seed", type=int, default=GATE_SEED,
                    help="门判定日 seed（默认 20260926；审计复验用预锁定 20260930/20261001）")
    ap.add_argument("--shadow-mode", default="greedy", choices=["greedy", "ls"],
                    help="战役 L1：影子未来插入路由强度（greedy=首个可行位；ls=最优插入+relocate/swap）")
    ap.add_argument("--energy-pricing", default="amortized", choices=["amortized", "marginal"],
                    help="战役 L2b：影子能耗定价（amortized=每单摊待命分摊；marginal=只计 door/COP+precool）")
    ap.add_argument("--future-policy", default="reveal", choices=["reveal", "density"],
                    help="战役 L2c：影子未来接单顺序（reveal=按揭示时刻；density=按价值密度降序打包）")
    ap.add_argument("--standby-orders", type=float, default=None,
                    help="战役 L2b 校准网格：边际定价下的固定待命预留=cooling_share×N（None=纯边际）")
    ap.add_argument("--overage-price", type=float, default=None,
                    help="战役 S3-3：影子终局超额连续价格（元/kWh；None=终局超预算不扣罚的旧行为）")
    ap.add_argument("--incr-eval", action="store_true",
                    help="战役 S3-4：ls 候选 O(1) 摘要预筛（等价加速；预筛只跳过可证明不可行者，"
                         "接受集合与旧实现一致）")
    ap.add_argument("--anytime-vote", action="store_true",
                    help="战役 S3-4-D3：anytime 部分投票（超时语义变更，已注册；默认关=旧语义）")
    ap.add_argument("--arms", default="uncond_hist,cond_hist,explicit_feat",
                    help="逗号分隔臂集合；A1 强动态对照经 a1_consensus,a1_rollout 显式加入")
    ap.add_argument("--a1-v-ckpt", default=None,
                    help="a1_rollout 臂的 V 模型 npz（train_a1_rollout_v.py 产物；"
                         "含该臂时必填）")
    ap.add_argument("--a1-h", type=float, default=2.0,
                    help="a1_rollout 臂的 rollout 截断小时数（规格固定 2.0）")
    ap.add_argument("--softknn-mult", type=float, default=1.0,
                    help="S3-6a 软 k-NN 带宽倍数（σ²=mult×训练日 h_med；主值 1.0）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 worker 数（(p_c × 臂) 任务并行）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    # 2026-09-30 修复：输出目录必须在 worker 启动前创建——旧实现到报告写入时才
    # makedirs，worker 首日写 progress 文件即 FileNotFoundError → 整批崩溃。
    os.makedirs(args.out, exist_ok=True)

    contract = make_c0_contract_v2()
    src_seal = source_seal(SOURCE_FILES_STEP2)   # Q-04：启动封存源码 hash（结束复查见 report 前）
    train_ds = generate_dataset(args.train_instances, args.n_orders, TRAIN_SEED)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, train_energy_mean = compute_budget_and_dwell(
        train_ds, contract, args.num_vehicles, TRAIN_SEED, args.rho, hist_tc,
        k=args.k_neighbors)
    gate_ds = add_v2_initial_quality(
        generate_dataset(args.gate_instances, args.n_orders, args.gate_seed), contract)

    samplers = {
        "uncond_hist": UncondHistoricalSampler(hist),
        "cond_hist": CondHistoricalSampler(hist, n_neighbors=args.k_neighbors),
        "explicit_feat": ExplicitFeatureSampler(hist),
    }
    arm_names = [s.strip() for s in args.arms.split(',') if s.strip()]
    for name in arm_names:
        if name not in ARM_SEEDS:
            raise SystemExit("unknown arm: %s (可选 %s)"
                             % (name, ','.join(sorted(ARM_SEEDS))))
    if 'a1_rollout' in arm_names and not args.a1_v_ckpt:
        raise SystemExit("a1_rollout 臂需要 --a1-v-ckpt（V 模型 npz）")
    a1_v_sha = None
    if args.a1_v_ckpt:
        import hashlib as _hl
        if not os.path.isfile(args.a1_v_ckpt):
            raise SystemExit("--a1-v-ckpt 不存在: " + args.a1_v_ckpt)
        a1_v_sha = _hl.sha256(open(args.a1_v_ckpt, 'rb').read()).hexdigest()
    arm_specs = {}   # name -> (sampler, replanner_cls, extra_kwargs)
    for name in arm_names:
        if name == 'a1_consensus':
            arm_specs[name] = (samplers['cond_hist'], ConsensusReplanner, {})
        elif name == 'a1_rollout':
            arm_specs[name] = (samplers['cond_hist'], RolloutVReplanner,
                               {'v_ckpt': args.a1_v_ckpt, 'h': args.a1_h})
        elif name == 'soft_knn':
            arm_specs[name] = (SoftKNNHistoricalSampler(hist, sigma2_mult=args.softknn_mult),
                               SaaReplanner, None)
        else:
            arm_specs[name] = (samplers[name], SaaReplanner, None)

    penalty_items = ([(args.penalty, PENALTIES[args.penalty])]
                     if args.penalty else list(PENALTIES.items()))
    if args.penalty and args.penalty not in PENALTIES:
        raise SystemExit("unknown --penalty: " + args.penalty)

    tasks = []
    for pname, p_c in penalty_items:
        for name, (sampler, cls, extra) in arm_specs.items():
            tasks.append((pname, name, gate_ds, contract, B, args.capacity,
                          args.num_vehicles, cooling_share, sampler, p_c,
                          args.K, args.time_limit, ARM_SEEDS[name], args.shadow_mode,
                          args.energy_pricing, args.future_policy,
                          args.standby_orders, args.overage_price,
                          args.incr_eval, args.anytime_vote, cls, extra, args.out))
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            task_results = pool.map(_worker, tasks)
    else:
        task_results = [_worker(t) for t in tasks]

    per_day = {}
    results = {}
    for pname, name, rows in task_results:
        results.setdefault(pname, {})["arms"] = results.setdefault(
            pname, {}).get("arms", {})
        results[pname]["arms"][name] = summarize(rows, args.gate_seed, args.gate_instances)
        per_day.setdefault(name, {})[pname] = rows

    for pname in results:
        arms = results[pname]["arms"]
        has_main = ("cond_hist" in arms and "uncond_hist" in arms)
        if has_main:
            cond_rows = per_day["cond_hist"][pname]
            uncond_rows = per_day["uncond_hist"][pname]
            st = paired_all_days(cond_rows, uncond_rows, args.gate_seed, args.gate_instances)
            both_ok = (arms["cond_hist"]["hard_feasible_rate"] == 1.0
                       and arms["uncond_hist"]["hard_feasible_rate"] == 1.0
                       and arms["cond_hist"]["adjudicable"]
                       and arms["uncond_hist"]["adjudicable"]
                       and bool(st.get("adjudicable")))
            passed = bool(both_ok and np.isfinite(st["mean"])
                          and st["mean"] >= DELTA_STAR and st["ci_lo"] > 0)
            results[pname]["main_comparison"] = {
                "cond_minus_uncond": st,
                "delta_star": DELTA_STAR,
                "criterion": "mean>=delta_star AND ci_lo>0 AND both hard_feasible_rate==1.0 "
                             "AND all_days_adjudicable",
                "passed": passed,
                "violations_present": (not (arms["cond_hist"]["hard_feasible_rate"] == 1.0
                                           and arms["uncond_hist"]["hard_feasible_rate"] == 1.0)),
                "adjudicable": bool(st.get("adjudicable")),
            }
        # 三层对比（2026-10-03）：条件优势层之上，另报每个非基线臂 vs cond_hist 的
        # 逐日配对（同判据结构）——A1 两臂与 S3-6a soft_knn 的「最强条件非学习」候选比较，
        # 不并入 A2 主门。
        for arm_name in arm_specs:
            if arm_name == 'cond_hist' or arm_name not in arms:
                continue
            if 'cond_hist' in arms:
                st_x = paired_all_days(per_day[arm_name][pname],
                                       per_day["cond_hist"][pname],
                                       args.gate_seed, args.gate_instances)
                results[pname][arm_name + "_minus_cond_hist"] = st_x
        results[pname]["per_day"] = {name: per_day[name][pname] for name in arm_specs}

    seal_end(src_seal)   # Q-04：运行结束后复查源码（中途变更 → source_stable=False）
    report = {
        "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
        "seeds": {"train": TRAIN_SEED, "gate": args.gate_seed,
                  "final_test_reserved": list(FINAL_TEST_SEEDS)},
        "budget": {"B": float(B), "cooling_share": float(cooling_share),
                   "train_accept_all_c0_energy_mean": float(train_energy_mean)},
        # Q-04（2026-09-29，严格版）：运行身份封存——启动+结束双读源码、
        # 完整 contract_hash、共享行为配置；后续跨运行配对逐项核对
        "identity": {
            "config": {k2: v for k2, v in vars(args).items() if k2 != "out"},
            "shared_config": {"time_limit": args.time_limit,
                              "capacity": args.capacity,
                              "num_vehicles": args.num_vehicles,
                              "n_orders": args.n_orders,
                              "anytime_vote": bool(args.anytime_vote),
                              "arms": arm_names},
            "a1": ({"v_ckpt_sha256": a1_v_sha, "h": args.a1_h,
                    "arms": [n for n in arm_names if n in A1_ARM_NAMES]}
                   if any(n in A1_ARM_NAMES for n in arm_names) else None),
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
            "seeds": {"train": TRAIN_SEED, "gate": args.gate_seed,
                      "final_test_reserved": list(FINAL_TEST_SEEDS)},
        },
        "main_penalty": MAIN_PENALTY,
        "delta_star": DELTA_STAR,
        "gate": results,
        "note": "A-v1 步骤 2 无训练条件信息门：三臂（uncond_hist/cond_hist/explicit_feat）共用 SAA 决策闭环；"
                "主比较=cond_hist−uncond_hist；不筛天（逐日报告全部结果与违规类型）；"
                "p_c 三档独立决策运行；旧 40 天（seed 20260923/24）为开发证据，不参与本门。",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
