"""O0-D / O0-CC：sequential action-space oracle 评估入口（实例级多进程）。

对比 baseline（JF1-H）与 greedy sequential oracle 在相同 strict-online 协议下的
终局 outcome，可选 local baseline-state oracle 双报告。coldchain objective 通过具名
profile 冻结 scale/λ（默认 pilot，或 --objective-profile 加载 devmean 冻结版本）。

实例级并行：每个 worker 独立完成一个实例的 baseline + sequential +（可选）local，
worker 内 action rollout 保持串行。固定 seed 与确定性 continuation 保证 workers 数
不改变结果（parity 验收）。任一 worker 失败时整体不静默成功。

用法：
    python scripts/evaluation/run_action_oracle.py \
        --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
        --capacity 50 --num_vehicles 25 --objective coldchain \
        --objective-profile results/o0cc/devcal_scale/objective_profile.json \
        --max_instances 8 --workers 8 --out results/o0cc/val8
"""
import argparse, os, csv, json, sys, time, hashlib, multiprocessing, traceback
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from sequential_oracle import (make_oracle_hook, sequential_oracle_plan,
                               mutable_vehicle_ids, decision_pool, customer_order_key)
from counterfactual_teacher import (_eval, lex_key, _snapshot_with_force,
                                    rollout_baseline, _incumbent_plans, rollout_action)
from action_contract import enumerate_actions_from_plans
from recourse_snapshot import capture_recourse_snapshot
from coldchain_contract import (default_pilot_contract, default_pilot_profile,
                                apply_objective_profile, ObjectiveProfile)
from hard_gate import service_ok, select_improving, hard_vector_from_outcome, outcome_protocol_error
from service_first import paired_bootstrap_ci


def _load_profile(args):
    if args.objective == 'coldchain' and args.objective_profile is None:
        raise ValueError("coldchain objective 必须显式提供 --objective-profile（v2），"
                         "禁止静默回退 pilot profile")
    if args.objective_profile is None:
        return default_pilot_profile()
    with open(args.objective_profile) as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile（non-hard-feasible JF1-H baseline）；"
                         "请使用 v2")
    profile = ObjectiveProfile(
        name=data['name'],
        distance_scale=float(data['distance_scale']),
        quality_scale=float(data['quality_scale']),
        energy_scale=float(data['energy_scale']),
        lambda_quality=float(data['lambda_quality']),
        lambda_energy=float(data['lambda_energy']),
        scale_source=data.get('scale_source', 'pilot'),
        dev_statistics=data.get('dev_statistics'),
    )
    if profile.profile_hash != data.get('profile_hash'):
        raise ValueError(f"profile reload hash 不一致：recomputed={profile.profile_hash} "
                         f"file={data.get('profile_hash')}")
    return profile


def _profile_worker_manifest(profile):
    return profile.to_manifest()  # 完整 manifest（含 dev_statistics + profile_hash）


def _profile_from_worker_manifest(m):
    profile = ObjectiveProfile(
        name=m['name'], distance_scale=m['distance_scale'],
        quality_scale=m['quality_scale'], energy_scale=m['energy_scale'],
        lambda_quality=m['lambda_quality'], lambda_energy=m['lambda_energy'],
        scale_source=m.get('scale_source', 'pilot'),
        dev_statistics=m.get('dev_statistics'),
    )
    if profile.profile_hash != m.get('profile_hash'):
        raise ValueError("worker profile hash mismatch: recomputed="
                         f"{profile.profile_hash} file={m.get('profile_hash')}")
    return profile


def _make_env(dataset, capacity, num_vehicles, continuation, objective, profile):
    if objective == 'coldchain':
        contract = apply_objective_profile(default_pilot_contract(), profile)
    else:
        contract = None
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=continuation, coldchain_contract=contract)


def _cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _parity_ok(a, b, objective):
    """keep rollout 与 baseline 终局 parity：完整 hard vector + D/Q/E/J_CC 一致（不能只比距离）。"""
    if objective == 'distance':
        return (bool(a['complete']) == bool(b['complete'])
                and abs(float(a['distance_cost']) - float(b['distance_cost'])) <= 1e-9)
    return (hard_vector_from_outcome(a) == hard_vector_from_outcome(b)
            and abs(float(a['distance_km']) - float(b['distance_km'])) <= 1e-9
            and abs(float(a['quality_loss']) - float(b['quality_loss'])) <= 1e-9
            and abs(float(a['energy_kwh']) - float(b['energy_kwh'])) <= 1e-9
            and abs(float(a['coldchain_cost']) - float(b['coldchain_cost'])) <= 1e-9)


def _local_instance(dataset, capacity, num_vehicles, objective, profile, inst_idx):
    """独立 local baseline-state oracle（阶段 C 步骤 5）。

    baseline（JF1-H-F）跑一遍收集「存在 replan 需求」的决策点 snapshot；每个 snapshot 生成
    一次 incumbent plan；每个 customer 从该原计划独立枚举单动作并独立恢复 rollout。keep
    rollout（no-op）必须与 baseline 终局一致。返回 instance 层 event 平均 delta 列表 + 记录。
    """
    env = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile)
    snapshots = []

    def hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        replan_ids = {v.vehicle_id for v in veh
                      if v.status in ('idle', 'ready') and v.needs_replan}
        if replan_ids:
            snapshots.append(capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac))

    env.snapshot_hook = hook
    traces, served = env.run(inst_idx)
    base_outcome = _eval(env, inst_idx, traces, objective, served_mask=served)
    _err = outcome_protocol_error(base_outcome, objective, strict_repair=True)
    if _err is not None:
        raise RuntimeError(f"PROTOCOL_ERROR: local baseline outcome invalid ({_err})")
    if not service_ok(base_outcome, objective):
        raise RuntimeError("local baseline 未通过 service Gate")
    base_cost = _cost(base_outcome, objective)

    event_deltas = []
    records = []
    for snap in snapshots:
        renv = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile)
        keep_outcome = rollout_baseline(renv, snap, objective)
        _err = outcome_protocol_error(keep_outcome, objective, strict_repair=True)
        if _err is not None:
            raise RuntimeError(f"PROTOCOL_ERROR: local KEEP outcome invalid ({_err})")
        keep_cost = _cost(keep_outcome, objective)
        # keep == baseline 终局 parity 断言（完整 hard vector + D/Q/E/J_CC）
        if not _parity_ok(keep_outcome, base_outcome, objective):
            raise RuntimeError(f"instance {inst_idx} event {snap['event_id']}: "
                               f"keep rollout != baseline terminal (parity broken)")
        incumbent = _incumbent_plans(renv, snap, renv.replanner)
        mutable_ids = mutable_vehicle_ids(snap)
        pool = sorted(decision_pool(snap), key=customer_order_key(renv, inst_idx))
        customer_deltas = []
        for customer in pool:
            cands, _ = enumerate_actions_from_plans(renv, inst_idx, incumbent, customer,
                                                    allowed_vehicle_ids=mutable_ids)
            cost_id_pairs = []
            for cand in cands:
                if not cand.feasible:
                    continue
                outcome, _ = rollout_action(renv, snap, cand.action, renv.replanner, incumbent,
                                            objective=objective,
                                            allowed_vehicle_ids=mutable_ids,
                                            mutable_ids=mutable_ids)
                _err = outcome_protocol_error(outcome, objective, strict_repair=True)
                if _err is not None:
                    raise RuntimeError(f"PROTOCOL_ERROR: local candidate {cand.action.action_id()} "
                                       f"outcome invalid ({_err})")
                if not service_ok(outcome, objective):
                    continue
                cost_id_pairs.append((_cost(outcome, objective), cand.action.action_id()))
            # 无改善动作 → delta=0（KEEP/DEFER），不把该 customer 从统计样本中删除
            best = select_improving(cost_id_pairs, keep_cost)
            customer_deltas.append((best[0] - keep_cost) if best is not None else 0.0)
        if customer_deltas:
            event_deltas.append(float(np.mean(customer_deltas)))
        records.append({'event_id': int(snap['event_id']), 'n_customers': len(pool),
                        'n_customer_deltas': len(customer_deltas),
                        'mean_customer_delta': (float(np.mean(customer_deltas))
                                                if customer_deltas else None),
                        'keep_cost': keep_cost})
    return {'event_deltas': event_deltas, 'records': records, 'base_cost': base_cost}


def _worker_instance(args):
    (inst_idx, data_path, capacity, num_vehicles, objective,
     profile_manifest, do_local, out_dir, code_sha256) = args
    # 单进程 BLAS/NumPy 线程数 = 1，避免 worker 内再超额并行
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
        os.environ[v] = '1'
    try:
        t0 = time.time()
        dataset = dict(np.load(data_path))
        profile = (_profile_from_worker_manifest(profile_manifest)
                   if profile_manifest is not None else None)

        # 统一 factory：每条 rollout 分支独立 JF1-H-F continuation（阶段 C 步骤 1，
        # 隔离 deferred 集合与累计审计状态，避免跨分支串扰）
        env = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile)
        traces, served_base = env.run(inst_idx)
        base = _eval(env, inst_idx, traces, objective, served_mask=served_base)
        _err = outcome_protocol_error(base, objective, strict_repair=True)
        if _err is not None:
            raise RuntimeError(f"PROTOCOL_ERROR: baseline outcome invalid ({_err})")

        log = []
        oenv = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile)
        renv = _make_env(dataset, capacity, num_vehicles, make_continuation(), objective, profile)
        oenv.oracle_hook = make_oracle_hook(renv, renv.replanner, objective, log=log)
        traces, served_orac = oenv.run(inst_idx)
        orac = _eval(oenv, inst_idx, traces, objective, served_mask=served_orac)
        _err = outcome_protocol_error(orac, objective, strict_repair=True)
        if _err is not None:
            raise RuntimeError(f"PROTOCOL_ERROR: oracle outcome invalid ({_err})")

        local = None
        if do_local:
            local = _local_instance(dataset, capacity, num_vehicles, objective, profile, inst_idx)

        runtime = time.time() - t0
        for rec in log:
            rec['inst_idx'] = inst_idx
        result = {'inst_idx': inst_idx, 'baseline': base, 'oracle': orac,
                  'local': local, 'log': log, 'runtime': runtime,
                  'code_sha256': code_sha256, 'error': None}
    except Exception as exc:
        result = {'inst_idx': inst_idx, 'error': repr(exc) + '\n' + traceback.format_exc(),
                  'runtime': time.time() - t0 if 't0' in locals() else None}

    if out_dir:
        inst_dir = os.path.join(out_dir, 'instances')
        os.makedirs(inst_dir, exist_ok=True)
        # 原子写：临时文件写完关闭后 rename，避免部分写入被误认为完成
        tmp = os.path.join(inst_dir, f'.tmp_inst_{inst_idx}_{os.getpid()}.json')
        final = os.path.join(inst_dir, f'inst_{inst_idx}.json')
        with open(tmp, 'w') as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    return result


def main():
    parser = argparse.ArgumentParser(description='O0-D/O0-CC sequential action oracle')
    parser.add_argument('--data', required=True)
    parser.add_argument('--capacity', type=float, default=50.0)
    parser.add_argument('--num_vehicles', type=int, default=25)
    parser.add_argument('--objective', choices=['distance', 'coldchain'], default='distance')
    parser.add_argument('--objective-profile', default=None)
    parser.add_argument('--max_instances', type=int, default=16)
    parser.add_argument('--instance-ids', type=str, default=None,
                        help='逗号分隔的实例 id（精确选样/续跑）；提供时忽略 max_instances')
    parser.add_argument('--role', choices=['regression', 'dev_proto', 'dev_gate', 'val'],
                        default='regression',
                        help='运行角色：regression=协议回归(不产生 GO)；dev_proto=探索；'
                             'dev_gate/val=正式 Gate')
    parser.add_argument('--workers', type=int, default=1,
                        help='实例级并行 worker 数（默认 1 串行；服务器用 8）')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # 计算版本身份：在进程启动时（长计算之前）hash 自身源码并保存在内存中，实例记录与
    # 最终 manifest 都引用它。绝不能在运行结束时重读磁盘——运行期间文件被覆盖会谎报版本。
    code_sha256 = _sha256_file(os.path.abspath(__file__))

    dataset = dict(np.load(args.data))
    actual_n = dataset['coords'].shape[0]
    if args.instance_ids is not None:
        idxs = [int(x) for x in args.instance_ids.split(',') if x.strip() != '']
        if not idxs:
            raise ValueError("--instance-ids 不能为空")
        if len(idxs) != len(set(idxs)):
            raise ValueError(f"--instance-ids 含重复：{idxs}")
        if any(i < 0 or i >= actual_n for i in idxs):
            raise ValueError(f"instance-ids 越界：{idxs}（actual_n={actual_n}）")
        n = len(idxs)
    else:
        if args.max_instances <= 0:
            raise ValueError(f"max_instances 必须为正：{args.max_instances}")
        if args.max_instances > actual_n:
            raise ValueError(f"max_instances={args.max_instances} 超过数据实例数 {actual_n}；"
                             f"禁止静默缩小样本，请用 --instance-ids 显式选样")
        idxs = list(range(args.max_instances))
        n = len(idxs)
    profile = _load_profile(args)
    profile_manifest = _profile_worker_manifest(profile) if args.objective == 'coldchain' else None

    print(f"=== O0 oracle (objective={args.objective}, n={n}, workers={args.workers}) ===",
          flush=True)
    if args.objective == 'coldchain':
        print(f"  [profile] {profile.name} hash={profile.profile_hash[:16]} "
              f"scale=({profile.distance_scale:.3f},{profile.quality_scale:.4f},"
              f"{profile.energy_scale:.2f}) λ=({profile.lambda_quality},{profile.lambda_energy})",
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    args_list = [(i, args.data, args.capacity, args.num_vehicles, args.objective,
                  profile_manifest, args.local, args.out, code_sha256) for i in idxs]

    if args.workers <= 1:
        results = [_worker_instance(a) for a in args_list]
        launch = 'serial'
    else:
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(_worker_instance, args_list)
        launch = 'multiprocessing.Pool(%d)' % args.workers

    failed = [r for r in results if r.get('error')]
    if failed:
        for r in failed:
            print(f"  [FAIL] instance {r['inst_idx']}:\n{r['error']}", file=sys.stderr)
        raise RuntimeError(f"{len(failed)}/{n} instances failed; run aborted (no silent success)")

    results = sorted(results, key=lambda r: r['inst_idx'])
    base = [r['baseline'] for r in results]
    orac = [r['oracle'] for r in results]
    # local：instance-level 统计单位（每 instance = 其 event 平均 delta 的平均）
    local_inst_deltas = None
    n_local_no_eligible = 0
    if args.local:
        local_inst_deltas = []
        for r in results:
            loc = r.get('local')
            ed = (loc or {}).get('event_deltas', [])
            if ed:
                local_inst_deltas.append(float(np.mean(ed)))
            else:
                n_local_no_eligible += 1
    log = [rec for r in results for rec in r['log']]
    log.sort(key=lambda rec: (rec['inst_idx'], rec.get('seq', 0)))  # 真实执行顺序
    runtimes = [r['runtime'] for r in results]

    def _full_service_ok(o):
        # service Gate = outcome service_ok + repair-level（terminal_unresolved/ownership）
        return (service_ok(o, args.objective)
                and int(o.get('terminal_unresolved', 0)) == 0
                and int(o.get('ownership_violations', 0)) == 0)

    base_service = all(_full_service_ok(o) for o in base)
    orac_service = all(_full_service_ok(o) for o in orac)
    # ownership 破坏 = 协议错误（不是 service fail）
    own_viol = [r['inst_idx'] for r in results
                if (r.get('baseline') or {}).get('ownership_violations', 0) > 0
                or (r.get('oracle') or {}).get('ownership_violations', 0) > 0]
    if own_viol:
        raise RuntimeError(f"PROTOCOL_ERROR: ownership_violations>0 实例 {own_viol}")
    paired_idx = [i for i in range(n)
                  if _full_service_ok(base[i]) and _full_service_ok(orac[i])]
    deltas = [_cost(orac[i], args.objective) - _cost(base[i], args.objective)
              for i in paired_idx]
    ci_lo, ci_hi = paired_bootstrap_ci(deltas) if deltas else (float('nan'), float('nan'))
    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    base_mean = float(np.mean([_cost(base[i], args.objective) for i in paired_idx])) if paired_idx else float('nan')
    orac_mean = float(np.mean([_cost(orac[i], args.objective) for i in paired_idx])) if paired_idx else float('nan')

    # 逐实例终局非退化：oracle 不得比 baseline 更差（固定浮点容差，非随决策数累加的退化预算）
    nonregress_violations = []
    for i in paired_idx:
        tol = 1e-6 * max(1.0, abs(_cost(base[i], args.objective)))
        d = _cost(orac[i], args.objective) - _cost(base[i], args.objective)
        if d > tol:
            nonregress_violations.append((i, _cost(base[i], args.objective),
                                          _cost(orac[i], args.objective), d))
    if nonregress_violations:
        raise RuntimeError(f"PROTOCOL_ERROR: {len(nonregress_violations)} 实例 oracle 终局劣于 "
                           f"baseline（超出浮点容差）：{nonregress_violations[:5]}")

    if args.role == 'regression':
        verdict = ('PROTOCOL_PASS' if (base_service and orac_service) else 'SERVICE_GATE_FAIL')
    elif not (base_service and orac_service):
        verdict = 'SERVICE_GATE_FAIL'
    elif mean_delta < 0 and ci_hi < 0:
        verdict = 'GO'
    elif mean_delta < 0:
        verdict = 'EVIDENCE_INSUFFICIENT'
    else:
        verdict = 'NO_HEADROOM'

    print(f"\n  [service] baseline={base_service} oracle={orac_service} "
          f"(paired n={len(paired_idx)}/{n})")
    print(f"  [cost] baseline={base_mean:.4f} oracle={orac_mean:.4f} "
          f"Δ={mean_delta:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
    print(f"  [role={args.role}] [verdict] {verdict}")
    if local_inst_deltas is not None:
        ld = np.array(local_inst_deltas)
        lci = paired_bootstrap_ci(ld) if len(ld) else (float('nan'), float('nan'))
        print(f"  [local headroom] instance-level mean_delta={np.mean(ld):+.4f}  "
              f"CI={list(lci)}  n_instances={len(ld)}  no_eligible={n_local_no_eligible}")

    field = 'distance_cost' if args.objective == 'distance' else 'coldchain_cost'
    component_fields = (['distance_cost', 'quality_loss', 'energy_kwh', 'num_unsalable',
                         'thermal_violation_count', 'thermal_violation_duration_h']
                        if args.objective == 'coldchain' else [])

    with open(os.path.join(args.out, 'per_instance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['instance_id', f'baseline_{field}', f'oracle_{field}', 'delta',
                  'baseline_complete', 'oracle_complete', 'runtime']
        for cf in component_fields:
            header += [f'baseline_{cf}', f'oracle_{cf}']
        w.writerow(header)
        for i in range(n):
            row = [results[i]['inst_idx'], f"{_cost(base[i], args.objective):.4f}",
                   f"{_cost(orac[i], args.objective):.4f}",
                   f"{_cost(orac[i], args.objective) - _cost(base[i], args.objective):+.4f}",
                   int(base[i]['complete']), int(orac[i]['complete']),
                   f"{runtimes[i]:.2f}"]
            for cf in component_fields:
                row += [f"{base[i][cf]:.4f}", f"{orac[i][cf]:.4f}"]
            w.writerow(row)

    summary = {
        'objective': args.objective,
        'role': args.role,
        'instance_ids': idxs,
        'objective_profile': profile.to_manifest() if args.objective == 'coldchain' else None,
        'n': n,
        'workers': args.workers,
        'launch_mode': launch,
        'baseline_service_ok': bool(base_service),
        'oracle_service_ok': bool(orac_service),
        'paired_n': len(paired_idx),
        'baseline_mean': base_mean,
        'oracle_mean': orac_mean,
        'mean_delta': mean_delta,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'verdict': verdict,
        'n_oracle_decision_records': len(log),
        'coverage': {
            'n_candidates': int(sum(r.get('num_candidates', 0) for r in log)),
            'n_feasible': int(sum(r.get('num_feasible', 0) for r in log)),
            'n_rolled_out': int(sum(r.get('num_rolled_out', 0) for r in log)),
            'n_hard_pass': int(sum(r.get('num_hard_pass', 0) for r in log)),
        },
        'n_accepted_actions': int(sum(1 for r in log
                                      if not str(r.get('selected_action', '')).startswith('__'))),
        'n_keep': int(sum(1 for r in log if r.get('selected_action') == '__KEEP__')),
        'n_defer': int(sum(1 for r in log if r.get('selected_action') == '__DEFER__')),
        'first_plan_change_seq': next(
            (r['seq'] for r in log
             if not str(r.get('selected_action', '')).startswith('__')), None),
        'n_failed': 0,
        'per_instance_runtime_mean': float(np.mean(runtimes)) if runtimes else None,
        'per_instance_runtime': [round(t, 2) for t in runtimes],
    }
    if local_inst_deltas is not None:
        ld = np.array(local_inst_deltas)
        summary['local_mean_delta'] = float(np.mean(ld)) if len(ld) else float('nan')
        summary['local_ci'] = list(paired_bootstrap_ci(ld) if len(ld) else (float('nan'), float('nan')))
        summary['local_n_instances'] = int(len(ld))
        summary['local_n_no_eligible'] = int(n_local_no_eligible)
    for cf in component_fields:
        cf_deltas = [orac[i][cf] - base[i][cf] for i in paired_idx]
        cf_lo, cf_hi = paired_bootstrap_ci(cf_deltas) if cf_deltas else (float('nan'), float('nan'))
        summary[f'{cf}_mean_delta'] = float(np.mean(cf_deltas)) if cf_deltas else float('nan')
        summary[f'{cf}_ci'] = [cf_lo, cf_hi]
        print(f"  [{cf}] Δ={summary[f'{cf}_mean_delta']:+.4f}  CI=[{cf_lo:+.4f},{cf_hi:+.4f}]")

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if log:
        with open(os.path.join(args.out, 'oracle_log.jsonl'), 'w') as f:
            for rec in log:
                f.write(json.dumps(rec) + '\n')

    # manifest：workers / 启动方式 / 代码+profile+data hash / 失败数 / 逐实例 runtime
    manifest = {
        'objective': args.objective,
        'workers': args.workers,
        'launch_mode': launch,
        'code_sha256': code_sha256,
        'data_sha256': _sha256_file(args.data),
        'profile_hash': profile.profile_hash if args.objective == 'coldchain' else None,
        'n_failed': 0,
        'n_instances': n,
        'per_instance_runtime': [round(t, 2) for t in runtimes],
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  saved: {args.out}/per_instance.csv + summary.json + manifest.json"
          + (" + oracle_log.jsonl" if log else ""))

    # 失败即停：service Gate 失败返回非零退出码（驱动据此停止后续 cell）
    if not (base_service and orac_service):
        print(f"  [FAIL] SERVICE_GATE_FAIL：baseline_service={base_service} "
              f"oracle_service={orac_service}，退出码非零", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
