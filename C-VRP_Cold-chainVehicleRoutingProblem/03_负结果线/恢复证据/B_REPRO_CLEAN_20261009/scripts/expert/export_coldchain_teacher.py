"""M0 完整候选 teacher 导出器（路线 B / R1）。

对每个基础实例：跑 JF1-H-F baseline 收集「存在 replan 需求」的决策点 snapshot，确定性
抽取有限个决策上下文（snapshot × customer）；对每个上下文的每个 customer，枚举**全部**
候选动作，用隔离 continuation + 共同 continuation rollout 到终局，保存每个候选的终局
D/Q/E/J、hard vector、certificate 与派生标签；并显式保存 KEEP/DEFER 基准伪动作。

输出三层版本化数据集（contexts + candidates + manifest + qc），fail-closed：协议错误、
合同身份缺失/漂移、空上下文均非零退出且不发布 COMPLETE。原子发布。

数据合同见 `docs/当前规划/工程实现/表征探针接口与数据规范.md`。本脚本是 teacher 数据导出，
不是把 DEV-GATE oracle_log 反推成训练集，也不运行任何模型。

用法：
    python scripts/expert/export_coldchain_teacher.py \
        --data <fresh_train_teacher.npz> --dataset-role train_teacher \
        --manifest <split_manifest.json> \
        --objective coldchain --objective-profile <profile.json> \
        --coldchain-contract <contract.json> \
        --max-instances 8 --max-contexts-per-instance 4 \
        --out <teacher_dataset_dir>
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import make_continuation
from counterfactual_teacher import (_eval, _incumbent_plans, rollout_baseline, rollout_action)
from action_contract import enumerate_actions_from_plans, find_customer_slot
from recourse_snapshot import capture_recourse_snapshot
from sequential_oracle import mutable_vehicle_ids, decision_pool, customer_order_key
from coldchain_contract import (default_pilot_contract, load_coldchain_contract,
                                apply_objective_profile, ObjectiveProfile)
from hard_gate import (service_ok, outcome_protocol_error, select_improving,
                       hard_vector_from_outcome)

DATASET_SCHEMA = 'o0cc-teacher-dataset-v1'
FEATURE_SCHEMA_VERSION = 'v1'

# 可见特征 allow-list 摘要（进入模型前必须显式白名单；完整快照只用于审计/恢复）。
ALLOW_LIST_SUMMARY = {
    'feature_schema_version': FEATURE_SCHEMA_VERSION,
    'allowed': {
        'orders': ['coord', 'demand', 'tw_start', 'tw_end', 'service_time', 'temp_class',
                   'initial_quality'],
        'fleet': ['anchor', 'ready_time', 'committed_time', 'load', 'status',
                  'committed_leg', 'mutable_suffix', 'cabin_temperature',
                  'zone_load', 'manifest_summary', 'cumulative_coldchain'],
        'action': ['customer', 'slot_kind', 'slot_anchor', 'position', 'predecessor',
                   'successor', 'incumbent', 'certificate'],
    },
    'forbidden_as_input': [
        'future coords/demand/tw/temp/reveal for unrevealed orders',
        'candidate rollout terminal D/Q/E/J_CC / hard vector / reject reason / selected',
        'future snapshots / oracle choice / global-normalization statistics',
        'dataset_role / filename / seed encodings that leak scenario or label',
    ],
}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b.encode('utf-8')).hexdigest()


def _jsonable(obj):
    """递归把 numpy 类型转成可 JSON 序列化的 Python 类型。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _load_contract(path):
    if path is None:
        return default_pilot_contract()
    return load_coldchain_contract(path)


def _load_profile(path):
    with open(path) as f:
        data = json.load(f)
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
        raise ValueError(f"profile reload hash mismatch: {profile.profile_hash[:12]} "
                         f"vs {data.get('profile_hash', '')[:12]}")
    return profile, data


def _make_env(dataset, capacity, num_vehicles, objective, profile, contract):
    if objective == 'coldchain':
        contract = apply_objective_profile(contract or default_pilot_contract(), profile)
    else:
        contract = None
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=make_continuation(), coldchain_contract=contract)


def _cost(outcome, objective):
    return float(outcome['distance_cost'] if objective == 'distance'
                 else outcome['coldchain_cost'])


def _action_payload(action):
    return {
        'customer': int(action.customer),
        'slot_kind': action.slot.kind,
        'slot_anchor': int(action.slot.anchor),
        'position': int(action.position),
        'predecessor': int(action.predecessor),
        'successor': int(action.successor),
        'incumbent': bool(action.incumbent),
    }


def _action_hash(payload):
    canon = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return _sha256_bytes(canon)


def _context_id(dataset_hash, inst_idx, event_id, state_hash, customer):
    canon = json.dumps([dataset_hash, inst_idx, event_id, state_hash, customer],
                       separators=(',', ':'), ensure_ascii=True)
    return _sha256_bytes(canon)


def _certificate_dict(cand):
    return {
        'feasible': bool(cand.feasible),
        'reason': cand.reason,
        'incremental_distance': cand.incremental_distance,
        'tw_slack': cand.tw_slack,
        'cap_slack': cand.cap_slack,
        'return_slack': cand.return_slack,
        'temperature_compatible': cand.temperature_compatible,
        'total_capacity_slack': cand.total_capacity_slack,
        'manifest_consistent': cand.manifest_consistent,
        'projected_energy_kwh': cand.projected_energy_kwh,
        'projected_quality_loss': cand.projected_quality_loss,
        'projected_thermal_margin': cand.projected_thermal_margin,
        'coldchain_reject_reason': cand.coldchain_reject_reason,
    }


def _outcome_record(outcome):
    """完整终局 outcome（D/Q/E/J + hard/repair 字段），用于标签与审计。"""
    keys = [
        'complete', 'n_unserved', 'n_duplicate', 'tw_feasible', 'capacity_feasible',
        'depot_return_feasible', 'temperature_hard_feasible', 'all_orders_picked',
        'all_cargo_delivered_to_depot', 'terminal_manifests_empty',
        'distance_accounting_consistent', 'trace_accounting_consistent',
        'distance_cost', 'distance_km', 'quality_loss', 'energy_kwh', 'coldchain_cost',
        'num_unsalable', 'thermal_violation_count', 'thermal_violation_duration_h',
        'ownership_violations', 'terminal_unresolved', 'contract_hash',
    ]
    return {k: _jsonable(outcome.get(k)) for k in keys if k in outcome}


def _empty_certificate():
    return {
        'feasible': True, 'reason': None, 'incremental_distance': None,
        'tw_slack': None, 'cap_slack': None, 'return_slack': None,
        'temperature_compatible': None, 'total_capacity_slack': None,
        'manifest_consistent': None, 'projected_energy_kwh': None,
        'projected_quality_loss': None, 'projected_thermal_margin': None,
        'coldchain_reject_reason': None,
    }


def _select_snapshots(snapshots, max_contexts, sampling):
    """按 sampling 选择用于导出的决策点快照。

    earliest：原顺序（默认，保留历史行为）。spread：跨事件均匀取（早/中/晚覆盖），
    快照数 >= max_contexts 时取 max_contexts 个均匀分布的 snapshot（各 1 context）；
    快照数 < max_contexts 时取全部，剩余 context 由调用方分摊到各快照。
    """
    if sampling == 'earliest' or not snapshots:
        return snapshots
    n = len(snapshots)
    if n >= max_contexts:
        idxs = np.unique(np.linspace(0, n - 1, max_contexts).round().astype(int))
        return [snapshots[int(i)] for i in idxs]
    return list(snapshots)


def _export_instance(env, inst_idx, objective, profile, contract, dataset_hash,
                     dataset_role, max_contexts, context_sampling='earliest',
                     instance_seed=None, scene_instance_id=None):
    """跑 baseline 收集 snapshot，按 (snapshot × customer) 抽 ≤max_contexts 个 context。"""
    snapshots = []

    def hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        replan_ids = {v.vehicle_id for v in veh
                      if v.status in ('idle', 'ready') and v.needs_replan}
        if replan_ids:
            snapshots.append(capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac))

    env.snapshot_hook = hook
    traces, served = env.run(inst_idx)
    base_outcome = _eval(env, inst_idx, traces, objective, served_mask=served)
    base_err = outcome_protocol_error(base_outcome, objective, strict_repair=True)
    if base_err is not None:
        raise RuntimeError(f"inst {inst_idx}: baseline PROTOCOL_ERROR: {base_err}")

    contexts = []
    candidates = []
    n_ctx = 0
    sel_snaps = _select_snapshots(snapshots, max_contexts, context_sampling)
    per_snap = 0
    if context_sampling == 'spread' and sel_snaps:
        per_snap = max(1, (max_contexts + len(sel_snaps) - 1) // len(sel_snaps))
    for snap in sel_snaps:
        if n_ctx >= max_contexts:
            break
        event_id = int(snap['event_id'])
        state_hash = snap['state_hash']
        renv = _make_env(env.dataset, env.capacity, env.num_vehicles, objective,
                         profile, contract)
        keep_outcome = rollout_baseline(renv, snap, objective)
        keep_err = outcome_protocol_error(keep_outcome, objective, strict_repair=True)
        if keep_err is not None:
            raise RuntimeError(f"inst {inst_idx} event {event_id}: KEEP PROTOCOL_ERROR: {keep_err}")
        if not _parity_ok(keep_outcome, base_outcome, objective):
            raise RuntimeError(f"inst {inst_idx} event {event_id}: KEEP != baseline terminal")

        incumbent = _incumbent_plans(renv, snap, renv.replanner)
        mutable_ids = mutable_vehicle_ids(snap)
        pool = sorted(decision_pool(snap), key=customer_order_key(renv, inst_idx))
        keep_cost = _cost(keep_outcome, objective)

        n_in_snap = 0
        for customer in pool:
            if n_ctx >= max_contexts:
                break
            if context_sampling == 'spread' and n_in_snap >= per_snap:
                break
            n_ctx += 1
            n_in_snap += 1
            customer = int(customer)
            cid = _context_id(dataset_hash, inst_idx, event_id, state_hash, customer)
            incumbent_exists = find_customer_slot(incumbent, customer) is not None
            cands, _ = enumerate_actions_from_plans(renv, inst_idx, incumbent, customer,
                                                    allowed_vehicle_ids=mutable_ids)

            rows = []          # 本 context 的所有候选（含 KEEP/DEFER 伪动作）
            rolled = []        # (cost, action_id, rec) for service_ok 且非协议错误
            incumbent_seen = False
            for cand in cands:
                payload = _action_payload(cand.action)
                is_incumbent = bool(cand.action.incumbent)
                if is_incumbent:
                    incumbent_seen = True
                aid = '__KEEP__' if is_incumbent else cand.action.action_id()
                rec = {
                    'context_id': cid,
                    'action_id': aid,
                    'is_pseudo': is_incumbent,
                    'pseudo': 'KEEP' if is_incumbent else None,
                    'action_hash': _action_hash(payload),
                    'action': payload,
                    'certificate': _certificate_dict(cand),
                    'feasible': bool(cand.feasible),
                    'rolled_out': False,
                    'service_ok': False,
                    'protocol_error': False,
                    'delta_vs_keep': None,
                    'strictly_improving': False,
                    'rank_within_context': None,
                    'selected_by_teacher': False,
                }
                if not cand.feasible:
                    # 枚举负例：保存 certificate，不做 rollout。
                    rows.append(rec)
                    continue
                outcome, ph = rollout_action(
                    renv, snap, cand.action, renv.replanner, incumbent,
                    objective=objective, allowed_vehicle_ids=mutable_ids, mutable_ids=mutable_ids)
                err = outcome_protocol_error(outcome, objective, strict_repair=True)
                if is_incumbent:
                    # KEEP 候选的 rollout 必须等于该 context 的 baseline（D/Q/E/J + hard + contract）。
                    if not _parity_ok(outcome, keep_outcome, objective):
                        raise RuntimeError(
                            f"inst {inst_idx} event {event_id} customer {customer}: "
                            f"__KEEP__ rollout != context baseline")
                    if outcome.get('contract_hash') != keep_outcome.get('contract_hash'):
                        raise RuntimeError(
                            f"inst {inst_idx} event {event_id} customer {customer}: "
                            f"__KEEP__ contract_hash 与 baseline 不一致")
                rec['rolled_out'] = True
                rec['candidate_plan_hash'] = ph
                rec['outcome'] = _outcome_record(outcome)
                rec['hard_vector'] = hard_vector_from_outcome(outcome)
                rec['service_ok'] = bool(service_ok(outcome, objective))
                rec['protocol_error'] = err is not None
                rec['delta_vs_keep'] = _cost(outcome, objective) - keep_cost
                if err is None and rec['service_ok']:
                    rolled.append((_cost(outcome, objective), aid, rec))
                rows.append(rec)

            # 未分配客户：显式 DEFER 基准伪动作（先构造 payload，再对其计算 hash）。
            if not incumbent_seen:
                defer_action = {'customer': customer, 'kind': 'defer'}
                defer_rec = {
                    'context_id': cid, 'action_id': '__DEFER__', 'is_pseudo': True,
                    'pseudo': 'DEFER',
                    'action_hash': _action_hash(defer_action),
                    'action': defer_action,
                    'certificate': _empty_certificate(),
                    'feasible': True, 'rolled_out': False,
                    'outcome': _outcome_record(keep_outcome),
                    'hard_vector': hard_vector_from_outcome(keep_outcome),
                    'service_ok': bool(service_ok(keep_outcome, objective)),
                    'protocol_error': False,
                    'delta_vs_keep': 0.0,
                    'strictly_improving': False,
                    'rank_within_context': None,
                    'selected_by_teacher': False,
                }
                rows.append(defer_rec)

            # 同 context 排序 + 稳定最优改善动作。
            rolled.sort(key=lambda x: (x[0], x[1]))
            best = select_improving([(c, a) for c, a, _ in rolled], keep_cost)
            best_id = best[1] if best is not None else None
            tau = 1e-9 + 1e-9 * max(1.0, abs(keep_cost))
            for k, (_, aid, rec) in enumerate(rolled):
                rec['strictly_improving'] = rec['delta_vs_keep'] < -tau
                rec['rank_within_context'] = k + 1
                rec['selected_by_teacher'] = (aid == best_id)
            if best_id is None:
                # 无改善动作 → teacher 选择 KEEP/DEFER（仅当 service_ok 且无协议错误）。
                for rec in rows:
                    if (rec.get('is_pseudo') and rec.get('service_ok')
                            and not rec.get('protocol_error')):
                        rec['selected_by_teacher'] = True
            candidates.extend(rows)

            ctx = {
                'context_id': cid,
                'dataset_role': dataset_role,
                'dataset_hash': dataset_hash,
                'inst_idx': inst_idx,
                'instance_seed': instance_seed,
                'scene_instance_id': scene_instance_id,
                'event_id': event_id,
                'clock': float(snap['clock']),
                'state_hash': state_hash,
                'customer': customer,
                'customer_rank': len(contexts),
                'incumbent_exists': incumbent_exists,
                'snapshot': _jsonable(snap),
                'baseline_outcome': _outcome_record(keep_outcome),
                'keep_cost': keep_cost,
                'n_candidates': len(rows),
                'raw_contract_hash': contract.contract_hash,
                'effective_contract_hash': (apply_objective_profile(contract, profile)
                                            .contract_hash),
                'feature_schema_version': FEATURE_SCHEMA_VERSION,
            }
            contexts.append(ctx)
    return contexts, candidates, base_outcome


def _parity_ok(a, b, objective):
    if objective == 'distance':
        return (bool(a['complete']) == bool(b['complete'])
                and abs(float(a['distance_cost']) - float(b['distance_cost'])) <= 1e-9)
    return (hard_vector_from_outcome(a) == hard_vector_from_outcome(b)
            and abs(float(a['distance_km']) - float(b['distance_km'])) <= 1e-9
            and abs(float(a['quality_loss']) - float(b['quality_loss'])) <= 1e-9
            and abs(float(a['energy_kwh']) - float(b['energy_kwh'])) <= 1e-9
            and abs(float(a['coldchain_cost']) - float(b['coldchain_cost'])) <= 1e-9)


def _atomic_write_text(path, text):
    tmp = path + f'.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_jsonl(path, records):
    _atomic_write_text(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                     for r in records))


def _match_manifest_cell(manifest, dataset_hash):
    """按 NPZ 文件 hash **唯一**匹配 manifest cell；文件名不作为身份替代。"""
    cells = manifest.get('cells', [])
    matches = [c for c in cells if c.get('sha256') == dataset_hash]
    if len(matches) != 1:
        return None  # 0 或 >1（歧义）都视为不匹配
    return matches[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='基础实例 NPZ（fresh TRAIN/DEV-TEACHER）')
    ap.add_argument('--dataset-role', required=True, choices=['train_teacher', 'dev_teacher'])
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', required=True)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--max-instances', type=int, default=8)
    ap.add_argument('--max-contexts-per-instance', type=int, default=4)
    ap.add_argument('--context-sampling', choices=['earliest', 'spread'], default='earliest',
                    help='earliest=按最早事件截断（默认）；spread=跨事件均匀抽样（覆盖早/中/晚）')
    ap.add_argument('--manifest', default=None, help='split manifest（cell 匹配 + seed/scene）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    # 前置校验：参数合法 + 输出目录为空，避免先跑 rollout 才失败。
    if args.max_instances <= 0:
        raise SystemExit(f'--max-instances 必须为正：{args.max_instances}')
    if args.max_contexts_per_instance <= 0:
        raise SystemExit(f'--max-contexts-per-instance 必须为正：{args.max_contexts_per_instance}')
    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f'输出目录非空，拒绝覆盖：{out}')

    contract = _load_contract(args.coldchain_contract)
    profile, profile_data = _load_profile(args.objective_profile)
    # profile provenance.contract_hash 若存在，必须等于 raw contract。
    prof_contract = (profile_data.get('provenance') or {}).get('contract_hash')
    if prof_contract and prof_contract != contract.contract_hash:
        raise SystemExit(f'profile provenance.contract_hash 与 raw contract 不一致：'
                         f'{prof_contract[:12]} vs {contract.contract_hash[:12]}')

    dataset = dict(np.load(args.data))
    actual_n = int(dataset['coords'].shape[0])
    if args.max_instances > actual_n:
        raise ValueError(f"--max-instances={args.max_instances} > data n={actual_n}")

    dataset_hash = _sha256_file(args.data)
    instance_seeds = None
    scene_ids = None
    split_registered = False
    if args.manifest:
        manifest = json.load(open(args.manifest))
        cell = _match_manifest_cell(manifest, dataset_hash)
        if cell is None:
            raise SystemExit(f'--data {args.data} 未在 manifest 中按 hash 唯一匹配到 cell')
        all_seeds = [int(s) for s in cell.get('instance_seeds', [])]
        all_scene = list(cell.get('scene_instance_ids', []))
        if len(all_seeds) < args.max_instances:
            raise SystemExit(f'manifest instance_seeds 不足 {args.max_instances}：'
                             f'{len(all_seeds)}')
        if len(all_scene) < args.max_instances:
            raise SystemExit(f'manifest scene_instance_ids 不足 {args.max_instances}：'
                             f'{len(all_scene)}')
        if len(set(all_seeds[:args.max_instances])) != args.max_instances:
            raise SystemExit('manifest instance_seeds 非唯一')
        instance_seeds = all_seeds[:args.max_instances]
        scene_ids = all_scene[:args.max_instances]
        split_registered = True

    started = time.time()
    all_contexts, all_candidates = [], []
    for inst_idx in range(args.max_instances):
        env = _make_env(dataset, args.capacity, args.num_vehicles, args.objective,
                        profile, contract)
        contexts, candidates, _ = _export_instance(
            env, inst_idx, args.objective, profile, contract, dataset_hash,
            args.dataset_role, args.max_contexts_per_instance, args.context_sampling,
            instance_seed=instance_seeds[inst_idx] if instance_seeds else None,
            scene_instance_id=scene_ids[inst_idx] if scene_ids else None)
        all_contexts.extend(contexts)
        all_candidates.extend(candidates)
        print(f"inst {inst_idx}: contexts={len(contexts)} candidates={len(candidates)}",
              flush=True)

    # QC（fail-closed）
    n_ctx = len(all_contexts)
    n_cand = len(all_candidates)
    rolled = [c for c in all_candidates if c.get('rolled_out')]
    n_rolled = len(rolled)
    n_service_ok = sum(1 for c in rolled if c.get('service_ok'))
    n_proto_err = sum(1 for c in rolled if c.get('protocol_error'))
    n_selected = sum(1 for c in all_candidates if c.get('selected_by_teacher'))

    effective = apply_objective_profile(contract, profile).contract_hash
    rolled_hashes = [c.get('outcome', {}).get('contract_hash') for c in rolled]
    contract_missing = [c['context_id'] for c, h in zip(rolled, rolled_hashes) if h is None]
    contract_mismatch = [c['context_id'] for c, h in zip(rolled, rolled_hashes)
                         if h is not None and h != effective]
    contract_consistent = (not contract_missing) and (not contract_mismatch)

    failures = []
    if n_proto_err > 0:
        failures.append(f'protocol_error={n_proto_err}')
    if contract_missing:
        failures.append(f'contract_hash_missing={len(contract_missing)}')
    if contract_mismatch:
        failures.append(f'contract_hash_mismatch={len(contract_mismatch)}')
    if n_ctx == 0:
        failures.append('no_trainable_contexts')

    manifest = {
        'schema': DATASET_SCHEMA,
        'dataset_role': args.dataset_role,
        'objective': args.objective,
        'dataset_hash': dataset_hash,
        'data_path': args.data,
        'split_registered': split_registered,
        'raw_contract_hash': contract.contract_hash,
        'effective_contract_hash': effective,
        'profile_hash': profile.profile_hash,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'allow_list_summary': ALLOW_LIST_SUMMARY,
        'capacity': args.capacity,
        'num_vehicles': args.num_vehicles,
        'max_instances': args.max_instances,
        'max_contexts_per_instance': args.max_contexts_per_instance,
        'base_instances': list(range(args.max_instances)),
        'instance_seeds': instance_seeds,
        'scene_instance_ids': scene_ids,
        'n_contexts': n_ctx,
        'n_candidates': n_cand,
        'n_rolled_out': n_rolled,
        'n_service_ok': n_service_ok,
        'n_protocol_error': n_proto_err,
        'n_selected_by_teacher': n_selected,
        'contract_consistent': contract_consistent,
        'failures': failures,
        'wall_seconds': round(time.time() - started, 2),
        'generated_at_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'generated_command': ' '.join(sys.argv),
    }

    os.makedirs(os.path.join(out, 'qc'), exist_ok=True)
    _write_jsonl(os.path.join(out, 'contexts.jsonl'), all_contexts)
    _write_jsonl(os.path.join(out, 'candidates.jsonl'), all_candidates)
    _atomic_write_text(os.path.join(out, 'manifest.json'),
                       json.dumps(manifest, indent=2, ensure_ascii=False))
    gate = {
        'n_contexts': n_ctx, 'n_candidates': n_cand, 'n_rolled_out': n_rolled,
        'n_service_ok': n_service_ok, 'n_protocol_error': n_proto_err,
        'n_selected_by_teacher': n_selected,
        'contract_consistent': contract_consistent,
        'contract_hash_missing': len(contract_missing),
        'contract_hash_mismatch': len(contract_mismatch),
        'keep_parity': 'PASS',
        'fail_closed': not failures,
        'failures': failures,
    }
    _atomic_write_text(os.path.join(out, 'qc', 'gate_summary.json'),
                       json.dumps(gate, indent=2, ensure_ascii=False))
    _atomic_write_text(os.path.join(out, 'qc', 'parity_summary.json'),
                       json.dumps({'keep_parity': 'PASS',
                                   'branch_isolation': 'isolated_continuation_per_rollout'},
                                  indent=2, ensure_ascii=False))

    if failures:
        _atomic_write_text(os.path.join(out, 'FAILED'),
                           json.dumps({'status': 'FAILED', 'failures': failures,
                                       'n_contexts': n_ctx}, indent=2, ensure_ascii=False))
        print(f'teacher export FAILED: {failures}', file=sys.stderr)
        sys.exit(1)

    _atomic_write_text(os.path.join(out, 'COMPLETE'),
                       json.dumps({'status': 'COMPLETE', 'n_contexts': n_ctx,
                                   'n_candidates': n_cand,
                                   'manifest_sha256': _sha256_file(os.path.join(out, 'manifest.json'))},
                                  indent=2, ensure_ascii=False))
    print(f"teacher dataset: {out}")
    print(f"  contexts={n_ctx} candidates={n_cand} rolled={n_rolled} "
          f"service_ok={n_service_ok} proto_err={n_proto_err} selected={n_selected}")
    print(f"  effective_contract={effective[:16]} contract_consistent={contract_consistent}")


if __name__ == '__main__':
    main()
