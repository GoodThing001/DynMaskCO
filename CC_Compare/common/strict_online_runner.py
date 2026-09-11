"""公共 strict-online runner（B1）：事件推进、commit、写回校验、记录装配与落盘。

外部方法只通过 ExternalReplanner 产生 mutable plans；本 runner 负责：
  - 用项目 StrictOnlineEnv 作为唯一事件引擎（commit / service / return / cold-chain
    trace 全部由 env 产生，runner 不重实现任何物理）；
  - RecordingEnv 在每次 _plan_and_commit 后捕获 commit 层动作（COMMIT/CLOSE/WAIT）；
  - 从 BridgeReplanner 的 fleet plan 前后 diff 派生 KEEP/DEFER/INSERT/NEW_ROUTE 动作，
    并用 action_contract.certify_route 生成 certificate；
  - 终局 outcome 一律由 counterfactual_teacher._eval（→ evaluate_coldchain_trace）重算，
    不信任 adapter 自报；
  - 静态检查：diff 出「未 reveal 客户被写入 plan」即 PROTOCOL_ERROR（未来泄漏的直接证据）。

记录 schema 见 baseline_contract；校验见 record_validation（NaN/Inf/缺字段拒绝）。
"""
import json
import os
import sys
import time
import traceback

import numpy as np

_COMMON = os.path.dirname(os.path.abspath(__file__))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
import _bootstrap  # noqa: F401

from strict_online_env import StrictOnlineEnv
from recourse_snapshot import capture_recourse_snapshot
from action_contract import build_vehicle_plans, certify_route, plan_hash
from counterfactual_teacher import _eval
from hard_gate import (outcome_protocol_error, hard_vector_from_outcome,
                       service_ok)
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile)

import baseline_contract as bc
import trace_export as te
import ownership_audit as oa
from method_adapter import (BridgeReplanner, NativeBridgeReplanner,
                            NativeReplanner, ContractViolation)
from record_validation import validate_instance_record

# 计算链：决定行为的所有源码文件（公共合同 + adapter + 项目计算模块）。
# 任一文件变化 → code_hash 变化 → 复现身份失效（续跑/复用必须重验）。
# 组合规则（identity manifest rev2）：按**逻辑相对路径**排序后组合
# 「逻辑路径 + 文件 SHA-256」；不使用绝对路径（跨平台不稳定）或仅 basename
# （同名冲突）。
PROJECT_COMPUTE_FILES = [
    'simulation/strict_online_env.py',
    'simulation/action_contract.py',
    'simulation/recourse_snapshot.py',
    'simulation/counterfactual_teacher.py',
    'evaluation/authoritative_evaluator.py',
    'evaluation/hard_gate.py',
    'evaluation/coldchain_evaluator.py',
    'coldchain/coldchain_contract.py',
    'coldchain/coldchain_state.py',
]
COMMON_CORE_FILES = ['strict_online_runner.py', 'method_adapter.py',
                     'baseline_contract.py', 'record_validation.py',
                     'trace_export.py', 'ownership_audit.py', '_bootstrap.py']


class RecordingEnv(StrictOnlineEnv):
    """StrictOnlineEnv + 每次决策点 commit 后状态捕获（不改项目文件）。"""

    def run(self, *args, **kwargs):
        self.post_commit_captures = []
        return super().run(*args, **kwargs)

    def run_resumed(self, *args, **kwargs):
        self.post_commit_captures = []
        return super().run_resumed(*args, **kwargs)

    def _plan_and_commit(self, inst_idx, clock, vehicles, traces, served_mask,
                         all_customers):
        pre = {v.vehicle_id: (v.status, v.committed_next, v.return_finish)
               for v in vehicles}
        super()._plan_and_commit(inst_idx, clock, vehicles, traces, served_mask,
                                 all_customers)
        cap = []
        for v in vehicles:
            before = pre[v.vehicle_id]
            cap.append({
                'vehicle_id': int(v.vehicle_id),
                'status_before': str(before[0]),
                'committed_next_before': before[1],
                'return_finish_before': before[2],
                'status_after': str(v.status),
                'committed_next_after': v.committed_next,
                'return_finish_after': v.return_finish,
                'suffix_after': [int(x) for x in v.mutable_suffix],
            })
        self.post_commit_captures.append({
            'event_id': int(self.event_id),
            'clock': float(clock),
            'vehicles': cap,
        })


def load_objective_profile(path):
    """加载具名 objective profile（与项目 _load_profile 同口径：拒绝 v1、校验 hash）。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError('拒绝加载 INVALIDATED v1 profile（non-hard-feasible JF1-H baseline）；'
                         '请使用 v2')
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
        raise ValueError(f'profile reload hash 不一致：recomputed={profile.profile_hash} '
                         f'file={data.get("profile_hash")}')
    return profile


def make_contract(profile):
    if profile is None:
        return None
    return apply_objective_profile(default_pilot_contract(), profile)


def dataset_hash(dataset):
    keys = sorted(dataset.keys())
    h_parts = []
    for k in keys:
        h_parts.append(bc.array_hash(np.asarray(dataset[k])))
    return bc.sha256_hex('|'.join(f'{k}:{h}' for k, h in zip(keys, h_parts))
                         .encode('utf-8'))


def code_hash(adapter_module_path=None, adapter_compute_files=()):
    """计算链 code hash（identity manifest rev2）。

    逻辑相对路径命名空间：
      common/<file>   公共合同模块
      adapter/<file>  adapter 计算文件（相对 adapter 模块目录；无前缀条目）
      project/<file>  项目计算模块（相对 scripts 根；'project/' 前缀条目）
    按逻辑路径排序组合「逻辑路径 + 文件 SHA-256」，与平台绝对路径无关。
    """
    import hashlib
    files = []
    for f in COMMON_CORE_FILES:
        files.append((f'common/{f}', os.path.join(_COMMON, f)))
    if adapter_module_path:
        adir = os.path.dirname(os.path.abspath(adapter_module_path))
        for f in adapter_compute_files:
            if f.startswith('project/'):
                files.append((f, os.path.join(_bootstrap.PROJECT_SCRIPTS_ROOT,
                                              f[len('project/'):])))
            elif f.startswith('common/'):
                files.append((f, os.path.join(_COMMON, f[len('common/'):])))
            else:
                files.append((f'adapter/{f}', os.path.join(adir, f)))
    _scripts = _bootstrap.PROJECT_SCRIPTS_ROOT
    for f in PROJECT_COMPUTE_FILES:
        files.append((f'project/{f}', os.path.join(_scripts, f)))
    h = hashlib.sha256()
    for logical, p in sorted(files):
        h.update(logical.encode('utf-8'))
        h.update(b'\x00')
        h.update(bc.sha256_file(p).encode('ascii'))
        h.update(b'\x00')
    return h.hexdigest()


def _visible_set(snapshot):
    return {int(c) for c in snapshot['customer_universe']
            if bool(snapshot['visible_mask'][int(c)])}


def _wait_audit(env, record):
    """WAIT 动作审计（公共层，所有 adapter 共用）。

    对每条 execution 层 WAIT 记录：
      - 即时返仓 slack（WAIT 时 ready_time + travel(anchor→0) 距 depot_tw_end）；
      - 到下一实际事件的事后间隔（interval）；
      - 下一事件时重算的返仓 slack（用事件记录里的车辆状态）；
      - customer-anchor / loaded 计数。
    hard gate 只能发现返仓失败、不保证不会失败——本审计为规则修订提供证据。
    """
    inst_idx = record['instance_id']
    depot_tw_end = float(env.tw_end[inst_idx, 0])
    events = record['events']
    by_id = {e['event_id']: e for e in events}
    vmap = {v['vehicle_id']: v for v in events[0]['vehicles']}

    def slack(v):
        return (float(depot_tw_end)
                - (float(v['ready_time'])
                   + float(env.dist_mat[inst_idx, int(v['anchor_node']), 0])
                   / env.tw_speed))

    wait_actions = [a for a in record['actions']
                    if a['action_layer'] == 'execution'
                    and a['action_type'] == 'WAIT']
    records = []
    n_customer_anchor = 0
    n_loaded = 0
    for a in wait_actions:
        ev = by_id[a['event_id']]
        v = next(x for x in ev['vehicles'] if x['vehicle_id'] == a['vehicle_id'])
        idx = events.index(ev)
        next_ev = events[idx + 1] if idx + 1 < len(events) else None
        interval = (float(next_ev['clock']) - float(ev['clock'])
                    if next_ev is not None else None)
        v_next = None
        slack_next = None
        if next_ev is not None:
            match = [x for x in next_ev['vehicles']
                     if x['vehicle_id'] == a['vehicle_id']]
            if match:
                v_next = match[0]
                slack_next = slack(v_next)
        anchor_is_customer = int(v['anchor_node']) != 0
        loaded = float(v['load']) > 1e-9
        n_customer_anchor += int(anchor_is_customer)
        n_loaded += int(loaded)
        records.append({
            'event_id': a['event_id'],
            'vehicle_id': a['vehicle_id'],
            'anchor_node': v['anchor_node'],
            'load': v['load'],
            'ready_time': v['ready_time'],
            'slack_now': slack(v),
            'interval_to_next_event': interval,
            'next_event_id': (next_ev['event_id'] if next_ev is not None
                              else None),
            'next_anchor_node': (v_next['anchor_node'] if v_next else None),
            'next_ready_time': (v_next['ready_time'] if v_next else None),
            'slack_at_next_event': slack_next,
            'anchor_is_customer': anchor_is_customer,
            'loaded': loaded,
        })
    return {
        'records': records,
        'n_waits': len(records),
        'n_customer_anchor_waits': n_customer_anchor,
        'n_loaded_waits': n_loaded,
    }


def run_instance(dataset, capacity, num_vehicles, adapter_factory, inst_idx,
                 objective='coldchain', profile=None, seed=0,
                 adapter_module_path=None, data_sha256=None,
                 instance_seed=None, runtime_budget=None,
                 scene_instance_id=None):
    """运行一个实例，返回完整 baseline 记录（详见 baseline_contract）。

    adapter_factory：无参可调用，每次调用返回**独立**的 ExternalReplanner 实例
    （与项目 continuation 工厂同约定：每条 rollout 分支独立状态）。
    """
    t0 = time.time()
    np.random.seed(seed)
    adapter = adapter_factory()
    bridge = (NativeBridgeReplanner(adapter) if isinstance(adapter, NativeReplanner)
              else BridgeReplanner(adapter))
    contract = make_contract(profile)
    env = RecordingEnv(dataset, capacity, 1.0, num_vehicles,
                       replanner=bridge, coldchain_contract=contract)

    # adapter 显式声明计算依赖文件（参与决策/映射，进 code_hash）
    adapter_compute_files = tuple(getattr(adapter, 'compute_files', lambda: ())())

    identity = {
        'method_name': bridge.method_name,
        'method_revision': getattr(adapter, 'method_revision', '0'),
        'adapter_revision': bridge.adapter_revision,
        'checkpoint_hash': bridge.checkpoint_hash,
        'code_hash': code_hash(adapter_module_path, adapter_compute_files),
        'data_hash': data_sha256 or dataset_hash(dataset),
        'profile_hash': (profile.profile_hash if profile is not None else 'none'),
        'instance_id': int(inst_idx),
        'instance_seed': int(instance_seed if instance_seed is not None else seed),
        'objective': objective,
        'runtime_budget': runtime_budget,
        'random_seed': int(seed),
    }
    record = bc.new_record(identity)
    if scene_instance_id is not None:
        record['scene_instance_id'] = str(scene_instance_id)

    snapshots = []
    prev_visible = set()

    def hook(e, inst, clk, eid, rid, veh, tr, sm, ac):
        snap = capture_recourse_snapshot(e, inst, clk, eid, rid, veh, tr, sm, ac)
        snapshots.append(snap)
        replan_ids = {v.vehicle_id for v in veh
                      if v.status in ('idle', 'ready') and v.needs_replan}
        ev = bc.event_record(snap, replan_ids, veh)
        vis_now = _visible_set(snap)
        ev['revealed_customer_ids'] = sorted(vis_now - prev_visible)
        prev_visible.clear()
        prev_visible.update(vis_now)
        record['events'].append(ev)

    env.snapshot_hook = hook

    try:
        traces, served_mask = env.run(inst_idx)
    except ContractViolation:
        raise

    outcome = _eval(env, inst_idx, traces, objective, served_mask=served_mask)

    # ---- repair 层字段分离（原生 adapter 自带 repair 层时 _eval 已附加）----
    repair_own = outcome.pop('ownership_violations', None)
    repair_term = outcome.pop('terminal_unresolved', None)
    has_repair = repair_own is not None

    # ---- 公共 ownership audit（逐事件 + 终局）：真实整数，不填默认值 ----
    per_event_audit = []
    customer_universe = [int(c) for c in snapshots[0]['customer_universe']]
    plan_by_eid = {h['event_id']: h for h in bridge.plan_hook_records}
    for snap in snapshots:
        h = plan_by_eid.get(int(snap['event_id']))
        plans = h['plans_after'] if h is not None else {}
        committed_by = {}
        for vid, nxt in enumerate(snap['committed_next']):
            if int(nxt) not in (-1, 0):
                committed_by.setdefault(int(nxt), []).append(int(vid))
        vis_ids = [int(c) for c in snap['customer_universe']
                   if bool(snap['visible_mask'][int(c)])]
        violations, deferred = oa.audit_decision_point(
            plans, committed_by, snap['served_mask'], vis_ids, customer_universe)
        per_event_audit.append({
            'event_id': int(snap['event_id']),
            'violations': violations,
            'deferred': deferred,
            'n_violations': len(violations),
        })
    summary = oa.summarize_audit(per_event_audit)
    terminal = oa.audit_terminal(served_mask, customer_universe)

    outcome['ownership_violations'] = int(summary['ownership_violations'])
    outcome['terminal_unresolved'] = int(len(terminal))
    outcome['repair_applicable'] = bool(has_repair)
    outcome['audit_source'] = ('common_runner + repair layer' if has_repair
                               else 'common_runner')
    if has_repair:
        # repair 层审计保留为独立字段（JF1-H-F 自带 repair 机制的真实计数）
        outcome['repair_ownership_violations'] = int(repair_own)
        outcome['repair_terminal_unresolved'] = int(repair_term)
    record['audit'] = {
        'ownership_violations': int(summary['ownership_violations']),
        'violations_by_type': summary['violations_by_type'],
        'terminal_unresolved': int(len(terminal)),
        'terminal_unresolved_customers': terminal,
        'n_events_with_deferred': int(summary['n_events_with_deferred']),
        'per_event': per_event_audit,
    }
    if summary['ownership_violations'] > 0:
        record['protocol']['error'] = ('ownership audit 违规: '
                                       + str(summary['violations_by_type']))
        raise RuntimeError(f'PROTOCOL_ERROR: 实例 {inst_idx} ownership audit 违规 '
                           f'{summary["violations_by_type"]}')

    record['execution_trace'] = te.export_trace(traces, contract)
    record['outcome'] = outcome
    record['hard_vector'] = hard_vector_from_outcome(outcome)

    # ---- fleet plan hash / 模型输入 / 耗时 / fallback / pool 分区 / solve_meta
    # 按 event_id 匹配到事件记录 ----
    plan_by_eid = {h['event_id']: h for h in bridge.plan_hook_records}
    runtime_budget = float(adapter.runtime_budget) if getattr(adapter, 'runtime_budget',
                                                             None) is not None else None
    for ev in record['events']:
        h = plan_by_eid.get(ev['event_id'])
        ev['plan_hash_before'] = h['plan_hash_before'] if h else None
        ev['plan_hash_after'] = h['plan_hash_after'] if h else None
        ev['model_input_customers'] = (h['model_input_customers'] if h else [])
        ev['model_runtime_s'] = (h['model_runtime_s'] if h else 0.0)
        ev['fallback_triggered'] = (h['fallback_triggered'] if h else False)
        ev['budget_exceeded'] = (bool(runtime_budget is not None
                                      and ev['model_runtime_s'] > runtime_budget))
        if h is not None and h.get('view') is not None:
            ev['pool_customer_ids'] = list(h['view'].pool_customer_ids)
            ev['protected_customer_ids'] = list(h['view'].protected_customer_ids)
            ev['solve_meta'] = h['solve_meta']
        else:
            # 无 replan 的决策点（plan persistence）：无决策视图，
            # 分区审计不适用（None，校验层跳过）
            ev['pool_customer_ids'] = None
            ev['protected_customer_ids'] = None
            ev['solve_meta'] = None

    # ---- action 记录装配 ----
    actions = _derive_actions(env, record, bridge, snapshots, dataset, inst_idx)
    record['actions'] = actions

    # ---- 校验 ----
    err, checks = validate_instance_record(record, objective, strict_repair=True)
    record['protocol']['checks'] = checks
    if err is not None:
        record['protocol']['error'] = err
        raise RuntimeError(f'PROTOCOL_ERROR: 实例 {inst_idx} 记录校验失败：{err}')

    ok_replay, problems = te.trace_replay_check(record['execution_trace'], outcome,
                                                objective)
    if not ok_replay:
        record['protocol']['error'] = 'trace_replay_check: ' + '; '.join(problems)
        raise RuntimeError(f'PROTOCOL_ERROR: 实例 {inst_idx} trace 对账失败：{problems}')

    record['runtime_s'] = time.time() - t0
    # ---- 协议统计（区分 defer / fast_path / solver failure 三类语义）----
    events = record['events']
    record['wait_audit'] = _wait_audit(env, record)
    record['stats'] = {
        'n_events': len(events),
        'n_solver_calls': sum(1 for e in events
                              if e.get('solve_meta')
                              and not e['solve_meta'].get('empty_pool_fast_path')),
        'n_fast_path': sum(1 for e in events
                           if e.get('solve_meta')
                           and e['solve_meta'].get('empty_pool_fast_path')),
        'n_unplanned_customer_events': sum(
            1 for pe in record['audit']['per_event'] if pe['deferred']),
        'n_penalty_warnings': sum(
            int((e.get('solve_meta') or {}).get('n_penalty_warnings', 0))
            for e in events),
        'fallback_triggered_events': sum(1 for e in events
                                         if e['fallback_triggered']),
        'n_waits': record['wait_audit']['n_waits'],
        'n_customer_anchor_waits': record['wait_audit']['n_customer_anchor_waits'],
        'n_loaded_waits': record['wait_audit']['n_loaded_waits'],
    }
    record['decision_hash'] = bc.decision_hash(record)
    record['artifact_hash'] = bc.artifact_hash(record)
    return record


def _derive_actions(env, record, bridge, snapshots, dataset, inst_idx):
    """装配 action 记录（分层）：
      1) execution 层：RecordingEnv 捕获的每决策点 COMMIT/CLOSE/WAIT（每车一条）；
      2) plan_diff 层：KEEP/DEFER/INSERT/NEW_ROUTE（每个受影响客户一条，审计性解释）
         + certificate（scope=final_vehicle_suffix，验证变更后整条车辆 suffix）；
      3) 未来泄漏静态检查（防御纵深：Bridge 已结构性拒绝，此处二次确认）。
    writeback_ok 语义 = 「最终 fleet plan 与 proposal 一致」——plan_diff 记录即由
    写回后的 plans_after 派生，故恒为 True；DEFER（客户被移出计划）同样为 True。
    """
    actions = []
    events = record['events']

    # -- 1) execution 层 --
    for cap in env.post_commit_captures:
        eid = cap['event_id']
        for c in cap['vehicles']:
            if c['status_before'] not in ('idle', 'ready'):
                continue
            nxt = c['committed_next_after']
            if c['status_after'] == 'committed' and nxt not in (None, 0):
                atype, cust = 'COMMIT', int(nxt)
            elif c['status_after'] == 'returning' or (
                    c['status_after'] in ('idle', 'closed')
                    and c['return_finish_after'] != c['return_finish_before']):
                atype, cust = 'CLOSE', 0
            else:
                atype, cust = 'WAIT', None
            actions.append(bc.commit_action_record(
                eid, c['vehicle_id'], cust, atype, None,
                [], c['suffix_after'], 'execution', writeback_ok=True,
                commit_info={'status_after': c['status_after'],
                             'committed_next_after': nxt}))

    # -- 2) plan_diff 层 --
    visible_by_event = {ev['event_id']: ev for ev in events}
    for hk in bridge.plan_hook_records:
        eid = hk['event_id']
        ev = visible_by_event.get(eid)
        if ev is None:
            continue
        vis_ids = set(ev['visible_customer_ids'])
        pb, pa = hk['plans_before'], hk['plans_after']

        def locate(plans, cust):
            for vid, p in plans.items():
                if cust in p.suffix:
                    return vid, p.suffix.index(cust)
            return None, None

        customers = set()
        for p in list(pb.values()) + list(pa.values()):
            customers.update(p.suffix)
        for cust in sorted(customers):
            vb, pb_pos = locate(pb, cust)
            va, pa_pos = locate(pa, cust)
            # 防御纵深：Bridge 已结构性拒绝未来客户；此处二次确认
            if cust not in vis_ids:
                raise RuntimeError(
                    f'PROTOCOL_ERROR: 实例 {record["instance_id"]} 事件 {eid}：'
                    f'未 reveal 客户 {cust} 出现在 fleet plan（未来泄漏）')
            if vb is None and va is None:
                continue
            if vb == va and pb_pos == pa_pos:
                atype, pos = 'KEEP', pa_pos
            elif va is None:
                atype, pos = 'DEFER', None
            else:
                p_before = pb.get(va)
                atype = ('NEW_ROUTE'
                         if p_before is not None and p_before.anchor_node == 0
                         and not p_before.suffix else 'INSERT')
                pos = pa_pos
            cert = None
            if atype in ('INSERT', 'NEW_ROUTE') and va is not None:
                p_after = pa[va]
                cert = certify_route(env, inst_idx, p_after.anchor_node,
                                     p_after.anchor_time, p_after.anchor_load,
                                     p_after.suffix)
            actions.append(bc.commit_action_record(
                eid, vb if va is None else va, cust, atype, pos,
                list(pb.get(vb).suffix) if vb is not None else [],
                list(pa.get(va).suffix) if va is not None else [],
                'plan_diff', certificate=cert, writeback_ok=True))

    actions.sort(key=lambda a: (a['event_id'], a['action_layer'] != 'execution',
                                a['vehicle_id'],
                                a['customer_id'] is None, a['customer_id'] or 0))
    return actions


def run_batch(dataset, capacity, num_vehicles, adapter_factory, instance_ids,
              objective='coldchain', profile=None, seed=0, adapter_module_path=None,
              data_sha256=None, out_dir=None, runtime_budget=None):
    """串行批量运行（B1 协议阶段固定串行；服务器并行在 E1 阶段加 workers）。

    任一实例失败即抛错（no silent success，与项目运行控制同口径）。
    """
    records = []
    for i in instance_ids:
        rec = run_instance(dataset, capacity, num_vehicles, adapter_factory, i,
                           objective=objective, profile=profile, seed=seed,
                           adapter_module_path=adapter_module_path,
                           data_sha256=data_sha256,
                           runtime_budget=runtime_budget)
        records.append(rec)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            _atomic_write_json(os.path.join(out_dir, f'inst_{i}.json'), rec)

    summary = _summarize(records, objective)
    if out_dir:
        _atomic_write_json(os.path.join(out_dir, 'summary.json'), summary)
        _atomic_write_json(os.path.join(out_dir, 'manifest.json'), {
            'schema_version': bc.RECORD_SCHEMA_VERSION,
            'method_name': records[0]['method_name'],
            'n_instances': len(records),
            'code_hash': records[0]['code_hash'],
            'data_hash': records[0]['data_hash'],
            'profile_hash': records[0]['profile_hash'],
            'objective': objective,
            'n_failed': 0,
            'per_instance_runtime': [round(r['runtime_s'], 2) for r in records],
        })
    return records, summary


def _summarize(records, objective):
    cost_field = 'distance_cost' if objective == 'distance' else 'coldchain_cost'
    n = len(records)
    complete = sum(1 for r in records if bool(r['outcome']['complete']))
    service = sum(1 for r in records if service_ok(r['outcome'], objective)
                  and int(r['outcome'].get('terminal_unresolved', 0)) == 0
                  and int(r['outcome'].get('ownership_violations', 0)) == 0)
    costs = [float(r['outcome'][cost_field]) for r in records]
    stats = {k: int(sum(r['stats'][k] for r in records))
             for k in ('n_events', 'n_solver_calls', 'n_fast_path',
                       'n_unplanned_customer_events', 'n_penalty_warnings',
                       'fallback_triggered_events', 'n_waits',
                       'n_customer_anchor_waits', 'n_loaded_waits')}
    stats['native_complete_rate'] = float(complete / n) if n else None
    # fallback 率从实例记录实际计算（不写死）
    stats['fallback_instance_rate'] = (float(sum(
        1 for r in records if r['stats']['fallback_triggered_events'] > 0) / n)
        if n else None)
    stats['fallback_event_rate'] = (float(stats['fallback_triggered_events']
                                          / stats['n_events'])
                                    if stats['n_events'] else None)
    return {
        'objective': objective,
        'n': n,
        'n_complete': complete,
        'n_service_ok': service,
        'cost_mean': float(np.mean(costs)) if costs else None,
        'cost_per_instance': costs,
        'runtime_mean': float(np.mean([r['runtime_s'] for r in records])),
        'protocol_errors': [r['protocol']['error'] for r in records
                            if r['protocol']['error']],
        'stats': stats,
    }


def _atomic_write_json(path, obj):
    tmp = path + f'.tmp_{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
