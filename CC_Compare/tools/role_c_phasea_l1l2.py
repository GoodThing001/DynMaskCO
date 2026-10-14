#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 的 **L1/L2 归因测量驱动器**（预登记范围；不挑参数、不改协议）。

跑什么
------
在**基线捕获的那 8 个固定决策点**（digest `5b432f28…`）上，用
`role_c_phasea_instrument` 的叶子插桩，产出：

* **A 类**：`time_limit = ∞`（**关闭内部截止**）+ **外层**每决策点 `600 s` 硬杀
  + 整个 Phase A 作业 `12 000 s` 硬上限。到点由**测量进程之外**的父进程 `SIGKILL`
  并回收，再**从进度原件重放**截断位置（REV2 §5.1.2：不得只写"超时"）。
  若任一点被判 `inconclusive_censored` ⇒ **不进入** §6 瓶颈判定。
* **B 类**：保留冻结的 `10 s` 内部截止，逐决策点记录
  `action` / `boundary` / `forward.deadline_hit` / `lambda_B` / `e_o` /
  `forward_entered` / `evaluation_status`（三态互斥）/ 场景进入与完成数 /
  `stages_entered` / `timeout_reason`。
* **L1**：调用次数与候选数 —— **只作解释证据**，**不参与**主导判定。

⚠️ A 类与 B 类**分开报告，绝不合并成一个时间占比**（REV2 §5.2 末）。

为什么 A 类要 fork
------------------
"外层硬上限"的语义是：**杀手在被测进程之外**。父进程 `fork` 出子进程跑一次决策，
自己在 `600 s` 后 `SIGKILL` —— 这满足"已登记上限 = 已执行硬上限"，且**不是**
在测量路径里塞一个内部截止（那会改变被测语义）。管线只构建一次，
子进程走 COW，不必为每个决策点重付 ~650 s 的固定开销。

预声明的读法（**跑之前写死**，不是事后挑选）
------------------------------------------
1. `i* = argmax_i median_d share_i(d)`，`i` 取 §4.2 的 11 个叶子 **加上**
   `P9_measured`（未归因也是一种份额，且 §6.3 专门为它留了一档）。
2. 并列最大 ⇒ 取**叶子索引最小**者（§6.2）；`P9_measured` 不在叶子序列里，
   **仅在并列集合里没有任何叶子时**才被选中（即"有叶子并列就先记叶子"）。
3. 达到门槛还要求：`median_d share_{i*} ≥ 0.5` **且** `Σ_d 1[i* = argmax] ≥ 6/8`。
4. 任一决策点 `inconclusive_censored` / `inconclusive_accounting_error`
   ⇒ **不出瓶颈结论**，按对应 `inconclusive_*` 上报并停止。

不做什么
--------
* **不**改任何被测模块的字节；**不**改 `K`/截止/核集/邻域/`workers`；
* **不**据耗时挑参数；**不**跑 Phase B、烟测、40 天批次、最终测试；
* **不**自动启动 L3。
"""
from __future__ import annotations

import io
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_phasea_ceiling as pc                                # noqa: E402
import role_c_phasea_instrument as pin                            # noqa: E402
import role_c_phasea_partition as pp                              # noqa: E402
import role_c_timing as rt                                        # noqa: E402

__all__ = ['build_bundle', 'run_stage_a', 'run_stage_b', 'aggregate', 'main']

#: A 类：关闭内部截止（JSON 里写字符串，不写 inf）
DISABLED = 'disabled'

#: §6.3 所需的稳定性下限（`Σ_d 1[i* = argmax_i share_i(d)] ≥ 该值`；`M=8`）
STABILITY_MIN = 6

_PHASE_START = None


def _now():
    return time.monotonic()


def elapsed_total():
    return 0.0 if _PHASE_START is None else _now() - _PHASE_START


def _write_json(path, obj):
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    return p


def _read_json(path):
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


# --------------------------------------------------------------------------- 构建
def build_bundle(*, train_seed=20260925, gate_seed=20260926, n_orders=200,
                 n_vehicles=15, rho=0.6, capture_n=8, arm_seed=7002, K=10,
                 inner_gate='on', inst_idx=0, dataset_cache=None,
                 expect_frozen_anchors=True):
    """按**冻结管线**搭出一次管线 + 8 个固定决策点（与 `role_c_timing` 同一顺序）。

    返回 `{contract, sampler, budget, cooling_share, contexts, digest, ...}`。
    **只构建与捕获，不产生任何耗时结论。**
    """
    import role_c_harness as hn
    import role_c_t_series as ts
    ts.ensure_project_paths()
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from run_exp_reserve import BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH
    from run_exp_reserve import generate_dataset
    from scenario_saa import (CondHistoricalSampler, build_history,
                              build_history_times_classes, make_c0_contract_v2)

    contract = make_c0_contract_v2()
    train_ds, ds_source = rt._load_or_build_train_ds(n_orders, train_seed,
                                                     dataset_cache)
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, share, _ = compute_budget_and_dwell(train_ds, contract, int(n_vehicles),
                                          int(train_seed), float(rho), hist_tc, k=10)
    anchor = {'checked': False}
    if expect_frozen_anchors:
        from role_c_identity import frozen_anchors
        a = frozen_anchors()
        fb, fs = float(a['budget_B']), float(a['cooling_share'])
        rel_b = abs(float(B) - fb) / max(1e-12, abs(fb))
        abs_s = abs(float(share) - fs)
        if rel_b > rt.ANCHOR_REL_TOL or abs_s > rt.ANCHOR_ABS_TOL_SHARE:
            raise RuntimeError(
                '预算/冷却份额未复现冻结锚点 ⇒ 拒绝测量：B=%.13f（冻结 %.13f，相对差 '
                '%.3e）、cooling_share=%.13f（冻结 %.13f，绝对差 %.3e）'
                % (float(B), fb, rel_b, float(share), fs, abs_s))
        anchor = {'checked': True, 'frozen_B': fb, 'witnessed_B': float(B),
                  'rel_gap_B': rel_b, 'frozen_cooling_share': fs,
                  'witnessed_cooling_share': float(share),
                  'abs_gap_cooling_share': abs_s,
                  'note': '相对容差只用于判"同一规模"；实测差值如实报出'}
    gate_ds = add_v2_initial_quality(generate_dataset(1, int(n_orders), int(gate_seed)),
                                    contract)
    sampler = CondHistoricalSampler(hist, n_neighbors=10)
    dfn = hn.build_decide_fn(sampler=sampler, arm_seed=int(arm_seed), K=int(K),
                             contract=contract, cooling_share=float(share),
                             booking_horizon=BOOKING_HORIZON, budget=float(B),
                             inner_gate=inner_gate, time_limit=10.0,
                             certify_plan_fn=pin.certify_proxy)
    run = ts.run_fixture(decide_fn=dfn, ds=gate_ds, contract=contract, budget=float(B),
                         capacity=50.0, num_vehicles=int(n_vehicles),
                         booking_horizon=BOOKING_HORIZON, cooling_share=float(share),
                         inst_idx=int(inst_idx), capture_n=int(capture_n),
                         tw_speed=SPEED_KMH / KM_PER_UNIT)
    contexts = run.get('contexts') or []
    digest = _digest_of(contexts)
    return {'contract': contract, 'sampler': sampler, 'budget': float(B),
            'cooling_share': float(share), 'booking_horizon': BOOKING_HORIZON,
            'contexts': contexts, 'digest': digest, 'dataset_source': ds_source,
            'anchor_check': anchor, 'n_contexts': len(contexts),
            'n_records': len(run['records']),
            'frozen': {'train_seed': int(train_seed), 'gate_seed': int(gate_seed),
                       'n_orders': int(n_orders), 'n_vehicles': int(n_vehicles),
                       'rho': float(rho), 'capture_n': int(capture_n),
                       'arm_seed': int(arm_seed), 'K': int(K),
                       'inner_gate': inner_gate, 'inst_idx': int(inst_idx)}}


def _digest_of(contexts):
    """与 `role_c_timing.DecisionWorkFn.digest()` **同一算法**（保证可比）。"""
    import hashlib
    import role_c_t_series as ts
    rows = []
    for c in contexts:
        plan = c.get('plan') or {}
        rows.append(json.dumps({
            'inst_idx': c.get('inst_idx'), 'clock': round(float(c.get('clock') or 0), 9),
            'order': c.get('order'),
            'plan': {str(k): list(v) for k, v in sorted(plan.items())},
            'vehicles': [list(x) for x in
                         ts.vehicle_fingerprint(c.get('vehicles') or ())],
        }, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(('\n'.join(sorted(rows))).encode('utf-8')).hexdigest()


def _make_decide(bundle, *, internal_deadline):
    """按给定内部截止造 `decide_fn`（`DISABLED` ⇒ 关闭内部截止）。"""
    import role_c_harness as hn
    lim = float('inf') if internal_deadline == DISABLED else float(internal_deadline)
    return hn.build_decide_fn(sampler=bundle['sampler'],
                              arm_seed=bundle['frozen']['arm_seed'],
                              K=bundle['frozen']['K'], contract=bundle['contract'],
                              cooling_share=bundle['cooling_share'],
                              booking_horizon=bundle['booking_horizon'],
                              budget=bundle['budget'],
                              inner_gate=bundle['frozen']['inner_gate'],
                              time_limit=lim, certify_plan_fn=pin.certify_proxy)


# ------------------------------------------------------------------------ A 类
def _a_child(bundle, k, decide_fn, result_path, progress_path):
    """**子进程体**：跑一个决策点的 A 类观测，把结果写盘后 `_exit`。

    只做这一件事 ⇒ 被杀时父进程仍能从进度原件恢复截断位置。
    """
    try:
        ctx = bundle['contexts'][k]
        order = ctx.get('order')
        inst = pin.LeafInstrumentation(progress_path=progress_path,
                                       progress_every=200)
        with inst:
            inst.begin(subject={'decision_point': k})
            t0 = time.perf_counter()
            try:
                accept, diag = decide_fn(order, ctx)
                err = None
            except BaseException as exc:                          # noqa: BLE001
                accept, diag, err = None, None, '%s: %s' % (type(exc).__name__, exc)
            t1 = time.perf_counter()
            rec = inst.end()
        rec['wall_s'] = t1 - t0
        rec['decision_point'] = k
        rec['accept'] = None if accept is None else bool(accept)
        rec['error'] = err
        d = diag or {}
        rec['diag'] = _trim_diag(d)
        _write_json(result_path, rec)
        os._exit(0 if err is None else 3)
    except BaseException as exc:                                  # noqa: BLE001
        try:
            _write_json(result_path, {'decision_point': k, 'fatal':
                                      '%s: %s' % (type(exc).__name__, exc)})
        except Exception:                                         # noqa: BLE001
            pass
        os._exit(4)


_DIAG_KEYS = ('boundary', 'boundary_reason', 'reject_reason', 'e_o', 'lambda_B',
              'opportunity_cost', 'net_after_opportunity_cost', 'myopic_net',
              'certify_ok', 'certify_energy', 'formulation', 'inner_gate',
              'lambda_estimator', 'myopic_base', 'budget_B', 'elapsed_s')


def _trim_diag(d):
    """只留**判决诊断**字段（不含任何效用字段；效用在计时报告里本来就被禁）。"""
    out = {k: d.get(k) for k in _DIAG_KEYS if k in d}
    ld = d.get('lambda_diag') or {}
    out['lambda_diag'] = {k: ld.get(k) for k in
                          ('lambda_source', 'undefined', 'lambda_B', 'e_o')
                          if k in ld}
    fw = d.get('forward')
    if isinstance(fw, dict):
        out['forward'] = {k: fw.get(k) for k in
                          ('deadline_hit', 'n_scenarios', 'n_gate_reject',
                           'n_insert_reject', 'n_future_accept',
                           'n_infeasible_scenario', 'inner_gate',
                           'scenario_digest', 'value_is_nan')
                          if k in fw}
        v = fw.get('value')
        out['forward']['value_is_nan'] = (v is None) or (v != v)
    sc = d.get('scenario')
    if isinstance(sc, dict):
        out['scenario'] = {k: sc.get(k) for k in ('info_max_reveal', 'K')
                           if k in sc}
    lf = d.get('log_fields') or {}
    out['log_fields'] = {k: lf.get(k) for k in ('order_id', 'action', 'certified')
                         if k in lf}
    return out


def run_stage_a(bundle, *, out_dir, per_decision_cap_s, total_cap_s,
                fork=True, decide_fn=None):
    """A 类：每个决策点一个**子进程**，外层 `600 s` 硬杀；恢复截断位置。"""
    decide_fn = decide_fn or _make_decide(bundle, internal_deadline=DISABLED)
    os.makedirs(out_dir, exist_ok=True)
    per = []
    total_exceeded = False
    for k in range(bundle['n_contexts']):
        if elapsed_total() >= total_cap_s:
            total_exceeded = True
            break
        remain = total_cap_s - elapsed_total()
        cap = min(float(per_decision_cap_s), max(1.0, remain))
        rp = os.path.join(out_dir, 'a_dp%02d_result.json' % k)
        ppath = os.path.join(out_dir, 'a_dp%02d_progress.jsonl' % k)
        for p in (rp, ppath):
            if os.path.exists(p):
                os.remove(p)
        if not fork:
            per.append(_run_a_inprocess(bundle, k, decide_fn, rp, ppath, cap))
            continue
        per.append(_run_a_forked(bundle, k, decide_fn, rp, ppath, cap))
    return {'class': 'A', 'internal_deadline': DISABLED,
            'per_decision_cap_s': float(per_decision_cap_s),
            'total_cap_s': float(total_cap_s),
            'enforcement': ('subprocess_sigkill_by_parent' if fork
                            else 'in_process_no_hard_kill_FAKE_CLASS_A'),
            'hard_kill_enforced': bool(fork),
            'total_exceeded': bool(total_exceeded),
            'n_decision_points': bundle['n_contexts'],
            'n_completed': sum(1 for r in per if r.get('status') == 'completed'),
            'n_censored': sum(1 for r in per if r.get('status') == 'censored'),
            'per_decision': per}


def _run_a_forked(bundle, k, decide_fn, rp, ppath, cap):
    t0 = _now()
    pid = os.fork()
    if pid == 0:                                   # 子进程
        _a_child(bundle, k, decide_fn, rp, ppath)
        os._exit(5)                                # 不可达
    killed = False
    deadline = t0 + cap
    while True:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            break
        if _now() > deadline:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            os.waitpid(pid, 0)
            killed = True
            break
        time.sleep(0.05)
    wall = _now() - t0
    rec = _collect_a(k, rp, ppath, killed=killed, wall_s=wall, cap_s=cap)
    return rec


def _run_a_inprocess(bundle, k, decide_fn, rp, ppath, cap):
    """**仅本机冒烟用**：没有外层硬杀 ⇒ 明确标注不是合格 A 类。"""
    t0 = _now()
    try:
        _a_child_inline(bundle, k, decide_fn, rp, ppath)
    except BaseException as exc:                                  # noqa: BLE001
        pass
    wall = _now() - t0
    rec = _collect_a(k, rp, ppath, killed=False, wall_s=wall, cap_s=cap)
    rec['enforcement_note'] = ('**本机无 fork**：没有外层硬杀，'
                               '本记录**不是**合格的 A 类观测（不得用于判定）')
    rec['hard_kill_enforced'] = False
    return rec


def _a_child_inline(bundle, k, decide_fn, rp, ppath):
    ctx = bundle['contexts'][k]
    order = ctx.get('order')
    inst = pin.LeafInstrumentation(progress_path=ppath, progress_every=200)
    with inst:
        inst.begin(subject={'decision_point': k})
        t0 = time.perf_counter()
        accept, diag, err = None, None, None
        try:
            accept, diag = decide_fn(order, ctx)
        except BaseException as exc:                              # noqa: BLE001
            err = '%s: %s' % (type(exc).__name__, exc)
        t1 = time.perf_counter()
        rec = inst.end()
    rec.update({'wall_s': t1 - t0, 'decision_point': k,
                'accept': None if accept is None else bool(accept), 'error': err,
                'diag': _trim_diag(diag or {})})
    _write_json(rp, rec)


def _collect_a(k, rp, ppath, *, killed, wall_s, cap_s):
    """把子进程的产物 + 进度原件合成一条 A 类记录（硬杀时给出**截断位置**）。"""
    rec = {'decision_point': k, 'wall_s': wall_s, 'cap_s': cap_s}
    if os.path.isfile(rp):
        try:
            rec.update(_read_json(rp))
        except Exception as exc:                                  # noqa: BLE001
            rec['result_unreadable'] = str(exc)
    marks = []
    if os.path.isfile(ppath):
        try:
            with io.open(ppath, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        marks.append(json.loads(line))
        except Exception as exc:                                  # noqa: BLE001
            rec['progress_unreadable'] = str(exc)
    if killed:
        rep = pin.active_leaf_from_progress(marks)
        rec['status'] = 'censored'
        rec['inconclusive'] = 'inconclusive_censored'
        rec['leaf_at_truncation'] = rep['leaf_at_truncation']
        rec['active_stack_tail'] = rep['active_stack_tail']
        rec['scenario_index_at_truncation'] = rep['ordered_at_truncation']
        rec['n_scenarios_completed_at_truncation'] = rep['n_scenarios_completed']
        rec['best_insert_calls_at_truncation'] = \
            rep['best_insert_calls_at_truncation']
        rec['elapsed_s_at_truncation'] = min(wall_s, cap_s)
        rec['truncation_note'] = ('外层监督器 `SIGKILL` 并回收；'
                                  '**不是**内部截止触发（A 类内部截止 = disabled）')
        rec.pop('verdict', None)
        rec.pop('shares', None)
    else:
        rec['status'] = 'completed' if rec.get('verdict') else 'incomplete'
    return rec


# ------------------------------------------------------------------------ B 类
def run_stage_b(bundle, *, decide_fn=None, per_decision_cap_s=10.0):
    """B 类：保留 `10 s` 内部截止，逐决策点记录三态诊断（**与 A 类分开报告**）。"""
    decide_fn = decide_fn or _make_decide(bundle, internal_deadline=10.0)
    per = []
    for k in range(bundle['n_contexts']):
        ctx = bundle['contexts'][k]
        order = ctx.get('order')
        inst = pin.LeafInstrumentation()
        timed_out = False
        out = None
        with inst:
            inst.begin(subject={'decision_point': k})
            try:
                res = rt._run_with_limit(lambda: decide_fn(order, ctx),
                                         float(per_decision_cap_s))
            except rt.DecisionTimeout:
                res = (None, True, float(per_decision_cap_s), True)
            out, timed_out, wall, hard = res
            rec = inst.end()
        accept, diag = (out if isinstance(out, tuple) else (None, None))
        d = _trim_diag(diag or {})
        stages = rec.get('stages_entered') or []
        fw = d.get('forward') or {}
        counts = inst.timer._counts
        forward_entered = int(counts.get('P7d', 0)) > 0
        if not forward_entered:
            status = 'not_entered'
        elif timed_out or bool(fw.get('deadline_hit')):
            status = 'timed_out'
        else:
            status = 'completed'
        per.append({
            'decision_point': k,
            'action': (d.get('log_fields') or {}).get('action'),
            'boundary': d.get('boundary'),
            'reject_reason': d.get('reject_reason'),
            'forward_entered': forward_entered,
            'evaluation_status': status,
            'deadline_hit': bool(fw.get('deadline_hit')),
            'timeout_reason': ('sigalrm' if timed_out else
                               ('forward_deadline' if fw.get('deadline_hit')
                                else None)),
            'n_scenarios_entered': rec['l1'].get('role_c_forward._ordered'),
            'n_scenarios_completed': rec['l1'].get(
                'complete_fn_calls_in_forward'),
            'stages_entered': stages,
            'lambda_B': d.get('lambda_B'),
            'e_o': d.get('e_o'),
            'certify_ok': d.get('certify_ok'),
            'wall_s': wall,
            'per_decision_cap_s': float(per_decision_cap_s),
            'exceeded_cap': bool(wall is not None
                                 and wall > float(per_decision_cap_s)),
            'partition': {'verdict': rec['verdict'],
                          'failed_conditions': rec['failed_conditions'],
                          'tolerance_s': rec['tolerance_s'],
                          't_decision': rec['t_decision'],
                          'sum_exclusive_online': rec['sum_exclusive_online'],
                          'unattributed_online': rec['unattributed_online'],
                          'sum_exclusive_offline': rec['sum_exclusive_offline'],
                          'unattributed_offline': rec['unattributed_offline'],
                          'p9_measured': rec['p9_measured'],
                          'p9_residual': rec['p9_residual'],
                          'shares': rec['shares'],
                          'exclusive': rec['offline']['exclusive'],
                          # inclusive 也记（§4.1 两套都记），但**不得**用于份额
                          'inclusive': rec.get('inclusive') or {},
                          'inclusive_usable_for_shares': False},
            'l1': rec['l1'],
        })
    return {'class': 'B', 'internal_deadline': 10.0,
            'per_decision_cap_s': float(per_decision_cap_s),
            'enforcement': 'hard_sigalrm', 'n_decision_points': bundle['n_contexts'],
            'n_timed_out': sum(1 for r in per if r['evaluation_status'] == 'timed_out'),
            'n_not_entered': sum(1 for r in per
                                 if r['evaluation_status'] == 'not_entered'),
            'n_completed': sum(1 for r in per
                               if r['evaluation_status'] == 'completed'),
            'per_decision': per}


# ---------------------------------------------------------------------- 判定
def _i_star(shares_per_dp):
    """§6.2：`i* = argmax_i median_d share_i(d)`；并列取**最小叶子索引**。"""
    import statistics
    labels = list(pp.LEAF_ORDER) + [pp.UNATTRIBUTED]
    med = {}
    for lab in labels:
        vals = [float(s.get(lab, 0.0)) for s in shares_per_dp]
        med[lab] = statistics.median(vals) if vals else 0.0
    top = max(med.values()) if med else 0.0
    tied = [l for l in labels if abs(med[l] - top) <= 1e-12]
    leaves = [l for l in tied if l in pp.LEAF_ORDER]
    pick = pp.tie_break_leaf(leaves) if leaves else (tied[0] if tied else None)
    return pick, med, tied


def aggregate(a_report, b_report, *, digest_registered):
    """出一份**分开报告**的归因判定（A 类定份额；B 类给在线诊断；L1 只解释）。"""
    out = {'label': 'P1A-RoleC-PhaseA-attribution',
           'digest_registered': digest_registered,
           'a_class': {k: v for k, v in a_report.items() if k != 'per_decision'},
           'b_class': {k: v for k, v in b_report.items() if k != 'per_decision'},
           'decision_points': []}
    for r in a_report['per_decision']:
        out['decision_points'].append({
            'decision_point': r['decision_point'], 'status': r.get('status'),
            'wall_s': r.get('wall_s'),
            'leaf_at_truncation': r.get('leaf_at_truncation'),
            'scenario_index_at_truncation': r.get('scenario_index_at_truncation'),
            'n_scenarios_completed_at_truncation':
                r.get('n_scenarios_completed_at_truncation'),
            'verdict': r.get('verdict'),
            'failed_conditions': r.get('failed_conditions'),
            't_decision': r.get('t_decision'),
            'tolerance_s': r.get('tolerance_s'),
            'p9_measured': r.get('p9_measured'),
            'p9_residual': r.get('p9_residual'),
            'shares': r.get('shares'),
            'stages_entered': r.get('stages_entered'),
            'l1': r.get('l1'),
        })
    reasons = []
    a_per = a_report['per_decision']
    completed = [r for r in a_per if r.get('status') == 'completed']
    if any(r.get('status') == 'censored' for r in a_per):
        reasons.append('存在 inconclusive_censored ⇒ **不进入瓶颈判定**（§5.1.2）')
    if any(r.get('verdict') == 'inconclusive_accounting_error' for r in a_per):
        reasons.append('存在 inconclusive_accounting_error ⇒ 不进入瓶颈判定')
    if a_report.get('total_exceeded'):
        reasons.append('整个 Phase A 作业触及 12 000 s 上限 ⇒ 未跑完全部决策点')
    if not completed:
        out['conclusion'] = ('inconclusive_censored'
                             if any(r.get('status') == 'censored' for r in a_per)
                             else 'inconclusive_accounting_error')
        out['conclusion_reasons'] = reasons or ['无完成的 A 类决策点']
        out['bottleneck'] = None
        return out
    shares_per_dp = [r['shares'] for r in completed]
    pick, med, tied = _i_star(shares_per_dp)
    hits = sum(1 for s in shares_per_dp
               if _argmax_label(s) == pick)
    out['criterion'] = {
        'labels': list(pp.LEAF_ORDER) + [pp.UNATTRIBUTED],
        'median_share': med, 'i_star': pick, 'tied_max': tied,
        'median_share_of_i_star': med.get(pick),
        'stability_hits': hits, 'stability_required': STABILITY_MIN,
        'n_completed_decision_points': len(completed),
        'tie_break_rule': ('并列最大取最小叶子索引；`P9_measured` 不在叶子序列里，'
                           '仅当并列集合里没有叶子时才被选中'),
        'note': ('分母唯一 = 各决策点自己的 `T_decision`；只用**互斥 exclusive** 份额；'
                 '轮转重复**不是**独立状态'),
    }
    if reasons:
        out['conclusion'] = ('inconclusive_censored'
                             if any(r.get('status') == 'censored' for r in a_per)
                             else 'inconclusive_accounting_error')
        out['conclusion_reasons'] = reasons
        out['bottleneck'] = None
        return out
    if pick == pp.UNATTRIBUTED and med.get(pick, 0.0) >= 0.5:
        out['conclusion'] = 'inconclusive_attribution_incomplete'
        out['conclusion_reasons'] = [
            '`P9_measured`（未归因）为最大项且中位数份额 ≥ 0.5 ⇒ 归因不完整：'
            '漏测吞掉大头，**不等于**没有瓶颈']
        out['bottleneck'] = None
        return out
    if med.get(pick, 0.0) >= 0.5 and hits >= STABILITY_MIN:
        structural = pick in ('P7a', 'P7b', 'P7d')
        out['conclusion'] = ('dominant_in_structural_group_slack_undetermined'
                             if structural else 'dominant_bottleneck_identified')
        out['bottleneck'] = pick
        out['conclusion_reasons'] = [
            ('`%s` 中位数份额 ≥ 0.5 且稳定性 %d/%d' % (pick, hits, len(completed))),
            ('该叶子属**结构性乘数组**（§6.5）⇒ 记为**实现级优化余量尚未确定**，'
             '**不声称**其内部计算不可优化' if structural else
             '该叶子不属结构性乘数组')]
        if structural:
            out['ratchet_wording'] = (
                'Phase A 未发现符合预声明门槛、且有明确语义等价优化路径的单一主导瓶颈。'
                '本轮单杠杆优化路线按资源门收束；不据此推断所有可能的计算优化均不可行。')
        return out
    out['conclusion'] = 'below_threshold'
    out['bottleneck'] = None
    out['conclusion_reasons'] = [
        ('`%s` 中位数份额 = %.4f < 0.5' % (pick, med.get(pick, 0.0)))
        if med.get(pick, 0.0) < 0.5 else
        ('稳定性 %d/%d < %d' % (hits, len(completed), STABILITY_MIN))]
    out['fork_1_wording'] = (
        'Phase A 未发现符合预声明门槛、且有明确语义等价优化路径的单一主导瓶颈。'
        '本轮单杠杆优化路线按资源门收束；不据此推断所有可能的计算优化均不可行。')
    return out


def _argmax_label(shares):
    labels = list(pp.LEAF_ORDER) + [pp.UNATTRIBUTED]
    top = max(float(shares.get(l, 0.0)) for l in labels)
    tied = [l for l in labels if abs(float(shares.get(l, 0.0)) - top) <= 1e-12]
    leaves = [l for l in tied if l in pp.LEAF_ORDER]
    return pp.tie_break_leaf(leaves) if leaves else (tied[0] if tied else None)


# ------------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse
    global _PHASE_START
    ap = argparse.ArgumentParser(description='P1-A Role C Phase A L1/L2 归因测量')
    ap.add_argument('--out', required=True)
    ap.add_argument('--out-dir', required=True, help='逐决策点原始产物目录')
    ap.add_argument('--stage', default='A', choices=['A', 'B'])
    ap.add_argument('--per-decision-cap', type=float, default=None)
    ap.add_argument('--total-cap', type=float, default=None)
    ap.add_argument('--dataset-cache', default='/tmp/rolec_train_ds.npz')
    ap.add_argument('--no-fork', action='store_true',
                    help='**仅本机冒烟**：没有外层硬杀，产物标注为不合格 A 类')
    a = ap.parse_args(argv)
    ceiling = pc.load_phasea_ceiling()
    cls = ceiling['classes'][a.stage]
    cap = float(a.per_decision_cap if a.per_decision_cap is not None
                else cls['per_decision_cap_s'])
    total = float(a.total_cap if a.total_cap is not None
                  else ceiling['phase_total_cap_s'])
    _PHASE_START = _now()
    bundle = build_bundle(dataset_cache=a.dataset_cache)
    if bundle['digest'] != ceiling['decision_points']['digest']:
        raise SystemExit('决策点 digest 不符：实测 %s != 登记 %s ⇒ 拒绝测量'
                         % (bundle['digest'], ceiling['decision_points']['digest']))
    if a.stage == 'A':
        rep = run_stage_a(bundle, out_dir=a.out_dir, per_decision_cap_s=cap,
                          total_cap_s=total - (elapsed_total()),
                          fork=not a.no_fork)
        rep['obligations'] = {
            'internal_deadline': DISABLED,
            'external_per_decision_cap_s': cap,
            'phase_total_cap_s': total,
            'note': ('A 类关闭内部截止；`600 s` 是**外层**硬上限；'
                     '截断位置必须给出（不得只写超时）'),
        }
    else:
        rep = run_stage_b(bundle, per_decision_cap_s=cap)
    rep['bundle'] = {k: v for k, v in bundle.items() if k != 'contexts'}
    rep['phase_elapsed_s'] = elapsed_total()
    p = _write_json(a.out, rep)
    print('WROTE: %s' % p)
    print(json.dumps({k: rep[k] for k in rep
                      if k in ('class', 'n_completed', 'n_censored',
                               'n_timed_out', 'n_not_entered', 'enforcement',
                               'total_exceeded', 'phase_elapsed_s')},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
