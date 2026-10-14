#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**T 系列回归**：把 Role C 的 T1–T9 跑成一份结构化报告（**不看效用**）。

本模块**只做编排**：真正判定在 `role_c_decision_audit`（T2–T9 检查）、
`role_c_opportunity_cost`（T9 纯函数层）、`role_c_identity`（T1 身份闭包）、
`check_control_config`（T8 配对字段）里。这里负责：

  1. 在**合成夹具**上驱动真实 `StrictOnlineEnv` + 真实 `certify_plan` + 真实
     `RoleCReplanner`，产出**真实落盘的决策日志**（不是手编记录）；
  2. 跑基准臂与三个消融臂（`opportunity_cost` / `lookahead` / `scenarios`），
     统计**激活计数**；
  3. 把 T1–T9 的结论汇总成 `{'checks', 'problems', 'not_applicable', 'passed'}`。

⚠️ **两条不许违反的边界**：
  * **不看效用**：本模块产出的报告里**没有任何效用字段**，也不读 `J`；
    动作一致率/激活计数都是**行为**统计，不是效果。
  * **`not_applicable` ≠ FAIL**：若在夹具上 `raw > 0` 从未发生（预算不紧），
    T2/T3 必须报 `not_applicable`。这正是规格 §5（T2 条，审计 D3b 限定）要求的：
    "机制没接进"与"机制接好了但资源不稀缺"在只看动作序列时**不可区分**，
    所以要用激活门控，而不是把后者判成失败。

由此本模块提供一个**双夹具**用法：`slack`（预算宽松 ⇒ 预期 `not_applicable`）与
`tight`（预算吃紧 ⇒ 预期机制被真实激活并给出可判定的结论）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

import role_c_decision_audit as da
import role_c_forward as fwd
import role_c_harness as hn
import role_c_identity as ri
import role_c_opportunity_cost as oc

__all__ = ['TSeriesError', 'ABLATION_SWITCHES', 'EXT_DIR', 'ensure_project_paths',
           'run_fixture', 'activation_counts', 't2b_stats', 't2a_preset_fixtures',
           't8_shared_config_check', 'run_t_series']

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXT_DIR = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')

ABLATION_SWITCHES = tuple(oc.ABLATIONS)     # ('opportunity_cost', 'lookahead', 'scenarios')


class TSeriesError(ValueError):
    """T 系列编排输入不合法。fail-closed。"""


def ensure_project_paths():
    """把项目子目录插进 `sys.path`（与 `test_role_c_env_connectivity` 同一做法）。"""
    for sub in ('scripts/simulation', 'scripts/coldchain', 'scripts/evaluation',
                'scripts/data'):
        p = os.path.join(EXT_DIR, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# 在真实 env 上跑一个臂
# ---------------------------------------------------------------------------
def run_fixture(*, decide_fn, ds, contract, budget, capacity, num_vehicles,
                booking_horizon, cooling_share, inst_idx=0):
    """用**真实** `StrictOnlineEnv` + 真实 `certify_plan` 驱动一个臂，返回落盘日志与诊断。

    返回 `{'records', 'diags', 'served', 'n_served'}`。**不返回任何效用**。
    """
    ensure_project_paths()
    from strict_online_env import StrictOnlineEnv
    from run_exp_energy_c0 import certify_plan
    import role_c_replanner as rp
    from role_c_decision_log import DecisionLogger

    logger = DecisionLogger(float(budget))

    def certify_fn(plan, ctx):
        try:
            ok, energy = certify_plan(ctx['env'], ctx['inst_idx'], ctx['clock'],
                                      ctx['vehicles'], ctx['served_mask'], plan,
                                      ctx['contract'], ctx['budget_B'])
        except ValueError:
            return False, None          # 物理硬违反 ⇒ 按不可行处理（与既有实现同）
        return bool(ok), energy

    diags = []

    def _wrapped(order, ctx):
        acc, d = decide_fn(order, ctx)
        diags.append(d)
        return acc, d

    policy = rp.RoleCReplanner(budget=float(budget), capacity=float(capacity),
                               booking_horizon=float(booking_horizon), contract=contract,
                               cooling_share=float(cooling_share), decide_fn=_wrapped,
                               certify_fn=certify_fn, logger=logger)
    env = StrictOnlineEnv(ds, capacity=float(capacity), num_vehicles=int(num_vehicles),
                          replanner=policy, coldchain_contract=contract,
                          booking_horizon=float(booking_horizon))
    traces, served_mask = env.run(int(inst_idx))
    return {'records': [dict(r) for r in logger.records], 'diags': diags,
            'served': served_mask,
            'n_served': int(np.sum(np.asarray(served_mask)[1:]))}


# ---------------------------------------------------------------------------
# 激活计数（T2/T3/T4 的 activation 门控输入）
# ---------------------------------------------------------------------------
def activation_counts(diags):
    """从逐决策诊断统计**机制被真正触发**的次数（不是效果，只是"跑到没跑到"）。

    * `opportunity_cost` / `lookahead`：`lambda_raw > 0` 的决策数
      —— 规格 §5 T2 条（审计 D3b）明确要求用**未截断**的 `raw` 计数当激活量，
      因为截断后的 `λ_B` 为 0 也可能是"raw>0 但被 clip"或"raw 本来就 ≤0"；
    * `scenarios`：**前瞻真的被求值**的决策数（未被边界/插入短路掉）
      —— 场景机制只有被跑到才谈得上"接没接进去"。
    """
    n_raw = 0
    n_scen = 0
    for d in diags or []:
        ld = (d or {}).get('lambda_diag') or {}
        raw = ld.get('lambda_raw')
        try:
            if raw is not None and float(raw) > 0.0:
                n_raw += 1
        except (TypeError, ValueError):
            pass
        if (d or {}).get('forward') is not None:
            n_scen += 1
    return {'opportunity_cost': n_raw, 'lookahead': n_raw, 'scenarios': n_scen}


def t2b_stats(diags_main, actions_ablated=None):
    """T2b 的逐决策统计（**raw>0 与截断后的有效 `λ_B` 必须分开**）。

    `actions_ablated` = `opportunity_cost` 消融臂的动作序列（同序对齐）；给了才统计
    "排序/动作是否改变"。
    """
    n_raw = n_eff = n_score = 0
    acts = []
    for d in diags_main or []:
        d = d or {}
        ld = d.get('lambda_diag') or {}
        acts.append(d.get('log_fields', {}).get('action') or d.get('action'))
        raw = ld.get('lambda_raw')
        lam = ld.get('lambda_B')
        try:
            if raw is not None and float(raw) > 0.0:
                n_raw += 1
        except (TypeError, ValueError):
            pass
        try:
            if lam is not None and float(lam) > 0.0:
                n_eff += 1
            if lam is not None and d.get('e_o') is not None \
                    and abs(float(lam) * float(d['e_o'])) > 1e-12:
                n_score += 1
        except (TypeError, ValueError):
            pass
    n_act = 0
    if actions_ablated is not None:
        n_act = sum(1 for a, b in zip(acts, actions_ablated) if a != b)
    return {'n_decisions': len(diags_main or []), 'n_raw_positive': n_raw,
            'n_lambda_effective_positive': n_eff, 'n_score_changed': n_score,
            'n_ordering_or_action_changed': n_act}


def t2a_preset_fixtures():
    """**事前构造**的 T2a 压力夹具（不是真实开发日）。

    覆盖：正常正差分 / 零能耗 / 负差分（噪声⇒clip） / `B − e_o < 0`（越界） / 大 e_o。
    """
    return [{'budget': 100.0, 'e_o': 5.0, 'v_hi': 60.0, 'v_lo': 40.0},
            {'budget': 100.0, 'e_o': 0.0, 'v_hi': 60.0, 'v_lo': 40.0},
            {'budget': 100.0, 'e_o': 5.0, 'v_hi': 40.0, 'v_lo': 60.0},
            {'budget': 3.0, 'e_o': 5.0, 'v_hi': 60.0, 'v_lo': 40.0},
            {'budget': 100.0, 'e_o': 0.001, 'v_hi': 60.0, 'v_lo': 40.0}]


def _extract_const(name, path=None):
    """从 `check_control_config.py` **AST 抽取**顶层常量（不 import 该模块）。

    ⚠️ **为什么不 import**（2026-10-14 实测，真实缺陷）：`check_control_config.py`
    在**模块级**执行

        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', ...)

    于是**旧** `TextIOWrapper` 被丢弃，GC 时按所有权**关闭底层 buffer** ⇒
    此后**任何** `print` 都抛 `ValueError: I/O operation on closed file`。
    任何 import 该模块的进程都会中招（本模块第一版就是这样崩的）。
    本仓库已有同类先例（`test_role_c_controls` 用 AST 抽取上游 `_marginal_lambda`
    而不是 import），这里照做：**只读常量，不引入副作用**，也**不改动**该文件。
    """
    import ast
    p = path or os.path.join(ROOT, 'CC_Compare', 'tools', 'check_control_config.py')
    with open(p, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise TSeriesError('`check_control_config.py` 里找不到顶层常量 %r（fail-closed：'
                       '不得因为抽不到就改用一份手抄的清单）' % (name,))


# ---------------------------------------------------------------------------
# T8：共享配对字段
# ---------------------------------------------------------------------------
def t8_shared_config_check(*, config=None, frozen=None):
    """T8：Role C 与对照的**共享字段**必须逐字段相同（否则拒绝配对）。

    对照的是 `check_control_config.PAIR_FIELDS` 这张**已被审计过的**清单（不另立一份，
    也不手抄）—— 通过 AST 抽取它的值（见 `_extract_const` 里记录的 import 副作用）。
    `budget_B` / `cooling_share` / 合同 / 数据在冻结文件的 `shared_identity_anchor` 下，
    故先把两处拍平成一个视图再比。

    冻结文件里**缺**的字段（如 `arm_seed`）**不冒充相等**，而是列进 `gaps` 如实上报。
    """
    pair_fields = _extract_const('PAIR_FIELDS')
    # `arm_seed` 由 2026-10-14 用户裁定补录进冻结文件（见 `role_c_identity` 模块头），
    # 但 `check_control_config.PAIR_FIELDS`（他方已登记模块）**没有**它 ⇒ 本层**额外**
    # 强制它，并把它标为 `beyond_pair_fields`，供用户决定是否也加进那张清单。
    extra_fields = ('arm_seed',)
    f = frozen if frozen is not None else ri.load_frozen_control()
    cfg = config if config is not None else ri.role_c_solver_config(frozen=f)
    mine = dict(cfg['shared'])
    mine.update({k: v for k, v in cfg['shared_identity_anchor'].items()
                 if k not in ('seeds', 'main_penalty')})
    flat = dict(f['frozen_config'])
    flat.update({k: v for k, v in f['shared_identity_anchor'].items()
                 if k not in ('seeds', 'main_penalty')})
    rows, mismatches = [], []
    for name in tuple(pair_fields) + tuple(extra_fields):
        beyond = name not in pair_fields
        if name not in flat:
            rows.append({'field': name, 'role_c': mine.get(name), 'control': None,
                         'status': 'not_recorded_in_frozen_control',
                         'beyond_pair_fields': beyond})
            continue
        same = mine.get(name) == flat[name]
        rows.append({'field': name, 'role_c': mine.get(name), 'control': flat[name],
                     'status': 'ok' if same else 'MISMATCH',
                     'beyond_pair_fields': beyond})
        if not same:
            mismatches.append(name)
    gaps = [r['field'] for r in rows
            if r['status'] == 'not_recorded_in_frozen_control']
    return {'ok': not mismatches, 'rows': rows, 'mismatches': mismatches, 'gaps': gaps,
            'n_fields': len(pair_fields), 'extra_fields': list(extra_fields),
            'pair_fields_source': 'check_control_config.py (AST-extracted, not imported)'}


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def run_t_series(*, ds, contract, budget, capacity, num_vehicles, booking_horizon,
                 cooling_share, sampler, arm_seed, K=10, inner_gate=None,
                 time_limit=10.0, ls_kwargs=None, identity=True, inst_idx=0):
    """跑完整 T 系列回归，返回结构化报告（**无任何效用字段**）。

    基准臂 + 三个消融臂各跑一遍；T7 用**同参数重跑一次**基准臂得到重复日志。
    """
    ensure_project_paths()
    gate = inner_gate or fwd.INNER_GATE_PRIMARY

    def mk(**over):
        kw = dict(sampler=sampler, arm_seed=arm_seed, K=K, contract=contract,
                  cooling_share=cooling_share, booking_horizon=booking_horizon,
                  budget=budget, inner_gate=gate, ls_kwargs=ls_kwargs,
                  time_limit=time_limit)
        kw.update(over)
        return hn.build_decide_fn(**kw)

    common = dict(ds=ds, contract=contract, budget=budget, capacity=capacity,
                  num_vehicles=num_vehicles, booking_horizon=booking_horizon,
                  cooling_share=cooling_share, inst_idx=inst_idx)
    base = run_fixture(decide_fn=mk(), **common)
    repeat = run_fixture(decide_fn=mk(), **common)                 # T7
    abl = {
        'opportunity_cost': run_fixture(decide_fn=mk(ablate_opportunity_cost=True),
                                        **common),
        'lookahead': run_fixture(decide_fn=mk(ablate_lookahead=True), **common),
        'scenarios_k1': run_fixture(decide_fn=mk(ablate_scenarios='k1'), **common),
        'scenarios_shuffle': run_fixture(decide_fn=mk(ablate_scenarios='shuffle'),
                                         **common),
    }
    act = activation_counts(base['diags'])
    # T4 的 activation：两个场景臂合并统计（任一被跑到即算激活）
    act['scenarios'] = max(act['scenarios'],
                           activation_counts(abl['scenarios_k1']['diags'])['scenarios'],
                           activation_counts(abl['scenarios_shuffle']['diags'])['scenarios'])
    t2b = t2b_stats(base['diags'],
                    actions_ablated=[(d or {}).get('log_fields', {}).get('action')
                                     for d in abl['opportunity_cost']['diags']])
    audit = da.audit(
        base['records'], log_repeat=repeat['records'],
        ablations={'opportunity_cost': (base['records'],
                                        abl['opportunity_cost']['records']),
                   'lookahead': (base['records'], abl['lookahead']['records']),
                   'scenarios': (base['records'], abl['scenarios_k1']['records'])},
        ablation_activation=act, t2a_fixtures=t2a_preset_fixtures(), t2b_stats=t2b)
    report = {'inner_gate': gate, 'n_records': len(base['records']),
              'n_served': base['n_served'], 'activation': act, 't2b_stats': t2b,
              'audit': audit, 't8': t8_shared_config_check(),
              'runs': {'base': len(base['records']),
                       **{k: len(v['records']) for k, v in abl.items()}}}
    if identity:
        cfg = ri.role_c_solver_config(inner_gate=gate, arm_seed=arm_seed, K=K,
                                      time_limit=time_limit, ls_kwargs=ls_kwargs)
        report['t1'] = {'config_sha256': __import__('nonlearning_identity')
                        .solver_config_sha256(cfg)}
    checks = dict(audit['checks'])
    checks['T1_identity_closure'] = None if not identity else True
    checks['T8_pairing_fields'] = report['t8']['ok']
    report['checks'] = checks
    report['problems'] = list(audit['problems']) + (
        [] if report['t8']['ok'] else ['T8 共享字段不符：%s' % (report['t8']['mismatches'],)])
    report['not_applicable'] = list(audit['not_applicable'])
    report['passed'] = all(v is not False for v in checks.values()) \
        and not report['problems']
    report['findings'] = _audit_findings(audit)
    return report


def _audit_findings(audit):
    """把审计结果里**与它自己文档相矛盾**的地方如实挑出来（不改对方模块）。

    ⚠️ **2026-10-14 发现的缺陷（属 `role_c_decision_audit.check_ablation_engagement`，
    本模块只报告、不修改）**：该函数在"完全相同 + `activation<=0`"时把
    `r['engaged']` 置 `None`（意图 = 降级为 `not_applicable`、**不判 FAIL**，见其
    docstring 与 `audit()` 的 "结构性不适用被当成失败" 警告），但紧接着的条件是
    `if r['engaged'] is not True: problems.append(...)` —— `None` **也**命中
    ⇒ 降级被自己的 `problems` 抵消，`t234_ok = not problems` 变 `False`，
    `audit['passed']` 随之变 `False`。**规格 §5 T2 条要求"报 not_applicable，不报 FAIL"**，
    故这是实现与其文档/规格不一致，而不是"机制没接进去"。
    （它自己的测试只断言 `rows[0]['engaged'] is None`，不断言 `ok`/`problems`，故漏掉了。）

    本函数**不替它下结论**：只把"行级判定为 not_applicable、却出现在 `problems` 里"
    的开关点名列出，供用户裁定由谁修。
    """
    out = []
    downgraded = [r.get('switch') for r in (audit.get('ablation_rows') or [])
                  if r.get('engaged') is None]
    if downgraded:
        out.append({
            'finding': 'ablation_downgraded_but_listed_in_problems',
            'switches': sorted(x for x in downgraded if x),
            'where': 'role_c_decision_audit.check_ablation_engagement (L277-278)',
            'detail': "`engaged` 已置 None（= not_applicable），但 `if r['engaged'] is "
                      "not True: problems.append(...)` 仍命中 ⇒ 降级被抵消、"
                      "`t234_ok`/`audit['passed']` 变 False。规格要求此处**报 "
                      "not_applicable 而不报 FAIL**。",
            'suggested_fix': "把 `is not True` 改为 `is False`（其自身测试 "
                             "case (a)(b) 只断言行字段、不断言 ok/problems，故不受影响）",
            'not_fixed_by': 'role_c_t_series（不改他方已登记 SHA 的模块）'})
    return out
