#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 决策记录（decision log）的**只读审计**：T5 / T6 / T7 不变式。

对应 `CC_Compare/P1A_RoleC_设计规格_2026-10-14.md` §5：

  * **T5 认证顺序与因果边界** —— 不得"先提交后认证"；不得使用真实未来信息；
  * **T6 承诺与预算硬约束** —— 已接受承诺**不可撤销**；`certify_plan ≤ B` 全程成立；
  * **T7 确定性** —— 同 seed 两次运行的 `decision_hash` 序列一致。

设计原则（多轮踩坑后的固定做法）：

  1. **纯函数 + 显式记录** ⇒ 可离线单测，不需要跑实验；
  2. **fail-closed**：记录**缺字段**一律判 FAIL（"没记录"**不能**当成"没违反"）；
  3. **只判"机制是否被违反"，不判"有没有用"** —— 不读任何效用；
  4. 时间比较带容差 `tol`（默认 1e-6），避免浮点噪声误判。

记录 schema（每个决策事件一行 dict；见 `REQUIRED_FIELDS`）：
    inst_idx, day_id, clock            事件标识
    info_max_reveal                    该决策**用到**的未来信息的最大揭示时刻
    order_id                           本事件的订单（可为 None = 纯 WAIT 事件）
    action                             'accept' | 'reject' | 'wait' | 'defer'
    certified                          提交前是否通过 certify_plan
    certify_energy / budget_B          硬约束数值（action=='accept' 时必须 ≤ B）
    accepted_before / accepted_after   已接受承诺集合（不可撤销 ⇒ before ⊆ after）
    executed_before / executed_after   已执行前缀（冻结 ⇒ before ⊆ after）
    decision_hash                      行为指纹（T7 用；排除计时/身份字段）
"""
from __future__ import annotations

SCHEMA_VERSION = 1
TOL = 1e-6

REQUIRED_FIELDS = ('inst_idx', 'day_id', 'clock', 'info_max_reveal', 'action',
                   'certified', 'accepted_before', 'accepted_after',
                   'executed_before', 'executed_after', 'decision_hash')
# **条件必填**：`action=='accept'` 时必须存在 —— 否则无法证明 `certify_plan ≤ B`
# 与"接的是哪一单"。把条件要求**写进 schema**（而不是散落在各检查里），
# 这样缺字段一定走同一条 fail-closed 路径。
CONDITIONAL_FIELDS = {'accept': ('order_id', 'certify_energy', 'budget_B')}
ACTIONS = ('accept', 'reject', 'wait', 'defer')


def _missing(rec, idx):
    miss = [f for f in REQUIRED_FIELDS if f not in rec]
    if miss:
        return '记录 #%d 缺字段 %s（**没记录 ≠ 没违反**，fail-closed）' % (idx, miss)
    if rec.get('action') not in ACTIONS:
        return '记录 #%d 的 action=%r 非法（应为 %s）' % (idx, rec.get('action'), ACTIONS)
    if not isinstance(rec.get('certified'), bool):
        return '记录 #%d 的 certified=%r 非布尔' % (idx, rec.get('certified'))
    cmiss = [f for f in CONDITIONAL_FIELDS.get(rec['action'], ()) if f not in rec]
    if cmiss:
        return ('记录 #%d 的 action=%s 缺**条件必填**字段 %s'
                '（fail-closed：无法证明该动作合法）' % (idx, rec['action'], cmiss))
    return None


def _as_set(v):
    return set(v or ())


def check_order_and_causality(records, tol=TOL):
    """T5：**先评估→认证→再提交**；且**不得**使用真实未来信息。

    因果判据：`info_max_reveal <= clock + tol`。
    ⚠️ 这是**记录层**审计：它证明"记录里没有越界"，**不能**替代静态检查
    （静态检查证明"代码里没有读取未来的路径"）。两者都要。
    """
    problems = []
    for i, r in enumerate(records or []):
        bad = _missing(r, i)
        if bad:
            problems.append(bad)
            continue
        clock = float(r['clock'])
        if float(r['info_max_reveal']) > clock + tol:
            problems.append('T5 因果越界：记录 #%d 用到 reveal=%.6f 的信息，'
                            '但决策时刻 clock=%.6f（泄漏真实未来）'
                            % (i, float(r['info_max_reveal']), clock))
        if r['action'] == 'accept' and r['certified'] is not True:
            problems.append('T5 顺序违规：记录 #%d 在 certified=False 的情况下提交了 accept'
                            '（"先提交后认证"）' % i)
    return (not problems), problems


def check_commitments_and_budget(records, tol=TOL):
    """T6：承诺**不可撤销**（集合单调不减）+ 已执行前缀**冻结** + `certify ≤ B`。

    刻意**只查单调性**，不要求集合"按某种顺序增长"：只要 `before ⊆ after`，
    且已接受/已执行的元素**不再消失**。
    """
    problems = []
    for i, r in enumerate(records or []):
        bad = _missing(r, i)
        if bad:
            problems.append(bad)
            continue
        ab, aa = _as_set(r['accepted_before']), _as_set(r['accepted_after'])
        eb, ea = _as_set(r['executed_before']), _as_set(r['executed_after'])
        removed_acc = ab - aa
        if removed_acc:
            problems.append('T6 承诺被撤销：记录 #%d 的 accepted **消失了** %s'
                            % (i, sorted(removed_acc)[:6]))
        removed_exe = eb - ea
        if removed_exe:
            problems.append('T6 已执行前缀被改写：记录 #%d 的 executed **回退** %s'
                            % (i, sorted(removed_exe)[:6]))
        if r['action'] == 'accept':
            oid = r.get('order_id')
            if oid is not None and oid in ab:
                problems.append('T6 重复接单：记录 #%d 的 order_id=%r 已在 accepted_before 里'
                                % (i, oid))
            ce, B = r.get('certify_energy'), r.get('budget_B')
            if ce is None or B is None:
                problems.append('T6 缺数值：记录 #%d 的 accept 未记录 certify_energy/budget_B'
                                '（无法证明 ≤ B，fail-closed）' % i)
            elif float(ce) > float(B) + tol:
                problems.append('T6 预算越界：记录 #%d 的 certify_energy=%.6f > B=%.6f'
                                % (i, float(ce), float(B)))
        if r['action'] != 'accept':
            added = aa - ab
            if added:
                problems.append('T6 非接单事件却改变了承诺集合：记录 #%d action=%s 新增 %s'
                                % (i, r['action'], sorted(added)[:6]))
    return (not problems), problems


def check_determinism(log_a, log_b):
    """T7：同 seed 的两次运行，`decision_hash` 序列必须**逐位一致**。

    只比 `(inst_idx, day_id, decision_hash)`，**不**比耗时/时间戳/身份哈希
    （那些本来就会不同）。
    """
    def sig(log):
        out = []
        for i, r in enumerate(log or []):
            bad = _missing(r, i)
            if bad:
                return None, bad
            out.append((r['inst_idx'], r['day_id'], r['decision_hash']))
        return out, None

    sa, ba = sig(log_a)
    sb, bb = sig(log_b)
    if ba or bb:
        return False, [x for x in (ba, bb) if x]
    if len(sa) != len(sb):
        return False, ['T7 决策条数不同：%d vs %d' % (len(sa), len(sb))]
    diffs = [(k, x, y) for k, (x, y) in enumerate(zip(sa, sb)) if x != y]
    if diffs:
        return False, ['T7 第 %d 条决策不一致：%r vs %r' % (k, x, y) for k, x, y in diffs[:8]]
    return True, []


def _sig(log):
    """决策序列指纹：`(inst_idx, day_id, decision_hash)`。缺字段 ⇒ 返回 `(None, 问题)`。"""
    out = []
    for i, r in enumerate(log or []):
        bad = _missing(r, i)
        if bad:
            return None, bad
        out.append((r['inst_idx'], r['day_id'], r['decision_hash']))
    return out, None


def _accept_rate(log):
    n = len(log or [])
    if not n:
        return None
    return sum(1 for r in (log or []) if r.get('action') == 'accept') / float(n)


def compare_ablations(log_base, log_ablated, label=''):
    """**T2 / T3 / T4**：该消融开关是否**真的接进了决策路径**。

    判据**只有一条**（2026-10-14 用户纠错后的措辞）：两个决策序列**不得完全相同**。

      * **完全相同 ⇒ `engaged=False` ⇒ FAIL** —— 说明这个开关根本没接进决策路径
        （不是"效应小"，是"没接线"）；
      * **只要有一处不同 ⇒ `engaged=True`**，**无论差异多小** ——
        「差异多大 / 是否显著 / 方向对不对」**属科学结论**，本函数**一律不判**
        （绝不因为"变化不显著"判失败）。

    同时**如实报告**行为指标（接单率、动作一致率、差异条数）供人阅读。
    """
    sb, pb = _sig(log_base)
    sa, pa = _sig(log_ablated)
    if pb or pa:
        return {'label': label, 'engaged': None, 'problems': [x for x in (pb, pa) if x],
                'note': '记录不完整 ⇒ 无法判定是否接入（fail-closed，不算通过）'}
    n_diff = (None if len(sb) != len(sa) else
              sum(1 for x, y in zip(sb, sa) if x != y))
    key_b = {(i, d): h for i, d, h in sb}
    key_a = {(i, d): h for i, d, h in sa}
    common = set(key_b) & set(key_a)
    agree = (sum(1 for k in common if key_b[k] == key_a[k]) / float(len(common))
             if common else None)
    rb, ra = _accept_rate(log_base), _accept_rate(log_ablated)
    return {
        'label': label,
        'engaged': (sb != sa),
        'n_decisions_base': len(sb),
        'n_decisions_ablated': len(sa),
        'n_differing': n_diff,
        'action_agreement': agree,
        'accept_rate_base': rb,
        'accept_rate_ablated': ra,
        'delta_accept_rate': (None if (rb is None or ra is None) else ra - rb),
        'problems': [],
        'note': ('engaged=True ⇒ 开关已接入（**差异大小不作判据**）'
                 if sb != sa else
                 'engaged=False ⇒ **FAIL**：消融前后决策序列完全相同，开关没有接进决策路径'),
    }


def check_ablation_engagement(cases):
    """对三个开关各给一对日志，返回 `(ok, rows, problems)`。

    `cases` = `{开关名: (log_base, log_ablated)}`，开关名对应
    `role_c_opportunity_cost.ABLATIONS`（`opportunity_cost` / `lookahead` / `scenarios`）。
    ⚠️ **三个开关都必须 engaged**（都不得"完全相同"）；任一未接入 ⇒ 判 FAIL。
    """
    rows, problems = [], []
    for name, pair in (cases or {}).items():
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            problems.append('开关 %s 需要 (基准日志, 消融日志) 一对' % name)
            continue
        r = compare_ablations(pair[0], pair[1], label=name)
        r['switch'] = name
        rows.append(r)
        if r['engaged'] is not True:
            problems.append('开关 %s 未接入决策路径：%s' % (name, r.get('note')))
        problems.extend(r.get('problems') or [])
    if not cases:
        problems.append('未提供任何消融对照（cases 为空）⇒ 无法判定接入，fail-closed')
    return (not problems), rows, problems


def audit(records, log_repeat=None, ablations=None, tol=TOL):
    """跑全部可用检查，返回结构化报告（**不读任何效用**）。

    ⚠️ **`problems` 只放"真的违反了"**；「未提供对照 ⇒ 不适用」进 `not_applicable`。
    把"不适用"混进 `problems` 会重演本项目反复出现的错误：
    **结构性不适用被当成失败**（本轮实测踩到：干净日志因"未提供消融对照"被判有 problem）。
    """
    t5_ok, t5 = check_order_and_causality(records, tol)
    t6_ok, t6 = check_commitments_and_budget(records, tol)
    na = []
    t7_ok, t7 = None, []
    if log_repeat is None:
        na.append('T7 确定性：未提供重复运行日志（`log_repeat`）⇒ 不适用')
    else:
        t7_ok, t7 = check_determinism(records, log_repeat)
    t234_ok, t234_rows, t234 = None, [], []
    if ablations is None:
        na.append('T2/T3/T4 消融接入：未提供消融对照（`ablations`）⇒ 不适用')
    else:
        t234_ok, t234_rows, t234 = check_ablation_engagement(ablations)
    checks = {'T2_T3_T4_ablation_engagement': t234_ok,
              'T5_order_and_causality': t5_ok,
              'T6_commitments_and_budget': t6_ok,
              'T7_determinism': t7_ok}
    problems = list(t234) + list(t5) + list(t6) + list(t7)
    return {'schema_version': SCHEMA_VERSION,
            'n_records': len(records or []),
            'checks': checks,
            'ablation_rows': t234_rows,
            'problems': problems,
            'not_applicable': na,
            # 只有**已执行**的检查才参与判定；未提供对照不算失败
            'passed': all(v is not False for v in checks.values()),
            'note': '只判"机制是否被违反 / 开关是否接入"，**不判有没有用**（不读效用）。'
                    '缺字段一律 FAIL（没记录 ≠ 没违反）；消融**有差异即算接入，不判显著性**；'
                    '「不适用」进 `not_applicable`，**不**计入 `problems`。'}
