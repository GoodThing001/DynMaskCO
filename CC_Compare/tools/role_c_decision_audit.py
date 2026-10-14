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
        # ③ **执行 ⊆ 接受**：没接过的单不可能被执行（拒绝 = 永不服务）。
        #    与写入器 `role_c_decision_log` 的同一不变式保持同步。
        orphan = ea - aa
        if orphan:
            problems.append('T6 执行了**未接受**的订单：记录 #%d 的 executed_after 含 %s'
                            '（不在 accepted_after 里）' % (i, sorted(orphan)[:6]))
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


def check_ablation_engagement(cases, activation=None):
    """对三个开关各给一对日志，返回 `(ok, rows, problems)`。

    `cases` = `{开关名: (log_base, log_ablated)}`，开关名对应
    `role_c_opportunity_cost.ABLATIONS`（`opportunity_cost` / `lookahead` / `scenarios`）。

    ⚠️ **判据已按审计 D3.3 收紧（2026-10-14）**：`engaged=False`（消融前后**完全相同**）
    **只在"机制在本集合内确实被触发过"时才判 FAIL**。
    否则报 **`not_applicable`**，因为"完全相同"无法区分两种情形：

      1. 机制**没接进**决策路径（应 FAIL）；
      2. 机制**接好了**，但在被检验的状态集合上**资源根本不稀缺** ⇒ 正确实现下 `λ_B ≈ 0`
         ⇒ 决策序列**理应相同**（**不应** FAIL）。

    `activation` = `{开关名: 该集合内"机制被触发"的决策数}`（例如 `λ_B` 的
    `raw > 0` 计数），由 runner 从其诊断里统计后传入。**未提供或为 0 ⇒ not_applicable**。

    ⚠️ **2026-10-14 用户裁定③纠正了本函数原先的措辞（我原来的推论过头了）**：
    仓库实据（登记规则 v4 的 1000 条决策）**只能**支持一件事——
    **下界记账下的硬前置检查是松的**（`b_rem` 最小 197.27、`reject_budget` 触发 0 次、
    每日 `spent_before` 峰值 ≤72.2% B）。它**不能**推出"其余 709 次决策的资源价格接近零"：

      * **291/1000 触及网格下限 ≠ 另外 709 次 λ≈0** —— 触及下限只说明该次不受网格约束，
        不给出其余决策的价格大小；两者不能互相推断。
      * **`reject_budget = 0` 不证明不存在前瞻性机会成本** —— 一个**已经**在高耗能订单上
        持续拒单的策略，本就会把预算维持在充足水平，于是硬前置检查**永远不会**触发。
        "检查没响"因此可能是**机制在起作用的结果**，而不是"机制没有必要"的证据。
      * `spent_before` 用的是 `Σ e_lower`（**下界**），而权威检查是 `certify_plan` 的
        完整续行能量；且**旧规则日志未记录 `certify_energy`** ⇒ "资源是否真的稀缺"
        在本轮**尚未闭合**，须由独立的资源可识别性诊断回答，不得由本函数假定。

    ⇒ **"机制是否接进代码"的正证仍由 T2a 承担**（预设压力夹具上的数值行为 + T9③ 纯函数反证），
    真实开发日上的**效果可识别性**由 T2b 承担；两者都**不**用"结果序列是否不同"当唯一判据。
    """
    rows, problems = [], []
    if not cases:
        problems.append('未提供任何消融对照（cases 为空）⇒ 无法判定接入，fail-closed')
        return False, rows, problems
    for name, pair in (cases or {}).items():
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            problems.append('开关 %s 需要 (基准日志, 消融日志) 一对' % name)
            continue
        r = compare_ablations(pair[0], pair[1], label=name)
        r['switch'] = name
        act = None if activation is None else int(activation.get(name, 0))
        r['activation_count'] = act
        if r['engaged'] is False and (act is None or act <= 0):
            # 未证明机制被触发过 ⇒ **不判 FAIL**，改判 not_applicable
            r['engaged'] = None
            r['note'] = ('not_applicable：本集合内未观测到该机制被触发'
                         '（activation=%s）⇒ "完全相同"无法区分「未接线」与'
                         '「接线但资源不稀缺」。接入的正证请用 T9③（纯函数）。'
                         % ('未提供' if act is None else act))
        rows.append(r)
        # ⚠️ **2026-10-14 修复（缺陷已由用户裁定）**：原为 `is not True`，于是上面刚把
        # `engaged` 置 `None`（= **not_applicable**、本意"不判 FAIL"）的那一支**也**命中，
        # 降级被自己的 `problems` 抵消 ⇒ `t234_ok=False`、`audit['passed']=False` ——
        # 与本节 docstring 及 `audit()` 的"结构性不适用被当成失败"警告直接矛盾，
        # 也与规格 §5 T2 条"报 not_applicable，**不报 FAIL**"矛盾。
        # 只有**明确判定为未接入**（`engaged is False`）才计入 problems。
        if r['engaged'] is False:
            problems.append('开关 %s 未接入决策路径：%s' % (name, r.get('note')))
        problems.extend(r.get('problems') or [])
    return (not problems), rows, problems


def check_t2a_shadow_price_correctness(fixtures):
    """**T2a 实现正确性门**（在**预设资源压力输入**上；**不要求**最终动作不同）。

    `fixtures` ＝ `[{'budget', 'e_o', 'v_hi', 'v_lo'}, ...]`，由运动者**事前构造**的压力夹具
    （不是真实开发日）。逐条核对数值行为**符合公式**：

      ① `lambda_raw == (v_hi − v_lo) / e_o`；
      ② `lambda_B == max(lambda_raw, 0)`（下限截断）；
      ③ `lambda_B >= 0`；
      ④ `e_o <= 0 ⇒ lambda_B == 0`（不得凭空产生机会成本）；
      ⑤ `B − e_o < 0 ⇒ infeasible=True` 且评估点截到 0；
      ⑥ **消融差值**：`λ_B≡0` 与不消融的**评分**（`net_after`）必须不同（当 `λ_B·e_o != 0`）——
         这里检查的是"价格确实进入了评分"，**不是**"动作必须改变"。

    用户裁定③原文："这里要求数值行为符合公式，不要求最终动作必须不同。"
    返回 `(ok, rows, problems)`。
    """
    from role_c_opportunity_cost import shadow_price_budget, decide
    rows, problems = [], []
    if not fixtures:
        problems.append('未提供任何 T2a 压力夹具 ⇒ 无法判定实现正确性，fail-closed')
        return False, rows, problems
    for i, fx in enumerate(fixtures):
        B, e = float(fx['budget']), float(fx['e_o'])
        jhi, jlo = float(fx['v_hi']), float(fx['v_lo'])

        def _fn(b, jhi=jhi, jlo=jlo, B=B):
            return jhi if abs(b - B) < 1e-12 else jlo

        lam, d = shadow_price_budget(B, e, _fn)
        exp_raw = 0.0 if e <= 0 else (jhi - jlo) / e
        exp_lam = max(exp_raw, 0.0)
        checks = {
            'raw_matches_formula': abs(d['lambda_raw'] - exp_raw) < 1e-9,
            'clip_matches_max0': abs(lam - exp_lam) < 1e-9,
            'nonnegative': lam >= 0.0,
            'zero_energy_zero_cost': (e > 0) or (lam == 0.0 and d['lambda_B'] == 0.0),
            'infeasible_and_clamped': (B - e >= 0) or (d['infeasible'] is True
                                                       and d['b_lo_eval'] == 0.0),
        }
        # ⑥ 消融差值必须改变**评分**（价格进入评分），不要求动作改变
        mn = 1.0
        _acc_on, d_on = decide(myopic_net=mn, e_o=e, lambda_B=lam, certify_ok=True)
        _acc_off, d_off = decide(myopic_net=mn, e_o=e, lambda_B=lam, certify_ok=True,
                                 ablate_opportunity_cost=True)
        price_nonzero = abs(lam * e) > 1e-12
        score_differs = (d_on['net_after_opportunity_cost']
                         != d_off['net_after_opportunity_cost'])
        checks['ablation_changes_score_when_price_nonzero'] = (not price_nonzero) or score_differs
        ok_i = all(checks.values())
        rows.append({'fixture_index': i, 'budget': B, 'e_o': e, 'v_hi': jhi, 'v_lo': jlo,
                     'lambda_raw': d['lambda_raw'], 'lambda_B': lam,
                     'expected_raw': exp_raw, 'expected_lambda_B': exp_lam,
                     'infeasible': d['infeasible'], 'checks': checks, 'ok': ok_i})
        if not ok_i:
            problems.append('T2a 夹具 %d 数值行为不符公式：%s'
                            % (i, sorted(k for k, v in checks.items() if not v)))
    return (not problems), rows, problems


T2B_NOT_APPLICABLE = 'not_applicable'
T2B_ACTIVATED_NO_EFFECT = 'mechanism_activated_no_behavioural_effect'
T2B_ACTIVATED_WITH_EFFECT = 'mechanism_activated_with_behavioural_effect'
T2B_WIRING_DEFECT = 'wiring_defect'


def check_t2b_effect_identifiability(stats):
    """**T2b 效果可识别性门**（真实日；只做"可识别性"判定，**不判效果好坏**）。

    `stats` ＝ `{'n_decisions', 'n_raw_positive', 'n_lambda_effective_positive',
                'n_score_changed', 'n_ordering_or_action_changed'}`
    （全部由 runner 从其逐决策诊断统计；本函数**不读效用**）。

    ⚠️ 用户裁定③：必须区分 **`raw > 0`** 与**截断后的实际 `λ_B`** ——
    原始值为正但截断后与对照相同，**仍不构成可识别的价格干预**。

    判定（四级，**只有接线缺陷才 FAIL**）：

      * `raw > 0` 计数为 0 ⇒ `not_applicable`（本集合根本没有前瞻价格信号）；
      * `raw > 0` 但**有效** `λ_B > 0` 为 0 ⇒ `not_applicable`
        （截断后无有效价格 ⇒ 不构成干预，**不判 FAIL**）；
      * 有效价格 > 0 但**评分变化**为 0 ⇒ `wiring_defect`（价格没进评分，**FAIL**）；
      * 评分变化 > 0 但**排序/动作变化**为 0 ⇒ `mechanism_activated_no_behavioural_effect`
        （**不是接线失败，也不是效果通过** —— 如实记为"机制已激活、未观察到行为效应"）；
      * 动作也变化 ⇒ `mechanism_activated_with_behavioural_effect`
        （**可识别**；"效果是否为正"仍归 P1-A 效果门，**本函数不判**）。

    返回 `(verdict, row, problems)`；`problems` 只在 `wiring_defect` 时非空。
    """
    def _g(k):
        return int((stats or {}).get(k, 0) or 0)

    n_dec = _g('n_decisions')
    n_raw = _g('n_raw_positive')
    n_eff = _g('n_lambda_effective_positive')
    n_score = _g('n_score_changed')
    n_act = _g('n_ordering_or_action_changed')
    row = {'n_decisions': n_dec, 'n_raw_positive': n_raw,
           'n_lambda_effective_positive': n_eff, 'n_score_changed': n_score,
           'n_ordering_or_action_changed': n_act,
           'note_raw_vs_clipped': ('raw>0 只说明未截断前为正；判定用截断后的实际 λ_B'
                                   '（n_lambda_effective_positive）')}
    if n_raw <= 0:
        row['reason'] = '本集合内 raw>0 计数为 0 ⇒ 没有前瞻价格信号'
        return T2B_NOT_APPLICABLE, row, []
    if n_eff <= 0:
        row['reason'] = ('raw>0 但截断后有效 λ_B>0 为 0 ⇒ 不构成可识别的价格干预'
                         '（注意：这不是接线失败）')
        return T2B_NOT_APPLICABLE, row, []
    if n_score <= 0:
        row['reason'] = '有效价格>0 但候选评分无变化 ⇒ 价格没有进入评分路径（接线缺陷）'
        return T2B_WIRING_DEFECT, row, ['T2b 接线缺陷：有效 λ_B>0 未改变候选评分']
    if n_act <= 0:
        row['reason'] = ('评分已变但排序/动作未变 ⇒ 机制已激活、**未观察到行为效应**；'
                         '既不算接线失败，也不算效果通过')
        return T2B_ACTIVATED_NO_EFFECT, row, []
    row['reason'] = ('价格→评分→排序/动作 三级均可观察 ⇒ 机制可识别；'
                     '效果方向与显著性归 P1-A 效果门，本函数不判')
    return T2B_ACTIVATED_WITH_EFFECT, row, []


def audit(records, log_repeat=None, ablations=None, ablation_activation=None, tol=TOL,
          t2a_fixtures=None, t2b_stats=None):
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
        t234_ok, t234_rows, t234 = check_ablation_engagement(
            ablations, activation=ablation_activation)
    # ---- T2a 实现正确性（预设压力夹具；不要求动作不同）----
    t2a_ok, t2a_rows = None, []
    if t2a_fixtures is None:
        na.append('T2a 影子价格实现正确性：未提供压力夹具（`t2a_fixtures`）⇒ 不适用')
    else:
        t2a_ok, t2a_rows, t2a_problems = check_t2a_shadow_price_correctness(t2a_fixtures)
    # ---- T2b 效果可识别性（真实日；raw>0 与截断后有效 λ_B 必须分开）----
    t2b_verdict, t2b_row = None, {}
    if t2b_stats is None:
        na.append('T2b 效果可识别性：未提供逐决策统计（`t2b_stats`）⇒ 不适用')
    else:
        t2b_verdict, t2b_row, t2b_problems = check_t2b_effect_identifiability(t2b_stats)
    checks = {'T2_T3_T4_ablation_engagement': t234_ok,
              'T2a_shadow_price_correctness': t2a_ok,
              'T5_order_and_causality': t5_ok,
              'T6_commitments_and_budget': t6_ok,
              'T7_determinism': t7_ok}
    problems = list(t234) + list(t5) + list(t6) + list(t7)
    if t2a_fixtures is not None:
        problems += list(t2a_problems)
    if t2b_stats is not None:
        problems += list(t2b_problems)          # 只有 wiring_defect 才非空
    return {'schema_version': SCHEMA_VERSION,
            'n_records': len(records or []),
            'checks': checks,
            'ablation_rows': t234_rows,
            't2a_rows': t2a_rows,
            't2b_verdict': t2b_verdict,
            't2b_row': t2b_row,
            'problems': problems,
            'not_applicable': na,
            # 只有**已执行**的检查才参与判定；未提供对照不算失败
            'passed': all(v is not False for v in checks.values()),
            'note': '只判"机制是否被违反 / 开关是否接入"，**不判有没有用**（不读效用）。'
                    '缺字段一律 FAIL（没记录 ≠ 没违反）；消融**有差异即算接入，不判显著性**；'
                    '「不适用」进 `not_applicable`，**不**计入 `problems`。'
                    'T2b 的 `mechanism_activated_no_behavioural_effect` '
                    '既不算接线失败，也不算效果通过。'}
