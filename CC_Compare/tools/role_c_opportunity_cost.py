#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**预算机会成本估计器**与接单判据（纯函数；不读效用、不跑实验）。

依据 `CC_Compare/P1A_RoleC_设计规格_2026-10-14.md` §2.1 / §2.1.1（2026-10-14 用户纠错后）：

    写法 (i)【本模块，唯一允许】：
        myopic_net(o) = revenue_o − fuel_o + reject_loss_o        # 接/拒两动作的**直接净增益**
        λ_B(o)        = clip( [Ĵ(B) − Ĵ(B − e_o)] / e_o , 0, +∞ )  # 预算边际价值，≥0
        接单 ⟺ myopic_net(o) − λ_B(o)·e_o > 0   且   certify_plan ≤ B

  ⚠️ **2026-10-14 符号修正（D1，审计见
  `CC_Compare/P1A_RoleC_效用公式与消融门审计_2026-10-14.md` §1）**：
  原式写的是 `revenue_o − fuel_o − reject_loss_o`，符号是**错的**。逐项推导：

      J(接 o) − J(拒 o) = R_o − c_f·ΔD + p_{c,o} + Δ(未来项)

  因为 A-v1 效用是 `J = Σ_served R − fuel − Σ_rejected p_c`
  （`coldchain_evaluator_a1.py:142-145`），拒单才付 `p_c` ⇒ **接单避免了这笔罚金，是正向增量**。
  p_c=0 时两种写法数值相同（主档不可见），p_c>0 时现式**系统性过度拒单**，
  偏差 `2·p_{c,o}`（p_c=15 档 ⇒ 30 > P1-A 效果门 δ=20）。单点反例：p_c=15、R=10、F=0
  ⇒ 正确 `+25` 判接单，现式 `−5` 判拒单。

  ⚠️ **能耗项不在效用里**：`fuel` 是 `distance_km × FUEL_COST_PER_KM`（**行驶燃料**），
  C0 能耗 `energy_kwh` 只作硬约束 `≤ B` 由 `certify_plan` 把关。
  因此 `λ_B·e_o` 定价的是**约束的稀缺性**，与 `myopic_net` 无重叠 ⇒ **不构成重复计入**。
  （接口约束：若哪天把能耗货币化进 utility，`λ_B·e_o` **才会**变成重复计入。）

  ⚠️ **禁用写法 (iii)**：`V̂(接单) − V̂(拒单) − λ_B·e_o`。
  因为 `V̂(·, B)` 已在**同一预算 B** 下 rollout，"占用 e_o 会挤掉未来 e_o"**已经体现在 V̂ 的差里**；
  再减一次 `λ_B·e_o` = 同一笔预算稀缺**扣两遍** ⇒ **系统性过度拒单**（随 e_o 放大）。
  本模块用**API 层硬护栏**挡住它：`decide()` 一旦同时收到 `lambda_B` 与 `lookahead_delta` 就**抛错**。

  ⚠️ `future_value_fn`（即 `Ĵ`）**只允许含未来**，**不得**含本单的 `revenue_o`／`reject_loss_o`，
  否则又会把 `myopic_net` 重复计入。

只做两件事：估计 `λ_B`、按 (i) 判决。**不**负责路线搜索（见规格 §2.2），
**不**读任何效用结果，**不**触碰任何 gate。
"""
from __future__ import annotations

# 消融开关（与规格 §2 一一对应；由 runner 传入，本模块只消费）
ABLATIONS = ('opportunity_cost', 'lookahead', 'scenarios')

# λ 的两种估计器（规格 §2.6）：**Role C 必须用前者**。
#   SHADOW  = [Ĵ(B) − Ĵ(B−e_o)]/e_o          —— 内生影子价格（单位 utility/kWh）
#   DENSITY = 既有 knn/valuee 的背包密度阈值   —— 外生（单位 revenue/kWh），
#             实现见 run_exp_reserve.py:108 `_marginal_lambda`（排序+贪心填充+取边界密度）
# 判决式两者**形式相同**（`REV − λ·e > 0`）⇒ 若 Role C 用了 DENSITY 却仍称"机制独立"，
# 那就是规格 §9.1 禁止的"换名重用同一决策函数"。此标签使该情形**可被机械检出**。
LAMBDA_ESTIMATOR_SHADOW = 'shadow_price_budget_response'
LAMBDA_ESTIMATOR_DENSITY = 'density_threshold_knapsack'
LAMBDA_ESTIMATORS = (LAMBDA_ESTIMATOR_SHADOW, LAMBDA_ESTIMATOR_DENSITY)

# --------------------------------------------------------------------------- #
# **事前确定**的边界规则（2026-10-14 用户裁定 §三；**不得**按结果调整）
#
#   B1 `e_o == 0`                ⇒ `λ_B = 0`（无占用即无机会成本；不得凭空产生）
#   B2 `e_o > B`（预算扰动越界）  ⇒ **强制拒单**（接单本身就让计划超预算），
#                                  记 `boundary='B2_budget_out_of_range'`
#   B3 `λ_B` 未定义（V̂ 非有限 / 差分为 NaN·Inf）
#                                ⇒ **强制拒单**（保守方向），
#                                  记 `boundary='B3_lambda_undefined'`
#   B4 差值为负（采样噪声）       ⇒ `λ_B` 截到 0（见 `shadow_price_budget`）
#
# ⚠️ B3 的方向是**刻意选的**：若把"未定义"当作 `λ_B = 0`，判据就退化成"可行即接"
#    ⇒ 系统性**偏向接单**。宁可保守拒单，并把该情形**计数**出来供审计。
# --------------------------------------------------------------------------- #
BOUNDARY_RULES = ('B1_e_o_zero', 'B2_budget_out_of_range', 'B3_lambda_undefined',
                  'B4_negative_diff_clipped')


def clipped_stats(diags):
    """对一批 `λ_B` 诊断，报告 `clipped`（`raw<0`）与 `undefined` 的**计数与比例**。

    ⚠️ **这是诊断量，不是门**（2026-10-14 审计 D2.4）：
    原 T9 断言"`λ_B ≥ 0`"是**同义反复** —— `shadow_price_budget` 里
    `lam = max(raw, 0.0)` 已经 clip 过，该断言永远不可能失败，因此不构成证据。
    有信息量的是**未截断**的 `lambda_raw`：`raw<0` 的比例偏高说明
    采样噪声大或 `V̂` 的设定有问题（"预算变少反而更好"在经济学上不该出现）。
    **不设阈值**：它只被报告，不用来判 PASS/FAIL。
    """
    ds = list(diags or [])
    n = len(ds)
    n_clip = sum(1 for d in ds if (d or {}).get('clipped'))
    n_und = sum(1 for d in ds if (d or {}).get('undefined'))
    return {'n': n,
            'n_clipped': n_clip,
            'n_undefined': n_und,
            'clipped_ratio': (n_clip / float(n)) if n else None,
            'undefined_ratio': (n_und / float(n)) if n else None,
            'note': '诊断量：`clipped` = 未截断差值为负的比例（噪声/设定可疑）；'
                    '`undefined` = λ_B 不可用的比例（触发 B3 强制拒单）。**不设阈值**。'}


def apply_budget_boundary(e_o, budget_B, lambda_B, tol=1e-9):
    """应用**事前确定**的边界规则；返回 `(force_reject, rule, reason)`。

    只处理需要决策上下文的 B2 / B3（B1 / B4 在 `shadow_price_budget` 内处理）。
    纯函数、与效用无关、与结果无关，可单测。
    """
    e = float(e_o or 0.0)
    if e > float(budget_B) + float(tol):
        return True, 'B2_budget_out_of_range', (
            'e_o=%.6f > B=%.6f ⇒ 预算扰动越界，按事前规则 **强制拒单**'
            % (e, float(budget_B)))
    if lambda_B is None:
        return True, 'B3_lambda_undefined', (
            'λ_B 未定义（V̂ 非有限/差分未定义）⇒ 按事前规则 **强制拒单**'
            '（不得当作 0 —— 那会系统性偏向接单）')
    try:
        lam = float(lambda_B)
    except (TypeError, ValueError):
        return True, 'B3_lambda_undefined', 'λ_B 非数值（%r）⇒ **强制拒单**' % (lambda_B,)
    if lam != lam or lam in (float('inf'), float('-inf')):     # NaN / ±Inf
        return True, 'B3_lambda_undefined', 'λ_B 非有限（%r）⇒ **强制拒单**' % (lambda_B,)
    return False, '', ''

DOUBLE_COUNT_REFUSAL = (
    '禁止重复计机会成本：不得把前瞻价值差（lookahead_delta）与 λ_B·e_o **同时**扣减 '
    '（规格 §2.1.1 写法 (iii)）。请二选一：'
    '写法 (i) 传 myopic_net + lambda_B（本规格）；或写法 (ii) 只传 V̂(接单)−V̂(拒单)，'
    '此时不要再传 λ_B。')

# --- C2 诊断审计（2026-10-14 用户裁定 ⑤ / 登记件 §8.2 + §14.4）----------------------
# 目的：让"一次真实的硬超时"与"普通 λ_B 未定义"在**记录层**可分。
# 纪律：**只记录，不改控制流**；判决、返回值形状、`undefined` 取值、B3 强制拒单规则均不变。
_EVENT_SEQ = [0]
_DECISION_TIMEOUT_TYPES = [None]          # 惰性解析；None = 尚未尝试，() = 不可用


def _event_id():
    """本进程内单调的事件号（§3.3 去重用；同时保证"两条事件不被并成一条"）。"""
    _EVENT_SEQ[0] += 1
    return 'exc-%d' % _EVENT_SEQ[0]


def _decision_timeout_types():
    """惰性取 `DecisionTimeout` 的**类型**（按类型判定，不靠文案）。

    不可用 ⇒ 返回 `()`，并让 `hard_deadline_exception` 记为 **None（未知）**，
    **不得**写 False —— 未观察到的信息不得写成否定证据。
    """
    if _DECISION_TIMEOUT_TYPES[0] is None:
        try:
            import role_c_timing as _rt
            _DECISION_TIMEOUT_TYPES[0] = (_rt.DecisionTimeout,)
        except Exception:                                      # noqa: BLE001
            _DECISION_TIMEOUT_TYPES[0] = ()
    return _DECISION_TIMEOUT_TYPES[0]


def _budget_point_of(future_value_fn):
    """**只读**推断异常落在哪个预算点（只在 `except` 体内调用，不在 `try` 内插桩）。

    `FutureValueFn.calls` **只在成功返回后**追加 ⇒ 长度为 0 判 `hi`、否则判 `lo`。
    推断不可得（注入的不是该类型）⇒ `'unknown'`。
    """
    calls = getattr(future_value_fn, 'calls', None)
    if not isinstance(calls, list):
        return 'unknown'
    try:
        return 'hi' if len(calls) == 0 else 'lo'
    except Exception:                                          # noqa: BLE001
        return 'unknown'


def _exception_audit(exc, *, stage, future_value_fn=None):
    """结构化异常来源（附加字段；**不改动**任何既有键的语义）。"""
    types = _decision_timeout_types()
    if not types:
        hard = None                        # 类型不可用 ⇒ **未知**，不是 False
    else:
        hard = bool(isinstance(exc, types))
    return {
        'exception_type': type(exc).__name__,
        'exception_stage': stage,
        'budget_point': _budget_point_of(future_value_fn),
        'hard_deadline_exception': hard,
        'evaluation_status': ('timed_out' if hard is True else
                              ('errored' if hard is False else 'unknown')),
        'event_id': _event_id(),
        'exception_audit_version': 'c2-v1',
    }


def shadow_price_budget(budget, e_o, future_value_fn, *, clip_nonneg=True):
    """估计预算的边际价值 `λ_B`。返回 `(lambda_B, diag)`。

    * `budget`  = 本次决策可用的 C0 能量预算 `B`（kWh）。
    * `e_o`     = 接单会占用的边际能耗（`c0_marginal_energy`，kWh）。
    * `future_value_fn(b)` = **只含未来**的续行价值 `Ĵ(b)`（**不得**含本单收益）。
      注入式（而不是在内部调用场景机制）是为了**可纯函数单测**，也便于 T9 用常数函数反证。

    边界处理（写死，避免未定义行为）：
      * `e_o <= 0` ⇒ `λ_B = 0`（没有占用就没有机会成本；**不得**凭空产生）；
      * `B − e_o < 0` ⇒ 评估点截到 `0` 并置 `infeasible=True`（此时 `certify_plan ≤ B` 也会否决，
        所以不影响判决，但必须在诊断里留痕）；
      * 差值为负（采样噪声）⇒ 截到 `0`（可用预算变少**不可能**提高效用；
        负影子价格会**反向补贴**接单）。
    """
    e = float(e_o or 0.0)
    if e <= 0.0:
        # ⚠️ 诊断键必须**统一**（2026-10-14 实测踩到）：本早退分支原先缺 `undefined`/`clipped`，
        # 于是调用方读 `diag['undefined']` 会 KeyError —— 同一函数不同分支给出不同 schema
        # 是很容易漏的接口不一致。
        return 0.0, {'lambda_raw': 0.0, 'lambda_B': 0.0, 'e_o': e,
                     'v_hi': None, 'v_lo': None, 'b_lo': float(budget),
                     'b_lo_eval': float(budget), 'infeasible': False,
                     'undefined': False, 'clipped': False,
                     'reason': 'e_o<=0 ⇒ 无占用，机会成本恒为 0（B1）'}
    b = float(budget)
    b_lo = b - e
    infeasible = b_lo < 0.0
    b_lo_eval = max(b_lo, 0.0)
    try:
        v_hi = float(future_value_fn(b))
        v_lo = float(future_value_fn(b_lo_eval))
    except Exception as exc:                                   # noqa: BLE001
        # ─── C2 诊断审计（2026-10-14 用户裁定 ⑤；白名单见登记件 §8.2 + §14.4）───
        # **只记录，不改控制流**：下面的返回值、`undefined` 取值与 B3 强制拒单规则
        # 一字不动；仅**新增** `exception_audit` 子字典（附加字段）。
        return None, {'lambda_raw': None, 'lambda_B': None, 'e_o': e, 'b_lo': b_lo,
                      'undefined': True, 'reason': 'V̂ 求值抛错：%s' % exc,
                      'exception_audit': _exception_audit(
                          exc, stage='shadow_price_budget.future_value',
                          future_value_fn=future_value_fn)}
    vals = (v_hi, v_lo)
    if any((x != x) or x in (float('inf'), float('-inf')) for x in vals):
        # B3：V̂ 非有限 ⇒ λ_B **未定义**（返回 None，由边界规则强制拒单）
        return None, {'lambda_raw': None, 'lambda_B': None, 'e_o': e,
                      'v_hi': v_hi, 'v_lo': v_lo, 'b_lo': b_lo,
                      'undefined': True,
                      'reason': 'V̂ 非有限（v_hi=%r, v_lo=%r）⇒ λ_B 未定义' % vals}
    raw = (v_hi - v_lo) / e
    if raw != raw or raw in (float('inf'), float('-inf')):
        return None, {'lambda_raw': raw, 'lambda_B': None, 'e_o': e,
                      'v_hi': v_hi, 'v_lo': v_lo, 'b_lo': b_lo,
                      'undefined': True, 'reason': '差分非有限（raw=%r）' % raw}
    lam = max(raw, 0.0) if clip_nonneg else raw
    return lam, {'lambda_raw': raw, 'lambda_B': lam, 'e_o': e,
                 'v_hi': v_hi, 'v_lo': v_lo, 'b_lo': b_lo,
                 'b_lo_eval': b_lo_eval, 'infeasible': infeasible,
                 'undefined': False,
                 'clipped': bool(clip_nonneg and raw < 0.0),
                 'reason': 'clip 到 0（负影子价格只可能来自采样噪声）' if (clip_nonneg and raw < 0.0)
                           else ''}


def decide(*, myopic_net, e_o, lambda_B, certify_ok,
           ablate_opportunity_cost=False, lookahead_delta=None):
    """按规格 §2.1.1 **写法 (i)** 判决。返回 `(accept, diag)`。

    ⚠️ `lookahead_delta` **只用于触发护栏**：非 None 即抛错（写法 (iii) 已禁用）。
    若确实要用写法 (ii)，请调用方**不要**传 `lambda_B`，而直接比较 `V̂` 差值。
    """
    if lookahead_delta is not None:
        raise ValueError(DOUBLE_COUNT_REFUSAL)
    lam = 0.0 if ablate_opportunity_cost else float(lambda_B or 0.0)
    e = float(e_o or 0.0)
    mn = float(myopic_net)
    cost = lam * e
    net_after = mn - cost
    hard_ok = bool(certify_ok)
    accept = bool(hard_ok and net_after > 0.0)
    return accept, {
        'myopic_net': mn,
        'e_o': e,
        'lambda_B': lam,
        'opportunity_cost': cost,
        'net_after_opportunity_cost': net_after,
        'certify_ok': hard_ok,
        'ablate_opportunity_cost': bool(ablate_opportunity_cost),
        'formulation': '(i) myopic_net − λ_B·e_o > 0 ∧ certify_ok',
        'reject_reason': ('' if accept else
                          ('certify_plan 未通过（硬约束）' if not hard_ok
                           else '净收益 ≤ 0（已扣预算机会成本）')),
    }


def myopic_net(revenue, fuel, reject_loss):
    """**接单 − 拒单**的直接净增益 = `revenue − fuel + reject_loss`（规格 §2.1 左式）。

    ⚠️ 2026-10-14 符号修正（D1）：`reject_loss` 是**加号**。
    推导：`J = Σ_served R − fuel − Σ_rejected p_c` ⇒ 接单避免了 `p_c` 的支出 ⇒ 正向增量。
    旧实现写 `− reject_loss`，在 p_c>0 档系统性过度拒单（偏差 `2·p_c`）。
    `fuel` 口径 = **边际**行驶燃料 `FUEL_COST_PER_KM × Δdistance(插入 o)`，
    不是分摊/整日燃料（否则近视基准量级错，归因失真）。
    **不得**在此再扣机会成本。

    ⚠️ 本函数只接受**分量**（revenue / fuel / reject_loss）。若调用方手上的量
    **已经是** `J_accept − J_reject`（其中已含拒单罚金），必须改调
    `myopic_net_from_full_margin` —— 否则会在修好 D1 之后**又加一次** `p_c`
    （偏差 `2·p_c`），这正是 2026-10-14 用户裁定①要求补符号单测的原因。
    """
    return float(revenue) - float(fuel) + float(reject_loss)


# 近视基准的**两种来源**：必须显式声明，供 `arms_are_matched` 做语义身份检查。
#   components  ：调用方给 R / ΔF / p_c 三个分量 ⇒ 由 `myopic_net` 组装
#   full_margin ：调用方给的量**已经等于** J_accept − J_reject（已含 p_c）⇒ 原样使用
MYOPIC_BASE_COMPONENTS = 'components_revenue_fuel_reject_loss'
MYOPIC_BASE_FULL_MARGIN = 'full_accept_minus_reject_margin'
MYOPIC_BASES = (MYOPIC_BASE_COMPONENTS, MYOPIC_BASE_FULL_MARGIN)

# 第二因素探索臂的近视基准（`REV` 版，对齐上游 knn/valuee）。
# ⚠️ 它**不得**再用于 `B-shadow − B-density` 的单因素归因比较（2026-10-14 用户裁定①）：
#    该差值会同时改变「λ 估计方式」与「近视基准」两件事，只能登记为**双因素**探索性消融。
MYOPIC_BASE_REVENUE_ONLY = 'revenue_only_legacy'


def myopic_net_from_full_margin(margin, *, reject_loss=None):
    """调用方给的量**已经是** `J_accept − J_reject`（含拒单罚金）⇒ **原样返回**。

    对应 2026-10-14 用户裁定①的符号单测：修好 D1（`+p_c`）之后，若把"已经完全含 p_c
    的差值"再喂给 `myopic_net(...)`，就会产生**新的重复计入**。因此本函数**不接受**
    `reject_loss`：一旦传入即抛错，逼迫调用方二选一。
    """
    if reject_loss is not None:
        raise ValueError(
            '传入了已经是 J_accept − J_reject 的完整差值，**不得**再加 reject_loss：'
            '该差值已含拒单罚金，再加一次 = 重复计入 p_c（偏差 2·p_c）。'
            '若手上只有分量，请改用 myopic_net(revenue, fuel, reject_loss)。')
    return float(margin)


def myopic_net_from_components_or_margin(*, myopic_base, revenue=None, fuel=None,
                                        reject_loss=None, margin=None):
    """按显式声明的 `myopic_base` 选择组装方式，返回 `(value, base_label)`。"""
    if myopic_base == MYOPIC_BASE_COMPONENTS:
        if None in (revenue, fuel, reject_loss):
            raise ValueError('myopic_base=components 需要 revenue / fuel / reject_loss 三者')
        return myopic_net(revenue, fuel, reject_loss), MYOPIC_BASE_COMPONENTS
    if myopic_base == MYOPIC_BASE_FULL_MARGIN:
        if margin is None:
            raise ValueError('myopic_base=full_margin 需要 margin')
        return (myopic_net_from_full_margin(margin, reject_loss=reject_loss),
                MYOPIC_BASE_FULL_MARGIN)
    raise ValueError('未知 myopic_base=%r；应为 %s' % (myopic_base, ' / '.join(MYOPIC_BASES)))


def decide_candidate(*, order, candidate_plan, budget_B,
                     myopic_net_fn, marginal_energy_fn, future_value_fn, certify_fn,
                     ablate_opportunity_cost=False, ablate_lookahead=False,
                     lambda_estimator=LAMBDA_ESTIMATOR_SHADOW,
                     myopic_base=MYOPIC_BASE_COMPONENTS):
    """**Role C 的候选评估核心**（规格 §2.1 + §2.4 的"候选评估 → certify → 提交"）。

    环境相关的部分**全部注入**（`myopic_net_fn` / `marginal_energy_fn` /
    `future_value_fn` / `certify_fn`），因此本函数**不依赖 env、不跑实验**，可完全离线单测；
    真实 runner 只需把项目实现传进来。

    ⚠️ **两个消融开关并不独立（重要，2026-10-14 发现）**：

      * `ablate_opportunity_cost`（A 版）= **保留**场景前瞻，但令 `λ_B ≡ 0`
        ⇒ 判据退化为 `myopic_net > 0 ∧ certify_ok`（"可行即接"）；
      * `ablate_lookahead`     = **去掉**场景前瞻 ⇒ `Ĵ` 无从估计 ⇒ `λ_B` **必然也只能是 0**。

    ⇒ 二者**都会**让 `λ_B = 0`，区别在于 **A 版仍持有场景机制**。因此 A–F 表里的
    **A 版必须定义为"有场景 + `λ_B` 强制为 0"**，而**不是**"去掉前瞻" ——
    否则 `B − A` 会把"去掉前瞻"与"去掉机会成本"两件事**混淆**（confound）。
    本函数在诊断里用 `lambda_source` 区分两者，便于 T2/T3 分别归因。

    返回 `(result_dict, diag)`；`result_dict['accept']` 为最终判决。
    """
    e_o = float(marginal_energy_fn(order))
    if lambda_estimator not in LAMBDA_ESTIMATORS:
        raise ValueError('未知 lambda_estimator=%r；应为 %s'
                         % (lambda_estimator, ' / '.join(LAMBDA_ESTIMATORS)))
    # ---- 事前边界规则（B2 预算越界 / B3 λ 未定义）先判：命中即**强制拒单**，不再看净收益 ----
    boundary = None
    if e_o > float(budget_B) + 1e-9:
        boundary = (True, 'B2_budget_out_of_range',
                    'e_o=%.6f > B=%.6f ⇒ 预算扰动越界，按事前规则强制拒单'
                    % (e_o, float(budget_B)))
    if boundary is None:
        lam0, _d0 = (None, {}) if lambda_estimator == LAMBDA_ESTIMATOR_DENSITY else \
            shadow_price_budget(budget_B, e_o, future_value_fn)
    else:
        lam0, _d0 = 0.0, {}
    if lambda_estimator == LAMBDA_ESTIMATOR_DENSITY:
        # 用既有阈值法时**不允许**再走本函数的前瞻差分（那会误导成"内生"）；
        # 显式拒绝，逼迫调用方要么用真实影子价格，要么明确走既有臂。
        raise ValueError(
            'Role C **不得**使用 density_threshold_knapsack 估计 λ（规格 §2.6）：'
            '判决式与既有 knn/valuee 相同，若 λ 也沿用其外生背包密度阈值，则 Role C 退化为'
            '该臂的同胞，违反预声明 §9.1「不得只是换名重用同一决策函数」。'
            '若确实要做该对照，请用独立臂 B-density（另立预注册），不要伪装成 Role C。')
    if boundary is None:
        reject_b, rule, reason = apply_budget_boundary(e_o, budget_B, lam0)
        if reject_b:
            boundary = (reject_b, rule, reason)
    if boundary is not None:
        _, rule, reason = boundary
        certify_ok0, certify_energy0 = certify_fn(candidate_plan)
        diag = {'boundary': rule, 'boundary_reason': reason, 'e_o': e_o,
                'lambda_B': None, 'opportunity_cost': None,
                'net_after_opportunity_cost': None,
                'myopic_net': float(myopic_net_fn(order)),
                'certify_ok': bool(certify_ok0), 'certify_energy': certify_energy0,
                'lambda_diag': {'lambda_source': 'boundary_short_circuit',
                                'undefined': rule == 'B3_lambda_undefined'},
                'lambda_estimator': lambda_estimator,
                'myopic_base': myopic_base,
                'budget_B': float(budget_B), 'ablate_lookahead': bool(ablate_lookahead),
                'reject_reason': reason,
                'formulation': '(i) myopic_net − λ_B·e_o > 0 ∧ certify_ok',
                'log_fields': {'order_id': getattr(order, 'order_id', None),
                               'action': 'reject', 'certified': bool(certify_ok0),
                               'certify_energy': certify_energy0}}
        # ─── C2 诊断审计传递（2026-10-14 用户裁定 ⑤；登记件 §14.4 追加白名单）───
        # 本分支原先**丢弃** `_d0`（只按 `rule` 重建 `lambda_diag`）⇒ 结构化异常来源
        # 进不了最终决策记录。此处**仅**把 `_d0` 的审计信息**附加**到 `diag`：
        # 不动 `boundary` / `boundary_reason` / `reject_reason` / 认证调用 / 返回值形状。
        _audit = (_d0 or {}).get('exception_audit')
        if _audit is not None:
            diag['exception_audit'] = _audit
        return False, diag
    lam = lam0
    if ablate_lookahead:
        lam, lam_diag = 0.0, {'lambda_raw': 0.0, 'lambda_B': 0.0, 'e_o': e_o,
                              'lambda_source': 'ablate_lookahead',
                              'reason': '无前瞻 ⇒ Ĵ 无从估计 ⇒ λ_B 置 0（与 A 版同为 0，但机制不同）'}
    else:
        lam_diag = _d0
        lam_diag['lambda_source'] = ('opportunity_cost' if not ablate_opportunity_cost
                                     else 'ablate_opportunity_cost')
    mn = float(myopic_net_fn(order))
    # §2.4 顺序：候选评估（含 λ_B）→ 认证 → 提交
    certify_ok, certify_energy = certify_fn(candidate_plan)
    accept, diag = decide(myopic_net=mn, e_o=e_o, lambda_B=lam, certify_ok=certify_ok,
                          ablate_opportunity_cost=ablate_opportunity_cost)
    diag.update({
        'lambda_diag': lam_diag,
        'lambda_estimator': lambda_estimator,
        'myopic_base': myopic_base,
        'certify_energy': certify_energy,
        'budget_B': float(budget_B),
        'ablate_lookahead': bool(ablate_lookahead),
        # 供写入器/审计器直接使用；`info_max_reveal` 由调用方按真实信息边界填写
        'log_fields': {'order_id': getattr(order, 'order_id', None),
                       'action': 'accept' if accept else 'reject',
                       'certified': bool(certify_ok),
                       'certify_energy': certify_energy},
    })
    return accept, diag


def log_decision(logger, diag, *, inst_idx, day_id, clock, info_max_reveal, decision_hash):
    """把 `decide_candidate` 的结果写进 `DecisionLogger`（写入即校验）。

    ⚠️ 拒单也**必须**记录 `certified`（写入器要求 bool）：拒单不需要认证，
    故按"认证状态未知/未认证"记为 `False`，**不**伪造 True。
    """
    fields = diag['log_fields']
    return logger.log_event(inst_idx=inst_idx, day_id=day_id, clock=clock,
                            info_max_reveal=info_max_reveal,
                            action=fields['action'], certified=fields['certified'],
                            decision_hash=decision_hash,
                            order_id=(fields['order_id'] if fields['action'] == 'accept' else None),
                            certify_energy=(fields['certify_energy']
                                            if fields['action'] == 'accept' else None))
