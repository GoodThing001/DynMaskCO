#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**对照臂**：`B-density`（外生背包密度阈值 λ）。

为什么需要（`P1A_RoleC_设计规格` §2.6，2026-10-14 读源码发现）：
Role C 的判决式 `myopic_net − λ·e > 0` 与既有 `knn` / `valuee20/30` **形式相同**
（`run_exp_energy_c0.py:358-380`）。因此只做 `B − A`（`λ_B ≡ 0`）**不足以**支撑
"资源机会成本"这一机制主张 —— 它排除的是"可行即接"，而真正的对手是
"**一个调好的密度阈值**"。

⇒ 必须增加 `B-density`：λ 用既有臂的**背包密度阈值**（外生，单位 revenue/kWh），
其余（场景、候选池、信息集、路线修复、判据、预算认证、执行规则）与 `B-shadow` 全同。
比较 **`B‑shadow − B‑density`**。
若该差 ≈ 0，正确表述是"**资源门槛机制**有用、**内生化**无增量"。

⚠️ **2026-10-14 用户裁定①**：`B-density` 的近视基准必须与 Role C **同一个**
（`m_o = R_o − c_f·ΔD_o + p_{c,o}`），否则该差值同时改变「λ 来源」与「近视基准」
两件事 ⇒ 只能登记为**双因素探索性消融**，**不得**用于证明"内生化"的独立作用。
原 `REV` 版本**不删除**，通过 `myopic_net_fn=None` 显式构造，并由 `factor_status()`
标为 `two_factor_exploratory`。历史结果保持原裁决，不追溯改成单因素实验。

⚠️ **本模块是"对照臂"，不是 Role C**：`role_c_opportunity_cost.decide_candidate`
**拒绝**用密度阈值估计 λ（会抛错），以免 Role C 退化为 `knn` 的同胞。
两条路径分开、标签分开（`lambda_estimator`），审计可机械区分。

`density_threshold_lambda` 是上游 `run_exp_reserve.py:108 _marginal_lambda` 的**忠实移植**
（测试里用 AST 抽取上游源码 `exec` 后**逐点数值对拍**，见 `test_role_c_controls.py`）。

只做 λ 与判决，**不读效用、不碰 env、不跑实验**。
"""
from __future__ import annotations

import numpy as np

from role_c_opportunity_cost import (LAMBDA_ESTIMATOR_DENSITY, decide)
import role_c_opportunity_cost as rc


def density_threshold_lambda(vals, energys, remaining_budget):
    """既有臂的 λ：按 **价值/能耗密度降序**贪心填满剩余预算，返回**第一个装不下**的密度。

    与上游 `_marginal_lambda` 语义一致（含 `R <= 1e-12 ⇒ 0`、全装得下 ⇒ 0）。
    单位是 **revenue/kWh**（外生阈值），**不是** utility/kWh 的影子价格。
    """
    vals = np.asarray(vals, dtype=float)
    energys = np.asarray(energys, dtype=float)
    R = float(remaining_budget)
    if len(vals) == 0 or R <= 1e-12:
        return 0.0
    order = np.argsort(-(vals / np.maximum(energys, 1e-12)))
    rem = R
    for i in order:
        if energys[i] <= rem + 1e-12:
            rem -= energys[i]
        else:
            return float(vals[i] / energys[i])
    return 0.0


def decide_b_density_control(*, order, candidate_plan, budget_B,
                             revenue_fn, marginal_energy_fn, certify_fn,
                             future_vals, future_energys, myopic_net_fn=None,
                             remaining_budget=None):
    """`B-density` 对照臂：用**外生密度阈值** λ 判决 `myopic_net − λ·e > 0 ∧ certify`。

    `future_vals` / `future_energys` = 场景集合里**未来订单**的收益与边际能耗
    （由 runner 从冻结的场景机制取出，保证与 `B-shadow` **同源同 K**）。
    `remaining_budget` 缺省为 `budget_B`。

    ⚠️ **近视基准必须与 Role C 完全一致（2026-10-14 用户裁定 + 审计 D2b）**：
    用户要求两个 B 臂"**只替换 λ 估计器**"。因此本臂**默认且应当**接收与
    `decide_candidate` **同一个** `myopic_net_fn`（= `R − fuel + p_c`）。
    若沿用 `revenue_fn`（只给 `REV`），`B‑shadow − B‑density` 会**同时**改变
    「λ 来源」与「近视基准」两件事 ⇒ 无法归因（这正是 `arms_are_matched` 现在会拦下的情形）。

    ⚠️ **保真度说明（必须随结果披露）**：改用 Role C 的近视基准后，本臂**不再**等于
    上游 `knn/valuee` 的 `REV − λe`。若论文需要"忠实上游阈值臂"的对照，
    该对照由表中**既有的 `knn` / `valuee20/30` 臂**承担，**不要**用本臂冒充。
    保留 `revenue_fn` 入参仅用于显式构造"上游口径"的对照，且此时
    `arms_are_matched` 会判**不可归因**（应当如此）。
    """
    e_o = float(marginal_energy_fn(order))
    rem = float(budget_B if remaining_budget is None else remaining_budget)
    lam = density_threshold_lambda(future_vals, future_energys, rem)
    if myopic_net_fn is None:
        mn = float(revenue_fn(order))
        base_note = 'REV（**上游口径**；与 Role C 基准不同 ⇒ 不可归因，仅作第二因素探索）'
        base_label = rc.MYOPIC_BASE_REVENUE_ONLY
    else:
        mn = float(myopic_net_fn(order))
        base_note = 'R − fuel + p_c（与 Role C **同一基准**）'
        base_label = rc.MYOPIC_BASE_COMPONENTS
    certify_ok, certify_energy = certify_fn(candidate_plan)
    # 复用同一判决实现（`decide`），只是 λ 的来源不同 ⇒ 基准一致时两臂只差这一件事
    accept, diag = decide(myopic_net=mn, e_o=e_o, lambda_B=lam, certify_ok=certify_ok)
    diag.update({
        'arm': 'B_density_control',
        'lambda_estimator': LAMBDA_ESTIMATOR_DENSITY,
        'myopic_base': base_label,
        'lambda_diag': {'lambda_B': lam, 'lambda_source': 'density_threshold_knapsack',
                        'n_future_orders': int(len(np.asarray(future_vals))),
                        'remaining_budget': rem},
        'certify_energy': certify_energy,
        'budget_B': float(budget_B),
        'myopic_net_note': base_note,
    })
    return accept, diag


def arms_are_matched(diag_shadow, diag_density):
    """核对两臂**只差 λ 来源**（其余全同）：用于 `B‑shadow − B‑density` 的可归因性检查。

    返回 `(ok, diffs)`；`diffs` 列出除 λ 之外**不应存在**的差异。

    ⚠️ **2026-10-14 修正（D2b）**：原实现只比 `e_o` / `certify_ok` / `budget_B`，
    **恰好漏掉了唯一真正不同的字段 `myopic_net`** —— 参考臂用 `revenue_fn`（= `REV`），
    而 Role C 用 `myopic_net_fn`（= `REV − fuel + p_c`）。于是它会对一对
    **近视基准不同**的臂宣称"只差 λ、可归因"（旧测试用例 ⑤ 正是如此示范）。
    现把 `myopic_net` 纳入比对：基准不同 ⇒ `ok=False`。
    若确要保留上游 `REV` 基准，则须把"近视基准"**登记为第二个被检验因素**，
    不得再把 `B‑shadow − B‑density` 解释成"内生化"的单独效应。
    审计见 `CC_Compare/P1A_RoleC_效用公式与消融门审计_2026-10-14.md` §2.2。

    ⚠️ **2026-10-14 用户裁定①补充**：除数值 `myopic_net` 外，还比**语义身份**
    `myopic_base`（`components…` / `full_accept_minus_reject_margin` / `revenue_only_legacy`）。
    仅数值偶然相等、而组装方式不同，仍判不可归因 —— 否则"我恰好调成了同一个数"
    会被当成"基准相同"。
    """
    diffs = []
    for k in ('e_o', 'certify_ok', 'budget_B', 'myopic_net', 'myopic_base'):
        a, b = diag_shadow.get(k), diag_density.get(k)
        if a != b:
            diffs.append((k, a, b))
    if diag_shadow.get('lambda_estimator') == diag_density.get('lambda_estimator'):
        diffs.append(('lambda_estimator', diag_shadow.get('lambda_estimator'),
                      diag_density.get('lambda_estimator')))
    return (not diffs), diffs


def factor_status(diag_shadow, diag_density):
    """声明 `B‑shadow − B‑density` 这个差值**能证明什么**（用户裁定①）。

    * `single_factor`：两臂近视基准语义一致（同 `myopic_base`）⇒ 差值可归因于
      **λ 估计方式（内生化）** 这一件事，是**主对照**；
    * `two_factor_exploratory`：两臂近视基准不同（典型：B-density 用 `REV` 上游口径）
      ⇒ 差值同时改变「λ 来源」与「近视基准」两件事，**只能登记为第二因素的探索性消融**，
      **不得**再用它证明"内生化"的独立作用。

    历史结果保持原裁决，**不追溯**改成单因素实验。
    """
    ok, diffs = arms_are_matched(diag_shadow, diag_density)
    if ok:
        return 'single_factor', diffs
    if any(k in ('myopic_net', 'myopic_base') for k, _a, _b in diffs):
        return 'two_factor_exploratory', diffs
    return 'not_comparable', diffs
