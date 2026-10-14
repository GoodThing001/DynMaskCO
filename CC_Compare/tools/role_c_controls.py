#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**对照臂**：`B-density`（外生背包密度阈值 λ）。

为什么需要（`P1A_RoleC_设计规格` §2.6，2026-10-14 读源码发现）：
Role C 的判决式 `myopic_net − λ·e > 0` 与既有 `knn` / `valuee20/30` **形式相同**
（`run_exp_energy_c0.py:358-380`）。因此只做 `B − A`（`λ_B ≡ 0`）**不足以**支撑
"资源机会成本"这一机制主张 —— 它排除的是"可行即接"，而真正的对手是
"**一个调好的密度阈值**"。

⇒ 必须增加 `B-density`：λ 用既有臂的**背包密度阈值**（外生，单位 revenue/kWh），
其余（场景、路线修复、判据、预算）与 `B-shadow` 全同。比较 **`B‑shadow − B‑density`**。
若该差 ≈ 0，正确表述是"**资源门槛机制**有用、**内生化**无增量"。

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
                             future_vals, future_energys, remaining_budget=None):
    """`B-density` 对照臂：用**外生密度阈值** λ 判决 `REV − λ·e > 0 ∧ certify`。

    `future_vals` / `future_energys` = 场景集合里**未来订单**的收益与边际能耗
    （由 runner 从冻结的场景机制取出，保证与 `B-shadow` **同源同 K**）。
    `remaining_budget` 缺省为 `budget_B`。
    """
    e_o = float(marginal_energy_fn(order))
    rem = float(budget_B if remaining_budget is None else remaining_budget)
    lam = density_threshold_lambda(future_vals, future_energys, rem)
    rev = float(revenue_fn(order))
    certify_ok, certify_energy = certify_fn(candidate_plan)
    # 复用同一判决实现（`decide`），只是 λ 的来源不同 ⇒ 两臂只差这一件事
    accept, diag = decide(myopic_net=rev, e_o=e_o, lambda_B=lam, certify_ok=certify_ok)
    diag.update({
        'arm': 'B_density_control',
        'lambda_estimator': LAMBDA_ESTIMATOR_DENSITY,
        'lambda_diag': {'lambda_B': lam, 'lambda_source': 'density_threshold_knapsack',
                        'n_future_orders': int(len(np.asarray(future_vals))),
                        'remaining_budget': rem},
        'certify_energy': certify_energy,
        'budget_B': float(budget_B),
        'myopic_net_note': 'REV − λ·e（本对照臂用 REV；fuel/reject_loss 属评价口径，'
                           '**不**进判决 —— 与既有 knn/valuee 一致）',
    })
    return accept, diag


def arms_are_matched(diag_shadow, diag_density):
    """核对两臂**只差 λ 来源**（其余全同）：用于 `B‑shadow − B‑density` 的可归因性检查。

    返回 `(ok, diffs)`；`diffs` 列出除 λ 之外**不应存在**的差异。
    """
    diffs = []
    for k in ('e_o', 'certify_ok', 'budget_B'):
        a, b = diag_shadow.get(k), diag_density.get(k)
        if a != b:
            diffs.append((k, a, b))
    if diag_shadow.get('lambda_estimator') == diag_density.get('lambda_estimator'):
        diffs.append(('lambda_estimator', diag_shadow.get('lambda_estimator'),
                      diag_density.get('lambda_estimator')))
    return (not diffs), diffs
