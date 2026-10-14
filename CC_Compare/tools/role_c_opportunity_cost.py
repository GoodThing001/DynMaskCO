#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**预算机会成本估计器**与接单判据（纯函数；不读效用、不跑实验）。

依据 `CC_Compare/P1A_RoleC_设计规格_2026-10-14.md` §2.1 / §2.1.1（2026-10-14 用户纠错后）：

    写法 (i)【本模块，唯一允许】：
        myopic_net(o) = revenue_o − fuel_o − reject_loss_o        # 只含"接这一单"的直接净收益
        λ_B(o)        = clip( [Ĵ(B) − Ĵ(B − e_o)] / e_o , 0, +∞ )  # 预算边际价值，≥0
        接单 ⟺ myopic_net(o) − λ_B(o)·e_o > 0   且   certify_plan ≤ B

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
DOUBLE_COUNT_REFUSAL = (
    '禁止重复计机会成本：不得把前瞻价值差（lookahead_delta）与 λ_B·e_o **同时**扣减 '
    '（规格 §2.1.1 写法 (iii)）。请二选一：'
    '写法 (i) 传 myopic_net + lambda_B（本规格）；或写法 (ii) 只传 V̂(接单)−V̂(拒单)，'
    '此时不要再传 λ_B。')


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
        return 0.0, {'lambda_raw': 0.0, 'lambda_B': 0.0, 'e_o': e,
                     'v_hi': None, 'v_lo': None, 'b_lo': float(budget),
                     'infeasible': False, 'reason': 'e_o<=0 ⇒ 无占用，机会成本恒为 0'}
    b = float(budget)
    b_lo = b - e
    infeasible = b_lo < 0.0
    b_lo_eval = max(b_lo, 0.0)
    v_hi = float(future_value_fn(b))
    v_lo = float(future_value_fn(b_lo_eval))
    raw = (v_hi - v_lo) / e
    lam = max(raw, 0.0) if clip_nonneg else raw
    return lam, {'lambda_raw': raw, 'lambda_B': lam, 'e_o': e,
                 'v_hi': v_hi, 'v_lo': v_lo, 'b_lo': b_lo,
                 'b_lo_eval': b_lo_eval, 'infeasible': infeasible,
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
    """`revenue − fuel − reject_loss`（规格 §2.1 左式；**不得**在此再扣机会成本）。"""
    return float(revenue) - float(fuel) - float(reject_loss)
