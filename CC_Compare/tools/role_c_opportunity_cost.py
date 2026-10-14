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


def decide_candidate(*, order, candidate_plan, budget_B,
                     myopic_net_fn, marginal_energy_fn, future_value_fn, certify_fn,
                     ablate_opportunity_cost=False, ablate_lookahead=False):
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
    if ablate_lookahead:
        lam, lam_diag = 0.0, {'lambda_raw': 0.0, 'lambda_B': 0.0, 'e_o': e_o,
                              'lambda_source': 'ablate_lookahead',
                              'reason': '无前瞻 ⇒ Ĵ 无从估计 ⇒ λ_B 置 0（与 A 版同为 0，但机制不同）'}
    else:
        lam, lam_diag = shadow_price_budget(budget_B, e_o, future_value_fn)
        lam_diag['lambda_source'] = ('opportunity_cost' if not ablate_opportunity_cost
                                     else 'ablate_opportunity_cost')
    mn = float(myopic_net_fn(order))
    # §2.4 顺序：候选评估（含 λ_B）→ 认证 → 提交
    certify_ok, certify_energy = certify_fn(candidate_plan)
    accept, diag = decide(myopic_net=mn, e_o=e_o, lambda_B=lam, certify_ok=certify_ok,
                          ablate_opportunity_cost=ablate_opportunity_cost)
    diag.update({
        'lambda_diag': lam_diag,
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
