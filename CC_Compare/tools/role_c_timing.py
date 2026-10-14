#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**不看效用**的资源计时器（规格 §6）。

它只做三件事，且每一件都有**机械边界**：

  1. **先过资源门**：登记未通过校验 ⇒ 抛 `CeilingNotRegistered`，**拒绝运行**
     （规格 §6："资源上限登记完成之后，才做资源计时"）；
  2. **核对实参**：进程环境的四项线程上限与 CPU 亲和性必须与登记**一致**，
     不一致 ⇒ 抛 `TimingRefused` —— 否则测出来的耗时**不可解释**
     （资格矩阵 6 个受争用臂的教训：配置层面与 v1 完全相同、从记录里看不出差异）；
  3. **只记时间与资源量**：逐决策墙钟 → `measure_series` 的
     `per_decision`（mean/P50/**P95**/max），`warmup` 单独计次且**不进统计**；
     报告里**不得**出现任何效用类字段（`resource_timing` 会递归拒绝 ⇒ `UtilityLeakError`）。

⚠️ **本模块不挑选任何参数**：不据计时结果挑 `workers`/`K`/档位。计时只产出
"在当前登记配置下，单决策耗时是多少"这一件事。

用法（登记完成之后）：

    python -c "import role_c_timing; ..."   # 或由 runner 调用 run_timing(work_fn=...)

`work_fn` 是"跑一次决策"的可调用对象，由调用方注入（本模块**不**自己构造 env）。
"""
from __future__ import annotations

import os

import resource_timing as rt
import role_c_resource_ceiling as ceil

__all__ = ['TimingRefused', 'run_timing', 'validate_preconditions']


class TimingRefused(RuntimeError):
    """前置条件不满足 ⇒ 拒绝计时（fail-closed；不是"耗时偏大"这类软判断）。"""


def _default_loadavg():
    try:
        return [round(x, 3) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def validate_preconditions(ceiling=None, *, env=None, affinity=None, path=None):
    """校验资源门与实参。返回 `(ceiling, problems)`；**不抛**（便于先看原因）。"""
    if ceiling is None:
        try:
            ceiling = ceil.load_ceiling(path)
        except ceil.CeilingNotRegistered as exc:
            return None, [str(exc)]
    ok, problems = ceil.validate_ceiling(ceiling)
    if not ok:
        return ceiling, problems
    problems = list(problems) + ceil.live_environment_problems(
        ceiling, env=env, affinity=affinity)
    return ceiling, problems


def run_timing(*, work_fn, n_decisions, ceiling=None, warmup=0, label='role_c_decision',
               path=None, env=None, affinity=None, env_record_fn=None,
               affinity_fn=None, loadavg_fn=None):
    """跑一次**不看效用**的单决策级计时，返回报告 dict。

    参数
    ----
    work_fn : 可调用，执行**一次决策**（由调用方注入：真实 runner 或合成夹具）。
        其返回值会被 `resource_timing._check_no_utility` 检查 ⇒ 返回效用类字段即抛错。
    n_decisions : 采样次数（>0）。
    ceiling : 已登记的上限 dict；不给则从登记文件读（未登记 ⇒ 抛 `CeilingNotRegistered`）。
    warmup : 预热次数，**单独计次且不进统计**（排除编译/缓存冷启动污染稳态耗时）。
    env / affinity / env_record_fn / affinity_fn / loadavg_fn : 可注入，便于离线测试。

    返回：`{ceiling_id, occupancy, per_decision, n, timings, environment,
            live_environment, loadavg_before, loadavg_after, note, ...}` —— 无任何效用字段。
    """
    if not callable(work_fn):
        raise TimingRefused('work_fn 必须可调用（执行一次决策）')
    if not isinstance(n_decisions, int) or isinstance(n_decisions, bool) \
            or n_decisions < 1:
        raise TimingRefused('n_decisions 必须是 >=1 的 int，得到 %r' % (n_decisions,))
    c, problems = validate_preconditions(ceiling, env=env, affinity=affinity, path=path)
    if problems:
        raise TimingRefused(
            '拒绝计时 —— 前置条件不满足（规格 §6："资源上限登记完成之后"）：\n  - '
            + '\n  - '.join(problems))
    env_record_fn = env_record_fn or rt.environment_record
    affinity_fn = affinity_fn or (lambda: sorted(os.sched_getaffinity(0))
                                  if hasattr(os, 'sched_getaffinity') else None)
    loadavg_fn = loadavg_fn or _default_loadavg

    before = loadavg_fn()
    rep = rt.measure_series(work_fn, n=int(n_decisions), label=label, warmup=int(warmup))
    after = loadavg_fn()
    rep.update({
        'ceiling_id': c.get('ceiling_id'),
        'n': int(n_decisions),
        'n_decisions': int(n_decisions),
        'occupancy': ceil.occupancy(c),
        'affinity_actual': affinity_fn(),
        'live_environment': env_record_fn(),
        'live_environment_problems': [],
        'loadavg_before': before,
        'loadavg_after': after,
        'note': ('只含时间与资源量；**不含效用**。资源门登记：%s。'
                 '⚠️ 不得据计时结果挑 workers / K / 档位（规格 §6）。'
                 % (c.get('ceiling_id'),)),
    })
    # 再核一次实参：**跑完之后**进程环境仍须与登记一致（防中途被改）
    rep['live_environment_problems'] = ceil.live_environment_problems(
        c, env=env, affinity=affinity_fn())
    if rep['live_environment_problems']:
        raise TimingRefused('计时过程中环境与登记不符 ⇒ 该次耗时不可解释：\n  - '
                            + '\n  - '.join(rep['live_environment_problems']))
    rt._check_no_utility(rep)          # 自检：报告本身不得夹带效用
    return rep
