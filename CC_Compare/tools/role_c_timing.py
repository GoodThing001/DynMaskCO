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

import io
import json
import os
import signal
import time

import resource_timing as rt
import role_c_resource_ceiling as ceil

__all__ = ['TimingRefused', 'DecisionTimeout', 'CONTENTION_RULE',
           'MAX_EXTERNAL_BUSY_RATIO', 'run_timing', 'validate_preconditions',
           'make_decision_work_fn', 'write_report', 'DecisionWorkFn',
           'affinity_contention', 'run_supervised']

# 竞争判据（**事前写死并记录在报告里**，不是事后挑选的阈值）：
# 实测期间**外部**任务在 affinity 核集上的平均占用比 > 该值 ⇒ 本次计时**不合格**
# （保留资源登记，但**不**产生合格结论；**不得**临场换核集或加 workers）。
MAX_EXTERNAL_BUSY_RATIO = 0.5
# 冻结锚点的**规模**容差（相对）：只用于判"是不是同一规模"，不用于判"完全相等"。
# 实测：`cooling_share` 逐位相同；`B` 相对差 2.5e-8（浮点/版本层面微差）；规模错配是 8 倍量级。
ANCHOR_REL_TOL = 1e-6
ANCHOR_ABS_TOL_SHARE = 1e-9
CONTENTION_RULE = ('external_busy_ratio_on_affinity_set > '
                   'MAX_EXTERNAL_BUSY_RATIO(%.1f) ⇒ qualified=False'
                   % MAX_EXTERNAL_BUSY_RATIO)


class TimingRefused(RuntimeError):
    """前置条件不满足 ⇒ 拒绝计时（fail-closed；不是"耗时偏大"这类软判断）。"""


class DecisionTimeout(RuntimeError):
    """单决策超出 `single_decision_time_limit_s` ⇒ 终止该次并**回收**（计入超时次数）。"""


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


def _read_proc_stat():
    """读 `/proc/stat` 的每核累计 jiffies（Linux）；不可用 ⇒ None。"""
    try:
        with io.open('/proc/stat', encoding='utf-8') as f:
            out = {}
            for line in f:
                if not line.startswith('cpu') or line.startswith('cpu '):
                    continue
                parts = line.split()
                idx = int(parts[0][3:])
                vals = [int(x) for x in parts[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                out[idx] = (sum(vals), idle)
            return out or None
    except Exception:                                         # noqa: BLE001
        return None


def affinity_contention(affinity=None, *, sample_s=1.0):
    """实测 **affinity 核集上"外部"任务的占用比**（用于判"明显竞争"）。

    ⚠️ 为什么不能用 `loadavg`：它反映**整机**负载，**不能**证明被钉住的 0–7 核上没有
    其他任务（用户 2026-10-14 明确指出这一点）。故这里读 `/proc/stat` 的**每核** jiffies
    差值，只统计 affinity 集合内的核。

    返回 `{'available', 'external_busy_ratio', 'reason', 'per_cpu'}`；
    非 Linux（无 `/proc/stat`）⇒ `available=False` 且**如实**写明"无法判定"，
    **不得**当成"无竞争"。
    """
    a = sorted(int(c) for c in (affinity or []))
    if not a:
        return {'available': False, 'external_busy_ratio': None,
                'reason': '未给出 affinity 核集 ⇒ 无法判定'}
    s1 = _read_proc_stat()
    if not s1:
        return {'available': False, 'external_busy_ratio': None,
                'reason': '/proc/stat 不可用（非 Linux？）⇒ **无法判定**是否有竞争；'
                          '不得当作"无竞争"'}
    time.sleep(float(sample_s))
    s2 = _read_proc_stat() or {}
    busy = 0
    total = 0
    per = {}
    for c in a:
        if c not in s1 or c not in s2:
            continue
        dt = (s2[c][0] - s1[c][0])
        di = (s2[c][1] - s1[c][1])
        if dt <= 0:
            continue
        busy += (dt - di)
        total += dt
        per[str(c)] = round((dt - di) / float(dt), 4)
    if total <= 0:
        return {'available': False, 'external_busy_ratio': None,
                'reason': '采样期内无 jiffies 差 ⇒ 无法判定', 'per_cpu': per}
    return {'available': True, 'external_busy_ratio': round(busy / float(total), 4),
            'n_cpus_sampled': len(per), 'per_cpu': per, 'sample_s': float(sample_s),
            'reason': ''}


def _run_with_limit(fn, limit_s):
    """在**硬时限**下跑一次 `fn`（POSIX `SIGALRM`）；返回 `(out, timed_out, elapsed)`。

    非 POSIX（无 `SIGALRM`/`setitimer`）⇒ 退化为**协作式**：只能事后判定"超了"，
    **无法**中断阻塞调用。此时报告里 `enforcement='cooperative'` 且该次计时**不合格**
    （"已登记上限"≠"已执行硬上限"，用户 2026-10-14 明确指出）。
    """
    t0 = time.perf_counter()
    if not (hasattr(signal, 'SIGALRM') and hasattr(signal, 'setitimer')):
        out = fn()
        return out, False, time.perf_counter() - t0, False

    def _handler(_signum, _frame):
        raise DecisionTimeout('单决策超过 %.3fs 硬时限 ⇒ 终止并回收' % limit_s)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(limit_s))
    try:
        out = fn()
        return out, False, time.perf_counter() - t0, True
    except DecisionTimeout:
        return None, True, time.perf_counter() - t0, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def run_timing(*, work_fn, n_decisions, ceiling=None, warmup=0, label='role_c_decision',
               path=None, env=None, affinity=None, env_record_fn=None,
               affinity_fn=None, loadavg_fn=None, decision_points_digest=None,
               contention_check=True, contention_sample_s=1.0,
               max_external_busy_ratio=MAX_EXTERNAL_BUSY_RATIO,
               total_wall_s=None, per_decision_limit_s=None, monotonic=None,
               allow_undigested=False):
    """跑一次**不看效用**的单决策级计时，返回报告 dict（含**超时监督**与**竞争判定**）。

    参数（除新增的监督项外与既有语义一致）
    ----
    work_fn : 可调用，执行**一次决策**（由调用方注入）。返回值会被
        `resource_timing._check_no_utility` 检查 ⇒ 返回效用类字段即抛错。
    n_decisions : 采样次数（>0）。
    ceiling : 已登记的上限 dict；不给则从登记文件读（未登记 ⇒ 抛 `CeilingNotRegistered`）。
    warmup : 预热次数，**单独计次且不进统计**。
    decision_points_digest : **待测决策点集合的指纹**（用户 2026-10-14 要求"计时前固定集合、
        不得在看到某些状态耗时高后剔除"）。若 `work_fn` 自带 `digest()` 则自动取用；
        两者都没有 ⇒ **拒绝计时**（除非显式 `allow_undigested=True`，且报告里如实标注）。
    contention_check : 计时**前**实测 affinity 核集上**外部**任务的占用比
        （`loadavg` 不能证明 0–7 核没有别的任务）。有竞争 ⇒ 报告 `qualified=False`，
        但**保留**资源登记，且**不**临场换核集/加 workers。
    total_wall_s / per_decision_limit_s : 缺省取登记值（43200 / 10.0）。
        **单决策**用 `SIGALRM` **硬中断并回收**；**总墙钟**在每次调用前复查，
        并在单决策时限里取 `min(单决策, 剩余总时间)` ⇒ 总上限也是**硬**的（POSIX）。
    monotonic : 计时时钟（可注入，便于离线测试"超总预算"分支）。

    返回：**无任何效用字段**；含 `per_decision`（mean/P50/P95/max）、
    **完整单次耗时** `records`、`n_timeouts`、真实 `affinity_actual`、`thread_caps_actual`、
    `loadavg_before/after`、`contention`、`enforcement` 与 `qualified`/`unqualified_reasons`。
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
    clock = monotonic or time.monotonic
    per_lim = float(per_decision_limit_s if per_decision_limit_s is not None
                    else c['single_decision_time_limit_s'])
    tot_lim = float(total_wall_s if total_wall_s is not None
                    else c['total_wall_clock_limit_s'])
    # ---- 决策点集合必须**先固定**（不得事后剔除"耗时高的状态"）----
    digest = decision_points_digest
    if digest is None and hasattr(work_fn, 'digest'):
        digest = work_fn.digest()
    if digest is None and not allow_undigested:
        raise TimingRefused(
            '未提供 `decision_points_digest` 且 work_fn 无 `digest()` ⇒ 拒绝计时：'
            '用户 2026-10-14 要求"计时前固定待测决策点集合，不能在看到某些状态耗时较高后'
            '将其剔除"。若确为纯合成夹具，请显式 `allow_undigested=True` 并接受报告中'
            '如实标注 `decision_points_digest=None`。')
    aff = affinity_fn()
    # ---- 计时**前**的竞争检查（loadavg 不足以证明钉住的核没被别人用）----
    contention = {'checked': False}
    if contention_check:
        try:
            contention = affinity_contention(aff, sample_s=contention_sample_s)
        except Exception as exc:                              # noqa: BLE001
            contention = {'available': False, 'external_busy_ratio': None,
                          'reason': '竞争探测失败：%s' % exc}
        contention['checked'] = True
        contention['rule'] = CONTENTION_RULE
        contention['max_external_busy_ratio'] = float(max_external_busy_ratio)
    before = loadavg_fn()
    t_start = clock()
    records = []
    hard = None
    with_timeouts = 0
    overruns = 0
    errors = 0
    total_exceeded = False
    # 预热（不计入统计，但仍受总预算与单决策硬时限约束 —— 预热也不能挂住）
    for _ in range(int(warmup)):
        if clock() - t_start >= tot_lim:
            total_exceeded = True
            break
        remain = max(1e-6, tot_lim - (clock() - t_start))
        try:
            _run_with_limit(work_fn, min(per_lim, remain))
        except DecisionTimeout:
            with_timeouts += 1
        except Exception:                                     # noqa: BLE001
            errors += 1
    for i in range(int(n_decisions)):
        if total_exceeded or (clock() - t_start) >= tot_lim:
            total_exceeded = True
            break
        remain = max(1e-6, tot_lim - (clock() - t_start))
        t0 = clock()
        try:
            out, timed_out, elapsed, _hard = _run_with_limit(work_fn, min(per_lim, remain))
            hard = _hard if hard is None else hard
            if timed_out:
                with_timeouts += 1
                records.append({'i': i, 'elapsed_s': elapsed, 'status': 'timeout',
                                'exceeded_per_decision_limit': True})
                continue
            rt._check_no_utility(out)          # 效用泄漏 ⇒ 立刻抛（不静默）
            over = elapsed > per_lim
            if over:
                overruns += 1
            records.append({'i': i, 'elapsed_s': elapsed, 'status': 'ok',
                            'exceeded_per_decision_limit': over})
        except DecisionTimeout:
            with_timeouts += 1
            records.append({'i': i, 'elapsed_s': clock() - t0, 'status': 'timeout',
                            'exceeded_per_decision_limit': True})
        except rt.UtilityLeakError:
            raise
        except Exception as exc:                              # noqa: BLE001
            errors += 1
            records.append({'i': i, 'elapsed_s': clock() - t0, 'status': 'error',
                            'error': type(exc).__name__,
                            'exceeded_per_decision_limit': False})
    total_elapsed = clock() - t_start
    after = loadavg_fn()
    ok_times = [r['elapsed_s'] for r in records if r['status'] == 'ok']
    rep = {
        'label': label, 'unit': 's', 'warmup_calls': int(warmup),
        'n_requested': int(n_decisions), 'n_completed': len(ok_times),
        'n_timeouts': with_timeouts, 'n_overruns': overruns, 'n_errors': errors,
        'total_wall_exceeded': bool(total_exceeded),
        'total_elapsed_s': total_elapsed,
        'total_wall_limit_s': tot_lim, 'per_decision_limit_s': per_lim,
        'per_decision': rt._stats(ok_times),
        'records': records,                 # **完整单次耗时**（含超时/失败，不剔除）
        'n': int(n_decisions), 'n_decisions': int(n_decisions),
        'ceiling_id': c.get('ceiling_id'), 'occupancy': ceil.occupancy(c),
        'affinity_actual': aff,
        'thread_caps_actual': {k: (env or os.environ).get(k)
                               for k in ceil.THREAD_CAP_FIELDS},
        'live_environment': env_record_fn(),
        'live_environment_problems': [],
        'loadavg_before': before, 'loadavg_after': after,
        'contention': contention,
        'decision_points_digest': digest,
        'enforcement': 'hard_sigalrm' if hard else 'cooperative',
        'note': ('只含时间与资源量；**不含效用**。资源门登记：%s。'
                 '⚠️ 不得据计时结果挑 workers / K / 档位（规格 §6）；'
                 'P95 是资源规划证据，**不是**放宽 10 s 决策硬限制的理由。'
                 % (c.get('ceiling_id'),)),
    }
    rep['live_environment_problems'] = ceil.live_environment_problems(
        c, env=env, affinity=aff)
    reasons = []
    if rep['live_environment_problems']:
        reasons.append('实参与登记不符：%s' % rep['live_environment_problems'])
    if rep['enforcement'] != 'hard_sigalrm':
        reasons.append('enforcement=%s（非 POSIX 硬中断）⇒ "已登记上限"不等于'
                       '"已执行硬上限"' % rep['enforcement'])
    if rep['total_wall_exceeded']:
        reasons.append('总墙钟上限被触及（%.1fs ≥ %.1fs）⇒ 未跑完全部采样'
                       % (total_elapsed, tot_lim))
    if rep['n_completed'] < int(n_decisions):
        reasons.append('完成 %d / 请求 %d' % (rep['n_completed'], int(n_decisions)))
    if contention.get('checked') and contention.get('available'):
        if float(contention['external_busy_ratio']) > float(max_external_busy_ratio):
            reasons.append('affinity 核集上外部占用 %.4f > %.4f ⇒ 有明显竞争'
                           % (contention['external_busy_ratio'],
                              max_external_busy_ratio))
    elif contention_check:
        reasons.append('竞争无法判定（%s）⇒ 不产生合格结论'
                       % (contention.get('reason') or '未知'))
    rep['unqualified_reasons'] = reasons
    rep['qualified'] = not reasons
    rt._check_no_utility(rep)          # 自检：报告本身不得夹带效用
    return rep


class DecisionWorkFn(object):
    """**单决策**计时单元：在**冻结的真实决策点**上重复求值同一个 `decide_fn`。

    为什么这样切：规格 §6 要的是"单决策耗时分布"，而 `decide_fn(order, ctx)` 对给定
    `ctx` 是**幂等**的（它只读 ctx、在副本上做路线搜索、不改 ctx），因此可以反复调用
    同一个真实决策点得到干净的分布 —— 不需要跑整天的轨迹。

    ⚠️ 两个必须说清的边界：
      * `contexts` 必须来自**真实**决策点（`role_c_t_series.run_fixture(capture_n=…)`
        的深拷贝），不是手搓的假状态；
      * 重复同一个状态在缓存局部性上**偏乐观** —— 故默认**轮转多个**决策点
        （`n_contexts` 个）而不是死磕一个；报告里写明用了几 个点。
    返回值**只含行为标签**（动作/是否认证/λ 是否有定义），**不含任何效用** ——
    `resource_timing` 会递归拒绝效用字段。
    """

    def __init__(self, decide_fn, contexts, *, actions_only=True):
        if not callable(decide_fn):
            raise TimingRefused('decide_fn 必须可调用')
        ctxs = [c for c in (contexts or [])]
        if not ctxs:
            raise TimingRefused(
                'contexts 为空 ⇒ 没有真实决策点可测。请先用 '
                'role_c_t_series.run_fixture(..., capture_n=N) 捕获真实上下文；'
                '**不得**用手搓的假状态冒充单决策耗时。')
        self.decide_fn = decide_fn
        self.contexts = ctxs
        self.actions_only = bool(actions_only)
        self.i = 0
        self.calls = 0

    def __len__(self):
        return len(self.contexts)

    def digest(self):
        """**待测决策点集合的指纹**（计时前固定；事后不可剔除"耗时高"的状态）。

        对每个冻结上下文取 `(inst_idx, clock, order, plan, 车辆指纹)` 的规范文本后聚合哈希。
        车辆指纹复用 `role_c_t_series.vehicle_fingerprint`（懒导入，避免模块级循环依赖）。
        """
        import hashlib
        import role_c_t_series as ts
        rows = []
        for c in self.contexts:
            plan = c.get('plan') or {}
            rows.append(json.dumps({
                'inst_idx': c.get('inst_idx'), 'clock': round(float(c.get('clock') or 0), 9),
                'order': c.get('order'),
                'plan': {str(k): list(v) for k, v in sorted(plan.items())},
                'vehicles': [list(x) for x in
                             ts.vehicle_fingerprint(c.get('vehicles') or ())],
            }, sort_keys=True, ensure_ascii=False))
        return hashlib.sha256(('\n'.join(sorted(rows))).encode('utf-8')).hexdigest()

    def __call__(self):
        ctx = self.contexts[self.i % len(self.contexts)]
        self.i += 1
        self.calls += 1
        order = ctx.get('order')
        accept, diag = self.decide_fn(order, ctx)
        d = diag or {}
        ld = d.get('lambda_diag') or {}
        out = {
            'decision_index': self.calls,
            'context_index': (self.i - 1) % len(self.contexts),
            'order_present': order is not None,
            'action': 'accept' if accept else 'reject',
            'certified': bool(d.get('certify_ok')),
            'lambda_defined': bool(ld.get('undefined') is False),
            'lambda_positive': bool(ld.get('lambda_B') not in (None, 0, 0.0)),
            'inner_gate': d.get('inner_gate'),
            'has_forward_diag': d.get('forward') is not None,
            'elapsed_scope': 'one_decision',
        }
        return out


def make_decision_work_fn(decide_fn, contexts, **kw):
    """构造单决策计时单元（见 `DecisionWorkFn`）。"""
    return DecisionWorkFn(decide_fn, contexts, **kw)


def run_supervised(argv, *, total_wall_s, cwd=None, env=None):
    """**外层监督器**：把计时子进程限制在 `total_wall_s` 内，超时**硬杀并回收**。

    为什么需要：进程内的 `SIGALRM` 只能中断**解释器可响应信号**的调用；若子进程卡在
    C 扩展/阻塞 IO 里，只有**外部**超时才能真正终止它。用户 2026-10-14 明确要求
    "由外层监督器强制限制总墙钟时间"。`subprocess.run(timeout=…)` 会在超时后 kill 子进程
    并回收，因此"已登记上限"同时成为"已执行硬上限"。
    """
    import subprocess
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, timeout=float(total_wall_s),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {'rc': p.returncode, 'timed_out': False,
                'elapsed_s': time.monotonic() - t0,
                'stdout_tail': (p.stdout or b'').decode('utf-8', 'replace')[-2000:],
                'stderr_tail': (p.stderr or b'').decode('utf-8', 'replace')[-2000:]}
    except subprocess.TimeoutExpired as exc:
        return {'rc': None, 'timed_out': True, 'elapsed_s': time.monotonic() - t0,
                'stdout_tail': (exc.stdout or b'').decode('utf-8', 'replace')[-2000:]
                if isinstance(exc.stdout, bytes) else (exc.stdout or '')[-2000:],
                'stderr_tail': (exc.stderr or b'').decode('utf-8', 'replace')[-2000:]
                if isinstance(exc.stderr, bytes) else (exc.stderr or '')[-2000:],
                'note': '总墙钟超限 ⇒ 子进程被外层监督器**硬杀并回收**；'
                        '本次计时按未完成处理，不产生合格结论。'}


def write_report(report, path):
    """把计时报告写成 JSON（**写入前再自检一次无效用字段**）。

    拒绝把带效用字段的对象写进计时产物 —— 这是"不看效用"的最后一道机械边界。
    """
    rt._check_no_utility(report)
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with io.open(p, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return p


def build_real_work_fn(*, train_seed=20260925, gate_seed=20260926, n_orders=200,
                       n_vehicles=15, rho=0.6, capture_n=8, arm_seed=7002, K=10,
                       inner_gate='on', time_limit=10.0, inst_idx=0,
                       expect_frozen_anchors=True):
    """按**冻结管线**（`diag_local_timing.py:27-46`）搭出真实计时单元。

    ⚠️ 与冻结配置逐项对齐，否则测的不是要跑的东西：
      * 训练历史 `generate_dataset(200, 200, train_seed=20260925)` → `build_history`；
      * `B, cooling_share = compute_budget_and_dwell(train_ds, contract, 15, train_seed,
        0.6, hist_tc, k=10)` —— **必须复现冻结锚点**
        `B=710.6548152249378` / `cooling_share=3.7208946347769882`（`expect_frozen_anchors`
        为真时不一致即抛错：宁可失败，也不要拿一条不确定的管线去测时间）；
      * 实例 = **单实例**（`generate_dataset(1, n_orders, gate_seed=20260926)`）
        + `add_v2_initial_quality`；
      * `StrictOnlineEnv(..., num_vehicles=15, tw_speed=SPEED_KMH/KM_PER_UNIT,
        booking_horizon=BOOKING_HORIZON)`。

    返回 `{'work_fn', 'budget', 'cooling_share', 'n_records', 'n_served', ...}`。
    **只建单元与捕获决策点，不产生任何耗时结论**（计时由 `run_timing` 做）。
    """
    import role_c_harness as hn
    import role_c_t_series as ts
    ts.ensure_project_paths()
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from run_exp_reserve import (BOOKING_HORIZON, KM_PER_UNIT, SPEED_KMH,
                                 generate_dataset)
    from scenario_saa import (CondHistoricalSampler, build_history,
                              build_history_times_classes, make_c0_contract_v2)

    contract = make_c0_contract_v2()
    train_ds = generate_dataset(int(n_orders), int(n_orders), int(train_seed))
    hist = build_history(train_ds)
    hist_tc = build_history_times_classes(train_ds)
    B, share, _ = compute_budget_and_dwell(train_ds, contract, int(n_vehicles),
                                          int(train_seed), float(rho), hist_tc, k=10)
    if expect_frozen_anchors:
        from role_c_identity import frozen_anchors
        a = frozen_anchors()
        fb, fs = float(a['budget_B']), float(a['cooling_share'])
        rel_b = abs(float(B) - fb) / max(1e-12, abs(fb))
        abs_s = abs(float(share) - fs)
        # ⚠️ 用**相对**容差判"是否同一规模"，并把**实测值/冻结值/两个差值**一并带回报告 ——
        #    不把差异藏在容差里。本地实测：`cooling_share` 逐位相同，
        #    `B` 相差 1.76e-5（相对 2.5e-8），属浮点/版本层面的微差而非规模错配
        #    （错配是 8 倍量级：89.6 vs 710.7）。
        if rel_b > ANCHOR_REL_TOL or abs_s > ANCHOR_ABS_TOL_SHARE:
            raise TimingRefused(
                '预算/冷却份额未复现冻结锚点 ⇒ 拒绝用这条管线测时间：'
                'B=%.13f（冻结 %.13f，相对差 %.3e）、cooling_share=%.13f（冻结 %.13f，'
                '绝对差 %.3e）' % (float(B), fb, rel_b, float(share), fs, abs_s))
        anchor_check = {'frozen_B': fb, 'witnessed_B': float(B), 'rel_gap_B': rel_b,
                        'frozen_cooling_share': fs,
                        'witnessed_cooling_share': float(share),
                        'abs_gap_cooling_share': abs_s,
                        'rel_tol_B': ANCHOR_REL_TOL,
                        'abs_tol_cooling_share': ANCHOR_ABS_TOL_SHARE,
                        'note': '相对容差只用于判"同一规模"；实测差值如实报出，未隐藏'}
    else:
        anchor_check = {'checked': False,
                        'reason': 'expect_frozen_anchors=False（**仅管线自检**用）'}
    gate_ds = add_v2_initial_quality(generate_dataset(1, int(n_orders), int(gate_seed)),
                                    contract)
    sampler = CondHistoricalSampler(hist, n_neighbors=10)
    dfn = hn.build_decide_fn(sampler=sampler, arm_seed=int(arm_seed), K=int(K),
                            contract=contract, cooling_share=float(share),
                            booking_horizon=BOOKING_HORIZON, budget=float(B),
                            inner_gate=inner_gate, time_limit=float(time_limit))
    run = ts.run_fixture(decide_fn=dfn, ds=gate_ds, contract=contract, budget=float(B),
                         capacity=50.0, num_vehicles=int(n_vehicles),
                         booking_horizon=BOOKING_HORIZON, cooling_share=float(share),
                         inst_idx=int(inst_idx), capture_n=int(capture_n),
                         tw_speed=SPEED_KMH / KM_PER_UNIT)
    wf = make_decision_work_fn(dfn, run.get('contexts') or [])
    return {'work_fn': wf, 'budget': float(B), 'cooling_share': float(share),
            'n_records': len(run['records']), 'n_served': run['n_served'],
            'n_contexts': len(wf), 'gate_seed': int(gate_seed),
            'train_seed': int(train_seed), 'n_orders': int(n_orders),
            'n_vehicles': int(n_vehicles), 'arm_seed': int(arm_seed), 'K': int(K),
            'inner_gate': inner_gate, 'inst_idx': int(inst_idx),
            'anchor_check': anchor_check}


def main(argv=None):
    """CLI：**只测时间与资源量**的单决策计时（登记未通过 ⇒ 拒绝运行）。

        python CC_Compare/tools/role_c_timing.py --out <report.json> \\
            --n-decisions 200 --capture-n 8 [--dry-run]

    `--dry-run` 只搭单元与捕获决策点（**不产生耗时结论**）。
    超时监督与竞争检查始终生效；**不改核集、不加 workers、不挑参数**。
    退出码：0 = 合格；5 = 跑完但不合格（原因见 `unqualified_reasons`）。
    """
    import argparse
    ap = argparse.ArgumentParser(description='P1-A Role C 单决策资源计时（不看效用）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-decisions', type=int, default=200)
    ap.add_argument('--capture-n', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--n-orders', type=int, default=200)
    ap.add_argument('--n-vehicles', type=int, default=15)
    ap.add_argument('--train-seed', type=int, default=20260925)
    ap.add_argument('--gate-seed', type=int, default=20260926)
    ap.add_argument('--arm-seed', type=int, default=7002)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--inner-gate', default='on')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    built = build_real_work_fn(train_seed=a.train_seed, gate_seed=a.gate_seed,
                              n_orders=a.n_orders, n_vehicles=a.n_vehicles,
                              capture_n=a.capture_n, arm_seed=a.arm_seed, K=a.K,
                              inner_gate=a.inner_gate)
    print(json.dumps({'built': {k: v for k, v in built.items() if k != 'work_fn'},
                      'work_fn_len': len(built['work_fn']),
                      'decision_points_digest': built['work_fn'].digest()},
                     ensure_ascii=False, indent=2))
    if a.dry_run:
        print('DRY-RUN: 只搭单元与捕获决策点，**未产生任何耗时结论**')
        return 0
    rep = run_timing(work_fn=built['work_fn'], n_decisions=a.n_decisions,
                     warmup=a.warmup, label='role_c_decision_%dorders' % a.n_orders)
    rep['work_fn_build'] = {k: v for k, v in built.items() if k != 'work_fn'}
    p = write_report(rep, a.out)
    print('WROTE: %s' % p)
    print(json.dumps({'n_completed': rep['n_completed'], 'n_timeouts': rep['n_timeouts'],
                      'n_overruns': rep['n_overruns'],
                      'per_decision': rep['per_decision'],
                      'enforcement': rep['enforcement'], 'qualified': rep['qualified'],
                      'unqualified_reasons': rep['unqualified_reasons']},
                     ensure_ascii=False, indent=2))
    return 0 if rep['qualified'] else 5


if __name__ == '__main__':
    import sys
    sys.exit(main())
