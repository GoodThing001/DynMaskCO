#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**不看效用**的资源计时 harness（P1-A §6）。

授权边界（2026-10-14 用户裁决）：**允许**做不看效用的资源计时；
**禁止**据计时结果挑参数/K，也**禁止**把效用读进计时记录。

本模块把"不看效用"做成**机械边界**，而不是靠自觉：

  * `measure()` 只接受一个可调用对象，**只**返回时间/资源统计；
  * 若该可调用对象返回的对象里出现**效用类字段**
    （`utility` / `profit` / `revenue` / `fuel` / `reject_loss` / `energy` / `diff` …），
    **直接抛错**（`UtilityLeakError`）—— 计时环节**根本不允许**效用进入记录；
  * 输出**不含**任何效用数值，只含耗时分布与资源/线程配置。

为什么必须记录**线程与设备配置**：资格矩阵里 6 个"受 CPU 争用"的臂，
根因是 **worker 数 × 每 worker 线程数**导致机器被过度订阅
（`lih_51015` 曾达 54.225 次超时/天、单决策 50.79 s）。
⇒ 只报"耗时"而不报线程/设备，是无法解释的；`environment_record()` 必须随记录一起落盘。

⚠️ 本模块**不跑任何实验**：它只是测量工具，由持有授权的 runner 调用。
"""
from __future__ import annotations

import os
import statistics
import time

# 计时记录里**禁止出现**的字段名（效用类）——子串匹配，小写比较
UTILITY_LIKE = ('utility', 'profit', 'revenue', 'fuel', 'reject_loss', 'rejectloss',
                'energy', 'objective', 'j_cc', 'jcc', 'diff', 'delta', 'gain')


class UtilityLeakError(RuntimeError):
    """计时结果里出现了效用类字段 ⇒ 拒绝（计时环节不得读效用）。"""


def _check_no_utility(result):
    """递归检查返回值里是否含效用类键；命中即抛错。

    允许返回：标量、None、或**不含**效用类键的 dict/list（例如内部诊断）。
    """
    bad = []

    def walk(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if any(t in kl for t in UTILITY_LIKE):
                    bad.append('%s%s' % (path, k))
                walk(v, '%s%s.' % (path, k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, '%s[%d].' % (path, i))
        elif isinstance(obj, (set, frozenset)):
            for v in obj:
                walk(v, '%s{}.' % path)

    walk(result)
    if bad:
        raise UtilityLeakError(
            '计时环节检测到**效用类字段** %s ⇒ 拒绝记录。'
            '资源计时**只**测时间与资源量（P1-A §6）；若确实需要效用，'
            '那是另一套流程，不得混进计时记录。' % sorted(set(bad))[:8])


def environment_record():
    """硬件/线程/设备配置（**必须**随计时记录一起落盘，否则耗时无法解释）。"""
    rec = {
        'cpu_count': os.cpu_count(),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'XLA_PYTHON_CLIENT_PREALLOCATE': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
    }
    try:                                    # Linux：实际可用核数（cgroup/亲和性可能更小）
        rec['sched_getaffinity'] = len(os.sched_getaffinity(0))
    except Exception:                       # noqa: BLE001  （Windows 无此接口）
        rec['sched_getaffinity'] = None
    return rec


def _stats(xs):
    ys = sorted(float(x) for x in xs)
    if not ys:
        return None
    n = len(ys)

    def pct(p):
        # 最近秩法（不插值）——计时分布用最近秩更直观，且不因插值产生"漂亮"数字
        import math
        k = max(0, min(n - 1, int(math.ceil(p / 100.0 * n)) - 1))
        return ys[k]

    return {'n': n, 'mean': statistics.fmean(ys), 'min': ys[0], 'p50': pct(50),
            'p95': pct(95), 'max': ys[-1]}


def measure(fn, *, warmup=0, repeats=1, label='', per_call_args=None):
    """测量 `fn` 的墙钟耗时，返回**不含效用**的报告。

    `fn()` 的返回值会被 `_check_no_utility` 检查（效用类字段 ⇒ 抛错）；
    `warmup` 次不计入统计（用于排除 JIT/编译/缓存冷启动 —— 项目里这一步必须做，
    否则"稳态耗时"会被首次编译严重污染）。
    """
    if repeats <= 0:
        raise ValueError('repeats 必须 > 0')
    for _ in range(int(warmup)):
        _check_no_utility(fn() if per_call_args is None else fn(**per_call_args))
    xs, leak = [], None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        out = fn() if per_call_args is None else fn(**per_call_args)
        xs.append(time.perf_counter() - t0)
        try:
            _check_no_utility(out)
        except UtilityLeakError as exc:      # 记录下来但仍然抛（不静默）
            leak = exc
            break
    if leak is not None:
        raise leak
    rep = {'label': label, 'unit': 's', 'warmup_calls': int(warmup),
           'repeats': int(repeats), **_stats(xs)}
    rep['timings'] = xs
    rep['environment'] = environment_record()
    rep['note'] = ('只含时间与资源量；**不含任何效用**。'
                   '⚠️ 不得据计时结果挑参数/K（P1-A §6）。')
    return rep


def measure_series(fn, *, n, label='', warmup=0):
    """对 `fn` 连续调用 `n` 次、**逐次**计时（用于"单决策耗时分布"）。

    与 `measure` 的区别：`measure` 关心"这一次调用多快"，本函数关心
    "每次调用的耗时分布"（规格要求报告 **平均与 P95 单决策耗时**）。
    """
    if n <= 0:
        raise ValueError('n 必须 > 0')
    for _ in range(int(warmup)):
        _check_no_utility(fn())
    xs = []
    for _ in range(int(n)):
        t0 = time.perf_counter()
        out = fn()
        xs.append(time.perf_counter() - t0)
        _check_no_utility(out)
    return {'label': label, 'unit': 's', 'warmup_calls': int(warmup),
            'per_decision': _stats(xs), 'timings': xs,
            'environment': environment_record(),
            'note': '单决策耗时分布（mean/P95）；**不含效用**。'}


def _self_test():
    """自检：统计量形状正确 + **效用泄漏必被拒**。"""
    calls = {'n': 0}

    def ok():
        calls['n'] += 1
        return {'diag': calls['n']}          # 非效用字段 ⇒ 允许

    rep = measure(ok, warmup=2, repeats=5, label='selftest')
    print('  measure: n=%d（期望 5）warmup 调用被排除=%s'
          % (rep['n'], calls['n'] - rep['n'] == 2))
    assert rep['n'] == 5 and calls['n'] == 7
    assert rep['min'] <= rep['p50'] <= rep['p95'] <= rep['max']
    print('  统计量单调 min≤p50≤p95≤max：OK')

    def leaky():
        return {'utility': 123.0}

    try:
        measure(leaky, repeats=1)
        print('  ✗ 效用泄漏未被拒绝'); return 1
    except UtilityLeakError as e:
        print('  ✅ 效用泄漏被拒：%s' % str(e)[:60])
    try:
        measure(lambda: {'nested': {'revenue': 1.0}}, repeats=1)
        print('  ✗ 嵌套效用泄漏未被拒绝'); return 1
    except UtilityLeakError:
        print('  ✅ 嵌套效用泄漏被拒')
    ser = measure_series(lambda: None, n=3, label='s')
    assert ser['per_decision']['n'] == 3
    print('  measure_series: per_decision n=3 OK')
    print('  environment_record keys=%s' % sorted(environment_record().keys()))
    print('\nSELF_TEST_PASS')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_self_test())
