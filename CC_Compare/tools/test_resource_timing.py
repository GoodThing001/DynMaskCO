#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`resource_timing` 的回归（**不看效用**是机械边界，必须被证明）。

最有价值的一条：**效用泄漏必须被拒绝**（含嵌套结构、大小写变体），
否则"不看效用"就只是一句自觉承诺。

纯合成调用；**不读效用、不跑实验**。
运行：python CC_Compare/tools/test_resource_timing.py   （退出码 0 = 全过）
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resource_timing as rt                                  # noqa: E402


def main():
    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    # ---------- ① 统计量形状与调用计数 ----------
    calls = {'n': 0}

    def ok():
        calls['n'] += 1
        return {'diag': 1}

    rep = rt.measure(ok, warmup=2, repeats=5, label='t')
    ck('① 调用次数 = warmup + repeats（warmup 不计入统计）',
       calls['n'] == 7 and rep['n'] == 5, 'calls=%d n=%d' % (calls['n'], rep['n']))
    ck('① 统计量单调 min ≤ p50 ≤ p95 ≤ max',
       rep['min'] <= rep['p50'] <= rep['p95'] <= rep['max'])
    ck('① 报告含环境/线程记录（否则耗时无法解释）',
       'environment' in rep and 'cpu_count' in rep['environment'])
    ck('① 报告自述不含效用', '不含' in rep.get('note', '') and '效用' in rep.get('note', ''))
    ck('① repeats<=0 ⇒ 拒绝', _raises(lambda: rt.measure(ok, repeats=0)))

    # ---------- ② ★ 效用泄漏必须被拒（这是"不看效用"的机械保证）----------
    for bad in ({'utility': 1.0}, {'profit': 2.0}, {'REVENUE': 3.0}, {'j_cc': 1.0},
                {'a': {'utility': 1.0}}, {'a': [{'gain': 1.0}]},
                {'energy': 5.0}, {'delta': 0.1}, {'reject_loss': 0.0}):
        ck('② 效用类字段被拒：%r' % (bad,), _raises(lambda b=bad: rt.measure(
            lambda: b, repeats=1), rt.UtilityLeakError))
    # 允许的返回：标量 / None / 非效用 dict
    for good in (None, 1.0, {'diag': 1}, {'n_solves': 3}, {'elapsed': 0.1}, [1, 2]):
        ck('② 非效用返回被允许：%r' % (good,),
           not _raises(lambda g=good: rt.measure(lambda: g, repeats=1)))
    ck('② warmup 阶段的效用泄漏同样被拒（不能只查统计阶段）',
       _raises(lambda: rt.measure(lambda: {'utility': 1.0}, warmup=1, repeats=1),
               rt.UtilityLeakError))

    # ---------- ③ measure_series：逐次计时 ----------
    ser = rt.measure_series(lambda: {'diag': 1}, n=4, warmup=1, label='s')
    ck('③ measure_series 逐次 n 条', ser['per_decision']['n'] == 4
       and len(ser['timings']) == 4)
    ck('③ measure_series n<=0 ⇒ 拒绝', _raises(lambda: rt.measure_series(lambda: None, n=0)))
    ck('③ measure_series 也查效用',
       _raises(lambda: rt.measure_series(lambda: {'utility': 1.0}, n=1),
               rt.UtilityLeakError))

    # ---------- ④ 计时确实是计时（对已知睡眠的调用）----------
    def sleep_1ms():
        time.sleep(0.001)
        return None

    rep = rt.measure(sleep_1ms, warmup=0, repeats=3, label='sleep')
    ck('④ 对 sleep(1ms) 的测量落在合理区间（0.8–20 ms）',
       all(0.0008 <= x <= 0.02 for x in rep['timings']), rep['timings'])

    # ---------- ⑤ 自检也能跑 ----------
    rc = rt._self_test()
    ck('⑤ 模块自带 _self_test 返回 0', rc == 0)

    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


def _raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:                                          # noqa: BLE001
        return False


if __name__ == '__main__':
    sys.exit(main())
