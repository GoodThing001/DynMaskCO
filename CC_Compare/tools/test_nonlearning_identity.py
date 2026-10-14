#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`nonlearning_identity` + `qualification_gate` 的 `nonlearning_solver` 联测。

覆盖（纯合成数据、临时文件；**不读任何 gate、不涉及效用**）：
  ① 闭包可独立重算且一致 ⇒ `qualified_full=True`
  ② **篡改**任一依赖文件 ⇒ 独立重算不符 ⇒ 不合格（fail-closed）
  ③ **增删**依赖文件 ⇒ 闭包哈希变化（闭包真的覆盖文件集）
  ④ 依赖清单为空 / 配置为空 ⇒ 不得冒充"已闭合"（返回 None）
  ⑤ 浮点规范化：`10.0` 与 `10.00` 同值；`10` 与 `10.0` **异值**（int/float 可区分）
  ⑥ 文件清单**顺序无关**
  ⑦ 缺 `solver_config_sha256` / 缺共享身份 ⇒ 不合格
  ⑧ 声明 `checkpoint_sha256` ⇒ 不合格（非学习求解器不得伪造 checkpoint）
  ⑨ 端到端接入 `qualification_gate.evaluate` ⇒ 无 checkpoint 也合格；有 fallback 则否决

运行：python CC_Compare/tools/test_nonlearning_identity.py   （退出码 0 = 全过）
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ⚠️ 不要重绑 sys.stdout：`qualification_gate` 不自绑，但保持同一约定，避免双重 TextIOWrapper。
import nonlearning_identity as ni                              # noqa: E402
from qualification_gate import evaluate                        # noqa: E402

SHARED = {'contract_sha256': 'c' * 64, 'data_sha256': 'd' * 64,
          'data_meta_sha256': 'e' * 64}
CFG = {'K': 10, 'time_limit': 10.0, 'arm_seed': 7001, 'future_policy': 'density',
       'shadow_mode': 'greedy', 'energy_pricing': 'marginal', 'overage_price': None,
       'incr_eval': False, 'reject_penalty': [5, 10, 15]}


def _mk(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    return p


def main():
    tmp = tempfile.mkdtemp(prefix='nonlearn_')
    f1 = _mk(tmp, 'solver.py', 'def solve(): pass\n')
    f2 = _mk(tmp, 'provider.py', 'def order(sp): pass\n')
    f3 = _mk(tmp, 'vendor.py', 'VERSION = "1.0"\n')
    files = [f1, f2, f3]

    cases = []

    def ck(name, ok, detail=''):
        cases.append((name, bool(ok), detail))

    ident = ni.build_identity(files, CFG, base=tmp, **SHARED)
    actual = ni.recompute_actual_hashes(files, CFG, base=tmp)

    # ① 一致 ⇒ 通过
    ok, why = ni.verify_closure(ident, files, CFG, base=tmp)
    ck('① 闭包独立重算一致 ⇒ PASS', ok, str(why))
    ck('① 不含 checkpoint_sha256', 'checkpoint_sha256' not in ident)
    ck('① 两个必填字段都被产出',
       bool(ident['provider_dependency_sha256']) and bool(ident['solver_config_sha256']))

    # ② 篡改依赖文件 ⇒ 重算不符
    _mk(tmp, 'solver.py', 'def solve(): return 1  # tampered\n')
    ok2, why2 = ni.verify_closure(ident, files, CFG, base=tmp)
    ck('② 篡改依赖文件 ⇒ 判不合格（fail-closed）', ok2 is False,
       '; '.join(why2[:2]))
    ck('② 篡改后重算哈希确实变了',
       ni.recompute_actual_hashes(files, CFG, base=tmp)['provider_dependency_sha256']
       != actual['provider_dependency_sha256'])
    _mk(tmp, 'solver.py', 'def solve(): pass\n')          # 还原

    # ③ 增删依赖文件 ⇒ 闭包变化
    h3 = ni.dependency_closure_sha256(files, base=tmp)
    h4 = ni.dependency_closure_sha256(files + [f3], base=tmp)
    h5 = ni.dependency_closure_sha256(files[:2], base=tmp)
    ck('③ 加入一个文件 ⇒ 闭包变化', h4 != h3)
    ck('③ 去掉一个文件 ⇒ 闭包变化', h5 != h3)

    # ④ 空清单/空配置 ⇒ None（不得冒充已闭合）
    ck('④ 空依赖清单 ⇒ None', ni.dependency_closure_sha256([], base=tmp) is None)
    ck('④ 空配置 ⇒ None', ni.solver_config_sha256({}) is None)
    ck('④ 空配置进入 verify ⇒ 不合格',
       ni.verify_closure(ident, [], {}, base=tmp)[0] is False)

    # ⑤ 浮点/整型规范化
    ck('⑤ 10.00 与 10.0 同哈希',
       ni.solver_config_sha256({'x': 10.00}) == ni.solver_config_sha256({'x': 10.0}))
    ck('⑤ 10 与 10.0 **异**哈希（int/float 可区分）',
       ni.solver_config_sha256({'x': 10}) != ni.solver_config_sha256({'x': 10.0}))
    ck('⑤ 键顺序无关',
       ni.solver_config_sha256({'a': 1, 'b': 2}) == ni.solver_config_sha256({'b': 2, 'a': 1}))
    ck('⑤ True/False 与 1/0 **异**哈希（bool 不等于 int）',
       ni.solver_config_sha256({'x': True}) != ni.solver_config_sha256({'x': 1}))

    # ⑥ 文件清单顺序无关
    ck('⑥ 依赖文件顺序无关',
       ni.dependency_closure_sha256([f3, f1, f2], base=tmp) == h3)

    # ⑦ 缺字段 ⇒ 不合格
    ck('⑦ 缺 solver_config_sha256 ⇒ 不合格',
       ni.verify_closure(dict(ident, solver_config_sha256=None), files, CFG,
                         base=tmp)[0] is False)
    ck('⑦ 缺共享身份（合同哈希）⇒ 不合格',
       ni.verify_closure(dict(ident, contract_sha256=None), files, CFG,
                         base=tmp)[0] is False)

    # ⑧ 伪造 checkpoint ⇒ 不合格
    ck('⑧ 声明 checkpoint_sha256 ⇒ 不合格（不得伪造）',
       ni.verify_closure(dict(ident, checkpoint_sha256='a' * 64), files, CFG,
                         base=tmp)[0] is False)

    # ⑨ 端到端接入 qualification_gate
    per_day = [dict(i=k, utility=100.0 + k, hard_feasible=True, failures=[],
                    n_served=60, n_rejected=140) for k in range(40)]
    ev = dict(identity=ident, pairing={'identity_verified': True}, per_day=per_day,
              solver_meta={'n_solves': 800, 'n_fail': 0},
              participation={'required': False, 'present': False},
              actual_hashes=actual, method_class='nonlearning_solver')
    r = evaluate(ev)
    ck('⑨ 端到端：无 checkpoint 的非学习求解器 ⇒ qualified_full=True',
       r['qualified'] is True and r['qualified_full'] is True,
       'vetoes=%s' % (r['vetoes'] or '—'))
    ck('⑨ 端到端：required_identity_fields 不含 checkpoint',
       'checkpoint_sha256' not in r['required_identity_fields'])

    r2 = evaluate(dict(ev, actual_hashes=None))
    ck('⑨ 端到端：未提供独立重算 ⇒ qualified_full=False（fail-closed）',
       r2['qualified_full'] is False)
    r3 = evaluate(dict(ev, solver_meta={'n_solves': 800, 'n_fail': 3}))
    ck('⑨ 端到端：有 fallback ⇒ 被 Q_method 否决',
       r3['criteria']['Q_method']['ok'] is False and r3['qualified'] is False)

    passed = sum(1 for _n, ok, _d in cases if ok)
    for n, ok, detail in cases:
        if ok:
            print('  PASS %s' % n)
        else:
            print('  FAIL %s%s' % (n, ('  ← %s' % detail) if detail else ''))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
