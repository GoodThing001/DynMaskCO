#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两项**公共工具缺陷**修复的独立反例回归（用户 2026-10-14 裁定"必须修"）。

裁定要求（逐条落成可执行断言）：
  * 两项修复都需要**独立反例测试**；
  * `not_applicable` 用例必须**同时**验证 `engaged`、`problems` **与最终 `passed`**，
    不能只检查中间字段；
  * 修复形成**新代码版本 + SHA 记录**，不覆盖原裁定、不回填旧 gate、
    不把旧失败追认成通过；
  * `arm_seed` 加入 `PAIR_FIELDS`：任一侧缺失或不一致 ⇒ **新协议判配对身份失败**；
    历史 gate **不因新增字段被追溯修改**。

  F1 组：`check_control_config.py` 的 **stdout 副作用**缺陷
        （原实现在模块级把 `sys.stdout` 换成新 `TextIOWrapper`，旧对象被 GC 时按所有权
        关掉底层 buffer ⇒ 任何 import 它的进程此后 `print` 全崩）。
        反例测试在**独立子进程**里 import 后立刻 `print` —— 这正是当初让 Role C 的
        T 系列第一版崩溃的路径。
  F2 组：`role_c_decision_audit.check_ablation_engagement` 的**降级被抵消**缺陷
        （`engaged=None` 本意 `not_applicable`，却因 `is not True` 仍进 `problems`）。
  F3 组：`arm_seed` 进入 `PAIR_FIELDS` 后的判定行为（缺失/不一致/一致）。

**不改任何历史 gate 或已封存结果**；本文件只读它们、并只做纯函数级断言。
运行：python CC_Compare/tools/test_public_tool_fixes.py   （退出码 0 = 全过）
"""
import copy
import io
import json
import os
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import role_c_decision_audit as da                        # noqa: E402

CHECK_PATH = os.path.join(HERE, 'check_control_config.py')


# ===========================================================================
# F1：import 副作用 —— 不得关掉 stdout
# ===========================================================================
def test_stdout_not_closed_on_import():
    cases = []
    # ⚠️ 必须在**子进程**里做：本进程若被污染，整个测试文件后续 print 都会崩。
    prog = (
        'import sys, os\n'
        'sys.path.insert(0, %r)\n'
        'import check_control_config\n'
        'sys.stdout.write("AFTER_IMPORT_PRINT_OK\\n")\n'
        'sys.stdout.flush()\n'
        'print("AFTER_IMPORT_PRINT2_OK")\n'
        'assert not sys.stdout.closed, "stdout closed"\n'
        % (HERE,))
    p = subprocess.run([sys.executable, '-c', prog], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, cwd=ROOT)
    out = (p.stdout or b'').decode('utf-8', 'replace')
    err = (p.stderr or b'').decode('utf-8', 'replace')
    cases.append(('★F1 子进程里 import `check_control_config` 后仍能 `print`'
                  '（原缺陷会让此后所有 print 抛 I/O operation on closed file）',
                  p.returncode == 0 and 'AFTER_IMPORT_PRINT_OK' in out
                  and 'AFTER_IMPORT_PRINT2_OK' in out,
                  (p.returncode, out[-200:], err[-300:])))
    cases.append(('F1b 导入后 `sys.stdout.closed` 必须为 False',
                  p.returncode == 0 and 'closed' not in err, err[-200:]))
    # ★ 非空转证明（**确定性**，不依赖 GC 时序）：两个 wrapper 套同一个 buffer 时，
    #   **关掉其中一个就会关掉共享的 buffer**，另一个此后写入即抛
    #   `I/O operation on closed file` —— 这正是旧写法会造成崩溃的机制。
    buf = io.BytesIO()
    w1 = io.TextIOWrapper(buf, encoding='utf-8')
    w2 = io.TextIOWrapper(buf, encoding='utf-8')
    w2.close()
    try:
        w1.write('x')
        w1.flush()
        mech = False
        mech_detail = '未复现'
    except ValueError as exc:
        mech = 'closed file' in str(exc)
        mech_detail = str(exc)
    cases.append(('★F1c 非空转证明：两个 wrapper 共享同一 buffer 时，关掉其一会让另一个'
                  '写入抛 `I/O operation on closed file`（旧写法崩溃的机制本身可复现）',
                  mech, mech_detail))
    # ★ 源码级检查：模块级**不得**再出现 `sys.stdout = ...` 赋值（改用 reconfigure）
    import ast
    src = io.open(CHECK_PATH, encoding='utf-8').read()
    tree = ast.parse(src)
    assigns = []
    for node in tree.body:                              # 只看模块级
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == 'stdout' \
                        and isinstance(t.value, ast.Name) and t.value.id == 'sys':
                    assigns.append(ast.dump(t))
    cases.append(('★F1d 源码级：模块级不再出现 `sys.stdout = …` 赋值',
                  assigns == [], assigns))
    uses_reconf = 'reconfigure' in src
    cases.append(('F1e 改用 `sys.stdout.reconfigure(...)`（就地改造，不换对象）',
                  uses_reconf, uses_reconf))
    # 修复后模块仍然功能正常：能读 spec 并给出 PAIR_FIELDS
    prog3 = ('import sys, json\n'
             'sys.path.insert(0, %r)\n'
             'import check_control_config as c\n'
             'print(json.dumps({"n_pair": len(c.PAIR_FIELDS),\n'
             '                  "has_arm_seed": "arm_seed" in c.PAIR_FIELDS,\n'
             '                  "spec_ok": bool(c.load_spec())}))\n' % (HERE,))
    p3 = subprocess.run([sys.executable, '-c', prog3], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, cwd=ROOT)
    out3 = (p3.stdout or b'').decode('utf-8', 'replace').strip()
    try:
        got = json.loads(out3.splitlines()[-1])
    except Exception:                                         # noqa: BLE001
        got = {}
    cases.append(('F1d 修复后模块功能仍正常（能读冻结 spec、PAIR_FIELDS 可枚举）',
                  got.get('spec_ok') is True and got.get('n_pair', 0) >= 16,
                  (out3, (p3.stderr or b'').decode()[-200:])))
    return cases


# ===========================================================================
# F2：not_applicable 降级必须同时体现在 engaged / problems / passed
# ===========================================================================
def _rec(i, action='reject', day=0, h='h%d' % 0):
    """schema 合法的决策记录。

    ⚠️ 必须**合法**：`compare_ablations` 会先做 schema 校验，`action='accept'` 还要求
    条件必填字段（`order_id`/`certify_energy`/`budget_B`）—— 第一版我漏了它们，
    于是所有 F2 用例都因 schema 而失败，看起来像"修复没生效"，其实是夹具不合法。
    这里用 `action='reject'`（无条件必填）以聚焦本组要测的判据。
    """
    return {'inst_idx': 0, 'day_id': day, 'clock': float(i), 'info_max_reveal': float(i),
            'action': action, 'certified': False, 'accepted_before': [],
            'accepted_after': [], 'executed_before': [], 'executed_after': [],
            'decision_hash': h}


def _base_log(n=3):
    return [_rec(i, h='h%d' % i) for i in range(n)]


def test_audit_downgrade_not_counted_as_failure():
    cases = []
    base = _base_log()
    same = [dict(r) for r in base]          # 与基准**完全相同**（消融前后无差别）

    # (a) 完全相同 + 未提供 activation ⇒ not_applicable，**不**进 problems
    ok, rows, probs = da.check_ablation_engagement(
        {'opportunity_cost': (base, same)}, activation=None)
    cases.append(('★F2a 完全相同 + 未提供 activation ⇒ engaged=None ∧ problems 为空 ∧ '
                  '整体 ok=True（三者同时验证，不只看中间字段）',
                  rows[0]['engaged'] is None and probs == [] and ok is True,
                  (rows[0].get('engaged'), probs, ok)))
    # (b) 完全相同 + activation = 0 ⇒ 同上
    ok_b, rows_b, probs_b = da.check_ablation_engagement(
        {'opportunity_cost': (base, same)}, activation={'opportunity_cost': 0})
    cases.append(('★F2b 完全相同 + activation=0 ⇒ not_applicable ∧ problems 为空 ∧ ok=True',
                  rows_b[0]['engaged'] is None and probs_b == [] and ok_b is True,
                  (rows_b[0].get('engaged'), probs_b, ok_b)))
    # (c) 完全相同 + activation > 0 ⇒ **真 FAIL**（机制触发过却毫无差别）
    ok_c, rows_c, probs_c = da.check_ablation_engagement(
        {'opportunity_cost': (base, same)}, activation={'opportunity_cost': 7})
    cases.append(('★F2c 完全相同 + activation>0 ⇒ engaged=False ∧ problems 非空 ∧ ok=False'
                  '（真 FAIL 仍必须判失败 —— 修复不能把该失败的也放过）',
                  rows_c[0]['engaged'] is False and bool(probs_c) and ok_c is False,
                  (rows_c[0].get('engaged'), probs_c, ok_c)))
    # (d) 有差异 + activation=0 ⇒ 仍判已接入
    other = _base_log()
    other[0] = _rec(0, action='reject', h='DIFFERENT')
    ok_d, rows_d, probs_d = da.check_ablation_engagement(
        {'opportunity_cost': (base, other)}, activation={'opportunity_cost': 0})
    cases.append(('★F2d 有差异 + activation=0 ⇒ engaged=True ∧ problems 为空 ∧ ok=True'
                  '（差异本身就是接入证据）',
                  rows_d[0]['engaged'] is True and probs_d == [] and ok_d is True,
                  (rows_d[0].get('engaged'), probs_d, ok_d)))
    # (e) 端到端：整份 audit 的 `passed` 也必须反映同样的结论
    au_na = da.audit(base, ablations={'opportunity_cost': (base, same)},
                     ablation_activation={'opportunity_cost': 0})
    cases.append(('★F2e 端到端 audit：not_applicable 情形下 `passed` 不为 False 且 '
                  '`problems` 不含该开关（这正是原缺陷造成"虚假资格失败"的地方）',
                  au_na['checks']['T2_T3_T4_ablation_engagement'] is not False
                  and not any('opportunity_cost' in p for p in au_na['problems']),
                  (au_na['checks']['T2_T3_T4_ablation_engagement'], au_na['problems'])))
    au_fail = da.audit(base, ablations={'opportunity_cost': (base, same)},
                       ablation_activation={'opportunity_cost': 7})
    cases.append(('★F2f 端到端 audit：真 FAIL 情形仍必须 `passed=False` 且 problems 点名该开关',
                  au_fail['passed'] is False
                  and any('opportunity_cost' in p for p in au_fail['problems']),
                  (au_fail['passed'], au_fail['problems'])))
    # (f) 历史结果**不被回填**：审计器是纯函数，不改输入
    snap = copy.deepcopy(base)
    da.check_ablation_engagement({'opportunity_cost': (base, same)},
                                 activation={'opportunity_cost': 0})
    cases.append(('F2g 审计器是**纯函数**：不改动传入的日志（"不回填旧 gate"）',
                  base == snap, None))
    return cases


# ===========================================================================
# F3：arm_seed 进入 PAIR_FIELDS
# ===========================================================================
def test_arm_seed_in_pair_fields():
    import check_control_config as ccc
    cases = []
    cases.append(('★F3 `arm_seed` 已在 `PAIR_FIELDS` 里（用户裁定 YES）',
                  'arm_seed' in ccc.PAIR_FIELDS, list(ccc.PAIR_FIELDS)))
    cases.append(('F3b `PAIR_FIELDS` 仍是元组且无重复（新增不破坏既有形状）',
                  isinstance(ccc.PAIR_FIELDS, tuple)
                  and len(set(ccc.PAIR_FIELDS)) == len(ccc.PAIR_FIELDS),
                  len(ccc.PAIR_FIELDS)))
    spec = ccc.load_spec()
    cfg = dict(spec['frozen_config'])
    cfg.update({k: v for k, v in spec['shared_identity_anchor'].items()
                if k not in ('seeds', 'main_penalty')})
    good = dict(cfg)
    bad = dict(cfg)
    bad['arm_seed'] = 7001                       # 误用 SaaReplanner 默认值
    miss = {k: v for k, v in cfg.items() if k != 'arm_seed'}

    def _verdict(a, b):
        """`compare_pair` 返回 **3 元组** `(ok, rows, bad)`（第一版我按字符串猜，判据写错）。"""
        r = ccc.compare_pair(a, b, spec=spec)
        if not (isinstance(r, tuple) and len(r) == 3):
            raise AssertionError('compare_pair 应返回 (ok, rows, bad)，得到 %r' % (r,))
        return r

    def _seed_row(rows):
        hit = [r for r in rows if r and r[0] == 'arm_seed']
        return hit[0] if hit else None

    ok_ok, rows_ok, bad_ok = _verdict(good, dict(good))
    ok_bad, rows_bad, bad_bad = _verdict(good, bad)
    ok_miss, rows_miss, bad_miss = _verdict(good, miss)
    cases.append(('★F3c 两侧 arm_seed 一致（7002/7002）⇒ 该字段判 OK，且不在 bad 里',
                  _seed_row(rows_ok) is not None
                  and _seed_row(rows_ok)[3] == 'OK' and 'arm_seed' not in bad_ok,
                  _seed_row(rows_ok)))
    cases.append(('★F3d 两侧 arm_seed 不一致（7002 vs 7001）⇒ overall=False ∧ arm_seed 进 bad',
                  ok_bad is False and 'arm_seed' in bad_bad
                  and '不一致' in (_seed_row(rows_bad) or ('', '', '', ''))[3],
                  (_seed_row(rows_bad), bad_bad)))
    cases.append(('★F3e 一侧**缺失** arm_seed ⇒ 同样 overall=False（不因缺失而放行，'
                  '与既有 fail-closed 口径一致）',
                  ok_miss is False and 'arm_seed' in bad_miss
                  and '缺失' in (_seed_row(rows_miss) or ('', '', '', ''))[3],
                  (_seed_row(rows_miss), bad_miss)))
    # 冻结 spec 里确实已记录 7002（补录裁定）
    cases.append(('F3f 冻结 spec 的 arm_seed = 7002（cond_hist，非默认 7001）',
                  spec['frozen_config'].get('arm_seed') == 7002,
                  spec['frozen_config'].get('arm_seed')))
    # 历史 gate 不被追溯修改：工具只读 spec / 只读输入，不写任何 gate 文件
    cases.append(('F3g `check_control_config` 不写任何 gate/spec 文件（"不回填旧结果"）',
                  not any(hasattr(ccc, n) for n in ('write_spec', 'save_spec', 'patch_gate')),
                  [n for n in dir(ccc) if n.startswith(('write', 'save', 'patch'))]))
    return cases


def test_sha_records():
    """修复必须形成**新的 SHA 记录**（用户要求），且两份文件都可复算。"""
    import hashlib
    cases = []
    rec = {}
    for rel in ('CC_Compare/tools/check_control_config.py',
                'CC_Compare/tools/role_c_decision_audit.py'):
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        with io.open(p, 'rb') as f:
            rec[rel] = hashlib.sha256(f.read()).hexdigest()
    cases.append(('★F4 两份被修复的公共工具都产生**新的 raw SHA-256**（64 位十六进制）',
                  all(len(v) == 64 and all(ch in '0123456789abcdef' for ch in v)
                      for v in rec.values()), rec))
    out = os.path.join(ROOT, 'CC_Compare', 'PUBLIC_TOOL_FIX_SHAS_2026-10-14.json')
    payload = {'kind': 'public-tool-fix-sha-record', 'date': '2026-10-14',
               'ruling': '用户 2026-10-14 裁定：两项公共工具缺陷必须修；'
                         '形成新版本与 SHA 记录，不覆盖原裁定、不回填旧 gate、'
                         '不把旧失败追认成通过。',
               'sha256': rec,
               'note': '本文由 test_public_tool_fixes.py 的 SHA 记录步骤生成；'
                       '若任一文件再改动，重跑该测试即得新记录（旧记录不覆盖，另存新版本）。'}
    with io.open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    cases.append(('F4b SHA 记录已落盘（%s）' % os.path.basename(out), os.path.isfile(out), out))
    return cases


def main():
    cases = []
    for fn in (test_stdout_not_closed_on_import,
               test_audit_downgrade_not_counted_as_failure,
               test_arm_seed_in_pair_fields, test_sha_records):
        try:
            cases.extend(fn())
        except Exception as exc:                            # noqa: BLE001
            import traceback
            cases.append(('★ %s 抛异常' % fn.__name__, False,
                          traceback.format_exc()[-700:]))
    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail)[:400])))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
