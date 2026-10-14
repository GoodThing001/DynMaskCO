#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_identity`（Role C 的 `nonlearning_solver` 身份闭包，T1）的测试。

T1 的判据（规格 §5）：**身份闭包可独立重算且一致**；`qualified_full=True`；
**篡改任一依赖文件 ⇒ `False`**。这里把它拆成可机械验证的若干条：

  I  清单与冻结配置的接地：清单**单一来源**、缺文件即报错、共享旋钮**逐字段**取自冻结文件
     （不重打 ⇒ 不会漂移）；`arm_seed` 缺口被**如实标注**（含出处），不是悄悄补上。
  T1a 身份块可构造、**不含 checkpoint**、`verify_role_c_identity` 通过；
  T1b `recompute_actual_hashes` 与身份块**逐位相同**（独立重算 = 同一真值）；
  T1c **篡改任一依赖文件** ⇒ 重算不等、`verify_closure` 失败、gate `qualified_full=False`
      且否决项指向闭包（用**真实闭包文件的副本**做篡改，不是假文件）；
  T1d gate 端到端：干净 ⇒ `qualified_full=True`；`source_stable=False` ⇒ False；
      声明 checkpoint ⇒ `Q_class_consistency` 否决；
  T1e **机制旋钮在哈希里**：改 `inner_gate` / LS 预算 / `arm_seed` / `K` ⇒ `solver_config_sha256`
      必变（规格 §2.2.2 ④：按结果偷换档位不可能静默）；
  T1f 求解器体检**不豁免**：`solver_meta.n_fail>0` ⇒ 否决（`nonlearning_solver` 不是豁免类型）。

合成夹具 + 真实闭包文件；**不跑实验、不读效用、不碰保留种子、不写冻结文件**。
运行：python CC_Compare/tools/test_role_c_identity.py   （退出码 0 = 全过）
"""
import copy
import os
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import nonlearning_identity as ni                          # noqa: E402
import role_c_harness as hn                                # noqa: E402
import role_c_identity as ri                               # noqa: E402


def good_per_day(n=40, served=60):
    return [dict(i=k, utility=100.0 + k, hard_feasible=True, failures=[],
                 n_served=served, n_rejected=140, energy_kwh=700.0,
                 budget_violated=False, timeouts=0, elapsed_s=1.0)
            for k in range(n)]


def good_pairing():
    return {'identity_verified': True, 'problems': []}


def good_solver_meta():
    # `nonlearning_solver` 的求解器体检**不豁免** ⇒ 必须给真实字段
    return {'n_solves': 7761, 'n_fail': 0}


# ===========================================================================
# I. 清单与冻结配置的接地
# ===========================================================================
def test_declaration_and_frozen_grounding():
    cases = []
    f = ri.load_frozen_control()
    deps = ri.dependency_files()
    decl = hn.ROLE_C_DEPENDENCY_ROLES
    cases.append(('I1 依赖清单**单一来源** = ROLE_C_DEPENDENCY_ROLES 的键（不另抄一份）',
                  len(deps) == len(decl)
                  and set(os.path.relpath(p, ri.repo_root()).replace(os.sep, '/')
                          for p in deps) == set(decl), (len(deps), len(decl))))
    cases.append(('I2 每个依赖文件都真实存在，且每一行都写明"为什么进来"',
                  all(os.path.isfile(p) for p in deps)
                  and all(isinstance(v, str) and v.strip() for v in decl.values()),
                  [r for r, v in decl.items() if not (isinstance(v, str) and v.strip())]))
    cases.append(('I3 空清单 ⇒ 拒绝（空清单会返回 None，那不是"已闭合"）',
                  ni.dependency_closure_sha256([], base=ri.repo_root()) is None, None))
    # 共享旋钮逐字段取自冻结文件
    sh = ri.shared_config(f)
    fc = f['frozen_config']
    miss = [k for k in fc if sh.get(k) != fc[k]]
    cases.append(('I4 共享旋钮**逐字段**等于冻结配置（不重打 ⇒ 不会漂移）',
                  miss == [], miss))
    cases.append(('I5 显式记录 booking_horizon（冻结要求，旧 gate 漏记）',
                  sh.get('booking_horizon') == 16.0
                  and f['must_record_causal_flag']['booking_horizon'] == 16.0,
                  sh.get('booking_horizon')))
    # ⚠️ arm_seed 缺口如实标注
    cfg = ri.role_c_solver_config()
    asd = cfg['role_c_scenario']['arm_seed']
    cases.append(('★I6 冻结配置**未记录** arm_seed ⇒ 身份里如实标注（含出处），不悄悄补上',
                  asd['recorded_in_frozen_control'] is False
                  and asd['provenance'] == ri.CONTROL_ARM_SEEDS_SOURCE
                  and 'arm_seed' not in fc, asd))
    cases.append(('I7 cond_hist 的 arm_seed = 7002（**不是** SaaReplanner 默认值 7001）',
                  ri.CONTROL_ARM_SEEDS['cond_hist'] == 7002
                  and asd['value'] == 7002, asd['value']))
    cases.append(('I8 身份里的共享锚点等于冻结锚点（合同/数据/数据元/预算/冷却份额）',
                  cfg['shared_identity_anchor'] == ri.frozen_anchors(f),
                  cfg['shared_identity_anchor']))
    try:
        ri.role_c_solver_config(arm_seed=None)
        cases.append(('I9 arm_seed=None ⇒ 报错', False, '未抛错'))
    except ri.IdentityError:
        cases.append(('I9 arm_seed=None ⇒ 报错', True, None))
    try:
        ri.role_c_solver_config(inner_gate='maybe')
        cases.append(('I10 未知 inner_gate ⇒ 报错', False, '未抛错'))
    except ri.IdentityError:
        cases.append(('I10 未知 inner_gate ⇒ 报错', True, None))
    return cases


# ===========================================================================
# T1a/T1b. 身份块与独立重算
# ===========================================================================
def test_identity_roundtrip():
    cases = []
    cfg = ri.role_c_solver_config()
    ident = ri.build_role_c_identity(config=cfg)
    cases.append(('T1a 身份块类型正确且**不含** checkpoint（nonlearning_solver）',
                  ident['method_class'] == 'nonlearning_solver'
                  and not ident.get('checkpoint_sha256')
                  and ident['source_stable'] is True, sorted(ident)))
    cases.append(('T1a2 闭包哈希与配置哈希都是 64 位十六进制，且依赖文件数 > 0',
                  len(ident['provider_dependency_sha256']) == 64
                  and len(ident['solver_config_sha256']) == 64
                  and ident['n_dependency_files'] == len(ri.dependency_files()),
                  ident['n_dependency_files']))
    ok, reasons = ri.verify_role_c_identity(ident, cfg)
    cases.append(('T1a3 跑批前自检通过（verify_closure：存在性/独立重算/三方一致）',
                  ok is True, reasons))
    actual = ni.recompute_actual_hashes(ri.dependency_files(), cfg,
                                        base=ri.repo_root())
    cases.append(('★T1b 独立重算 == 身份块**逐位相同**（T1 的"可独立重算且一致"）',
                  actual['provider_dependency_sha256']
                  == ident['provider_dependency_sha256']
                  and actual['solver_config_sha256'] == ident['solver_config_sha256'],
                  (actual, {k: ident[k] for k in actual})))
    # 顺序无关（同一集合不同传入顺序 ⇒ 同一哈希）
    h1 = ni.dependency_closure_sha256(list(ri.dependency_files()), base=ri.repo_root())
    h2 = ni.dependency_closure_sha256(list(reversed(ri.dependency_files())),
                                      base=ri.repo_root())
    cases.append(('T1b2 闭包哈希对清单顺序不敏感（先排序再聚合）', h1 == h2, (h1, h2)))
    return cases


# ===========================================================================
# T1c. 篡改任一依赖文件 ⇒ 必被发现（用**真实闭包文件的副本**）
# ===========================================================================
def test_tamper_detection_on_real_closure():
    cases = []
    cfg = ri.role_c_solver_config()
    ident = ri.build_role_c_identity(config=cfg)
    with tempfile.TemporaryDirectory() as tmp:
        # 把**真实闭包**按相对路径复制进临时树（不改动仓库里的任何文件）
        for rel in sorted(hn.ROLE_C_DEPENDENCY_ROLES):
            dst = os.path.join(tmp, rel.replace('/', os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, rel.replace('/', os.sep)), dst)
        ident_t = ri.build_role_c_identity(config=cfg, base=tmp)
        same = ident_t['provider_dependency_sha256'] == ident['provider_dependency_sha256']
        cases.append(('T1c0 同一内容在另一绝对路径下 ⇒ 闭包哈希相同（路径按相对根取）',
                      same is True, (ident_t['provider_dependency_sha256'],
                                     ident['provider_dependency_sha256'])))
        ok0, r0 = ri.verify_role_c_identity(ident_t, cfg, base=tmp)
        cases.append(('T1c1 未篡改的副本树 ⇒ 自检通过（证明 T1c 不是恒失败）',
                      ok0 is True, r0))
        # 逐个篡改**每一个**依赖文件（不只代表性的一个）
        victims = sorted(hn.ROLE_C_DEPENDENCY_ROLES)
        bad = []
        for rel in victims:
            p = os.path.join(tmp, rel.replace('/', os.sep))
            with open(p, 'rb') as fh:
                orig = fh.read()
            with open(p, 'ab') as fh:
                fh.write(b'\n# tamper-probe\n')
            actual = ni.recompute_actual_hashes(ri.dependency_files(tmp), cfg, base=tmp)
            ok, reasons = ri.verify_role_c_identity(ident_t, cfg, base=tmp)
            detected = (actual['provider_dependency_sha256']
                        != ident_t['provider_dependency_sha256']) and (ok is False)
            if not detected:
                bad.append(rel)
            with open(p, 'wb') as fh:
                fh.write(orig)
        cases.append(('★T1c2 篡改**任一**依赖文件（逐个 %d 个全试）都被发现：'
                      '重算不等 ∧ verify_closure 失败' % len(victims), bad == [], bad))
        # 篡改配置旋钮同样被发现
        cfg2 = copy.deepcopy(cfg)
        cfg2['role_c_mechanism']['inner_gate'] = 'off'
        actual2 = ni.recompute_actual_hashes(ri.dependency_files(tmp), cfg2, base=tmp)
        ok2, _ = ri.verify_role_c_identity(ident_t, cfg2, base=tmp)
        cases.append(('T1c3 篡改**配置**（inner_gate on→off）也被发现',
                      actual2['solver_config_sha256'] != ident_t['solver_config_sha256']
                      and ok2 is False, None))
    return cases


# ===========================================================================
# T1d. gate 端到端
# ===========================================================================
def test_gate_end_to_end():
    cases = []
    cfg = ri.role_c_solver_config()
    ident = ri.build_role_c_identity(config=cfg)
    res = ri.evaluate_identity(identity=ident, config=cfg, per_day=good_per_day(),
                               solver_meta=good_solver_meta(), pairing=good_pairing())
    cases.append(('★T1d 干净身份 + 40 天 + 求解器健康 ⇒ qualified_full=True（无否决）',
                  res['qualified_full'] is True and not res['vetoes'],
                  (res['qualified_full'], res['vetoes'])))
    cases.append(('T1d2 `Q_class_consistency` 通过：nonlearning_solver 未声明 checkpoint',
                  res['criteria']['Q_class_consistency']['ok'] is True,
                  res['criteria']['Q_class_consistency']))
    cases.append(('T1d3 模型参与度对 nonlearning_solver 是结构性不适用（未给也不否决）',
                  res['criteria']['Q_method']['ok'] is True
                  and '不适用' in res['criteria']['Q_method']['reason'],
                  res['criteria']['Q_method']))
    # source_stable=False
    ident_bad = ri.build_role_c_identity(config=cfg, source_stable=False)
    res_bad = ri.evaluate_identity(identity=ident_bad, config=cfg, per_day=good_per_day(),
                                  solver_meta=good_solver_meta(),
                                  pairing=good_pairing())
    cases.append(('T1d4 source_stable=False ⇒ qualified_full=False（共享身份照查不豁免）',
                  res_bad['qualified_full'] is False, res_bad['vetoes']))
    # 声明 checkpoint
    ident_ck = dict(ident, checkpoint_sha256='a' * 64)
    res_ck = ri.evaluate_identity(identity=ident_ck, config=cfg, per_day=good_per_day(),
                                 solver_meta=good_solver_meta(),
                                 pairing=good_pairing())
    cases.append(('★T1d5 给 nonlearning_solver 声明 checkpoint ⇒ `Q_class_consistency` 否决',
                  res_ck['qualified_full'] is False
                  and 'Q_class_consistency' in (res_ck['vetoes'] or []),
                  res_ck['vetoes']))
    # 求解器体检不豁免
    res_sf = ri.evaluate_identity(identity=ident, config=cfg, per_day=good_per_day(),
                                 solver_meta={'n_solves': 7761, 'n_fail': 3},
                                 pairing=good_pairing())
    cases.append(('★T1d6 求解器体检**不豁免**：n_fail=3 ⇒ 否决不通过',
                  res_sf['qualified_full'] is False, res_sf['vetoes']))
    # 篡改后的身份进 gate（重算不符）：身份块必须在**篡改前**构造，
    # 才模拟出"跑批后磁盘变了"——第一版在篡改后才构造，两边都取自篡改后的磁盘，
    # 于是自洽通过，等于没测。
    with tempfile.TemporaryDirectory() as tmp:
        for rel in sorted(hn.ROLE_C_DEPENDENCY_ROLES):
            dst = os.path.join(tmp, rel.replace('/', os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, rel.replace('/', os.sep)), dst)
        ident_t = ri.build_role_c_identity(config=cfg, base=tmp)     # ← 篡改**前**
        p = os.path.join(tmp, sorted(hn.ROLE_C_DEPENDENCY_ROLES)[0].replace('/', os.sep))
        with open(p, 'ab') as fh:
            fh.write(b'\n# tamper\n')
        ev = ri.gate_evidence(identity=ident_t, config=cfg, per_day=good_per_day(),
                              solver_meta=good_solver_meta(), pairing=good_pairing(),
                              base=tmp)
        import qualification_gate as qg
        res_t = qg.evaluate(ev, method_class=ri.ROLE_C_METHOD_CLASS)
    return cases + [('★T1d7 跑批后依赖被改（身份块取自改前、重算取改后）⇒ '
                     '`Q_dependency_closure.ok=False` ∧ `qualified_full=False`'
                     '（注意 closure **刻意不进 `vetoes`** —— 不事后改写历史臂资格）',
                     res_t['qualified_full'] is False
                     and res_t['Q_dependency_closure']['ok'] is False
                     and '重算' in res_t['Q_dependency_closure']['reason'],
                     res_t['Q_dependency_closure'])]


# ===========================================================================
# T1e. 机制旋钮在哈希里
# ===========================================================================
def test_mechanism_knobs_are_hashed():
    cases = []
    base = ri.role_c_solver_config()
    h0 = ni.solver_config_sha256(base)
    variants = [
        ('inner_gate on→off', dict(inner_gate='off')),
        ('LS max_passes 2→3', dict(ls_kwargs={'max_passes': 3})),
        ('LS max_moves 4000→10', dict(ls_kwargs={'max_moves': 10})),
        ('arm_seed 7002→7001', dict(arm_seed=7001)),
        ('K 10→1', dict(K=1)),
        ('time_limit 10→11', dict(time_limit=11.0)),
        ('energy_pricing marginal→amortized', dict(energy_pricing='amortized')),
        ('standby_orders None→10', dict(standby_orders=10)),
        ('future_policy reveal→density', dict(future_policy='density')),
    ]
    same = []
    for name, kw in variants:
        h = ni.solver_config_sha256(ri.role_c_solver_config(**kw))
        if h == h0:
            same.append(name)
    cases.append(('★T1e 规格 §2.2.2 ④：机制/共享旋钮**逐个**改变 ⇒ solver_config_sha256 必变'
                  '（按结果偷换 inner_gate 不可能静默）', same == [], same))
    cases.append(('T1e2 同一配置两次 ⇒ 同一哈希（规范化稳定）',
                  ni.solver_config_sha256(ri.role_c_solver_config()) == h0, None))
    cases.append(('T1e3 空配置 ⇒ None（不得冒充已冻结）',
                  ni.solver_config_sha256({}) is None, None))
    cases.append(('T1e4 身份里带 LS 邻域规格哈希与前瞻规格哈希（路线 A 的代码身份可复核）',
                  base['role_c_route_search']['source_spec_sha256']
                  == __import__('role_c_local_search').SOURCE_SPEC_SHA256
                  and base['role_c_forward']['spec_sha256']
                  == __import__('role_c_forward').FORWARD_SPEC_SHA256
                  and base['role_c_route_search']['neighborhood_id'] == 'role-c-ls-v1',
                  base['role_c_route_search']))
    return cases


def main():
    cases = []
    for fn in (test_declaration_and_frozen_grounding, test_identity_roundtrip,
               test_tamper_detection_on_real_closure, test_gate_end_to_end,
               test_mechanism_knobs_are_hashed):
        try:
            cases.extend(fn())
        except Exception as exc:                            # noqa: BLE001
            import traceback
            cases.append(('★ %s 抛异常' % fn.__name__, False,
                          traceback.format_exc()[-600:]))
    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
