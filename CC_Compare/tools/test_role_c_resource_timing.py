#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_resource_ceiling` + `role_c_timing` 的测试（规格 §6 的机械门）。

要证明的三件事，每件都用**反例**证明判据不是空转：

  G  **资源门校验 fail-closed**：缺字段 / 线程上限漏配 / `OMP != 1` /
     `workers × max(thread_caps) > cpu_count`（过度订阅的**定义性**不等式）/
     affinity 短于 workers / 单决策时限 ≠ 冻结 `time_limit` / 总上限缺失 /
     夹带效用字段 —— 逐条必须被拒；一个**合法**登记必须通过。
  R  **未登记 ⇒ 拒绝计时**：登记文件不存在或未通过校验 ⇒ `CeilingNotRegistered`；
     `run_timing` 抛 `TimingRefused`（**不是**"照跑但记个警告"）。
  T  **实参核对 + 只记时间**：进程环境/亲和性与登记不符 ⇒ 拒绝（耗时不可解释）；
     一致时 → `per_decision`（mean/P50/P95/max）、`warmup` **不进统计**、
     报告里**没有效用字段**；`work_fn` 返回效用 ⇒ `UtilityLeakError` 直接抛。

全部用**注入的假 work_fn / 假环境读取器**，因此**不跑真实决策路径、不产生任何耗时结论**。
运行：python CC_Compare/tools/test_role_c_resource_timing.py   （退出码 0 = 全过）
"""
import copy
import io
import json
import os
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import resource_timing as rt                              # noqa: E402
import role_c_resource_ceiling as ceil                     # noqa: E402
import role_c_timing as timing                             # noqa: E402

UTILITY_KEYS = {'utility', 'profit', 'revenue', 'fuel', 'reject_loss', 'j_cc',
                'delta_j', 'gain', 'energy'}


def good_ceiling(ncpu=None):
    """一个**合法**登记（针对假机器；真实登记必须由用户按服务器事实填）。"""
    ncpu = ncpu or (os.cpu_count() or 8)
    nw = min(8, max(1, ncpu // 4))
    aff = list(range(nw))
    return {
        'ceiling_id': 'test-ceiling-v1',
        'registered_by': 'test fixture（非真实登记）',
        'registered_at': '2026-10-14',
        'source_policy': 'test fixture',
        'hardware': {'machine': 'fixture', 'cpu_count': ncpu, 'mem_gb': 16.0},
        'workers': nw,
        'cpu_affinity': aff,
        'thread_caps': {'OMP_NUM_THREADS': 1, 'MKL_NUM_THREADS': 1,
                        'OPENBLAS_NUM_THREADS': 1, 'NUMEXPR_NUM_THREADS': 1},
        'single_decision_time_limit_s': 10.0,
        'total_wall_clock_limit_s': 3600.0,
    }


def _leaked(obj, path=''):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in UTILITY_KEYS:
                out.append('%s/%s' % (path, k))
            out += _leaked(v, '%s/%s' % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _leaked(v, '%s[%d]' % (path, i))
    return out


def _live_env(**over):
    e = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
         'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    e.update(over)
    return e


def test_gate_validation():
    cases = []
    good = good_ceiling()

    def bad(name, mut, expect_substr=None):
        c = copy.deepcopy(good)
        mut(c)
        ok, p = ceil.validate_ceiling(c)
        blob = ' | '.join(p)
        hit = (expect_substr is None) or (expect_substr in blob)
        cases.append(('G %s ⇒ 被拒%s' % (name, '' if hit else '（但理由不对）'),
                      ok is False and hit, p))

    ok, p = ceil.validate_ceiling(good)
    cases.append(('G0 合法登记通过（证明以下反例不是恒失败）', ok is True, p))
    cases.append(('G0b 无效用字段的登记才允许存在', _leaked(good) == [], _leaked(good)))

    bad('缺 workers', lambda c: c.pop('workers'), '缺必需字段')
    bad('缺 total_wall_clock_limit_s', lambda c: c.pop('total_wall_clock_limit_s'),
        '缺必需字段')
    bad('缺 thread_caps', lambda c: c.pop('thread_caps'), 'thread_caps 必须是 dict')
    bad('漏配 MKL_NUM_THREADS（只设 OMP 不够）',
        lambda c: c['thread_caps'].pop('MKL_NUM_THREADS'), 'MKL_NUM_THREADS')
    bad('漏配 OPENBLAS_NUM_THREADS',
        lambda c: c['thread_caps'].pop('OPENBLAS_NUM_THREADS'), 'OPENBLAS_NUM_THREADS')
    bad('OMP_NUM_THREADS != 1（规格明文）',
        lambda c: c['thread_caps'].__setitem__('OMP_NUM_THREADS', 4),
        '必须 = 1')
    bad('workers=0', lambda c: c.__setitem__('workers', 0), 'workers')
    bad('workers 是 bool', lambda c: c.__setitem__('workers', True), 'workers')
    bad('cpu_affinity 为空', lambda c: c.__setitem__('cpu_affinity', []), 'cpu_affinity')
    bad('affinity 短于 workers',
        lambda c: c.__setitem__('cpu_affinity', list(range(max(1, c['workers'] - 1)))),
        '少于 workers')
    bad('单决策时限 ≠ 冻结 time_limit',
        lambda c: c.__setitem__('single_decision_time_limit_s', 20.0),
        '冻结 time_limit')
    bad('总墙钟上限为 0',
        lambda c: c.__setitem__('total_wall_clock_limit_s', 0.0), '> 0')
    bad('hardware 为空 dict', lambda c: c.__setitem__('hardware', {}), 'hardware')
    bad('ceiling_id 为空', lambda c: c.__setitem__('ceiling_id', '  '), 'ceiling_id')
    bad('夹带效用字段', lambda c: c.__setitem__('utility', 1.0), '效用类字段')

    # 过度订阅的**定义性**不等式：构造一个 cpu_count 内仍超订阅的登记
    c = copy.deepcopy(good)
    ncpu = os.cpu_count() or 8
    c['workers'] = max(1, ncpu // 2)
    c['cpu_affinity'] = list(range(c['workers']))
    c['thread_caps']['MKL_NUM_THREADS'] = 4
    ok2, p2 = ceil.validate_ceiling(c)
    cases.append(('G1 workers × max(thread_caps) > cpu_count ⇒ 判过度订阅（定义性不等式）',
                  ok2 is False and any('过度订阅' in x for x in p2), p2))
    cases.append(('G1b occupancy 如实报出 requested/cpu_count',
                  ceil.occupancy(c)['cores_requested']
                  == c['workers'] * 4, ceil.occupancy(c)))
    return cases


def test_unregistered_refuses():
    cases = []
    missing = os.path.join(tempfile.gettempdir(), 'no_such_ceiling_%d.json' % os.getpid())
    if os.path.isfile(missing):
        os.remove(missing)
    try:
        ceil.load_ceiling(missing)
        cases.append(('R1 登记文件不存在 ⇒ CeilingNotRegistered', False, '未抛错'))
    except ceil.CeilingNotRegistered as exc:
        cases.append(('R1 登记文件不存在 ⇒ CeilingNotRegistered',
                      '未登记' in str(exc), str(exc)[:80]))
    st = ceil.registration_status(missing)
    cases.append(('R1b registration_status 如实报未登记（不抛）',
                  st['registered'] is False and st['problems'], st))
    # 内容不合法的登记文件 ⇒ 也是"未登记"
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'ceiling.json')
        with io.open(p, 'w', encoding='utf-8') as f:
            json.dump({'ceiling_id': 'x'}, f)
        try:
            ceil.load_ceiling(p)
            cases.append(('R2 登记存在但不合法 ⇒ 同样拒绝', False, '未抛错'))
        except ceil.CeilingNotRegistered as exc:
            cases.append(('R2 登记存在但不合法 ⇒ 同样拒绝',
                          '缺必需字段' in str(exc), str(exc)[:80]))
        # run_timing 必须拒绝（不能"照跑但记个警告"）
        try:
            timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=3, path=p)
            cases.append(('★R3 未登记 ⇒ run_timing 抛 TimingRefused', False, '未抛错'))
        except timing.TimingRefused as exc:
            cases.append(('★R3 未登记 ⇒ run_timing 抛 TimingRefused',
                          '拒绝计时' in str(exc), str(exc)[:90]))
        except ceil.CeilingNotRegistered as exc:
            cases.append(('★R3 未登记 ⇒ run_timing 拒绝（CeilingNotRegistered）',
                          True, str(exc)[:90]))
    # 真实仓库里当前**没有**登记文件 ⇒ 全局状态必须是"未登记"
    st_repo = ceil.registration_status()
    cases.append(('★R4 当前仓库**确实**没有登记文件 ⇒ 全局状态 = 未登记'
                  '（资源计时因此被机械挡住）',
                  st_repo['registered'] is False, st_repo))
    cases.append(('R4b 登记路径指向 CC_Compare/P1A_资源门登记.json',
                  ceil.CEILING_REL.endswith('P1A_资源门登记.json'), ceil.CEILING_REL))
    # ★ 模板即使被改名放到登记路径上也必须被拒 —— 否则复制模板就把门意外打开了
    cc_dir = os.path.dirname(HERE)                       # .../CC_Compare
    tpl_path = os.path.join(cc_dir, 'P1A_资源门登记_模板.json')
    reg_path = os.path.join(cc_dir, ceil.CEILING_REL.split('/')[-1])
    cases.append(('★R5 模板文件存在且**不在**登记路径上（门保持关闭）',
                  os.path.isfile(tpl_path)
                  and os.path.abspath(tpl_path).lower() != os.path.abspath(reg_path).lower()
                  and not os.path.isfile(reg_path), (tpl_path, reg_path)))
    with io.open(tpl_path, encoding='utf-8') as f:
        tpl = json.load(f)
    ok_t, probs_t = ceil.validate_ceiling(tpl)
    cases.append(('★R5b 模板本身**永远不被**校验通过（`is_template` 判据）',
                  ok_t is False and any('模板' in x for x in probs_t), probs_t))
    # 把模板原样放到登记路径 ⇒ load_ceiling 仍必须拒绝
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'ceiling.json')
        shutil.copy2(tpl_path, p)
        try:
            ceil.load_ceiling(p)
            cases.append(('★R5c 模板被改名放到登记路径 ⇒ load_ceiling 仍拒绝',
                          False, '未抛错'))
        except ceil.CeilingNotRegistered as exc:
            cases.append(('★R5c 模板被改名放到登记路径 ⇒ load_ceiling 仍拒绝',
                          '模板' in str(exc), str(exc)[:80]))
    return cases


def test_timing_harness():
    cases = []
    good = good_ceiling()
    ncpu = os.cpu_count() or 8
    aff = list(range(good['workers']))
    envc = _live_env()
    calls = {'n': 0}

    def work():
        calls['n'] += 1
        return {'diag': calls['n'], 'elapsed_label': 'fake'}   # 非效用 ⇒ 允许

    rep = timing.run_timing(work_fn=work, n_decisions=20, warmup=3, ceiling=good,
                            env=envc, affinity=aff,
                            env_record_fn=lambda: rt.environment_record(),
                            affinity_fn=lambda: aff, loadavg_fn=lambda: [1.0, 2.0, 3.0])
    cases.append(('T1 合法登记 + 实参一致 ⇒ 计时跑通', rep['n'] == 20, rep.get('n')))
    cases.append(('T2 warmup **不进统计**（3 次预热 + 20 次采样 = 23 次调用）',
                  calls['n'] == 23 and rep['warmup_calls'] == 3, (calls['n'], rep)))
    pd = rep['per_decision']
    cases.append(('T3 报告单决策分布 mean/P50/P95/max 且单调',
                  pd and pd['min'] <= pd['p50'] <= pd['p95'] <= pd['max']
                  and pd['n'] == 20, pd))
    cases.append(('★T4 报告里**没有任何效用字段**（"不看效用"的机械边界）',
                  _leaked(rep) == [], _leaked(rep)))
    cases.append(('T5 报告携带登记身份 + 占用率 + 前后 loadavg + 实际亲和性',
                  rep['ceiling_id'] == 'test-ceiling-v1'
                  and rep['occupancy']['cores_requested'] == good['workers']
                  and rep['loadavg_before'] == [1.0, 2.0, 3.0]
                  and rep['affinity_actual'] == aff, None))
    cases.append(('T5b 实参核对问题为空（一致）', rep['live_environment_problems'] == [],
                  rep['live_environment_problems']))

    # 实参不符 ⇒ 拒绝
    for name, kw, sub in (
            ('OMP 未设', dict(env=_live_env(OMP_NUM_THREADS=None)), 'OMP_NUM_THREADS'),
            ('MKL 值不符', dict(env=_live_env(MKL_NUM_THREADS='4')), 'MKL_NUM_THREADS'),
            ('affinity 不同', dict(affinity=[99]), '亲和性')):
        try:
            timing.run_timing(work_fn=work, n_decisions=2, ceiling=good,
                              env_record_fn=lambda: {}, affinity_fn=lambda: None,
                              loadavg_fn=lambda: None, **kw)
            cases.append(('★T6 %s ⇒ TimingRefused（耗时不可解释）' % name, False, '未抛错'))
        except timing.TimingRefused as exc:
            cases.append(('★T6 %s ⇒ TimingRefused（耗时不可解释）' % name,
                          sub in str(exc), str(exc)[:90]))

    # work_fn 返回效用 ⇒ UtilityLeakError
    try:
        timing.run_timing(work_fn=lambda: {'utility': 1.0}, n_decisions=2, ceiling=good,
                          env=envc, affinity=aff, env_record_fn=lambda: {},
                          affinity_fn=lambda: aff, loadavg_fn=lambda: None)
        cases.append(('★T7 work_fn 返回效用字段 ⇒ UtilityLeakError 直接抛',
                      False, '未抛错'))
    except rt.UtilityLeakError as exc:
        cases.append(('★T7 work_fn 返回效用字段 ⇒ UtilityLeakError 直接抛', True,
                      str(exc)[:70]))
    # 参数校验
    for name, kw in (('n_decisions=0', dict(n_decisions=0)),
                     ('n_decisions=True', dict(n_decisions=True))):
        try:
            timing.run_timing(work_fn=work, ceiling=good, env=envc, affinity=aff, **kw)
            cases.append(('T8 %s ⇒ TimingRefused' % name, False, '未抛错'))
        except timing.TimingRefused:
            cases.append(('T8 %s ⇒ TimingRefused' % name, True, None))
    try:
        timing.run_timing(work_fn='not-callable', n_decisions=2, ceiling=good,
                          env=envc, affinity=aff)
        cases.append(('T8b work_fn 不可调用 ⇒ TimingRefused', False, '未抛错'))
    except timing.TimingRefused:
        cases.append(('T8b work_fn 不可调用 ⇒ TimingRefused', True, None))
    # validate_preconditions 只报告不抛（便于先看原因）
    c2, p2 = timing.validate_preconditions(good, env=_live_env(OMP_NUM_THREADS='2'),
                                          affinity=aff)
    cases.append(('T9 validate_preconditions 只报告不抛，且点名不符项',
                  c2 is not None and any('OMP_NUM_THREADS' in x for x in p2), p2))
    return cases


def test_ceiling_template_roundtrip():
    """登记文件的**形状**可从建议文件照填 ⇒ 校验器能读（用临时文件走一遍完整路径）。"""
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'ceiling.json')
        c = good_ceiling()
        with io.open(p, 'w', encoding='utf-8') as f:
            json.dump(c, f, ensure_ascii=False, indent=2)
        loaded = ceil.load_ceiling(p)
        cases.append(('T10 合法登记文件可被 load_ceiling 读出', loaded == c,
                      sorted(loaded)))
        ok, probs = ceil.validate_ceiling(loaded)
        cases.append(('T10b 读出的登记仍通过校验（round-trip）', ok is True, probs))
        # 真实计时路径：登记文件 + 注入 work_fn
        rep = timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=5, path=p,
                                env=_live_env(), affinity=list(range(c['workers'])),
                                env_record_fn=lambda: {}, affinity_fn=lambda: None,
                                loadavg_fn=lambda: [0.5])
        cases.append(('T10c 走登记文件路径的计时也能跑通（不依赖 ceiling= 传参）',
                      rep['ceiling_id'] == c['ceiling_id'] and rep['n'] == 5, None))
    return cases


def test_decision_work_fn():
    """单决策计时单元：只在**真实捕获的**决策点上求值，返回值不含效用。"""
    cases = []
    seen = {'n': 0}

    def fake_decide(order, ctx):
        seen['n'] += 1
        return True, {'certify_ok': True, 'inner_gate': 'on',
                      'lambda_diag': {'lambda_B': 1.0, 'undefined': False},
                      'forward': {'n_scenarios': 3}}

    ctxs = [{'order': 1, 'clock': 0.0}, {'order': 2, 'clock': 2.0},
            {'order': 3, 'clock': 4.0}]
    wf = timing.make_decision_work_fn(fake_decide, ctxs)
    cases.append(('W1 单元长度 = 决策点数', len(wf) == 3, len(wf)))
    outs = [wf() for _ in range(5)]
    cases.append(('W2 轮转多个决策点（不死的复议一个状态）',
                  [o['context_index'] for o in outs] == [0, 1, 2, 0, 1],
                  [o['context_index'] for o in outs]))
    cases.append(('★W3 返回值**只含行为标签、不含效用**（计时边界）',
                  _leaked(outs[0]) == [], _leaked(outs[0])))
    cases.append(('W4 标签齐备（动作/认证/λ 是否定义/是否为正/前瞻是否跑到）',
                  set(outs[0]) >= {'action', 'certified', 'lambda_defined',
                                   'lambda_positive', 'inner_gate',
                                   'has_forward_diag', 'elapsed_scope'},
                  sorted(outs[0])))
    cases.append(('W5 decide_fn 被真正调用（不是空转）', seen['n'] == 5, seen['n']))
    for name, args in (('contexts 为空', ([],)),
                       ('contexts=None', (None,))):
        try:
            timing.make_decision_work_fn(fake_decide, *args)
            cases.append(('★W6 %s ⇒ TimingRefused（不得用手搓假状态冒充）' % name,
                          False, '未抛错'))
        except timing.TimingRefused as exc:
            cases.append(('★W6 %s ⇒ TimingRefused（不得用手搓假状态冒充）' % name,
                          '假状态' in str(exc), str(exc)[:70]))
    try:
        timing.make_decision_work_fn('nope', ctxs)
        cases.append(('W6b decide_fn 不可调用 ⇒ TimingRefused', False, '未抛错'))
    except timing.TimingRefused:
        cases.append(('W6b decide_fn 不可调用 ⇒ TimingRefused', True, None))
    # 决策诊断里带效用 ⇒ 计时单元**白名单**式取值，效用**结构上出不来**
    # （比"依赖递归检查去抓"更强：不靠"记得别传"，而是"根本没有那个通道"）
    wf_leak = timing.make_decision_work_fn(
        lambda order, ctx: (True, {'certify_ok': True, 'utility': 9.0,
                                   'lambda_diag': {'lambda_B': 1.0, 'undefined': False}}),
        ctxs)
    leak_out = wf_leak()
    cases.append(('★W7 决策诊断里带效用 ⇒ 计时单元白名单取值，效用**传不出来**',
                  _leaked(leak_out) == [] and 'utility' not in leak_out,
                  _leaked(leak_out)))
    rep_leak = timing.run_timing(
        work_fn=wf_leak, n_decisions=2, ceiling=good_ceiling(), env=_live_env(),
        affinity=list(range(good_ceiling()['workers'])), env_record_fn=lambda: {},
        affinity_fn=lambda: None, loadavg_fn=lambda: None)
    cases.append(('★W7b 因此即使诊断带效用，计时报告里也不会出现（并有 rt 递归检查兜底）',
                  _leaked(rep_leak) == [], _leaked(rep_leak)))
    # 报告写出：写入前再自检
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'r.json')
        rep = timing.run_timing(work_fn=wf, n_decisions=6, ceiling=good_ceiling(),
                                warmup=1, env=_live_env(),
                                affinity=list(range(good_ceiling()['workers'])),
                                env_record_fn=lambda: {}, affinity_fn=lambda: None,
                                loadavg_fn=lambda: None)
        out = timing.write_report(rep, p)
        cases.append(('W8 报告可写出且可读回（JSON round-trip）',
                      os.path.isfile(out)
                      and json.load(io.open(out, encoding='utf-8'))['n'] == 6, out))
        try:
            timing.write_report({'utility': 1.0}, os.path.join(tmp, 'bad.json'))
            cases.append(('★W9 写报告前再自检：带效用 ⇒ 拒绝写', False, '未抛错'))
        except rt.UtilityLeakError:
            cases.append(('★W9 写报告前再自检：带效用 ⇒ 拒绝写', True, None))
    return cases


def test_capture_real_contexts():
    """`run_fixture(capture_n=…)` 捕获的是**真实**决策点，且是**深拷贝**（冻结）。"""
    import role_c_t_series as ts
    ts.ensure_project_paths()          # 必须先注入项目路径，才能 import scenario_saa
    from scenario_saa import make_c0_contract_v2, ScenarioOrder
    cases = []
    contract = make_c0_contract_v2()
    env_mod = __import__('test_role_c_harness')
    ds = env_mod.make_fixture()
    scenario = [ScenarioOrder(oid=10 ** 6 + j, reveal=1.0, x=float(1 + j * 0.1), y=0.0,
                              demand=1.0, temp_class=1, tw_start=0.0, tw_end=100.0,
                              service_time=0.1) for j in range(4)]
    sampler = env_mod._FixedSampler(scenario)
    dfn = __import__('role_c_harness').build_decide_fn(
        sampler=sampler, arm_seed=7002, K=3, contract=contract,
        cooling_share=3.7208946347769882, booking_horizon=16.0, budget=710.6548152249378,
        certify_plan_fn=lambda *a, **k: (True, 1.0))
    run = ts.run_fixture(decide_fn=dfn, ds=ds, contract=contract,
                         budget=710.6548152249378, capacity=50.0, num_vehicles=2,
                         booking_horizon=16.0, cooling_share=3.7208946347769882,
                         capture_n=3)
    ctxs = run.get('contexts') or []
    cases.append(('C1 捕获到真实决策点（%d 个）' % len(ctxs), len(ctxs) >= 1, len(ctxs)))
    if ctxs:
        c = ctxs[0]
        cases.append(('C2 上下文含 decide_fn 需要的键（env/vehicles/served_mask/plan/'
                      'budget_B/contract/order）',
                      all(k in c for k in ('env', 'vehicles', 'served_mask', 'plan',
                                           'budget_B', 'contract', 'order')),
                      sorted(c)))
        cases.append(('★C3 vehicles 被**深拷贝并冻结**：跑完一天后重新指纹化，仍与捕获'
                      '当时一致（模拟器原地改车 ⇒ 不冻结就会拿到"决策之后"的状态；'
                      '本项目踩过 freeze_vehicles 那个坑）',
                      c.get('_vehicle_fingerprint') is not None
                      and ts.vehicle_fingerprint(c['vehicles'])
                      == c['_vehicle_fingerprint'],
                      (ts.vehicle_fingerprint(c['vehicles'])[:2],
                       (c.get('_vehicle_fingerprint') or ())[:2])))
        # 幂等性：同一 ctx 反复调用 decide_fn 不应报错（计时单元依赖这一点）
        try:
            a1 = dfn(c['order'], c)
            a2 = dfn(c['order'], c)
            cases.append(('★C4 decide_fn 对同一 ctx **幂等**（计时单元靠它做重复测量）',
                          isinstance(a1, tuple) and isinstance(a2, tuple), (a1[0], a2[0])))
        except Exception as exc:                            # noqa: BLE001
            cases.append(('★C4 decide_fn 对同一 ctx 幂等', False,
                          '%s: %s' % (type(exc).__name__, exc)))
    cases.append(('C5 捕获不引入效用字段（run_fixture 返回里不需有效用）',
                  _leaked(ctxs) == [], _leaked(ctxs)[:3]))
    # ★ 非空转证明：**活车**在跑完一天后指纹确实变了 ⇒ 捕获指纹不变只能归因于深拷贝
    fp_first = run.get('live_fingerprint_first')
    fp_last = run.get('live_fingerprint_last')
    cases.append(('★C6 非空转：**活车**跑完一天后指纹**确实变了**（车辆被原地修改）'
                  '⇒ C3 的"捕获指纹不变"只能归因于深拷贝，不是因为状态本来就没变',
                  fp_first is not None and fp_last is not None and fp_first != fp_last,
                  (str(fp_first)[:90], str(fp_last)[:90])))
    return cases


def main():
    cases = []
    for fn in (test_gate_validation, test_unregistered_refuses, test_timing_harness,
               test_ceiling_template_roundtrip, test_decision_work_fn,
               test_capture_real_contexts):
        try:
            cases.extend(fn())
        except Exception as exc:                            # noqa: BLE001
            import traceback
            cases.append(('★ %s 抛异常' % fn.__name__, False,
                          traceback.format_exc()[-700:]))
    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail)[:300])))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
