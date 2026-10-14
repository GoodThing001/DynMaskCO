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
            timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=3, path=p, decision_points_digest='test-fixture')
            cases.append(('★R3 未登记 ⇒ run_timing 抛 TimingRefused', False, '未抛错'))
        except timing.TimingRefused as exc:
            cases.append(('★R3 未登记 ⇒ run_timing 抛 TimingRefused',
                          '拒绝计时' in str(exc), str(exc)[:90]))
        except ceil.CeilingNotRegistered as exc:
            cases.append(('★R3 未登记 ⇒ run_timing 拒绝（CeilingNotRegistered）',
                          True, str(exc)[:90]))
    # ★ 用户 2026-10-14 授权后登记已落盘 ⇒ 全局状态必须是"已登记"，且**逐字段等于裁定值**
    st_repo = ceil.registration_status()
    cases.append(('★R4 资源门**已登记**（用户 2026-10-14 授权后落盘），'
                  '且 ceiling_id/占用率与裁定一致',
                  st_repo['registered'] is True
                  and st_repo.get('ceiling_id') == 'p1a-rolec-v1'
                  and st_repo['occupancy'] == {'workers': 8, 'max_thread_cap': 1,
                                               'cores_requested': 8, 'cpu_count': 96},
                  st_repo))
    reg = ceil.load_ceiling()
    cases.append(('★R4b 登记值**逐字段**等于用户批准值（workers/affinity/四项线程上限/'
                  'cpu_count/mem/单决策/总墙钟）',
                  reg['workers'] == 8 and reg['cpu_affinity'] == list(range(8))
                  and all(reg['thread_caps'][k] == 1 for k in ceil.THREAD_CAP_FIELDS)
                  and reg['hardware']['cpu_count'] == 96
                  and abs(reg['hardware']['mem_gb'] - 265.3) < 1e-9
                  and abs(reg['single_decision_time_limit_s'] - 10.0) < 1e-12
                  and abs(reg['total_wall_clock_limit_s'] - 43200.0) < 1e-9,
                  {k: reg[k] for k in ('workers', 'cpu_affinity', 'thread_caps',
                                       'single_decision_time_limit_s',
                                       'total_wall_clock_limit_s')}))
    cases.append(('★R4c 登记留存**授权来源**（不冒充执行人签名）',
                  isinstance(reg.get('authorization'), str)
                  and '2026-10-14' in reg['authorization']
                  and reg['registered_by'] == 'P1A-resource-review-2026-10-14'
                  and '12 小时' in reg['authorization'],
                  str(reg.get('authorization'))[:100]))
    cases.append(('R4d 登记路径指向 CC_Compare/P1A_资源门登记.json',
                  ceil.CEILING_REL.endswith('P1A_资源门登记.json'), ceil.CEILING_REL))
    # ★ 登记路径上必须是**真登记**（不是被改名的模板）
    cc_dir = os.path.dirname(HERE)                       # .../CC_Compare
    tpl_path = os.path.join(cc_dir, 'P1A_资源门登记_模板.json')
    reg_path = os.path.join(cc_dir, ceil.CEILING_REL.split('/')[-1])
    cases.append(('★R5 登记路径上是**真登记**（无 `is_template`），模板在另一个文件里',
                  os.path.isfile(tpl_path) and os.path.isfile(reg_path)
                  and os.path.abspath(tpl_path).lower()
                  != os.path.abspath(reg_path).lower()
                  and json.load(io.open(reg_path, encoding='utf-8')).get('is_template')
                  is None, (tpl_path, reg_path)))
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
                            decision_points_digest='test-fixture',
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
                              decision_points_digest='test-fixture',
                              env_record_fn=lambda: {}, affinity_fn=lambda: None,
                              loadavg_fn=lambda: None, **kw)
            cases.append(('★T6 %s ⇒ TimingRefused（耗时不可解释）' % name, False, '未抛错'))
        except timing.TimingRefused as exc:
            cases.append(('★T6 %s ⇒ TimingRefused（耗时不可解释）' % name,
                          sub in str(exc), str(exc)[:90]))

    # work_fn 返回效用 ⇒ UtilityLeakError
    try:
        timing.run_timing(work_fn=lambda: {'utility': 1.0}, n_decisions=2, ceiling=good,
                          decision_points_digest='test-fixture',
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
                          decision_points_digest='test-fixture',
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
                                decision_points_digest='test-fixture',
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
        decision_points_digest='test-fixture',
        affinity=list(range(good_ceiling()['workers'])), env_record_fn=lambda: {},
        affinity_fn=lambda: None, loadavg_fn=lambda: None)
    cases.append(('★W7b 因此即使诊断带效用，计时报告里也不会出现（并有 rt 递归检查兜底）',
                  _leaked(rep_leak) == [], _leaked(rep_leak)))
    # 报告写出：写入前再自检
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'r.json')
        rep = timing.run_timing(work_fn=wf, n_decisions=6, ceiling=good_ceiling(),
                                warmup=1, env=_live_env(),
                                decision_points_digest='test-fixture',
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


def test_registration_builder_and_cli():
    """登记**构建 + CLI**：校验不过就不落盘；登记自带机器事实 ⇒ 换机也可复验。"""
    cases = []
    ncpu = os.cpu_count() or 8
    # ---- 换机校验（关键正确性）：登记描述的是**服务器 96 核**，写登记的本机可能只有 8 核
    srv = ceil.build_registration(
        workers=8, cpu_affinity=list(range(8)), total_wall_clock_limit_s=43200.0,
        registered_by='user',
        hardware={'machine': 'server', 'cpu_count': 96, 'mem_gb': 265.3})
    ok_srv, prob_srv = ceil.validate_ceiling(srv)
    cases.append(('★X1 登记服务器 96 核 + workers=8/affinity 0-7 ⇒ 在**只有 %d 核的本机**上'
                  '也通过（核数取登记里的事实，不取 os.cpu_count()）' % ncpu,
                  ok_srv is True, prob_srv))
    cases.append(('X1b occupancy 用登记核数（96），不是本机核数',
                  ceil.occupancy(srv)['cpu_count'] == 96
                  and ceil.occupancy(srv)['cores_requested'] == 8,
                  ceil.occupancy(srv)))
    bad_srv = ceil.build_registration(
        workers=8, cpu_affinity=list(range(8)), total_wall_clock_limit_s=1.0,
        registered_by='user', hardware={'machine': 'tiny', 'cpu_count': 4})
    ok_bad, prob_bad = ceil.validate_ceiling(bad_srv)
    cases.append(('★X1c 反向：登记核数 4 + workers=8 ⇒ 判过度订阅（不因本机核数而放行）',
                  ok_bad is False and any('过度订阅' in x for x in prob_bad), prob_bad))
    # ---- 构建器缺省值
    cases.append(('X2 构建器把四项线程上限缺省为 1（不留空 ⇒ 不漏配）',
                  set(srv['thread_caps']) == set(ceil.THREAD_CAP_FIELDS)
                  and all(v == 1 for v in srv['thread_caps'].values()),
                  srv['thread_caps']))
    cases.append(('X3 构建器把单决策时限取**冻结** time_limit（10.0），不让人随手填',
                  abs(srv['single_decision_time_limit_s']
                      - ceil.load_frozen_time_limit()) < 1e-12,
                  srv['single_decision_time_limit_s']))
    # ---- affinity 解析
    cases.append(("X4 affinity 解析 '0-7' / '0,1,2' / '0-3,8'",
                  ceil._parse_affinity('0-7') == list(range(8))
                  and ceil._parse_affinity('0,1,2') == [0, 1, 2]
                  and ceil._parse_affinity('0-3,8') == [0, 1, 2, 3, 8],
                  [ceil._parse_affinity(s) for s in ('0-7', '0,1,2', '0-3,8')]))
    # ---- 不合法 ⇒ **不落盘**
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'reg.json')
        invalid = dict(srv, workers=0)
        try:
            ceil.write_registration(invalid, p)
            cases.append(('★X5 未通过校验 ⇒ 抛错**且不落盘**', False, '未抛错'))
        except ceil.CeilingError as exc:
            cases.append(('★X5 未通过校验 ⇒ 抛错**且不落盘**',
                          not os.path.isfile(p) and '不落盘' in str(exc), str(exc)[:70]))
        # ---- 合法 ⇒ 落盘且可被 load_ceiling 读回
        out = ceil.write_registration(srv, p)
        cases.append(('X6 合法登记落盘，且 load_ceiling 能读回、状态为已登记',
                      os.path.isfile(out)
                      and ceil.load_ceiling(p)['ceiling_id'] == srv['ceiling_id']
                      and ceil.registration_status(p)['registered'] is True, out))
        try:
            ceil.write_registration(srv, p)
            cases.append(('★X7 已存在 ⇒ 拒绝覆盖（重新登记属资源门变更）', False, '未抛错'))
        except ceil.CeilingError as exc:
            cases.append(('★X7 已存在 ⇒ 拒绝覆盖（重新登记属资源门变更）',
                          '拒绝覆盖' in str(exc), str(exc)[:70]))
        cases.append(('X7b --force 才允许覆盖',
                      os.path.isfile(ceil.write_registration(srv, p, force=True)), None))
    # ---- CLI：dry-run 不落盘；不合法返回 3
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'cli.json')
        rc = ceil.main(['--workers', '8', '--affinity', '0-7', '--total-wall', '43200',
                        '--cpu-count', '96', '--registered-by', 'user',
                        '--path', p, '--dry-run'])
        cases.append(('X8 CLI --dry-run 返回 0 且**不落盘**',
                      rc == 0 and not os.path.isfile(p), rc))
        rc2 = ceil.main(['--workers', '8', '--affinity', '0-7', '--total-wall', '43200',
                         '--cpu-count', '96', '--registered-by', 'user', '--path', p])
        cases.append(('X9 CLI 正常路径落盘（返回 0）',
                      rc2 == 0 and os.path.isfile(p), rc2))
        p3 = os.path.join(tmp, 'bad.json')
        rc3 = ceil.main(['--workers', '99', '--affinity', '0-7', '--total-wall',
                         '43200', '--cpu-count', '96', '--registered-by', 'user',
                         '--path', p3])
        cases.append(('★X10 CLI 在过度订阅时返回 3 且不落盘',
                      rc3 == 3 and not os.path.isfile(p3), rc3))
    # ---- CLI **不提供**默认上限值（上限是登记决策，不能被我默认掉）
    try:
        ceil.main([])
        cases.append(('★X11 CLI 不给上限参数 ⇒ 报错（**不提供默认值**）', False, '未抛错'))
    except SystemExit as exc:
        cases.append(('★X11 CLI 不给上限参数 ⇒ 报错（**不提供默认值**）',
                      exc.code != 0, exc.code))
    return cases


def test_supervision_and_timeouts():
    """**超时监督**（用户 2026-10-14 点名必须补的缺口）：单决策可终止并回收、总墙钟是硬上限、
    决策点集合先固定、竞争有判据，且报告含全部必需字段。"""
    import subprocess
    import sys as _sys
    import time as _time
    cases = []
    good = good_ceiling()
    ncpu_good_aff = list(range(good['workers']))
    env = _live_env()

    def mk(**kw):
        base = dict(ceiling=good, env=env, affinity=ncpu_good_aff,
                    env_record_fn=lambda: {}, affinity_fn=lambda: ncpu_good_aff,
                    loadavg_fn=lambda: [1.0], decision_points_digest='fixture-set-v1')
        base.update(kw)
        return base

    # ---- A. 决策点集合必须先固定（不得事后剔除"耗时高的状态"）----
    try:
        timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=2,
                          **mk(decision_points_digest=None))
        cases.append(('★S1 未提供 decision_points_digest 且 work_fn 无 digest() ⇒ 拒绝',
                      False, '未抛错'))
    except timing.TimingRefused as exc:
        cases.append(('★S1 未提供 decision_points_digest 且 work_fn 无 digest() ⇒ 拒绝',
                      '固定待测决策点' in str(exc), str(exc)[:80]))
    rep_u = timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=2,
                              **mk(decision_points_digest=None, allow_undigested=True))
    cases.append(('S1b 显式 allow_undigested=True 才放行，且报告中如实记 None',
                  rep_u['decision_points_digest'] is None, None))
    cases.append(('S1c 提供了 digest ⇒ 报告里带着它（事后可核对集合未被换掉）',
                  timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=1,
                                    **mk())['decision_points_digest']
                  == 'fixture-set-v1', None))

    # ---- B. 单决策超时：终止 + 回收 + 计数（不阻塞整轮）----
    slow = lambda: (_time.sleep(0.30), {'diag': 1})[1]        # noqa: E731
    rep_t = timing.run_timing(work_fn=slow, n_decisions=3, **mk(per_decision_limit_s=0.05,
                                                               contention_check=False))
    if hasattr(__import__('signal'), 'SIGALRM'):
        cases.append(('★S2 POSIX：单决策超限被 SIGALRM **硬中断并回收**（3 次都超时）',
                      rep_t['n_timeouts'] >= 1 and rep_t['n_completed'] == 0
                      and rep_t['enforcement'] == 'hard_sigalrm',
                      (rep_t['n_timeouts'], rep_t['n_completed'],
                       rep_t['enforcement'])))
    else:
        cases.append(('★S2 非 POSIX：**无法**硬中断，但超限被如实检出并计数'
                      '（`enforcement=cooperative` ⇒ 该次计时不合格）',
                      rep_t['n_overruns'] >= 1 and rep_t['enforcement'] == 'cooperative'
                      and rep_t['qualified'] is False
                      and any('enforcement' in x for x in rep_t['unqualified_reasons']),
                      (rep_t['n_overruns'], rep_t['enforcement'],
                       rep_t['unqualified_reasons'])))
    cases.append(('S2b 超时的**完整记录**保留在 records 里（不剔除）',
                  len(rep_t['records']) >= 1
                  and any(r['exceeded_per_decision_limit'] for r in rep_t['records']),
                  rep_t['records'][:2]))
    cases.append(('S2c 「已登记上限」≠「已执行硬上限」：非硬中断 ⇒ qualified=False',
                  (rep_t['qualified'] is True) if rep_t['enforcement'] == 'hard_sigalrm'
                  else (rep_t['qualified'] is False), rep_t['unqualified_reasons']))

    # ---- C. 总墙钟是硬上限：用注入时钟确定性地触发 ----
    tick = {'t': 0.0}

    def fake_clock():
        tick['t'] += 1.0
        return tick['t']
    rep_w = timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=100,
                              **mk(total_wall_s=5.0, monotonic=fake_clock,
                                   contention_check=False))
    cases.append(('★S3 总墙钟上限被触及 ⇒ 停下、`total_wall_exceeded=True`、'
                  '完成数 < 请求数、qualified=False',
                  rep_w['total_wall_exceeded'] is True
                  and rep_w['n_completed'] < 100 and rep_w['qualified'] is False
                  and any('总墙钟' in x for x in rep_w['unqualified_reasons']),
                  (rep_w['total_wall_exceeded'], rep_w['n_completed'],
                   rep_w['unqualified_reasons'])))

    # ---- D. 竞争判据（loadavg 不足以证明钉住的核没被别人用）----
    cont = timing.affinity_contention([0, 1, 2, 3], sample_s=0.05)
    cases.append(('S4 affinity_contention 在无 /proc/stat 时**如实报无法判定**'
                  '（不得当作"无竞争"）',
                  ('available' in cont)
                  and (cont['available'] or '无法判定' in (cont.get('reason') or '')
                       or '不可用' in (cont.get('reason') or '')), cont))
    old = timing.affinity_contention
    try:
        timing.affinity_contention = lambda *a, **k: {
            'available': True, 'external_busy_ratio': 0.9, 'per_cpu': {}, 'reason': '',
            'checked': True}
        rep_c = timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=2,
                                  **mk(contention_check=True))
        cases.append(('★S5 外部占用 0.9 > 0.5 ⇒ qualified=False，且**不**换核集/加 workers'
                      '（报告里 affinity_actual 仍是登记的 0–7）',
                      rep_c['qualified'] is False
                      and any('明显竞争' in x for x in rep_c['unqualified_reasons'])
                      and rep_c['affinity_actual'] == ncpu_good_aff,
                      (rep_c['unqualified_reasons'], rep_c['affinity_actual'])))
        timing.affinity_contention = lambda *a, **k: {
            'available': True, 'external_busy_ratio': 0.01, 'per_cpu': {}, 'reason': '',
            'checked': True}
        rep_c2 = timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=2,
                                   **mk(contention_check=True))
        cases.append(('S5b 外部占用 0.01 ⇒ 不再因竞争判不合格（判据非恒失败）',
                      not any('明显竞争' in x for x in rep_c2['unqualified_reasons']),
                      rep_c2['unqualified_reasons']))
    finally:
        timing.affinity_contention = old

    # ---- E. 报告必须含用户要求的全部字段 ----
    rep_f = timing.run_timing(work_fn=lambda: {'diag': 1}, n_decisions=4, warmup=1,
                              **mk(contention_check=False))
    need = ('records', 'per_decision', 'n_timeouts', 'n_overruns', 'affinity_actual',
            'thread_caps_actual', 'loadavg_before', 'loadavg_after', 'contention',
            'decision_points_digest', 'enforcement', 'qualified',
            'unqualified_reasons', 'total_elapsed_s', 'total_wall_limit_s',
            'per_decision_limit_s')
    cases.append(('★S6 报告含用户点名的全部字段（完整单次耗时/P95/超时次数/真实亲和性/'
                  '线程配置/机器负载/监督判据）', all(k in rep_f for k in need),
                  [k for k in need if k not in rep_f]))
    cases.append(('S6b per_decision 含 P95（资源规划证据）',
                  rep_f['per_decision'] and 'p95' in rep_f['per_decision'],
                  rep_f['per_decision']))
    cases.append(('S6c 报告仍**无任何效用字段**', _leaked(rep_f) == [], _leaked(rep_f)))

    # ---- F. 外层监督器：超时**硬杀并回收** ----
    sup = timing.run_supervised(
        [_sys.executable, '-c', 'import time; time.sleep(30)'], total_wall_s=1.0)
    cases.append(('★S7 外层监督器在总墙钟超限时**硬杀并回收**子进程（timed_out=True）',
                  sup['timed_out'] is True and sup['elapsed_s'] < 20.0, sup))
    sup2 = timing.run_supervised(
        [_sys.executable, '-c', 'print("ok")'], total_wall_s=30.0)
    cases.append(('S7b 正常完成的子进程如实返回 rc=0 且 timed_out=False',
                  sup2['timed_out'] is False and sup2['rc'] == 0
                  and 'ok' in sup2['stdout_tail'], sup2))
    return cases


def test_real_work_fn_pipeline():
    """真实计时单元的**管线自检**（小规模；不产生耗时结论）。

    关键：CLI **没有**跳过锚点校验的开关 —— 小规模必须**被拒绝**，
    否则就会拿一条与冻结配置不同规模的管线去测时间。
    """
    cases = []
    # ---- 小规模 + 关掉锚点校验 ⇒ 仅用于验证管线搭得起来 ----
    try:
        built = timing.build_real_work_fn(n_orders=20, n_vehicles=3, capture_n=2,
                                          expect_frozen_anchors=False)
    except Exception as exc:                                # noqa: BLE001
        import traceback
        cases.append(('★Z1 真实计时单元可搭建（小规模、关锚点校验）', False,
                      traceback.format_exc()[-500:]))
        return cases
    wf = built['work_fn']
    cases.append(('★Z1 真实计时单元可搭建（小规模）：捕获到真实决策点',
                  len(wf) >= 1 and built['n_records'] >= 1, built))
    cases.append(('Z1b 计时单元自带 `digest()` ⇒ `run_timing` 能自动固定决策点集合',
                  isinstance(wf.digest(), str) and len(wf.digest()) == 64, wf.digest()))
    cases.append(('Z1c 单元返回值只含行为标签（无任何效用字段）',
                  _leaked(wf()) == [], _leaked(wf())))
    # ---- 同一小规模但**打开**锚点校验 ⇒ 必须拒绝（判据非空转） ----
    try:
        timing.build_real_work_fn(n_orders=20, n_vehicles=3, capture_n=1,
                                 expect_frozen_anchors=True)
        cases.append(('★Z2 小规模 + 锚点校验开 ⇒ **拒绝**（不得拿非冻结规模的管线测时间）',
                      False, '未抛错'))
    except timing.TimingRefused as exc:
        cases.append(('★Z2 小规模 + 锚点校验开 ⇒ **拒绝**（不得拿非冻结规模的管线测时间）',
                      '未复现冻结锚点' in str(exc), str(exc)[:90]))
    # ---- CLI **不提供**跳过锚点校验的开关（避免误用） ----
    import ast as _ast
    src = io.open(os.path.join(HERE, 'role_c_timing.py'), encoding='utf-8').read()
    has_bypass = 'skip-anchor' in src or 'no-anchor' in src \
        or 'allow-nonfrozen' in src
    cases.append(('★Z3 CLI **没有**跳过锚点校验的开关（防"拿错规模测时间"）',
                  has_bypass is False, has_bypass))
    _ = _ast
    return cases


def main():
    cases = []
    for fn in (test_gate_validation, test_unregistered_refuses, test_timing_harness,
               test_ceiling_template_roundtrip, test_decision_work_fn,
               test_capture_real_contexts, test_registration_builder_and_cli,
               test_supervision_and_timeouts, test_real_work_fn_pipeline):
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
