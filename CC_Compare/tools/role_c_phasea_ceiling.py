#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A 的**专用资源门**（身份 `p1a-rolec-phasea-v1`）。

为什么不能复用旧的校验器（用户 2026-10-14 亲自指出的接口冲突）
-------------------------------------------------------------
`role_c_resource_ceiling.validate_ceiling` 硬性要求：

    single_decision_time_limit_s == frozen time_limit        # 即 10.0 s

那是**在线协议时限**。但 Phase A 的 A 类定义是：**内部截止关闭**、单点**外层监督**
上限 600 s、整个 Phase A 上限 12 000 s。把 A 类登记塞进旧校验器 ⇒ 要么被迫写成
10 s（**谎报**），要么把在线时限改掉（**改协议**）。两者都不可接受。

⇒ 本模块**另立** Phase A 专用的登记形状与校验器，并且：

  * **不改** `P1A_资源门登记.json`（`p1a-rolec-v1`）的任何字段；
  * **不改** 旧校验器的任何语义（旧文件仍用旧校验器验，见 `baseline_status()`）；
  * Phase A 登记里**禁止**出现顶层 `single_decision_time_limit_s`
    —— 该字段属于在线协议，出现在这里就是把两件事混为一谈（**拒绝**）。

用户 2026-10-14 的三条硬要求（机械化为校验项）
---------------------------------------------
  1. A 类 `internal_deadline = "disabled"`、`per_decision_cap_s = 600`；
  2. B 类 `internal_deadline = 10.0`（= 冻结 `time_limit`，**不变**）；
  3. `phase_total_cap_s = 12 000` 必须约束**整个 Phase A 作业**
     —— 决策点准备 + L1 + L2 + B 类诊断**全部**在内（`PHASE_SCOPE` 四项缺一即拒），
     而不只是 A 类的计算循环；
  4. 外层监督器**真正终止并回收**进程，且**保留截断记录**。

⚠️ 本模块**不跑任何测量**，只做登记与校验（`validate_phasea_ceiling` 是 fail-closed）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import resource_timing as rt
import role_c_resource_ceiling as base

__all__ = [
    'PHASEA_CEILING_REL', 'PHASEA_CEILING_ID', 'PARENT_CEILING_ID',
    'PHASE_SCOPE_REQUIRED', 'PHASEA_DECISION_POINTS_DIGEST',
    'PHASEA_N_DECISION_POINTS', 'CLASS_A_PER_DECISION_CAP_S',
    'PHASE_TOTAL_CAP_S', 'PhaseACeilingError', 'PhaseACeilingNotRegistered',
    'validate_phasea_ceiling', 'load_phasea_ceiling', 'phasea_status',
    'build_phasea_registration', 'write_phasea_registration',
    'baseline_status', 'baseline_recorded_digests', 'occupancy_phasea',
]

PHASEA_CEILING_REL = 'CC_Compare/P1A_资源门登记_PhaseA_v1.json'
PHASEA_CEILING_ID = 'p1a-rolec-phasea-v1'
PARENT_CEILING_ID = 'p1a-rolec-v1'

#: 12 000 s 必须覆盖的**整个 Phase A 作业**（用户 2026-10-14：不只是 A 类计算循环）
PHASE_SCOPE_REQUIRED = ('decision_point_preparation', 'L1_counting',
                        'L2_partitioned_timing', 'B_class_diagnostics')

#: 事前固定的待测决策点集合（用户 2026-10-14：**沿用原 M=8**，不替换/筛选/增加）
PHASEA_DECISION_POINTS_DIGEST = \
    '5b432f2826ea66622173ecc514378bae8d11013374d4e93ad7940a392b2de3dc'
PHASEA_N_DECISION_POINTS = 8

#: 用户 2026-10-14 给定的上限（**不是**由数据推导、**不得**事后调整）
CLASS_A_PER_DECISION_CAP_S = 600.0
PHASE_TOTAL_CAP_S = 12000.0

#: `p1a-rolec-v1` 的**双字节身份**（用户 2026-10-14 裁定 (b)：不 renormalize，
#: 并列保留两种表示；跨平台复核必须标明核验的是哪一种）。
BASELINE_DIGESTS = {
    'crlf_worktree': '6d0e57c6d46bf46a61005b8f4d27edd99d8641dd47675e508c39c155f1a635bf',
    'git_lf_blob': '020fc0e7f8177da5fb96d62316c09192417a843b034c44d1e7e8ac3f2d998083',
}

#: Phase A 登记里**禁止**出现的顶层字段（它们属于在线协议，混进来就是口径混乱）
FORBIDDEN_TOP_LEVEL = ('single_decision_time_limit_s',)


class PhaseACeilingError(ValueError):
    """Phase A 登记内容不合法。"""


class PhaseACeilingNotRegistered(RuntimeError):
    """Phase A 登记缺失或未通过校验 ⇒ **拒绝启动 Phase A 作业**。"""


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def occupancy_phasea(reg):
    """过度订阅判据（与旧门同一不等式：`workers × max(thread_caps) ≤ cpu_count`）。"""
    return base.occupancy(reg)


def validate_phasea_ceiling(reg):
    """校验 Phase A 专用登记。返回 `(ok, problems)` —— **fail-closed**。"""
    p = []
    if not isinstance(reg, dict):
        return False, ['登记必须是 dict，得到 %r' % (type(reg).__name__,)]
    if reg.get('is_template') is True:
        return False, ['这是**模板**，不是已核准的登记 ⇒ 拒绝。']

    # ① 身份：必须**另立**，且必须指认父门
    if reg.get('ceiling_id') != PHASEA_CEILING_ID:
        p.append('ceiling_id 必须 = %r（Phase A 专用身份；**不得**复用 %r），得到 %r'
                 % (PHASEA_CEILING_ID, PARENT_CEILING_ID, reg.get('ceiling_id')))
    if reg.get('parent_baseline_ceiling_id') != PARENT_CEILING_ID:
        p.append('parent_baseline_ceiling_id 必须 = %r（说明本门由哪一份基线门派生），'
                 '得到 %r' % (PARENT_CEILING_ID, reg.get('parent_baseline_ceiling_id')))

    # ② 禁止把在线协议时限搬到 Phase A 顶层（用户指出的口径冲突）
    for k in FORBIDDEN_TOP_LEVEL:
        if k in reg:
            p.append('顶层不得出现 %r —— 该字段是在线协议时限；Phase A 的 A/B 两类'
                     '时限必须分别写在 classes.A.internal_deadline（="disabled"）与 '
                     'classes.B.internal_deadline（= 冻结 time_limit）里，'
                     '否则会把"在线协议时限"与"离线诊断监督上限"混为一谈' % k)

    # ③ 两类观测的时限语义
    classes = reg.get('classes')
    if not isinstance(classes, dict):
        p.append('classes 必须是 dict（A 类 / B 类分开登记）')
        classes = {}
    a = classes.get('A') if isinstance(classes.get('A'), dict) else {}
    b = classes.get('B') if isinstance(classes.get('B'), dict) else {}
    if a.get('internal_deadline') != 'disabled':
        p.append('classes.A.internal_deadline 必须 = "disabled"（A 类关闭内部截止），'
                 '得到 %r' % (a.get('internal_deadline'),))
    if not _num(a.get('per_decision_cap_s')) \
            or abs(float(a.get('per_decision_cap_s')) - CLASS_A_PER_DECISION_CAP_S) > 1e-9:
        p.append('classes.A.per_decision_cap_s 必须 = %g s（用户给定，非推导），得到 %r'
                 % (CLASS_A_PER_DECISION_CAP_S, a.get('per_decision_cap_s')))
    if not _num(a.get('scenario_scope_scenarios')) \
            or int(a.get('scenario_scope_scenarios')) < 1:
        p.append('classes.A.scenario_scope_scenarios 必须是 >=1 的 int'
                 '（A 类覆盖该决策点的全部场景）')
    if a.get('counts_as_online_evidence') is not False:
        p.append('classes.A.counts_as_online_evidence 必须 = false —— A 类关闭了内部截止，'
                 '其耗时**不得**用来声称在线可行')
    frozen = base.load_frozen_time_limit()
    if not _num(b.get('internal_deadline')) \
            or abs(float(b.get('internal_deadline')) - float(frozen)) > 1e-9:
        p.append('classes.B.internal_deadline 必须 = 冻结 time_limit=%.6f（**不变**），'
                 '得到 %r' % (float(frozen), b.get('internal_deadline')))

    # ④ 12 000 s 必须覆盖**整个** Phase A 作业
    tot = reg.get('phase_total_cap_s')
    if not _num(tot) or abs(float(tot) - PHASE_TOTAL_CAP_S) > 1e-9:
        p.append('phase_total_cap_s 必须 = %g s（用户给定），得到 %r'
                 % (PHASE_TOTAL_CAP_S, tot))
    if _num(tot) and _num(a.get('per_decision_cap_s')) \
            and float(tot) < float(a['per_decision_cap_s']):
        p.append('phase_total_cap_s(%r) 小于单决策点上限(%r) ⇒ 自相矛盾'
                 % (tot, a.get('per_decision_cap_s')))
    scope = reg.get('phase_scope')
    if not isinstance(scope, (list, tuple)):
        p.append('phase_scope 必须是 list（12 000 s 覆盖哪些阶段）')
        scope = []
    miss = [s for s in PHASE_SCOPE_REQUIRED if s not in scope]
    if miss:
        p.append('phase_scope 缺 %s —— 用户 2026-10-14 明确要求 12 000 s 约束'
                 '**整个 Phase A 作业**（含决策点准备 / L1 / L2 / B 类诊断），'
                 '而不只是 A 类计算循环' % (miss,))
    extra = [s for s in scope if s not in PHASE_SCOPE_REQUIRED]
    if extra:
        p.append('phase_scope 多出 %s —— 12 000 s 预算**只**覆盖 %s。把别的阶段'
                 '（尤其 `L3_profiler`：规格规定 L3 仅在 L1/L2 互相矛盾时**另行申请**、'
                 '**不得自动触发**）塞进 Phase A 范围，等于用预算**隐性授权**它 ⇒ 拒绝'
                 % (extra, list(PHASE_SCOPE_REQUIRED)))
    if len(scope) != len(set(scope)):
        p.append('phase_scope 有重复项：%s' % (scope,))

    # ⑤ 外层监督器：真终止 + 回收 + 保留截断记录
    sup = reg.get('supervision')
    if not isinstance(sup, dict):
        p.append('supervision 必须是 dict（外层监督器规格）')
        sup = {}
    if sup.get('mode') != 'subprocess_hard_kill':
        p.append('supervision.mode 必须 = "subprocess_hard_kill"'
                 '（"已登记上限"≠"已执行硬上限"），得到 %r' % (sup.get('mode'),))
    if sup.get('kill_signal') not in ('SIGKILL', 'SIGKILL_OR_EQUIVALENT'):
        p.append('supervision.kill_signal 必须 = "SIGKILL"（或平台等价），得到 %r'
                 % (sup.get('kill_signal'),))
    if sup.get('reclaim') is not True:
        p.append('supervision.reclaim 必须 = true（超时必须**回收**进程）')
    if sup.get('preserve_truncation_record') is not True:
        p.append('supervision.preserve_truncation_record 必须 = true —— '
                 '截断必须保留"发生在哪个叶子/哪个场景"（REV2 §5.1.2），不得只写"超时"')

    # ⑥ 决策点集合事前固定（用户：沿用原 M=8，**不替换、不筛选、不增加状态**）
    dp = reg.get('decision_points')
    if not isinstance(dp, dict):
        p.append('decision_points 必须是 dict')
        dp = {}
    if dp.get('digest') != PHASEA_DECISION_POINTS_DIGEST:
        p.append('decision_points.digest 必须 = %r（基线捕获的固定集合），得到 %r'
                 % (PHASEA_DECISION_POINTS_DIGEST, dp.get('digest')))
    if not isinstance(dp.get('n'), int) or isinstance(dp.get('n'), bool) \
            or dp.get('n') != PHASEA_N_DECISION_POINTS:
        p.append('decision_points.n 必须 = %d（M=8），得到 %r'
                 % (PHASEA_N_DECISION_POINTS, dp.get('n')))
    if dp.get('policy') != 'fixed_no_replace_no_filter_no_add':
        p.append('decision_points.policy 必须 = "fixed_no_replace_no_filter_no_add"'
                 '（不得为新的成本归因重挑更容易执行的状态），得到 %r'
                 % (dp.get('policy'),))
    if dp.get('rotation_is_not_independent_state') is not True:
        p.append('decision_points.rotation_is_not_independent_state 必须 = true —— '
                 '轮转重复观测**不是** N 个独立状态（当前 M=8），报告必须标注')

    # ⑦ 对账容差事前写死、执行后不得调整
    rec = reg.get('reconciliation')
    if not isinstance(rec, dict):
        p.append('reconciliation 必须是 dict（对账容差与四条件）')
        rec = {}
    if rec.get('tolerance_rule') != 'max(5 ms, 1% * T_decision)':
        p.append('reconciliation.tolerance_rule 必须 = "max(5 ms, 1%% * T_decision)"，'
                 '得到 %r' % (rec.get('tolerance_rule'),))
    if rec.get('adjustable_after_execution') is not False:
        p.append('reconciliation.adjustable_after_execution 必须 = false'
                 '（回归失败时不得放宽容差或重定义 P9）')
    if rec.get('p9_measured_is_unattributed_time') is not True:
        p.append('reconciliation.p9_measured_is_unattributed_time 必须 = true '
                 '（P9_measured = G = 未归因时间）')
    if rec.get('p9_measured_equals_p9_residual') is not False:
        p.append('reconciliation.p9_measured_equals_p9_residual 必须 = false —— '
                 '用户 2026-10-14 裁定：P9_measured 是未归因时间，P9_residual 是'
                 '对账误差，**二者不应相等**；要求的是 residual 在容差内接近零')

    # ⑧ 资源事实与线程上限（沿用旧门的**同一条**过度订阅不等式，不放宽）
    ok_base, base_problems = base.validate_ceiling(_as_legacy_shape(reg))
    for msg in base_problems:
        # 旧校验器会抱怨"缺 single_decision_time_limit_s" —— 那是**预期的**口径分离，
        # 由本模块用 classes.B 承担；其余抱怨一律照抄（不得静默丢弃）。
        if 'single_decision_time_limit_s' in msg:
            continue
        p.append('资源事实（沿用旧门判据）：%s' % msg)
    del ok_base
    try:
        rt._check_no_utility(reg)
    except rt.UtilityLeakError as exc:
        p.append('登记里出现效用类字段：%s' % (exc,))

    # ⑨ 身份留痕
    for k in ('registered_by', 'registered_at', 'hardware', 'workers',
              'cpu_affinity', 'thread_caps'):
        if k not in reg:
            p.append('缺必需字段 %r' % k)
    if not isinstance(reg.get('authorization'), (str, dict)) \
            or not reg.get('authorization'):
        p.append('authorization 必须给出授权来源（谁批的、批了什么）——'
                 '非空字符串或 dict')
    return (not p), p


def _as_legacy_shape(reg):
    """把 Phase A 登记**投影**成旧校验器认识的外形，以复用其资源事实判据。

    ⚠️ 只投影**资源事实**（hardware/workers/affinity/thread_caps）。时限字段故意
    用冻结值填入，使旧校验器的时限规则**总是通过** —— Phase A 的时限语义由本模块
    的 `classes` 段独立裁定，**不**借旧校验器判定。
    """
    out = dict(reg)
    out.pop('single_decision_time_limit_s', None)
    out['single_decision_time_limit_s'] = base.load_frozen_time_limit()
    out['total_wall_clock_limit_s'] = reg.get('phase_total_cap_s')
    out['ceiling_id'] = reg.get('ceiling_id')
    return out


def load_phasea_ceiling(path=None):
    """读 Phase A 登记。不存在/未过校验 ⇒ 抛 `PhaseACeilingNotRegistered`。"""
    p = path or os.path.join(_repo_root(), PHASEA_CEILING_REL.replace('/', os.sep))
    if not os.path.isfile(p):
        raise PhaseACeilingNotRegistered(
            'Phase A 资源门**未登记**：%s 不存在 ⇒ 拒绝启动 Phase A 作业。' % p)
    with io.open(p, encoding='utf-8') as f:
        obj = json.load(f)
    ok, problems = validate_phasea_ceiling(obj)
    if not ok:
        raise PhaseACeilingNotRegistered(
            'Phase A 资源门登记**未通过校验** ⇒ 拒绝启动：\n  - ' + '\n  - '.join(problems))
    return obj


def phasea_status(path=None):
    """只报告状态（**不抛**）。"""
    p = path or os.path.join(_repo_root(), PHASEA_CEILING_REL.replace('/', os.sep))
    if not os.path.isfile(p):
        return {'registered': False, 'path': p, 'problems': ['登记文件不存在']}
    try:
        with io.open(p, encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as exc:                                    # noqa: BLE001
        return {'registered': False, 'path': p,
                'problems': ['无法解析 JSON：%s' % exc]}
    ok, problems = validate_phasea_ceiling(obj)
    return {'registered': bool(ok), 'path': p, 'problems': problems,
            'ceiling_id': obj.get('ceiling_id'),
            'occupancy': occupancy_phasea(obj),
            'phase_total_cap_s': obj.get('phase_total_cap_s')}


def build_phasea_registration(*, registered_by, registered_at=None, workers=8,
                              cpu_affinity=None, thread_caps=None, hardware=None,
                              authorization=None, source_policy=None,
                              per_decision_cap_s=CLASS_A_PER_DECISION_CAP_S,
                              phase_total_cap_s=PHASE_TOTAL_CAP_S,
                              decision_points_digest=PHASEA_DECISION_POINTS_DIGEST,
                              n_decision_points=PHASEA_N_DECISION_POINTS):
    """**纯函数**：组装 Phase A 登记 dict（不写盘、不判通过与否）。

    缺省值全部**照抄用户 2026-10-14 的给定值**，不在这里推导任何上限。
    """
    caps = dict(thread_caps or {})
    for k in base.THREAD_CAP_FIELDS:
        caps.setdefault(k, 1)
    hw = dict(hardware or {'cpu_count': 96,
                           'machine': '服务器 /home/hzeng/project/MASKCO-Main',
                           'mem_gb': 265.3})
    return {
        'ceiling_id': PHASEA_CEILING_ID,
        'parent_baseline_ceiling_id': PARENT_CEILING_ID,
        'registered_by': registered_by,
        'registered_at': registered_at or __import__('datetime').date.today().isoformat(),
        'source_policy': source_policy or (
            'P1A_RoleC_方案1_PhaseA_归因登记_2026-10-14.md REV2 §5.1/§8 + '
            '用户 2026-10-14 Phase A 有条件放行裁定（三项启动前置门）'),
        'hardware': hw,
        'workers': int(workers),
        'cpu_affinity': [int(c) for c in (cpu_affinity if cpu_affinity is not None
                                         else range(8))],
        'thread_caps': {k: int(caps[k]) for k in base.THREAD_CAP_FIELDS},
        'phase_total_cap_s': float(phase_total_cap_s),
        'phase_scope': list(PHASE_SCOPE_REQUIRED),
        'supervision': {
            'mode': 'subprocess_hard_kill',
            'kill_signal': 'SIGKILL',
            'reclaim': True,
            'preserve_truncation_record': True,
            'note': ('"已登记上限"必须同时是"已执行硬上限"：进程内的 SIGALRM 只能中断'
                     '可响应信号的调用；卡在 C 扩展/阻塞 IO 时只有外层超时能真正终止。'),
        },
        'classes': {
            'A': {
                'internal_deadline': 'disabled',
                'per_decision_cap_s': float(per_decision_cap_s),
                'scenario_scope_scenarios': 10,
                'counts_as_online_evidence': False,
                'purpose': '完整路径归因（成本在哪里）；关闭内部截止 ⇒ 不产生在线行为证据',
                'truncation_fields': ['leaf_at_truncation',
                                      'scenario_index_at_truncation',
                                      'n_scenarios_completed_at_truncation',
                                      'elapsed_s_at_truncation'],
            },
            'B': {
                'internal_deadline': float(base.load_frozen_time_limit()),
                'per_decision_cap_s': float(base.load_frozen_time_limit()),
                'purpose': '在线截止诊断（10 s 截止到来时执行到哪一步）—— 时限**不变**',
                'counts_as_online_evidence': True,
            },
        },
        'decision_points': {
            'n': int(n_decision_points),
            'digest': decision_points_digest,
            'policy': 'fixed_no_replace_no_filter_no_add',
            'rotation_is_not_independent_state': True,
            'note': ('沿用基线捕获的固定集合；轮转重复观测只描述测量波动，'
                     '不得人为扩大状态数量；任一点被截断即按 inconclusive_censored '
                     '停止，**不**拿其余点重新定义样本'),
        },
        'reconciliation': {
            'tolerance_rule': 'max(5 ms, 1% * T_decision)',
            'adjustable_after_execution': False,
            'p9_measured_is_unattributed_time': True,
            'p9_measured_equals_p9_residual': False,
            'conditions': ['1_exclusive_vs_offline', '2_unattributed_vs_offline',
                           '3_closure_consistency', '4_structure_invariants'],
            'note': ('条件 ③ 在记账正确时是代数恒等式 ⇒ 它抓实现层故障，不是独立证据；'
                     '独立性来自 ①②（离线区间扫描用另一套算法）'),
        },
        'authorization': dict(authorization) if isinstance(authorization, dict) \
            else (authorization or ''),
    }


def write_phasea_registration(reg, path=None, *, force=False):
    """校验**通过才**落盘；并**拒绝**覆盖父门文件（防手滑把基线登记改掉）。"""
    p = path or os.path.join(_repo_root(), PHASEA_CEILING_REL.replace('/', os.sep))
    parent = os.path.join(_repo_root(), base.CEILING_REL.replace('/', os.sep))
    if os.path.abspath(p) == os.path.abspath(parent):
        raise PhaseACeilingError(
            '拒绝写入：目标路径是**基线登记** %s。Phase A 必须写自己的文件，'
            '**不得**修改或覆盖 %s' % (parent, PARENT_CEILING_ID))
    ok, problems = validate_phasea_ceiling(reg)
    if not ok:
        raise PhaseACeilingError('登记未通过校验 ⇒ **不落盘**：\n  - '
                                 + '\n  - '.join(problems))
    if os.path.isfile(p) and not force:
        raise PhaseACeilingError(
            'Phase A 登记已存在（%s）⇒ 拒绝覆盖。资源门变更请换 ceiling_id 并显式 --force。' % p)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(reg, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    return p


# --------------------------------------------------------------- 父门未被改动（只读）
def baseline_recorded_digests():
    """`p1a-rolec-v1` 的双字节身份（用户裁定 (b)：两种表示并列，互不替代）。"""
    return dict(BASELINE_DIGESTS)


def baseline_status():
    """只读核验 `p1a-rolec-v1`：① 旧校验器仍判它通过；② 字节身份属于已记录的两种之一。

    返回 `{'path','exists','legacy_valid','legacy_problems','byte_identity',
    'sha256','matches_recorded','problems'}`。
    ⚠️ 本函数**只读**，绝不写基线登记。
    """
    p = os.path.join(_repo_root(), base.CEILING_REL.replace('/', os.sep))
    out = {'path': p, 'exists': os.path.isfile(p), 'legacy_valid': False,
           'legacy_problems': [], 'sha256': None, 'byte_identity': None,
           'matches_recorded': None, 'problems': []}
    if not out['exists']:
        out['problems'].append('基线登记不存在')
        return out
    raw = open(p, 'rb').read()
    sha = hashlib.sha256(raw).hexdigest()
    out['sha256'] = sha
    # 判字节身份：CRLF 工作树原件 or LF blob（两种都记录在案）
    if b'\r\n' in raw:
        out['byte_identity'] = 'crlf_worktree'
    else:
        out['byte_identity'] = 'git_lf_blob'
    out['matches_recorded'] = sha in BASELINE_DIGESTS.values()
    if not out['matches_recorded']:
        out['problems'].append(
            '字节身份 %s 不在已记录的双身份里（CRLF=%s / LF=%s）⇒ '
            '父门可能被改写' % (sha[:16], BASELINE_DIGESTS['crlf_worktree'][:16],
                              BASELINE_DIGESTS['git_lf_blob'][:16]))
    try:
        with io.open(p, encoding='utf-8') as f:
            reg = json.load(f)
    except Exception as exc:                                    # noqa: BLE001
        out['problems'].append('无法解析：%s' % exc)
        return out
    ok, problems = base.validate_ceiling(reg)
    out['legacy_valid'] = bool(ok)
    out['legacy_problems'] = list(problems)
    out['ceiling_id'] = reg.get('ceiling_id')
    out['single_decision_time_limit_s'] = reg.get('single_decision_time_limit_s')
    if not ok:
        out['problems'].append('基线登记**未通过旧校验器**：%s' % (problems,))
    if reg.get('ceiling_id') != PARENT_CEILING_ID:
        out['problems'].append('基线 ceiling_id 变了：%r' % (reg.get('ceiling_id'),))
    return out


def _default_authorization():
    return {
        'source': ('用户 2026-10-14 Phase A REV2 审查与最终裁定：'
                   '"原则上批准 Phase A，但实行有条件放行"'),
        'granted': ['建立专用资源登记', '实现测量工装', '完成仪器正确性回归'],
        'conditional': ('三项工程前置门全部满足后才允许按预登记范围启动 L1/L2 测量'),
        'not_granted': ['Phase B', '烟测', '40 天正式批次', '最终测试', 'L3（不得自动启动）'],
        'rulings': {'launch_phase_a': '有条件批准（先过三项工程门）',
                    'who_registers': '由执行端建立；数值已给定，无须重新征询',
                    'decision_points': '沿用原 M=8，固定 digest，不替换/不筛选/不增加',
                    'p9_equals_residual': '否 —— P9_measured 是未归因时间，'
                                          'P9_residual 是对账误差，正常应接近零',
                    'crlf_sha_disposition': '选择 (b)：不 renormalize、不改封存件，'
                                            '另行记录跨平台字节身份限制'},
        'verbatim': ('允许执行端建立专用资源登记、实现测量工装并完成仪器正确性回归。'
                     '只有下述工程前置条件全部满足后，才允许按预登记范围启动 L1/L2 测量。'
                     '不授权 Phase B、烟测、40 天正式批次或最终测试。'),
    }


def main(argv=None):
    """CLI：把 Phase A 专用登记写成文件（**校验不过就不落盘**）。

        python CC_Compare/tools/role_c_phasea_ceiling.py --status
        python CC_Compare/tools/role_c_phasea_ceiling.py --write [--dry-run]
        python CC_Compare/tools/role_c_phasea_ceiling.py --baseline-status
    """
    import argparse
    ap = argparse.ArgumentParser(description='P1-A Role C Phase A 专用资源门')
    ap.add_argument('--write', action='store_true', help='组装并写入 Phase A 登记')
    ap.add_argument('--dry-run', action='store_true', help='只校验并打印，不落盘')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--baseline-status', action='store_true')
    ap.add_argument('--registered-by', default='P1A-rolec-phasea-exec-2026-10-14')
    ap.add_argument('--path', default=None)
    a = ap.parse_args(argv)

    if a.baseline_status:
        print(json.dumps(baseline_status(), ensure_ascii=False, indent=2))
        return 0
    if a.status or not a.write:
        print(json.dumps({'phasea': phasea_status(a.path),
                          'baseline': baseline_status()},
                         ensure_ascii=False, indent=2))
        return 0
    reg = build_phasea_registration(registered_by=a.registered_by,
                                    authorization=_default_authorization())
    ok, problems = validate_phasea_ceiling(reg)
    print(json.dumps({'valid': ok, 'problems': problems,
                      'occupancy': occupancy_phasea(reg),
                      'phase_total_cap_s': reg['phase_total_cap_s'],
                      'classes': reg['classes']}, ensure_ascii=False, indent=2))
    if not ok:
        print('REFUSED: 登记未通过校验 ⇒ 不落盘')
        return 3
    if a.dry_run:
        print('DRY-RUN: 校验通过，未落盘')
        return 0
    p = write_phasea_registration(reg, a.path, force=a.force)
    print('WROTE: %s' % p)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
