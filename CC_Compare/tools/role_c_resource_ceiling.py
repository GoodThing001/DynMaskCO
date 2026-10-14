#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role C 的**资源上限登记**：机器可校验的登记文件（规格 §6 的"资源门"）。

为什么要有这个模块（不是形式主义）
--------------------------------
规格 §6 把资源计时**排在**"T 系列全绿 **且资源上限登记完成**"之后。若"已登记"只是
一句话，那它挡不住任何东西 —— 本项目已经吃过一次：资格矩阵里 **6 个"受 CPU 争用"臂**
的根因是 **worker 数 × 每 worker 线程数**把机器过度订阅（`lih_51015` 曾 **54.225 次
超时/天、单决策 50.79 s**），而当时代码身份与 v1 **完全相同、从记录里看不出差异**。
⇒ 把登记做成**文件 + 校验器**，并让计时器**未通过校验就拒绝运行**。

登记文件的**必需字段**（`REQUIRED_FIELDS`，缺一即拒绝 —— "没记录 ≠ 没违反"）
------------------------------------------------------------------------
  * `hardware`：机型/核数/内存（事实，不是选择）；
  * `workers`：并发 worker 数（**只影响墙钟、不影响轨迹**，故可自由登记）；
  * `cpu_affinity`：实际使用的 CPU 集（如 `[0..7]`）；
  * `thread_caps`：**必须显式给出四项** —— `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
    `OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS`。规格只点名 OMP 并规定其 **= 1**；
    另外三项**不设值就是漏配**（numpy 走 MKL/OpenBLAS ⇒ 只设 OMP **不足以**消掉
    过度订阅）⇒ 本校验器要求它们**存在且值合法**，但对具体值只做 §"占用率"约束；
  * `single_decision_time_limit_s`：**必须等于冻结配置的 `time_limit`**（当前 10.0
    —— 计时器与正式协议必须同一时限，否则测的不是要跑的东西）；
  * `total_wall_clock_limit_s`：总墙钟上限；
  * `ceiling_id` / `registered_by` / `registered_at`：身份与责任留痕。

**占用率硬约束（不含任何魔数）**：`workers × max(thread_caps.values()) ≤ cpu_count`
—— 这是过度订阅的**定义性**不等式，超了必然互相抢核。

⚠️ 本模块**不写**登记文件，也**不假定**任何上限值；它只校验。登记值由用户裁定后填入
`CC_Compare/P1A_资源门登记.json`（形状见 `CC_Compare/P1A_资源门登记建议_2026-10-14.md`）。
"""
from __future__ import annotations

import io
import json
import os

import resource_timing as rt

__all__ = ['CeilingError', 'CeilingNotRegistered', 'CEILING_REL', 'REQUIRED_FIELDS',
           'THREAD_CAP_FIELDS', 'load_frozen_time_limit', 'validate_ceiling',
           'load_ceiling', 'registration_status', 'live_environment_problems',
           'occupancy']

CEILING_REL = 'CC_Compare/P1A_资源门登记.json'

REQUIRED_FIELDS = ('ceiling_id', 'registered_by', 'registered_at', 'hardware',
                   'workers', 'cpu_affinity', 'thread_caps',
                   'single_decision_time_limit_s', 'total_wall_clock_limit_s',
                   'source_policy')

# 四项都必须**显式**给出（规格只点名 OMP；另外三项漏配会造成过度订阅）
THREAD_CAP_FIELDS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS')

_REQUIRED_THREAD_CAP_VALUE = {'OMP_NUM_THREADS': 1}   # 规格 §6 明文要求


class CeilingError(ValueError):
    """登记内容不合法。"""


class CeilingNotRegistered(RuntimeError):
    """登记文件缺失或未通过校验 —— **拒绝运行计时**（规格 §6）。"""


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frozen_time_limit():
    p = os.path.join(_repo_root(), 'CC_Compare', 'P1A_CONTROL_FROZEN.json')
    with io.open(p, encoding='utf-8') as f:
        return float(json.load(f)['frozen_config']['time_limit'])


def load_frozen_time_limit():
    """冻结配置里的 `time_limit`（登记必须与它一致）。"""
    return _frozen_time_limit()


def occupancy(ceiling):
    """`workers × max(thread_caps)` 与 `cpu_count` —— 过度订阅的判据。"""
    caps = (ceiling or {}).get('thread_caps') or {}
    vals = [int(v) for v in caps.values() if isinstance(v, int)
            and not isinstance(v, bool)] or [1]
    w = int((ceiling or {}).get('workers') or 0)
    return {'workers': w, 'max_thread_cap': max(vals),
            'cores_requested': w * max(vals),
            'cpu_count': os.cpu_count()}


def validate_ceiling(ceiling):
    """校验登记内容。返回 `(ok, problems)` —— **fail-closed**，缺一即不通过。"""
    p = []
    if not isinstance(ceiling, dict):
        return False, ['登记必须是 dict，得到 %r' % (type(ceiling).__name__,)]
    # ⓪ **模板永远不是登记**：即使被改名放到登记路径上，也必须被拒 ——
    #    否则有人把模板复制过去就会把资源门**意外打开**（那正是本模块要防的）。
    if ceiling.get('is_template') is True:
        return False, ['这是**模板**（`is_template: true`），不是已核准的登记 ⇒ 拒绝。'
                       '请按 CC_Compare/P1A_资源门登记建议_2026-10-14.md 的 §3 填入'
                       '用户裁定的值，并去掉 `is_template`。']
    # ① 必需字段齐备（"没记录 ≠ 没违反"）
    miss = [k for k in REQUIRED_FIELDS if k not in ceiling]
    if miss:
        p.append('缺必需字段 %s' % (miss,))
    # ② 不得夹带效用字段（登记是资源门，不是结果表）
    try:
        rt._check_no_utility(ceiling)
    except rt.UtilityLeakError as exc:
        p.append('登记里出现效用类字段：%s' % (exc,))
    # ③ 线程上限：四项显式 + OMP 必须 = 1
    caps = ceiling.get('thread_caps')
    if not isinstance(caps, dict):
        p.append('thread_caps 必须是 dict（四项线程上限）')
        caps = {}
    for k in THREAD_CAP_FIELDS:
        v = caps.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            p.append('thread_caps.%s 必须显式给出 >=1 的 int（得到 %r）：'
                     '**漏配**这三项会造成 BLAS 线程过度订阅 —— 规格只点名 OMP，'
                     '但 numpy 走 MKL/OpenBLAS，只设 OMP 不足以消掉该风险' % (k, v))
    for k, want in _REQUIRED_THREAD_CAP_VALUE.items():
        if caps.get(k) != want:
            p.append('thread_caps.%s 必须 = %r（规格 §6 明文），得到 %r'
                     % (k, want, caps.get(k)))
    # ④ workers / affinity
    w = ceiling.get('workers')
    if not isinstance(w, int) or isinstance(w, bool) or w < 1:
        p.append('workers 必须是 >=1 的 int，得到 %r' % (w,))
        w = 0
    aff = ceiling.get('cpu_affinity')
    if not isinstance(aff, (list, tuple)) or not aff:
        p.append('cpu_affinity 必须是非空 list（实际要用的 CPU 集）')
        aff = []
    else:
        if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in aff):
            p.append('cpu_affinity 只能含非负 int')
        if len(aff) < w:
            p.append('cpu_affinity 有 %d 个核，少于 workers=%d ⇒ 必然互相抢核'
                     % (len(aff), w))
        ncpu = os.cpu_count() or 0
        if ncpu and any(c >= ncpu for c in aff if isinstance(c, int)):
            p.append('cpu_affinity 含 >= cpu_count(%d) 的核号' % ncpu)
    # ⑤ 过度订阅的**定义性**不等式（不含魔数）
    occ = occupancy(ceiling)
    if w and occ['cores_requested'] > (os.cpu_count() or 0) > 0:
        p.append('workers(%d) × max(thread_caps)(%d) = %d > cpu_count(%d) '
                 '⇒ **过度订阅**（资格矩阵 6 个受争用臂的根因）'
                 % (w, occ['max_thread_cap'], occ['cores_requested'], occ['cpu_count']))
    # ⑥ 单决策时限必须与冻结协议一致
    tl = ceiling.get('single_decision_time_limit_s')
    frozen = _frozen_time_limit()
    if not isinstance(tl, (int, float)) or isinstance(tl, bool):
        p.append('single_decision_time_limit_s 必须是数值')
    elif abs(float(tl) - frozen) > 1e-9:
        p.append('single_decision_time_limit_s=%.6f 必须等于冻结 time_limit=%.6f'
                 '（计时必须测**要跑的那个**时限）' % (float(tl), frozen))
    # ⑦ 总墙钟上限
    tot = ceiling.get('total_wall_clock_limit_s')
    if not isinstance(tot, (int, float)) or isinstance(tot, bool) or float(tot) <= 0:
        p.append('total_wall_clock_limit_s 必须是 > 0 的数值，得到 %r' % (tot,))
    # ⑧ 身份留痕格式
    for k in ('ceiling_id',):
        v = ceiling.get(k)
        if not isinstance(v, str) or not v.strip():
            p.append('%s 必须是非空字符串（登记身份）' % k)
    hw = ceiling.get('hardware')
    if not isinstance(hw, dict) or not hw:
        p.append('hardware 必须是非空 dict（机型/核数/内存等事实）')
    return (not p), p


def load_ceiling(path=None):
    """读登记文件。**不存在 ⇒ 抛 `CeilingNotRegistered`**（不是返回空 dict）。"""
    p = path or os.path.join(_repo_root(), CEILING_REL.replace('/', os.sep))
    if not os.path.isfile(p):
        raise CeilingNotRegistered(
            '资源上限**未登记**：%s 不存在。规格 §6 要求"资源上限登记完成之后"才做资源\n'
            '计时 ⇒ 计时器拒绝运行。候选值见 CC_Compare/P1A_资源门登记建议_2026-10-14.md\n'
            '（§3 全部为候选，待用户逐项裁定后填成登记文件）。' % p)
    with io.open(p, encoding='utf-8') as f:
        obj = json.load(f)
    ok, problems = validate_ceiling(obj)
    if not ok:
        raise CeilingNotRegistered('资源上限登记**未通过校验**，拒绝运行计时：\n  - '
                                   + '\n  - '.join(problems))
    return obj


def registration_status(path=None):
    """只报告状态（**不抛**）：`{'registered', 'path', 'problems', 'occupancy'}`。"""
    p = path or os.path.join(_repo_root(), CEILING_REL.replace('/', os.sep))
    if not os.path.isfile(p):
        return {'registered': False, 'path': p,
                'problems': ['登记文件不存在'], 'occupancy': None}
    try:
        with io.open(p, encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as exc:                                  # noqa: BLE001
        return {'registered': False, 'path': p,
                'problems': ['无法解析 JSON：%s' % exc], 'occupancy': None}
    ok, problems = validate_ceiling(obj)
    return {'registered': bool(ok), 'path': p, 'problems': problems,
            'occupancy': occupancy(obj), 'ceiling_id': obj.get('ceiling_id')}


def live_environment_problems(ceiling, *, env=None, affinity=None):
    """**实参**与登记的对照（规格 §6/建议 §4：要记实际值，不是打算用的值）。

    返回问题清单（空 = 一致）。`env`/`affinity` 可注入以便离线测试。
    """
    env = os.environ if env is None else env
    p = []
    caps = (ceiling or {}).get('thread_caps') or {}
    for k in THREAD_CAP_FIELDS:
        want = caps.get(k)
        got = env.get(k)
        if got is None:
            p.append('进程环境里 %s 未设置（登记要求 %r）—— 运行时线程上限**未生效**，'
                     '耗时不可解释' % (k, want))
        else:
            try:
                if int(got) != int(want):
                    p.append('%s 实际=%r 与登记=%r 不符' % (k, got, want))
            except (TypeError, ValueError):
                p.append('%s 实际=%r 不是整数（登记=%r）' % (k, got, want))
    aff = ceiling.get('cpu_affinity') or []
    if affinity is None:
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except AttributeError:
            affinity = None
    if affinity is not None and aff:
        if sorted(aff) != sorted(affinity):
            p.append('CPU 亲和性实际=%s 与登记=%s 不符（未用 taskset 或核集不同）'
                     % (affinity, sorted(aff)))
    return p
