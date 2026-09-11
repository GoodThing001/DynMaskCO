"""OR5/OR6：逐实例记录复核（从落盘完整记录重新校验，不信任任何自报值）。

检查层：
  - 身份：instance / scene / seed、code / data / profile / checkpoint hash
    —— expected **全部外源**（DEV_MANIFEST + 冻结身份 + 计算链 hash），
    不从记录自身取（否则 seed/scene 自证，P0-3）；
  - 终局：outcome、hard vector、ownership、terminal、D/Q/E/J_CC 有限；
  - 过程：solver status **成功白名单**（SUCCESS/PARTIAL_SUCCESS/OPTIMAL/
    EMPTY_POOL_FAST_PATH），fallback 精确拒绝；
  - 统计：从 events 重算 solver/fast_path/fallback/warning 并与记录自报
    stats 对账（不一致即失败，P1）；
  - 复现身份：重算 decision_hash / artifact_hash 并与记录对账（篡改检测）；
  - 关键规则：**中间事件出现过 fallback 或失败状态，即使终局 complete 也判
    protocol failure**（fallback 是协议失败不是服务失败）。
"""
import math
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_contract as bc


class InstanceValidationError(RuntimeError):
    pass


# 成功白名单（其余一切状态都是协议失败）。OR-Tools 状态码 1/2/7 映射为
# SUCCESS/PARTIAL_SUCCESS/OPTIMAL；空池快速路径不调用求解器，同样合法。
OK_SOLVER_STATUSES = {'SUCCESS', 'PARTIAL_SUCCESS', 'OPTIMAL',
                      'EMPTY_POOL_FAST_PATH'}
CHECKPOINT_PREFIX = 'ortools=9.11.4210_native='


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def recompute_stats(events):
    """从 events 重算协议统计（不信任 rec['stats'] 自报）。"""
    n_solver = 0
    n_fast = 0
    n_fallback = 0
    n_warn = 0
    for ev in events or []:
        sm = ev.get('solve_meta') or {}
        if sm.get('empty_pool_fast_path'):
            n_fast += 1
        elif sm:
            n_solver += 1
        if ev.get('fallback_triggered'):
            n_fallback += 1
        n_warn += int(sm.get('n_penalty_warnings', 0) or 0)
    return {'n_solver_calls': n_solver, 'n_fast_path': n_fast,
            'fallback_triggered_events': n_fallback,
            'n_penalty_warnings': n_warn}


def recompute_hashes(rec):
    """重算 decision_hash / artifact_hash（与 baseline_contract 同口径）。"""
    return bc.decision_hash(rec), bc.artifact_hash(rec)


def validate_instance_record(rec, expected, recompute_identity=True):
    """expected（全部外源，不得取自已落盘记录）：
    {instance_id, scene_instance_id, instance_seed, data_sha256, profile_hash,
     code_hash, checkpoint_hash}。
    checkpoint_hash 是**完整期望值**（'ortools=9.11.4210_native=' + native
    extension 完整 64 位 SHA-256），由 pre-run 环境身份生成，逐实例精确相等
    （P0-2/P1-3：不能只比前缀或截断值，否则任意 native hash 都通过）。
    recompute_identity=True 时重算 decision_hash/artifact_hash 与记录对账
    （篡改检测；代价为整条记录 JSON 序列化）。
    返回 problem 列表（空 = 通过）。"""
    problems = []

    def check(cond, msg):
        if not cond:
            problems.append(msg)

    check(isinstance(rec, dict), 'record 非 dict')
    if not isinstance(rec, dict):
        return problems

    # ---- 完整公共合同 + trace replay（P0-3：先跑 common 完整 schema/event/
    # action/audit/hard-vector 派生校验 + trace 对账，再做 OR-Tools 特有检查）。
    # 懒加载：record_validation / trace_export 依赖项目脚本路径（经 _bootstrap
    # 引导），推迟到调用时（此时调用方已 import _bootstrap）避免模块导入顺序问题。
    try:
        import _bootstrap  # noqa: F401
        from record_validation import validate_instance_record as _common_validate
        import trace_export as _te
        _common_ready = True
    except Exception as exc:  # noqa: BLE001
        problems.append(f'公共合同校验模块加载失败: {type(exc).__name__}: {exc}')
        _common_ready = False
    if _common_ready:
        err, _ = _common_validate(rec, objective='coldchain', strict_repair=True)
        if err is not None:
            problems.append(f'common 合同校验失败: {err}')
        if rec.get('execution_trace') is not None and rec.get('outcome') is not None:
            ok, replay = _te.trace_replay_check(
                rec['execution_trace'], rec['outcome'], 'coldchain')
            if not ok:
                problems.append(f'trace replay 对账失败: {replay}')

    # ---- 身份（expected 外源）----
    check(rec.get('instance_id') == expected['instance_id'],
          f"instance_id {rec.get('instance_id')} != {expected['instance_id']}")
    check(rec.get('scene_instance_id') == expected['scene_instance_id'],
          f"scene_instance_id {rec.get('scene_instance_id')} != "
          f"{expected['scene_instance_id']}")
    check(rec.get('instance_seed') == expected['instance_seed'],
          f"instance_seed {rec.get('instance_seed')} != {expected['instance_seed']}")
    check(rec.get('data_hash') == expected['data_sha256'],
          'data hash 不一致（NPZ 漂移）')
    check(rec.get('profile_hash') == expected['profile_hash'],
          'profile hash 不一致（v2 profile 漂移）')
    check(rec.get('code_hash') == expected['code_hash'],
          'code_hash 不一致（计算链漂移）')
    check(rec.get('method_name') == 'ortools-rh-d', 'method_name 不一致')
    check(rec.get('checkpoint_hash') == expected['checkpoint_hash'],
          'checkpoint_hash 与冻结 native extension 不一致（环境漂移）')

    # ---- 终局 ----
    outcome = rec.get('outcome')
    check(isinstance(outcome, dict), 'outcome 缺失')
    if isinstance(outcome, dict):
        for k in ('distance_km', 'quality_loss', 'energy_kwh', 'coldchain_cost',
                  'distance_cost'):
            if k in outcome:
                check(_finite(outcome[k]), f'outcome.{k} 非有限')
        check(outcome.get('complete') is True, '终局未 complete')
        check(outcome.get('n_unserved') == 0, 'n_unserved != 0')
        check(outcome.get('n_duplicate') == 0, 'n_duplicate != 0')
    hv = rec.get('hard_vector')
    check(isinstance(hv, dict) and all(hv.values()) if isinstance(hv, dict)
          else False, 'hard vector 未全过')
    audit = rec.get('audit')
    check(isinstance(audit, dict), 'audit 缺失')
    if isinstance(audit, dict):
        check(audit.get('ownership_violations') == 0, 'ownership violations != 0')
        check(audit.get('terminal_unresolved') == 0, 'terminal unresolved != 0')

    # ---- 过程（失败状态白名单 + fallback 精确拒绝）----
    events = rec.get('events', [])
    for ev in events:
        sm = ev.get('solve_meta') or {}
        status = sm.get('solver_status')
        if status is not None:
            check(status in OK_SOLVER_STATUSES,
                  f"事件 {ev.get('event_id')} 出现非成功状态 {status}")
        check(ev.get('fallback_triggered') is not True,
              f"事件 {ev.get('event_id')} fallback_triggered")

    # ---- 统计重算对账（P1：不信任 stats 自报）----
    recomputed = recompute_stats(events)
    stats = rec.get('stats')
    check(isinstance(stats, dict), 'stats 缺失')
    if isinstance(stats, dict):
        for k in ('n_solver_calls', 'n_fast_path', 'fallback_triggered_events',
                  'n_penalty_warnings'):
            check(stats.get(k) == recomputed[k],
                  f'stats.{k} 自报 {stats.get(k)} != 重算 {recomputed[k]}')
    check(recomputed['n_solver_calls'] > 0, '无真实 solver call')

    # ---- 复现身份重算对账（篡改检测）----
    for k in ('decision_hash', 'artifact_hash'):
        check(isinstance(rec.get(k), str) and len(rec[k]) == 64,
              f'{k} 缺失/非法')
    if recompute_identity:
        try:
            d, a = recompute_hashes(rec)
        except Exception as exc:  # noqa: BLE001
            problems.append(f'hash 重算异常: {type(exc).__name__}: {exc}')
        else:
            check(rec.get('decision_hash') == d,
                  'decision_hash 与记录内容不一致（篡改/陈旧）')
            check(rec.get('artifact_hash') == a,
                  'artifact_hash 与记录内容不一致（篡改/陈旧）')

    return problems
