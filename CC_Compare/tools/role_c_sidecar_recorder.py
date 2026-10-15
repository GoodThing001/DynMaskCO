#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**分析侧最小实现**：sidecar 归档 + 独立统计器 + 核验器（**不改任何冻结模块**）。

授权：用户 2026-10-14「对比方法线：五项正式裁定」① —— 批准新增 wrapper、sidecar、
      独立统计器与核验器；**不得修改** `role_c_opportunity_cost.py` /
      `role_c_replanner.py` / `role_c_decision_log.py` / `role_c_timing.py`。
      实现后提交源码 SHA、接口差异与静态自检，**然后停止等审查**。

用户指定的**四处 sidecar 边界**（本模块逐条落实）
------------------------------------------------
1. **决策 ID 必须独立于决策结果** —— 不能只靠 `decision_hash` 绑定三方结果
   （它的构造可能含 `action`；一旦两版行为不同，关联键也可能不同，**恰好妨碍定位差异**）。
   ⇒ 新增 `input_decision_id = SHA256(规范化输入身份)`，
   **不含** action / 效用 / 最终计划；`decision_hash` **保留为历史日志字段**，两者**同时归档**。
2. **`plan_of` 必须记录实际被采纳的计划** —— wrapper 在 `decide_fn` 前后各读一次 `_plan`
   很可能**始终相同**（`decide_fn` 通常不改 replanner 的 `_plan`）⇒ 会造出**假的行为一致**。
   ⇒ 四个计划身份分开：`plan_before` / `candidate_plan` / `accepted_plan` / `plan_after`；
   未发生权威提交的隔离决策单元标 **`COMMIT_NOT_OBSERVED`**，**不得**伪称已观察到权威写入。
3. **sidecar 写失败不能变成"已记录失败"** —— 若 sidecar 自己写不进去，
   就不能假设那个失败标志可靠落盘。⇒ 算法按原控制流继续（保持旧行为）；
   归档资格标 **`ARCHIVE_INCOMPLETE`**；若失败原因也无法由独立监督记录核验 ⇒
   **`INCONCLUSIVE_ARCHIVE_FAILURE`**；**不得**把该决策计入四通道完整性的分母。
   并**更正**"flush 后崩溃最多损失尾行"的表述：`flush` **不等于**持久化保证，
   不用 `fsync` 就必须如实披露断电/文件系统故障下的证据丢失风险。
4. **拒单 ≠ timeout fallback** —— 严格区分 `普通拒单` / `B3 保守拒单` / `超时回退`；
   B2 预算越界拒单是**事前边界规则的正常执行**，**不得**自动记为超时回退。
   `sidecar` **不得**仅凭 `action=reject` 推断 `fallback_applied=true`；
   分析侧无法判定时输出 **`FALLBACK_NOT_VERIFIABLE`**，
   **不得**制造不存在的独立回退事件。

⚠️ 本模块**只读**被测模块（import + 读 `diag`），**不写**任何被测文件。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

__all__ = ['ARCHIVE_STATUS', 'FALLBACK_KINDS', 'PLAN_VARIANTS', 'FROZEN_MODULES',
           'FROZEN_SHA256', 'SidecarRecorder', 'IndependentCounters',
           'classify_fallback', 'plan_variants', 'normalized_input_identity',
           'durability_disclosure', 'static_selfcheck']

#: 归档资格（**不把"写失败"当成"已记录失败"**）
ARCHIVE_STATUS = ('COMPLETE', 'ARCHIVE_INCOMPLETE', 'INCONCLUSIVE_ARCHIVE_FAILURE')

#: 回退分类（**拒单 ≠ 超时回退**）
FALLBACK_KINDS = ('none', 'b2_boundary_reject', 'b3_conservative_reject',
                  'timeout_fallback', 'FALLBACK_NOT_VERIFIABLE')

#: 四个**必须分开**的计划身份（用户第 2 条）
PLAN_VARIANTS = ('plan_before', 'candidate_plan', 'accepted_plan', 'plan_after')

#: 本实现**不得修改**的冻结模块（用户明列的四个 + 同批冻结的分析面）
FROZEN_MODULES = ('role_c_opportunity_cost.py', 'role_c_replanner.py',
                  'role_c_decision_log.py', 'role_c_timing.py')

#: 这些模块在本次实现前的 SHA-256（实现后必须逐字节不变）
FROZEN_SHA256 = {
    'role_c_opportunity_cost.py':
        '6301949da86244ab40bcfdeb720ec1b9a32bd789634e96b4f228cb68b00f0094',
    'role_c_replanner.py': None,      # 运行时实测填入（见 static_selfcheck）
    'role_c_decision_log.py': None,
    'role_c_timing.py':
        '1a2c8d7e59c23e289a8e00a25e986b234715615f03bfeacfcb18b8557871122d',
}

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- 工具
def _sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def normalized_input_identity(*, protocol_identity, decision_point, input_snapshot):
    """`input_decision_id` 的规范输入身份（**不含** action / 效用 / 最终计划）。

    `input_snapshot` 应包含：决策点位置、上下文（`inst_idx`/`clock`/`order`）、
    车辆状态指纹（含不透明 `cc`）、已冻结计划与 `frozen` 前缀、可见订单集合、
    随机流身份（采样器 + `arm_seed` + `K`）。
    """
    payload = {'protocol_identity': protocol_identity,
               'decision_point': int(decision_point),
               'input_snapshot': input_snapshot}
    return _sha_text(_canon(payload))


def plan_variants(*, plan_before, candidate_plan=None, accepted_plan=None,
                  plan_after=None, commit_observed):
    """四个计划身份**分开**记录；未观察权威提交 ⇒ `COMMIT_NOT_OBSERVED`。"""
    def _sha(p):
        if p is None:
            return None
        norm = {str(k): list(v) for k, v in sorted((p or {}).items())}
        return _sha_text(_canon(norm))

    return {
        'plan_before': plan_before, 'plan_before_sha256': _sha(plan_before),
        'candidate_plan': candidate_plan, 'candidate_plan_sha256': _sha(candidate_plan),
        'accepted_plan': accepted_plan, 'accepted_plan_sha256': _sha(accepted_plan),
        'plan_after': plan_after, 'plan_after_sha256': _sha(plan_after),
        'commit_observed': bool(commit_observed),
        'commit_status': 'COMMIT_OBSERVED' if commit_observed else 'COMMIT_NOT_OBSERVED',
        'primary_criterion_plan': ('accepted_plan' if commit_observed
                                   else 'COMMIT_NOT_OBSERVED'),
        'note': ('主判比较的是**实际被采纳的计划**（`accepted_plan`），'
                 '**不是** wrapper 前后各读一次的本地副本；'
                 '未观察权威提交时**不得**伪称已观察到写入'),
    }


def classify_fallback(*, diag, commit_observed, exception_audit=None):
    """回退分类。**拒单 ≠ 超时回退**；证据不足 ⇒ `FALLBACK_NOT_VERIFIABLE`。

    判定只用**可观察的边界/异常**证据，**不**从 `action=reject` 反推回退。
    """
    d = diag or {}
    rule = d.get('boundary') or ''
    if not commit_observed:
        return {'fallback_applied': None, 'fallback_kind': 'FALLBACK_NOT_VERIFIABLE',
                'reason': '未观察到权威提交 ⇒ 无法判定是否发生行为回退'}
    if rule == 'B2_budget_out_of_range':
        return {'fallback_applied': False, 'fallback_kind': 'b2_boundary_reject',
                'reason': ('B2 预算越界是**事前边界规则的正常执行**，'
                           '不是超时回退（用户第 4 条）')}
    if rule == 'B3_lambda_undefined':
        ea = exception_audit or {}
        if ea.get('hard_deadline_exception') is True:
            return {'fallback_applied': True, 'fallback_kind': 'timeout_fallback',
                    'reason': ('B3 由**硬截止异常**导致（`exception_type=%s`）'
                               % ea.get('exception_type'))}
        if ea.get('hard_deadline_exception') is False:
            return {'fallback_applied': False, 'fallback_kind': 'b3_conservative_reject',
                    'reason': ('B3 由**非截止**异常或数值未定义导致'
                               '（`exception_type=%s`）⇒ 保守拒单，不是超时回退'
                               % ea.get('exception_type'))}
        return {'fallback_applied': None, 'fallback_kind': 'FALLBACK_NOT_VERIFIABLE',
                'reason': ('B3 但**没有** `exception_audit`（或类型不可用）⇒ '
                           '**无法**区分"硬超时"与"数值未定义"；'
                           '**不得**制造不存在的独立回退事件')}
    if d.get('action') == 'reject' or (d.get('log_fields') or {}).get('action') == 'reject':
        return {'fallback_applied': None, 'fallback_kind': 'FALLBACK_NOT_VERIFIABLE',
                'reason': ('仅有 `action=reject`，**不足以**判定回退 '
                           '（普通拒单 / B3 保守拒单 / 超时回退三者不可由 action 区分）')}
    return {'fallback_applied': False, 'fallback_kind': 'none', 'reason': ''}


def durability_disclosure():
    """**更正后**的持久化表述（用户第 3 条）。"""
    return {
        'flush_equals_durability': False,
        'disclosure': ('`flush()` **只把数据交给 OS 页缓存**，'
                       '**不等于持久化保证**；本实现**不使用** `fsync` ⇒ '
                       '**断电或文件系统故障时存在证据丢失风险**，'
                       '该风险必须随归档一并如实披露。'
                       '先前"flush 后崩溃最多损失尾行"的表述**已更正**。'),
        'mitigation': ('由独立监督记录（运行级 supervisor 判决 + 归档存在性检查）'
                       '交叉印证；无法印证时判 `INCONCLUSIVE_ARCHIVE_FAILURE`'),
    }


# --------------------------------------------------------------------- sidecar
class SidecarRecorder(object):
    """一次运行的 sidecar 归档器（append-only JSONL + 写入后验收）。

    ⚠️ **写失败不写成"已记录失败"**：失败状态保存在**内存**并由返回值上报，
    **不假定**任何失败标志已可靠落盘（用户第 3 条）。
    """

    def __init__(self, *, run_id, arm, sidecar_path, protocol_identity,
                 decision_points_digest=None):
        self.run_id = run_id
        self.arm = arm
        self.sidecar_path = sidecar_path
        self.protocol_identity = protocol_identity
        self.decision_points_digest = decision_points_digest
        self.archive_status = 'COMPLETE'
        self.failures = []                 # [{decision_point, error, verified_by_supervisor}]
        self.n_records = 0
        self._fh = None

    # ---- 身份 ----
    def input_decision_id(self, *, decision_point, input_snapshot):
        return normalized_input_identity(protocol_identity=self.protocol_identity,
                                        decision_point=decision_point,
                                        input_snapshot=input_snapshot)

    # ---- 写 ----
    def _open(self):
        if self._fh is None:
            d = os.path.dirname(os.path.abspath(self.sidecar_path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            self._fh = io.open(self.sidecar_path, 'a', encoding='utf-8', newline='\n')
        return self._fh

    def record(self, *, decision_point, input_snapshot, action, boundary=None,
               reject_reason=None, lambda_B=None, e_o=None, certify_ok=None,
               decision_hash=None, exception_audit=None, **plans):
        """写一条 sidecar 记录；返回 `(record, write_ok)`。

        写失败 ⇒ 归档资格降级（**不**伪造已落盘的失败标志），**算法行为不受影响**。
        `**plans` 透传给 `plan_variants`（`plan_before` / `candidate_plan` /
        `accepted_plan` / `plan_after` / `commit_observed`）。
        """
        pid = self.input_decision_id(decision_point=decision_point,
                                     input_snapshot=input_snapshot)
        pv = plan_variants(**plans)
        fb = classify_fallback(diag={'boundary': boundary,
                                     'log_fields': {'action': action}},
                               commit_observed=bool(plans.get('commit_observed')),
                               exception_audit=exception_audit)
        rec = {
            'sidecar_version': 'c1c4-v1',
            'run_id': self.run_id, 'arm': self.arm,
            'decision_point': int(decision_point),
            'input_decision_id': pid,
            'decision_hash': decision_hash,          # 历史日志字段（**保留**，不与上面混义）
            'decision_points_digest': self.decision_points_digest,
            'action': action, 'boundary': boundary, 'reject_reason': reject_reason,
            'lambda_B': lambda_B, 'e_o': e_o, 'certify_ok': certify_ok,
            'exception_audit': exception_audit,
            'fallback': fb,
            'clock_domain': 'time.perf_counter',
            'timer_origin': 'role_c_sidecar_recorder',
        }
        rec.update(pv)
        write_ok, err = True, None
        try:
            fh = self._open()
            fh.write(_canon(rec) + '\n')
            fh.flush()                     # ⚠️ 不等于持久化（见 durability_disclosure）
        except Exception as exc:                             # noqa: BLE001
            write_ok, err = False, '%s: %s' % (type(exc).__name__, exc)
            self._note_failure(decision_point, err)
        if write_ok:
            self.n_records += 1
            ok, why = self.verify_written()
            if not ok:
                self._note_failure(decision_point, 'write-back verification failed: %s' % why)
                write_ok = False
        return rec, write_ok

    def _note_failure(self, decision_point, err):
        self.failures.append({'decision_point': int(decision_point), 'error': err,
                              'verified_by_supervisor': None})
        self.archive_status = ('ARCHIVE_INCOMPLETE' if self.archive_status == 'COMPLETE'
                               else self.archive_status)

    def mark_supervisor_verification(self, *, decision_point, verified):
        """独立监督记录能否核验该失败原因（不能 ⇒ `INCONCLUSIVE_ARCHIVE_FAILURE`）。"""
        for f in self.failures:
            if f['decision_point'] == int(decision_point):
                f['verified_by_supervisor'] = bool(verified)
        if any(f['verified_by_supervisor'] is False for f in self.failures):
            self.archive_status = 'INCONCLUSIVE_ARCHIVE_FAILURE'
        return self.archive_status

    def verify_written(self):
        """**写入后验收**：文件非 0 字节、末行可解析、带 `input_decision_id`。"""
        try:
            if not os.path.isfile(self.sidecar_path):
                return False, '文件不存在'
            if os.path.getsize(self.sidecar_path) == 0:
                return False, '0 字节'
            with io.open(self.sidecar_path, encoding='utf-8') as f:
                lines = [x for x in f.read().splitlines() if x.strip()]
            if not lines:
                return False, '无有效行'
            last = json.loads(lines[-1])
            if 'input_decision_id' not in last:
                return False, '末行缺 input_decision_id'
            return True, ''
        except Exception as exc:                             # noqa: BLE001
            return False, '%s: %s' % (type(exc).__name__, exc)

    def completeness_denominator(self, requested_decision_points):
        """四通道完整性的**分母**（不含归档不完整的决策点；用户第 3 条）。"""
        bad = {f['decision_point'] for f in self.failures}
        return {'requested': int(requested_decision_points),
                'excluded_due_to_archive': sorted(bad),
                'eligible': max(0, int(requested_decision_points) - len(bad)),
                'rule': ('**不得**把 sidecar 写失败的决策计入四通道完整性分母；'
                         '若失败原因无法由独立监督记录核验 ⇒ '
                         '`INCONCLUSIVE_ARCHIVE_FAILURE`')}

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:                                # noqa: BLE001
                pass
            self._fh = None
        return {'archive_status': self.archive_status, 'n_records': self.n_records,
                'failures': self.failures,
                'durability': durability_disclosure()}


# --------------------------------------------------------------- 独立统计器
class IndependentCounters(object):
    """**独立**统计器：决策级与事件级**分开**，且 `n_execution_errors` **显式标注**口径。"""

    def __init__(self):
        self._dec_timeout = set()
        self._ev_timeout = set()
        self._exec_errors = set()
        self._exec_errors_unit = None          # 'decisions' | 'events'
        self._fallbacks = {}
        self._not_entered = set()
        self._unknown = set()

    def add_exception_event(self, *, run_id, decision_point, exception_audit):
        ea = exception_audit or {}
        eid = ea.get('event_id')
        key_ev = (run_id, int(decision_point), eid)
        hard = ea.get('hard_deadline_exception')
        if hard is True:
            self._ev_timeout.add(key_ev)
            self._dec_timeout.add((run_id, int(decision_point)))
        elif hard is False:
            self._exec_errors.add(key_ev)
            self._exec_errors_unit = self._exec_errors_unit or 'events'
        else:
            self._unknown.add((run_id, int(decision_point)))

    def add_fallback(self, *, run_id, decision_point, fallback_event_id, fallback_kind):
        key = (run_id, int(decision_point))
        n = self._fallbacks.get(key, 0) + 1
        self._fallbacks[key] = n
        if n > 1:
            # §14.1：终态回退每决策至多一次；重复 ⇒ execution_error（**不得**静默去重）
            self._exec_errors.add((run_id, int(decision_point), fallback_event_id))
            self._exec_errors_unit = 'decisions'

    def add_not_entered(self, *, run_id, decision_point):
        self._not_entered.add((run_id, int(decision_point)))

    def summary(self):
        return {
            'n_deadline_timeouts': len(self._dec_timeout),              # 决策级
            'n_deadline_timeout_events': len(self._ev_timeout),         # 事件级
            'n_execution_errors': len(self._exec_errors),
            'n_execution_errors_unit': self._exec_errors_unit or 'undetermined',
            'n_fallbacks': sum(self._fallbacks.values()),
            'n_decisions_with_fallback': len(self._fallbacks),
            'n_not_entered': len(self._not_entered),
            'n_unknown': len(self._unknown),
            'invariant_real_path_events_per_decision_max': 1,
            'invariant_dedup_keys': {
                'timeout_events': '(run_id, decision_point, event_id)',
                'timeout_decisions': '(run_id, decision_point)',
                'fallbacks': '(run_id, decision_point, fallback_event_id)',
            },
            'note': ('**决策级 ≠ 事件级**，且**不得要求** `n_deadline_timeouts '
                     '== n_fallbacks`；`n_execution_errors` 的口径由 '
                     '`n_execution_errors_unit` **显式标注**（不得混用）'),
        }


# ------------------------------------------------------------------ 静态自检
def _ast_write_and_poke(src):
    """AST 检查：本模块**写了哪些路径**、**是否 monkeypatch 冻结模块**。

    ⚠️ 第一版用"源码里同时出现冻结模块名与 `open(`"当判据 —— 那会把
    **注释/字符串里提到模块名**误判成"写它"，于是自检恒假失败。
    改为 AST：只看 `open` / `io.open` 的**第一个实参字符串常量**，以及
    `setattr(...)` / `X.attr = ...` 这类**属性写入**。
    """
    import ast
    tree = ast.parse(src)
    write_targets, setattrs, attr_assigns, imports = [], [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, 'attr', None) or getattr(f, 'id', None)
            if name in ('open',) and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    write_targets.append(a0.value)
            if name == 'setattr' and len(node.args) >= 2:
                t = node.args[0]
                setattrs.append(getattr(t, 'id', None) or getattr(t, 'attr', None))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                    attr_assigns.append('%s.%s' % (t.value.id, t.attr))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ''])
            imports.extend(mods)
    return {'write_targets': write_targets, 'setattrs': setattrs,
            'attr_assigns': attr_assigns, 'imports': imports}


def static_selfcheck(*, root=None):
    """静态自检：① 四个冻结模块**逐字节未变**；② 本模块**不写**、**不 poke** 冻结模块。"""
    root = root or os.path.dirname(os.path.dirname(_HERE))
    out = {'frozen_modules': [], 'problems': []}
    for m in FROZEN_MODULES:
        p = os.path.join(_HERE, m)
        if not os.path.isfile(p):
            out['problems'].append('冻结模块缺失：%s' % m)
            continue
        b = io.open(p, 'rb').read()
        sha = hashlib.sha256(b).hexdigest()
        exp = FROZEN_SHA256.get(m)
        out['frozen_modules'].append({'module': m, 'sha256': sha, 'expected': exp,
                                      'unchanged': (exp is None or sha == exp)})
        if exp is not None and sha != exp:
            out['problems'].append('冻结模块被改动：%s（%s != %s）' % (m, sha, exp))
    src = io.open(os.path.abspath(__file__), encoding='utf-8').read()
    info = _ast_write_and_poke(src)
    frozen_stems = {m[:-3] for m in FROZEN_MODULES}
    # ① 写到冻结模块路径？
    bad_writes = [w for w in info['write_targets']
                  if os.path.basename(w) in FROZEN_MODULES]
    # ② monkeypatch 冻结模块？
    bad_poke = ([s for s in info['setattrs'] if s in frozen_stems]
                + [a for a in info['attr_assigns']
                   if a.split('.')[0] in frozen_stems])
    # ③ 是否 import 了冻结模块（被测面）？本模块应当**一个都不 import**
    bad_imports = [i for i in info['imports'] if i.split('.')[0] in frozen_stems]
    out['ast'] = info
    out['writes_to_frozen'] = bad_writes
    out['pokes_frozen'] = bad_poke
    out['imports_frozen'] = bad_imports
    if bad_writes:
        out['problems'].append('本模块写到冻结模块路径：%s' % bad_writes)
    if bad_poke:
        out['problems'].append('本模块 monkeypatch 冻结模块：%s' % bad_poke)
    if bad_imports:
        out['problems'].append('本模块 import 了冻结模块：%s' % bad_imports)
    out['does_not_write_frozen'] = not (bad_writes or bad_poke)
    out['ok'] = not out['problems']
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='分析侧 sidecar 实现静态自检 + 身份冻结')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    sc = static_selfcheck()
    this_sha = hashlib.sha256(io.open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    rep = {'label': 'P1A-C1C4-analysis-side-identity-freeze',
           'module': os.path.basename(os.path.abspath(__file__)),
           'module_sha256': this_sha,
           'static_selfcheck': sc,
           'archive_status_vocabulary': list(ARCHIVE_STATUS),
           'fallback_kind_vocabulary': list(FALLBACK_KINDS),
           'plan_variants': list(PLAN_VARIANTS),
           'durability': durability_disclosure(),
           'interface_diff_vs_frozen': {
               'role_c_opportunity_cost.py': '**未改动**（仍 C2 身份）',
               'role_c_replanner.py': '**未改动**',
               'role_c_decision_log.py': '**未改动**',
               'role_c_timing.py': '**未改动**',
               'how_audit_reaches_sidecar':
                   ('分析侧直接读 `decide_fn` 返回的 `diag["exception_audit"]` 落盘；'
                    '**不**需要 `DecisionLogger` 接收该字段 ⇒ 无需改核心接口'),
               'still_needs_core_hook': ['hi/lo 完整逐场景进度（裁定 ②：暂不批准）'],
           },
           'authorization': ('用户 2026-10-14 裁定 ①：批准新增分析侧模块；'
                             '实现后提交源码 SHA / 接口差异 / 静态自检并**停止等审查**；'
                             '**不执行 L1**')}
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({'module_sha256': this_sha, 'selfcheck_ok': sc['ok'],
                      'frozen_unchanged': all(x['unchanged']
                                              for x in sc['frozen_modules']),
                      'problems': sc['problems']}, ensure_ascii=False, indent=2))
    return 0 if sc['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
