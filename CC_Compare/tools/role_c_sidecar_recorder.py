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

#: 四个冻结模块的**双身份**（用户 2026-10-14 指出：`expected=None` 会让自检恒真）。
#: ⚠️ 有 `eol=lf` 属性的模块 blob == 工作树；**没有**属性的（如 `role_c_replanner.py`）
#: 工作树是 CRLF ⇒ **blob ≠ 工作树**，两者必须**分别核验**（封存 §2.1 登记的是工作树字节）。
FROZEN_DUAL_IDENTITY = {
    'role_c_opportunity_cost.py': {
        'blob_sha256':
            '6301949da86244ab40bcfdeb720ec1b9a32bd789634e96b4f228cb68b00f0094',
        'worktree_sha256':
            '6301949da86244ab40bcfdeb720ec1b9a32bd789634e96b4f228cb68b00f0094',
        'blob_equals_worktree': True},
    'role_c_replanner.py': {
        'blob_sha256':
            '1835d79691d1820f1ac0ed9cf8b1022a2218823ae77b6458d1f4381bad3d8ae1',
        'worktree_sha256':
            '92cd185c8287cac711930665405401fe6f3fd4ad9f339d372c1effcc72566f98',
        'blob_equals_worktree': False},
    'role_c_decision_log.py': {
        'blob_sha256':
            'f618a2bd93dd8d8a23cd96095a7fcda37438c5037ba93ea65a75a5e721249515',
        'worktree_sha256':
            'f618a2bd93dd8d8a23cd96095a7fcda37438c5037ba93ea65a75a5e721249515',
        'blob_equals_worktree': True},
    'role_c_timing.py': {
        'blob_sha256':
            '1a2c8d7e59c23e289a8e00a25e986b234715615f03bfeacfcb18b8557871122d',
        'worktree_sha256':
            '1a2c8d7e59c23e289a8e00a25e986b234715615f03bfeacfcb18b8557871122d',
        'blob_equals_worktree': True},
}

#: 兼容旧名（指向双身份表的 blob 列；**不再**允许 None）
FROZEN_SHA256 = {k: v['blob_sha256'] for k, v in FROZEN_DUAL_IDENTITY.items()}

#: **决策前**允许出现在输入快照里的键（严格白名单；不在表内 ⇒ fail-closed）
INPUT_SNAPSHOT_ALLOWED = (
    'inst_idx', 'clock', 'order', 'vehicles_fingerprint',
    'plan', 'frozen',                      # 决策前**已冻结**的计划（保留）
    'visible_orders', 'visible_order_boundary',
    'sampler', 'arm_seed', 'K', 'rho',
    'protocol', 'source_identity', 'data_identity', 'booking_horizon',
)

#: **决策结果类**键（出现即 fail-closed；任意嵌套深度都算）
INPUT_SNAPSHOT_FORBIDDEN = (
    'action', 'accept', 'accepted', 'accepted_plan', 'candidate_plan', 'plan_after',
    'utility', 'revenue', 'fuel', 'reject_loss', 'profit', 'gain', 'delta',
    'objective', 'j_cc', 'jcc', 'energy', 'certify_ok', 'certify_energy',
    'lambda_b', 'lambda_raw', 'e_o', 'boundary', 'reject_reason', 'decision_hash',
    'fallback_applied', 'fallback_kind', 'exception_audit',
)

#: 提交状态的三种语义（必须区分"没有采纳新计划"与"没观察到决策完成"）
COMMIT_STATUS = ('COMMIT_OBSERVED', 'NO_NEW_PLAN_ADOPTED', 'COMMIT_NOT_OBSERVED')

#: 可接受的**权威提交证据**（布尔自述**不算**证据）
COMMIT_EVIDENCE_KINDS = ('commit_receipt', 'post_commit_plan',
                         'authoritative_write_receipt')

#: **真实终态回退**允许的 fallback_kind（其余一律不得增加 n_fallbacks）
REAL_TERMINAL_FALLBACK_KINDS = ('timeout_fallback',)


class InputIdentityError(ValueError):
    """输入快照不合规 ⇒ **fail-closed**（不靠调用方自觉）。"""


class CommitEvidenceError(ValueError):
    """提交状态与证据自相矛盾 ⇒ **fail-closed**。"""


class CounterInputError(ValueError):
    """计数器输入不合规（非法 fallback_kind / 重复 event_id / 缺事件身份）。"""

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- 工具
def _sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _scan_forbidden(obj, path='', hits=None):
    """递归扫描**任意嵌套**里的禁止键（不靠调用方自觉）。"""
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in INPUT_SNAPSHOT_FORBIDDEN:
                hits.append('%s%s' % (path, k))
            _scan_forbidden(v, '%s%s.' % (path, k), hits)
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _scan_forbidden(v, '%s[%d].' % (path, idx), hits)
    return hits


def validate_input_snapshot(input_snapshot):
    """校验输入快照**只含决策前信息**；不合规 ⇒ 抛 `InputIdentityError`（fail-closed）。

    规则（用户 2026-10-14 要求）：
      * 顶层键必须落在 `INPUT_SNAPSHOT_ALLOWED` 白名单内 ⇒ 未知键即抛；
      * **任意嵌套**出现 `INPUT_SNAPSHOT_FORBIDDEN` 里的结果类键 ⇒ 抛；
      * 决策前的**冻结计划**（`plan` / `frozen`）**保留**（它们是决策前信息）。
    """
    if not isinstance(input_snapshot, dict):
        raise InputIdentityError('输入快照必须是 dict，得到 %r'
                                % (type(input_snapshot).__name__,))
    unknown = [k for k in input_snapshot if k not in INPUT_SNAPSHOT_ALLOWED]
    if unknown:
        raise InputIdentityError(
            '输入快照含**白名单外**的键 %s ⇒ fail-closed。'
            '决策前身份只允许 %s；决策结果类字段一律不得进入输入身份'
            % (unknown, list(INPUT_SNAPSHOT_ALLOWED)))
    hits = _scan_forbidden(input_snapshot)
    if hits:
        raise InputIdentityError(
            '输入快照（含嵌套）出现**决策结果类**键 %s ⇒ fail-closed：'
            '它们必须在决策**之后**才产生，进入输入身份会让关联键随结果变化，'
            '**恰好妨碍定位差异**' % hits)
    return True


def normalized_input_identity(*, protocol_identity, decision_point, input_snapshot):
    """`input_decision_id` 的规范输入身份。

    **严格只含决策前信息**：`protocol_identity` + 决策点位置 + 经过
    `validate_input_snapshot` 校验的输入快照。
    **不含**（且出现即 fail-closed）action / 效用 / 候选结果 / 采纳结果。
    决策前**已冻结**的计划保留。
    """
    validate_input_snapshot(input_snapshot)
    payload = {'protocol_identity': protocol_identity,
               'decision_point': int(decision_point),
               'input_snapshot': input_snapshot}
    return _sha_text(_canon(payload))


def plan_variants(*, plan_before, candidate_plan=None, accepted_plan=None,
                  plan_after=None, commit_observed=False, commit_evidence=None,
                  decision_outcome_observed=None, action=None):
    """四个计划身份**分开**记录；提交状态必须由**权威证据**支撑（用户 2026-10-14 要求）。

    提交三态（`COMMIT_STATUS`）：
      * `COMMIT_OBSERVED`      —— 接受动作 **且** 有权威提交证据（回执 / 提交后计划）；
      * `NO_NEW_PLAN_ADOPTED`  —— 拒单，且**已观察到本次决策完成**（`accepted_plan=None` 合法），
                                  这是"**没有采纳新计划**"，**不是**"没观察到决策完成"；
      * `COMMIT_NOT_OBSERVED`  —— 未观察到本次决策完成（或缺少权威证据）。

    **fail-closed 的四种矛盾组合**（用户逐条点名）：
      ① `commit_observed=True` 但 `accepted_plan is None` ⇒ 抛；
      ② `commit_observed=False` 却传入非空 `accepted_plan` ⇒ 抛；
      ③ `commit_observed=True` 但**缺少**合法 `commit_evidence` ⇒ 抛
         （**布尔参数由调用方声明 ≠ 已观察到权威提交**）；
      ④ `commit_evidence` 非 dict 或 `kind` 不在 `COMMIT_EVIDENCE_KINDS` ⇒ 抛。
    """
    def _sha(p_):
        if p_ is None:
            return None
        norm = {str(k): list(v) for k, v in sorted((p_ or {}).items())}
        return _sha_text(_canon(norm))

    if commit_observed and accepted_plan is None:
        raise CommitEvidenceError(
            'commit_observed=True 但 accepted_plan is None ⇒ **自相矛盾**（用户点名的组合①）')
    if (not commit_observed) and accepted_plan is not None:
        raise CommitEvidenceError(
            'commit_observed=False 却传入非空 accepted_plan ⇒ **自相矛盾**（组合②）')
    if commit_observed:
        if not isinstance(commit_evidence, dict) or \
                commit_evidence.get('kind') not in COMMIT_EVIDENCE_KINDS:
            raise CommitEvidenceError(
                'commit_observed=True 但**缺少权威提交证据**（需要 %s 之一）⇒ '
                '布尔自述**不算**证据（组合③/④）' % (COMMIT_EVIDENCE_KINDS,))
    if decision_outcome_observed is None:
        # 接受动作默认要求已观察到决策完成；拒单需调用方明确声明
        decision_outcome_observed = bool(commit_observed) or (action == 'reject')
    if commit_observed:
        status, criterion = 'COMMIT_OBSERVED', 'accepted_plan'
    elif decision_outcome_observed and action == 'reject':
        status, criterion = 'NO_NEW_PLAN_ADOPTED', 'none_adopted'
    else:
        status, criterion = 'COMMIT_NOT_OBSERVED', 'COMMIT_NOT_OBSERVED'
    return {
        'plan_before': plan_before, 'plan_before_sha256': _sha(plan_before),
        'candidate_plan': candidate_plan, 'candidate_plan_sha256': _sha(candidate_plan),
        'accepted_plan': accepted_plan, 'accepted_plan_sha256': _sha(accepted_plan),
        'plan_after': plan_after, 'plan_after_sha256': _sha(plan_after),
        'commit_observed': bool(commit_observed),
        'decision_outcome_observed': bool(decision_outcome_observed),
        'commit_evidence': commit_evidence,
        'commit_status': status,
        'primary_criterion_plan': criterion,
        'note': ('主判比较的是**实际被采纳的计划**（`accepted_plan`，需权威提交证据）；'
                 '只有 `decide_fn()` 被调用、未发生真实提交时，'
                 '**最多**只能比较"**拟采纳计划**"（`candidate_plan`），'
                 '**不得**用 `COMMIT_OBSERVED` 证明计划已安装；'
                 '拒单的 `accepted_plan=None` 表示 `NO_NEW_PLAN_ADOPTED`，'
                 '与 `COMMIT_NOT_OBSERVED` **不是**同一件事'),
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
    """**独立**统计器：事件级与决策级错误**分别存储、分别导出**（用户 2026-10-14 修正口径）。

    ⚠️ 第一版的缺陷：`n_execution_errors` **混用**了事件级与决策级对象
    （`add_exception_event` 写入事件键并置 `unit='events'`，
    `add_fallback` 遇重复回退又往**同一个集合**写决策键并把 `unit` 改成 `'decisions'`）
    ⇒ 汇总可能把事件级错误标成决策级。**现改为两个独立集合、两个独立导出字段**。
    """

    def __init__(self):
        self._timeout_decisions = set()
        self._timeout_events = set()
        self._exec_error_events = set()        # 事件级（**独立存储**）
        self._exec_error_decisions = set()     # 决策级（**独立存储**）
        self._fallbacks = {}
        self._fallback_kinds = {}
        self._not_entered = set()
        self._unknown = set()

    # ---- 事件 ----
    def add_exception_event(self, *, run_id, decision_point, exception_audit):
        ea = exception_audit or {}
        eid = ea.get('event_id')
        if eid in (None, ''):
            raise CounterInputError(
                '异常事件**缺少事件身份**（`event_id` 为空）⇒ 拒绝计数：'
                '没有身份就无法去重，也就无法证明"事件 ↔ 决策"一一对应')
        key = (run_id, int(decision_point), eid)
        hard = ea.get('hard_deadline_exception')
        if hard is True:
            if key in self._timeout_events:
                raise CounterInputError('重复 event_id：%r ⇒ 拒绝重复计数' % (key,))
            self._timeout_events.add(key)
            self._timeout_decisions.add((run_id, int(decision_point)))
        elif hard is False:
            if key in self._exec_error_events:
                raise CounterInputError('重复 event_id：%r ⇒ 拒绝重复计数' % (key,))
            self._exec_error_events.add(key)
        else:
            self._unknown.add((run_id, int(decision_point)))

    # ---- 回退 ----
    def add_fallback(self, *, run_id, decision_point, fallback_event_id,
                     fallback_kind, evidence=None):
        """**只有可核验的真实终态回退**才增加 `n_fallbacks`。

        校验（用户逐条点名）：
          * `fallback_kind` 必须 ∈ `REAL_TERMINAL_FALLBACK_KINDS`（当前仅 `timeout_fallback`）
            ⇒ `none` / `b2_boundary_reject` / `b3_conservative_reject` /
            `FALLBACK_NOT_VERIFIABLE` **一律拒绝**（不得把边界拒单计作回退）；
          * 必须有 `evidence`（如 `{'hard_deadline_exception': True, 'event_id': ...}`）；
          * `fallback_event_id` 不得为空。
        """
        if fallback_kind not in REAL_TERMINAL_FALLBACK_KINDS:
            raise CounterInputError(
                '`fallback_kind=%r` **不是**可核验的真实终态回退（允许：%s）⇒ 拒绝计数：'
                '正常边界拒单/保守拒单/不可核验 都**不得**计作终态 fallback'
                % (fallback_kind, REAL_TERMINAL_FALLBACK_KINDS))
        if fallback_event_id in (None, ''):
            raise CounterInputError('缺少 `fallback_event_id` ⇒ 无法去重，拒绝计数')
        if not isinstance(evidence, dict) or \
                evidence.get('hard_deadline_exception') is not True:
            raise CounterInputError(
                '缺少可核验证据（需 `hard_deadline_exception=True`）⇒ 拒绝计数')
        key = (run_id, int(decision_point))
        n = self._fallbacks.get(key, 0) + 1
        self._fallbacks[key] = n
        self._fallback_kinds[key] = fallback_kind
        if n > 1:
            # §14.1：终态回退每决策至多一次；重复 ⇒ execution_error（**不静默去重**）
            self._exec_error_decisions.add((run_id, int(decision_point), fallback_event_id))

    def add_not_entered(self, *, run_id, decision_point):
        self._not_entered.add((run_id, int(decision_point)))

    def summary(self):
        return {
            'n_deadline_timeouts': len(self._timeout_decisions),        # 决策级
            'n_deadline_timeout_events': len(self._timeout_events),     # 事件级
            # ⚠️ **两个独立字段**，不再有混用的 `n_execution_errors`
            'n_execution_errors_events': len(self._exec_error_events),
            'n_execution_errors_decisions': len(self._exec_error_decisions),
            'n_execution_errors_units_are_separate': True,
            'n_fallbacks': sum(self._fallbacks.values()),
            'n_decisions_with_fallback': len(self._fallbacks),
            'fallback_kinds': {str(k): v for k, v in self._fallback_kinds.items()},
            'n_not_entered': len(self._not_entered),
            'n_unknown': len(self._unknown),
            'invariant_real_path_events_per_decision_max': 1,
            'invariant_dedup_keys': {
                'timeout_events': '(run_id, decision_point, event_id)',
                'timeout_decisions': '(run_id, decision_point)',
                'fallbacks': '(run_id, decision_point, fallback_event_id)',
            },
            'note': ('**决策级 ≠ 事件级**；执行错误**分别导出**为 '
                     '`n_execution_errors_events` 与 `n_execution_errors_decisions`，'
                     '**不得**混为同一个字段。**不得要求** `n_deadline_timeouts '
                     '== n_fallbacks`。仅 `timeout_fallback` 且带可核验证据才计入 `n_fallbacks`'),
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
    """静态自检：① 四个冻结模块**双身份**分别核验（blob 与工作树）；② 本模块不写不 poke。

    ⚠️ 第一版把 `role_c_replanner.py` / `role_c_decision_log.py` 的 `expected` 留成 `None`
    ⇒ `unchanged` **自动成立**（自检恒真）。现引用**完整 SHA**，并**分别**核验
    **Git blob 身份**与 **CRLF 工作树身份** —— 两者在有/无 `eol=lf` 属性时**不相等**。
    """
    import subprocess
    root = root or os.path.dirname(os.path.dirname(_HERE))
    out = {'frozen_modules': [], 'problems': []}
    for m, exp in FROZEN_DUAL_IDENTITY.items():
        p_ = os.path.join(_HERE, m)
        if not os.path.isfile(p_):
            out['problems'].append('冻结模块缺失：%s' % m)
            continue
        wt = io.open(p_, 'rb').read()
        wt_sha = hashlib.sha256(wt).hexdigest()
        blob = subprocess.run(['git', 'cat-file', 'blob', 'HEAD:%s' % _rel(m)],
                              cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        blob_sha = (hashlib.sha256(blob.stdout).hexdigest()
                    if blob.returncode == 0 else None)
        rec = {'module': m,
               'blob_sha256': blob_sha, 'blob_expected': exp['blob_sha256'],
               'blob_unchanged': blob_sha == exp['blob_sha256'],
               'worktree_sha256': wt_sha, 'worktree_expected': exp['worktree_sha256'],
               'worktree_unchanged': wt_sha == exp['worktree_sha256'],
               'blob_equals_worktree_expected': exp['blob_equals_worktree'],
               'crlf_bytes': wt.count(b'\r\n'),
               'identity_convention': ('blob==worktree' if exp['blob_equals_worktree']
                                       else 'worktree(CRLF)!=blob（封存 §2.1 登记的是工作树字节）')}
        rec['unchanged'] = rec['blob_unchanged'] and rec['worktree_unchanged']
        out['frozen_modules'].append(rec)
        if not rec['blob_unchanged']:
            out['problems'].append('冻结模块 **blob** 被改动：%s（%s != %s）'
                                   % (m, blob_sha, exp['blob_sha256']))
        if not rec['worktree_unchanged']:
            out['problems'].append('冻结模块 **工作树** 被改动：%s（%s != %s）'
                                   % (m, wt_sha, exp['worktree_sha256']))
    src = io.open(os.path.abspath(__file__), encoding='utf-8').read()
    info = _ast_write_and_poke(src)
    frozen_stems = {m[:-3] for m in FROZEN_MODULES}
    bad_writes = [w for w in info['write_targets']
                  if os.path.basename(w) in FROZEN_MODULES]
    bad_poke = ([s for s in info['setattrs'] if s in frozen_stems]
                + [a for a in info['attr_assigns']
                   if a.split('.')[0] in frozen_stems])
    bad_imports = [i_ for i_ in info['imports'] if i_.split('.')[0] in frozen_stems]
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
    out['no_none_expected'] = all(
        x.get('blob_expected') and x.get('worktree_expected')
        for x in out['frozen_modules'])
    if not out['no_none_expected']:
        out['problems'].append('自检里仍有 `expected=None`（会让 unchanged 自动成立）')
    out['does_not_write_frozen'] = not (bad_writes or bad_poke)
    out['ok'] = not out['problems']
    return out


def _rel(module_name):
    return 'CC_Compare/tools/%s' % module_name


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
