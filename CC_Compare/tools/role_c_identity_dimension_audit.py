#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**只读**：provider 身份歧义补充审计 + 敏感性档（`p_c=0`）14 条盘查。

授权：用户 2026-10-14「对比方法线：五项正式裁定」
      ④ 批准**独立**只读补充审计；**不修改**历史资格矩阵；原报告的五条裁决**保留**；
      ⑤ 批准只读深挖剩余 14 条（`cond_hist` 7 + `uncond_hist` 7，**全属敏感性档**）；
         **主档与 `p_c=0` 敏感性档严格分开**，不调整旧裁决。

用户给出的两处**方法学更正**（本模块落实）
------------------------------------------
1. `arm=cond_hist` 与 `provider=cada` **不必定矛盾**：它们可能是**两个不同维度**
   （`cond_hist` = 用哪种未来场景抽样方法；`cada` = 被评估的外部求解器 / provider）。
   ⇒ 五条记录的准确初步表述是 **`IDENTITY_DIMENSION_AMBIGUITY`**
   （表格把不同层级的身份塞进容易误用的 `method` 字段 ⇒ 下游可能错误聚合）。
   **只有**仓库原始 schema 明确规定 `method` 必须等于 `provider` 时，
   才可在独立审查后维持 `MISMATCH`。本模块**同时保留**原始 `recorded_tier`。
2. `per_day` 归属靠 utility 均值匹配 ⇒ 是**有证据支持的推断**，**不是**归档自带的
   原生 arm 身份 ⇒ 必须带 **`INFERRED_UNIQUE_MATCH`** 标记。

⚠️ 只读：不改矩阵、不改 gate、不重算效用、不重新判定旧结果、不访问服务器。
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_main_tier_cond_hist_review as base                 # noqa: E402

__all__ = ['IDENTITY_DIMENSIONS', 'PAIRING_KEY', 'audit_identity_dimension',
           'sweep_sensitivity_tier', 'main']

#: 身份维度（用户给出的复合键成分）
IDENTITY_DIMENSIONS = ('provider', 'variant', 'sampler', 'p_c', 'protocol',
                       'source_identity', 'data_identity')

#: 未来配对键（至少这些维度，再用 `day_id` 配对）
PAIRING_KEY = ('provider', 'variant', 'sampler', 'p_c', 'protocol',
               'source_identity', 'data_identity', 'day_id')

#: 原始报告里被判 `MISMATCH` 的那一类检查名
AMBIGUOUS_CHECK = 'arm_label_identifies_provider'


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _local(gate_path, root):
    rel = (gate_path or '').replace('/home/hzeng/project/MASKCO-Main/', '')
    return os.path.join(root, rel.replace('/', os.sep))


def _matrix(root):
    c = sorted(glob.glob(os.path.join(
        root, 'CC_Compare', 'audit', 'BASELINE_QUALIFICATION_MATRIX_*.json')))
    return c[-1] if c else None


def audit_identity_dimension(*, root=None, matrix=None):
    """把 5 条 `MISMATCH` **重新解释**为 `IDENTITY_DIMENSION_AMBIGUITY`（**保留**原 tier）。"""
    root = root or _root()
    mp = matrix or _matrix(root)
    d = json.load(io.open(mp, encoding='utf-8'))
    arms = d.get('arms') or []
    rows, n_mismatch = [], 0
    for a in arms:
        if (a.get('Q') or {}).get('Q_timeout_clean') is not False:
            continue
        if not (a.get('health') or {}).get('n_solves'):
            continue
        if 'Q_identity' in (a.get('Q_vetoes') or []):
            continue
        gp = a.get('gate_path') or ''
        if not os.path.isfile(_local(gp, root)):
            continue
        rec = base.review_record({'gate_path': gp, 'local': _local(gp, root),
                                  'arm': a.get('arm'), 'group': a.get('group'),
                                  'p_c': a.get('p_c'),
                                  'identity_problems': a.get('identity_problems') or [],
                                  'identity_verified': a.get('identity_verified')})
        c = (rec['checks'] or {}).get(AMBIGUOUS_CHECK) or {}
        recorded = c.get('tier')
        if recorded == 'MISMATCH':
            n_mismatch += 1
        rows.append({
            'gate': gp, 'provider': rec.get('provider'), 'arm': a.get('arm'),
            'sampler_label_is_the_arm': True,
            'recorded_tier': recorded,
            'corrected_reading': ('IDENTITY_DIMENSION_AMBIGUITY'
                                  if recorded == 'MISMATCH' else recorded),
            'correction_reason': (
                '`arm=cond_hist` 与 `provider=<外部求解器>` 可以是**两个维度**'
                '（抽样方法 vs 被评估 provider）⇒ **不必定**构成身份冲突；'
                '表格把不同层级的身份放进易误用的 `method` 字段，'
                '风险是**下游错误聚合**，而不是"记录错误"'),
            'mismatch_maintainable_only_if':
                '仓库原始 schema 明确规定 `method` 必须等于 `provider`（须独立审查确认）',
            'observed_dimensions': {
                'provider': rec.get('provider'),
                'sampler': a.get('arm'),
                'p_c': a.get('p_c'),
                'protocol': None,
                'source_identity': 'gate 只有路径、无哈希 ⇒ NOT_VERIFIABLE',
                'data_identity': 'gate 无 data_sha256 ⇒ NOT_VERIFIABLE',
            },
        })
    return {
        'matrix_path': os.path.relpath(mp, root).replace(os.sep, '/'),
        'n_records_reviewed': len(rows),
        'n_recorded_MISMATCH': n_mismatch,
        'n_reinterpreted_as_ambiguity':
            sum(1 for r in rows if r['corrected_reading'] == 'IDENTITY_DIMENSION_AMBIGUITY'),
        'records': rows,
        'identity_dimensions': list(IDENTITY_DIMENSIONS),
        'proposed_pairing_key': list(PAIRING_KEY),
        'pairing_rule': ('配对键**不应**只是 `gate_path` 或 provider 名称；'
                         '至少区分 (provider, variant, sampler, p_c, protocol, '
                         'source/data identity)，再用 `day_id` 建立配对 —— '
                         '既避免把不同 provider 当同一算法的重复运行，'
                         '也避免把同一 provider 的不同 coenrich 变体混为一谈'),
        'report_policy': ('**不修改**原报告 `P1A_对比方法_主档cond_hist六条_只读审查_2026-10-14.md` '
                          '的五条裁决；本件是其**带日期的只读补充**，'
                          '`recorded_tier` 原样保留'),
    }


def sweep_sensitivity_tier(*, root=None, matrix=None):
    """只读盘查**敏感性档** `p_c=0` 的 14 条（`cond_hist` 7 + `uncond_hist` 7）。

    ⚠️ **严格与主档分开**：不并入主档统计通过率/超时均值/方法排名。
    """
    root = root or _root()
    mp = matrix or _matrix(root)
    d = json.load(io.open(mp, encoding='utf-8'))
    arms = d.get('arms') or []
    rows = []
    for a in arms:
        if (a.get('Q') or {}).get('Q_timeout_clean') is not False:
            continue
        if 'p_c=0' not in str(a.get('p_c')):
            continue
        if a.get('method') not in ('cond_hist', 'uncond_hist'):
            continue
        if not (a.get('health') or {}).get('n_solves'):
            continue
        if 'Q_identity' in (a.get('Q_vetoes') or []):
            continue
        gp = a.get('gate_path') or ''
        if not os.path.isfile(_local(gp, root)):
            continue
        rec = base.review_record({'gate_path': gp, 'local': _local(gp, root),
                                  'arm': a.get('arm'), 'group': a.get('group'),
                                  'p_c': a.get('p_c'),
                                  'identity_problems': a.get('identity_problems') or [],
                                  'identity_verified': a.get('identity_verified')})
        c = rec['checks']
        pd_ident = c.get('per_day_arm_identified', {})
        rows.append({
            'gate': gp, 'method_column': a.get('method'), 'provider': rec.get('provider'),
            'tiers': rec['tiers'],
            'provider_label_ambiguity': c.get(AMBIGUOUS_CHECK, {}).get('tier'),
            'per_day_attribution': ('INFERRED_UNIQUE_MATCH'
                                    if pd_ident.get('tier') == 'VERIFIED'
                                    else pd_ident.get('tier')),
            'per_day_attribution_note': (
                '⚠️ 归属由 **utility 均值精确匹配**推断，**不是**归档自带的原生 arm 身份；'
                '即使数值恰好匹配也保留 `INFERRED_UNIQUE_MATCH` 标记'),
            'identity_chain_hash_level': c.get('identity_chain_hash_level', {}).get('tier'),
            'booking_horizon_in_gate': c.get('booking_horizon_in_gate', {}).get('tier'),
            'timeout_events_present':
                c.get('per_decision_timeout_events_present', {}).get('tier'),
            'config_time_limit': c.get('config_time_limit', {}).get('tier'),
            'strict_online_config_completeness':
                c.get('strict_online_config_completeness', {}).get('tier'),
        })
    agg = collections.Counter()
    for r in rows:
        for k, v in r['tiers'].items():
            agg[k] += v
    return {
        'matrix_path': os.path.relpath(mp, root).replace(os.sep, '/'),
        'tier': 'sensitivity (p_c=0)',
        'n_records': len(rows),
        'composition': dict(collections.Counter(r['method_column'] for r in rows)),
        'tier_totals': dict(agg),
        'records': rows,
        'separation_rule': ('**严格与主档分开**：本档**不得**并入主档的通过率、'
                            '超时均值或方法排名；不调整任何旧裁决'),
        'same_questions_answered': [
            'provider 命名混淆（同主档同一检查）',
            'per_day 归属不明（改用 INFERRED_UNIQUE_MATCH 标记）',
            '身份哈希不足（gate 只有路径）',
            'timeout 仅有汇总记录（无逐决策事件）',
            'booking_horizon 缺失（NOT_VERIFIABLE_FROM_ARCHIVED_GATE）',
        ],
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='provider 身份歧义 + 敏感性档 只读审计')
    ap.add_argument('--out', default=None)
    ap.add_argument('--matrix', default=None)
    a = ap.parse_args(argv)
    rep = {'label': 'P1A-comparison-identity-dimension-and-sensitivity-readonly-audit',
           'identity_dimension': audit_identity_dimension(matrix=a.matrix),
           'sensitivity_tier': sweep_sensitivity_tier(matrix=a.matrix),
           'authorization': ('用户 2026-10-14 裁定 ④⑤：只读；不改矩阵、不改 gate、'
                             '不重算效用、不重新判定旧结果、不访问服务器；'
                             '原报告五条裁决保留')}
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({
        'identity_dimension': {
            'n_records': rep['identity_dimension']['n_records_reviewed'],
            'recorded_MISMATCH': rep['identity_dimension']['n_recorded_MISMATCH'],
            'reinterpreted': rep['identity_dimension']['n_reinterpreted_as_ambiguity']},
        'sensitivity_tier': {
            'n_records': rep['sensitivity_tier']['n_records'],
            'composition': rep['sensitivity_tier']['composition'],
            'tier_totals': rep['sensitivity_tier']['tier_totals']},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
