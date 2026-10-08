# -*- coding: utf-8 -*-
"""A-v1 统一验收器（2026-09-30 v2，路线图验收清单 #1/#2 修订）。

纯读取 gate.json，独立重算资格与数值门，输出带原因码的 adjudication.json：
  - numeric_passed：原预声明效用判据（cond−uncond ≥20 且天级配对 CI 下界>0 且
    两臂 hard=1.0 且全日可裁决）——按 day_id 显式配对重算，不采信 runner 自报；
    explicit_feat 不进入数值门（原主判据只含 cond/uncond 两臂，仅作诊断报告）；
  - formal_adjudicable：身份（identity 存在且 source_stable=True + 必需字段显式核验 +
    与报告预算/配置一致性）+ 全日完整唯一 + 效用全有限 + 无硬违规 + 资源口径；
  - 资源口径（2026-09-30 15:00 修订）：09:09 批次启动先于 12:55 零超时规则，故默认
    --timeout-rule protocol（原协议：超时保计划+拒单+计入效用，允许非零超时）；
    12:55 零超时规则仅作后设敏感性标签（zero_timeout_ok），不追溯当前批次。
    未来批次若启动前声明零超时资格门，用 --timeout-rule sensitivity。

v2 修复（用户路线图）：①按 day_id 唯一映射重排效用向量后配对（原行序相减会错配）；
②numeric_passed 不再要求第三臂 explicit_feat；③bootstrap 种子读实际 gate seed；
④身份核验补必需字段显式检查与预算/配置一致性，不只信 source_stable 一个布尔。
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'coldchain')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp_reserve import _stat

TIERS = ['p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)']
DELTA_STAR = 20.0
_REQUIRED_IDENTITY_FIELDS = ('shared_source_sha256', 'contract_sha256',
                             'data_sha256', 'data_meta_sha256', 'budget_B',
                             'cooling_share', 'shared_config')


def _is_sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def _arm_vector(blk, arm, expected):
    """按 day_id 唯一映射重排后返回有序效用向量与资格标志（不依赖 JSON 行序）。"""
    rows = blk.get('per_day', {}).get(arm)
    if not isinstance(rows, list) or not rows:
        return dict(complete=False, finite=False, hard_ok=False, timeout_days=0,
                    max_timeouts=0, n_rows=0, utility=None, problems=['no_per_day_rows'])
    ids = [int(r['i']) for r in rows]
    dup = len(set(ids)) != len(ids)
    complete = (len(rows) == expected and not dup and sorted(ids) == list(range(expected)))
    # 按 day_id 建唯一映射 → 0..expected-1 有序向量（v2：修复原行序相减错配）
    by_id = {}
    problems = []
    for r in rows:
        if int(r['i']) in by_id:
            problems.append('duplicate_day_id_%d' % int(r['i']))
        by_id[int(r['i'])] = r
    u = np.array([by_id[i]['utility'] for i in range(expected)], float) if complete else None
    finite = bool(complete and np.all(np.isfinite(u)))
    hard_ok = bool(all(bool(r['hard_feasible']) for r in rows))
    tos = [int(r['timeouts']) for r in rows]
    return dict(complete=complete, finite=finite, hard_ok=hard_ok,
                timeout_days=int(sum(1 for t in tos if t > 0)),
                max_timeouts=int(max(tos)) if tos else 0,
                n_rows=len(rows), utility=u, problems=problems)


def _check_identity(d):
    """身份显式核验（v2）：存在性 + source_stable + 必需字段 + 报告一致性。"""
    problems = []
    idt = d.get('identity')
    if not isinstance(idt, dict):
        return False, ['identity_missing']
    for k in _REQUIRED_IDENTITY_FIELDS:
        if not idt.get(k):
            problems.append('identity_field_missing_%s' % k)
    if idt.get('source_stable') is not True:
        problems.append('identity_source_not_stable')
    start, end = idt.get('source_sha256'), idt.get('source_end_sha256')
    if not isinstance(start, dict) or not isinstance(end, dict) or start != end:
        problems.append('identity_source_hash_mismatch')
    for name in ('contract_sha256', 'data_sha256', 'data_meta_sha256'):
        if idt.get(name) and not _is_sha(idt[name]):
            problems.append('identity_%s_not_sha256' % name)
    # 与报告自身的预算/配置一致性（防止身份块与数值块脱节）
    budget = d.get('budget', {})
    if isinstance(idt.get('budget_B'), (int, float)) and 'B' in budget \
            and float(idt['budget_B']) != float(budget['B']):
        problems.append('identity_budget_B_mismatch_report')
    sc = idt.get('shared_config') or {}
    cfg = d.get('config', {})
    for key in ('time_limit', 'capacity', 'num_vehicles', 'n_orders'):
        if key in cfg and key in sc and float(sc[key]) != float(cfg[key]):
            problems.append('identity_shared_config_%s_mismatch' % key)
    seeds = d.get('seeds', {})
    if 'gate' in seeds and idt.get('seeds', {}).get('gate') is not None \
            and int(idt['seeds']['gate']) != int(seeds['gate']):
        problems.append('identity_gate_seed_mismatch')
    return (not problems), problems


def adjudicate(d, expected=40, timeout_rule='protocol'):
    """返回裁决 dict（可单测）；不写文件。"""
    problems = []
    sensitivity = []
    # 配置核对
    cfg = d.get('config', {})
    if int(cfg.get('gate_instances', -1)) != expected:
        problems.append('gate_instances_mismatch')
    if float(cfg.get('time_limit', -1)) != 10.0:
        problems.append('time_limit_not_10s')
    if int(cfg.get('K', -1)) != 10:
        problems.append('K_not_10')
    # 身份（显式核验）
    identity_ok, id_problems = _check_identity(d)
    problems.extend(id_problems)
    # bootstrap 种子读实际 gate seed（v2：不再硬编码 20260926）
    gate_seed = None
    try:
        gate_seed = int(d.get('seeds', {}).get('gate'))
    except (TypeError, ValueError):
        pass
    if gate_seed is None:
        problems.append('gate_seed_missing')

    declared_arms = [x.strip() for x in str(cfg.get('arms') or '').split(',') if x.strip()]
    extra_arms = [x for x in declared_arms if x not in
                  ('uncond_hist', 'cond_hist', 'explicit_feat')]
    if len(set(declared_arms)) != len(declared_arms):
        problems.append('duplicate_declared_arm')
    if 'a1_rollout' in extra_arms:
        a1_identity = (d.get('identity') or {}).get('a1') or {}
        v_path = cfg.get('a1_v_ckpt')
        v_sha = a1_identity.get('v_ckpt_sha256')
        if not v_path or not v_sha or not os.path.isfile(v_path):
            problems.append('a1_v_weight_or_hash_missing')
        else:
            h = hashlib.sha256()
            with open(v_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
            if h.hexdigest() != v_sha:
                problems.append('a1_v_weight_hash_mismatch')
        if a1_identity.get('h') != cfg.get('a1_h'):
            problems.append('a1_h_identity_mismatch')
        if set(a1_identity.get('arms') or []) != set(
                x for x in extra_arms if x in ('a1_consensus', 'a1_rollout')):
            problems.append('a1_arm_identity_mismatch')

    arm_checks, numeric, extra_comparisons = {}, {}, {}
    for pn in TIERS:
        p = d.get('gate', {}).get(pn)
        if p is None:
            problems.append('tier_missing_%s' % pn)
            continue
        arm_checks[pn] = {}
        for arm in ('uncond_hist', 'cond_hist', 'explicit_feat'):
            arm_checks[pn][arm] = _arm_vector(p, arm, expected)
        for arm in extra_arms:
            arm_checks[pn][arm] = _arm_vector(p, arm, expected)
        c = arm_checks[pn]['cond_hist']
        u = arm_checks[pn]['uncond_hist']
        # 原预声明主判据 = 两臂（cond/uncond）完整+有限+硬可行；explicit_feat 不参与
        two_arm_ok = (c['complete'] and c['finite'] and c['hard_ok']
                      and u['complete'] and u['finite'] and u['hard_ok'])
        if two_arm_ok and gate_seed is not None:
            st = _stat(c['utility'] - u['utility'], gate_seed)
        else:
            st = dict(mean=None, ci_lo=None, ci_hi=None)
        passed = bool(two_arm_ok and st['mean'] is not None
                      and st['mean'] >= DELTA_STAR and (st['ci_lo'] or 0) > 0)
        numeric[pn] = dict(mean=st['mean'], ci_lo=st['ci_lo'], ci_hi=st['ci_hi'],
                           numeric_passed=bool(passed), bootstrap_seed=gate_seed)
        # 资源：零超时仅作后设敏感性标签（v2：不追溯当前批次）
        any_timeout = any(v['timeout_days'] > 0 for v in arm_checks[pn].values())
        if any_timeout:
            sensitivity.append('zero_timeout_violated_%s' % pn)
        for arm in ('cond_hist', 'uncond_hist'):        # 主判据两臂资格问题
            v = arm_checks[pn][arm]
            if not v['complete']:
                problems.append('day_incomplete_%s_%s' % (pn, arm))
            if not v['finite']:
                problems.append('nonfinite_utility_%s_%s' % (pn, arm))
            if not v['hard_ok']:
                problems.append('hard_violation_%s_%s' % (pn, arm))
        for arm in ('explicit_feat',):                   # 第三臂仅诊断，不进资格
            v = arm_checks[pn][arm]
            if not v['complete'] or not v['finite'] or not v['hard_ok']:
                sensitivity.append('explicit_feat_diagnostic_%s_%s' % (
                    pn, 'incomplete' if not v['complete'] else
                    ('nonfinite' if not v['finite'] else 'hard')))
        for arm in extra_arms:
            v = arm_checks[pn][arm]
            if not v['complete']:
                problems.append('day_incomplete_%s_%s' % (pn, arm))
            if not v['finite']:
                problems.append('nonfinite_utility_%s_%s' % (pn, arm))
            if not v['hard_ok']:
                problems.append('hard_violation_%s_%s' % (pn, arm))
            if v['complete'] and v['finite'] and v['hard_ok'] and \
                    c['complete'] and c['finite'] and c['hard_ok'] and gate_seed is not None:
                st_extra = _stat(v['utility'] - c['utility'], gate_seed)
                extra_comparisons.setdefault(pn, {})[arm + '_minus_cond_hist'] = st_extra

    if timeout_rule == 'sensitivity':
        problems.extend(sensitivity)
    zero_timeout_ok = (not sensitivity)
    formal_adjudicable = (not problems)
    return {
        'expected_days': expected,
        'timeout_rule': timeout_rule,
        'identity_ok': identity_ok,
        'gate_seed': gate_seed,
        'problems': sorted(set(problems)),
        'zero_timeout_ok': zero_timeout_ok,
        'zero_timeout_sensitivity_labels': sorted(set(sensitivity)),
        'numeric_per_tier': numeric,
        'extra_arm_comparisons': extra_comparisons,
        'formal_adjudicable': formal_adjudicable,
        'note': ('numeric_passed（原预声明两臂效用判据）与 formal_adjudicable（身份/全日/'
                 '有限/硬约束/资源口径）为独立字段；二者同时满足才可作正式裁决，合格负结果'
                 '同样保留正式资格。explicit_feat 为第三臂诊断，不进入数值门。'
                 '资源口径：默认 protocol（原协议允许超时并计入效用）；12:55 零超时规则'
                 '仅作后设敏感性标签，未来批次若启动前声明资格门可用 --timeout-rule '
                 'sensitivity。'),
        'arm_checks': {pn: {a: {k: v2 for k, v2 in v.items() if k != 'utility'}
                            for a, v in arms.items()}
                       for pn, arms in arm_checks.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gate_json')
    ap.add_argument('--expected', type=int, default=40)
    ap.add_argument('--timeout-rule', choices=['protocol', 'sensitivity'],
                    default='protocol',
                    help='protocol=原协议（超时计入效用，默认）；sensitivity=零超时资格门'
                         '（仅限启动前已声明的未来批次）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    d = json.load(open(args.gate_json, encoding='utf-8'))
    adjudication = adjudicate(d, args.expected, args.timeout_rule)
    adjudication['gate_json'] = os.path.abspath(args.gate_json)
    out_path = args.out or (os.path.splitext(args.gate_json)[0] + '.adjudication.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(adjudication, f, indent=2)
    print(json.dumps(adjudication, indent=2))
    # 机器可读失败：formal 不过 → 退出码 2；仅数值不过但 formal 过 → 0（负结果也是正式结果）
    sys.exit(2 if not adjudication['formal_adjudicable'] else 0)


if __name__ == '__main__':
    main()
