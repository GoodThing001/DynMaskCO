#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**只读**审查：主档 `p_c=(5,10,15)` 的 6 条 `cond_hist` 记录（用户裁定 ④）。

授权（用户 2026-10-14「L1 就绪审计后的五项裁定」④）：
  **批准只读盘查**；仅限**本地已有资料**，**不访问服务器**，**不修改历史 gate**；
  结果按 **VERIFIED / NOT_VERIFIABLE / MISMATCH** 分层；
  **不得**重新计算效用、修改 gate 或重新判定旧结果。

用户指定的审查重点：
  ① gate 与对应 runner 的身份链；② 真实计时定义；③ fallback 语义；
  ④ `booking_horizon` 的**独立配置证据**；⑤ 相同方法在不同运行中的重复身份。

⚠️ 本模块**只读**：只打开 gate.json 读 JSON，不做任何写入。
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import sys

__all__ = ['TIERS', 'find_main_tier_cond_hist', 'review_record', 'main']

TIERS = ('VERIFIED', 'NOT_VERIFIABLE', 'MISMATCH')

#: 严格在线协议里**应当**在 gate 里可核验的配置项（缺席 = 不可从该 gate 核验）
PROTOCOL_CONFIG_KEYS = (
    'K', 'time_limit', 'booking_horizon', 'inner_gate', 'future_policy',
    'energy_pricing', 'gate_seed', 'arm_seed', 'rho', 'num_vehicles', 'n_orders',
)

#: 身份链所需的**哈希级**证据（只有路径 = 不可核验）
IDENTITY_HASH_KEYS = ('provider_ckpt_sha256', 'runner_source_sha256',
                      'config_sha256', 'identity_sha256')


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _local(gate_path, root):
    rel = (gate_path or '').replace('/home/hzeng/project/MASKCO-Main/', '')
    return os.path.join(root, rel.replace('/', os.sep))


def find_main_tier_cond_hist(*, root=None, matrix=None):
    """定位主档 `p_c=(5,10,15)` 的 `cond_hist` 记录（用户指定的 6 条）。"""
    root = root or _root()
    cands = sorted(glob.glob(os.path.join(
        root, 'CC_Compare', 'audit', 'BASELINE_QUALIFICATION_MATRIX_*.json')))
    mp = matrix or (cands[-1] if cands else None)
    d = json.load(io.open(mp, encoding='utf-8'))
    out = []
    for a in d.get('arms') or []:
        if (a.get('Q') or {}).get('Q_timeout_clean') is not False:
            continue
        if a.get('method') != 'cond_hist':
            continue
        if '5,10,15' not in str(a.get('p_c')):
            continue
        # 与"最干净子群"同口径（用户裁定的 20 臂）：health 产出 + 无身份否决 + gate 在本地
        if not (a.get('health') or {}).get('n_solves'):
            continue
        if 'Q_identity' in (a.get('Q_vetoes') or []):
            continue
        gp = a.get('gate_path') or ''
        if not os.path.isfile(_local(gp, root)):
            continue
        out.append({'gate_path': gp, 'local': _local(gp, root), 'arm': a.get('arm'),
                    'group': a.get('group'), 'p_c': a.get('p_c'),
                    'mean_timeouts_matrix':
                        (a.get('runner_reported') or {}).get('mean_timeouts'),
                    'identity_verified': a.get('identity_verified'),
                    'identity_keys': a.get('identity_keys') or [],
                    'identity_problems': a.get('identity_problems') or [],
                    'source_stable': a.get('source_stable')})
    return {'matrix_path': os.path.relpath(mp, root).replace(os.sep, '/'),
            'n_records': len(out), 'records': out}


def _identify_per_day_arm(blk):
    """**确定** `per_day` 这条无标签列表属于哪个臂（按 utility 均值精确匹配）。

    ⚠️ 为什么必须显式识别：`results[seed][p_c]` 里 `per_day` 是**单个列表**，
    与**多臂**的 `arms` 摘要并列，文件里**没有任何字段**说明它属于哪个臂。
    第一版审查看都没看就把 `per_day.timeouts` 与 `arms['cond_hist']` 对账，
    于是得出一个**假的 MISMATCH**（`ordering_accept` 的 0 vs `cond_hist` 的 0.75）。
    ⇒ 唯一性匹配失败时只能判 `NOT_VERIFIABLE`，**不得**臆断归属。
    """
    pd = blk.get('per_day')
    if not isinstance(pd, list) or not pd:
        return None, 'per_day 缺失或为空', None
    if 'utility' not in (pd[0] or {}):
        return None, 'per_day 无 utility ⇒ 无法识别归属', None
    u = sum(float(d.get('utility') or 0.0) for d in pd) / float(len(pd))
    matches = []
    for name, arm in (blk.get('arms') or {}).items():
        m = (arm.get('utility_all_days') or {}).get('mean')
        if m is not None and abs(float(m) - u) <= 1e-6:
            matches.append(name)
    if len(matches) == 1:
        return matches[0], '按 utility 均值精确匹配', u
    if matches:
        return None, '归属**歧义**：多个臂均值相同 %s' % matches, u
    return None, '无臂的 utility 均值与 per_day 一致', u


def review_record(rec):
    """单条记录的逐项分层。返回 `{'checks': {name: {'tier':…, 'detail':…}}}`。"""
    g = json.load(io.open(rec['local'], encoding='utf-8'))
    cfg = g.get('config') or {}
    checks = {}

    def put(name, tier, detail):
        assert tier in TIERS, tier
        checks[name] = {'tier': tier, 'detail': detail}

    # ① 身份链：gate 是否给出**哈希级**身份（而不是只有路径）
    has_hash = [k for k in IDENTITY_HASH_KEYS if k in cfg]
    path_only = [k for k in ('provider_ckpt', 'baseline_from') if cfg.get(k)]
    if has_hash:
        put('identity_chain_hash_level', 'VERIFIED',
            'gate 配置含哈希级身份：%s' % has_hash)
    elif path_only:
        put('identity_chain_hash_level', 'NOT_VERIFIABLE',
            'gate 里只有**路径**（%s），无哈希 ⇒ 无法仅从该 gate 校验 runner/权重身份；'
            '矩阵侧另有 identity 块（%s），两者不是同一证据源'
            % (path_only, (rec.get('identity_keys') or [])[:6]))
    else:
        put('identity_chain_hash_level', 'NOT_VERIFIABLE', 'gate 里没有身份字段')
    put('identity_problems_from_matrix',
        'MISMATCH' if rec.get('identity_problems') else 'VERIFIED',
        '矩阵 identity_problems=%s，identity_verified=%s'
        % (rec['identity_problems'], rec.get('identity_verified')))

    # ② 真实计时定义
    tl = cfg.get('time_limit')
    put('config_time_limit', 'VERIFIED' if tl == 10.0 else
        ('MISMATCH' if tl is not None else 'NOT_VERIFIABLE'),
        'config.time_limit=%r' % (tl,))
    seed = (g.get('seeds') or {}).get('dev')
    res = (g.get('results') or {}).get(str(seed)) or {}
    pc_key = rec.get('p_c') if rec.get('p_c') in res else next(iter(res), None)
    blk = (res or {}).get(pc_key) or {}
    arms = blk.get('arms') or {}
    arm = arms.get('cond_hist') or {}
    sm = blk.get('solver_meta') or {}
    put('per_decision_timing_present',
        'VERIFIED' if sm.get('max_decide_s') is not None else 'NOT_VERIFIABLE',
        'solver_meta.max_decide_s=%r、mean_solve_time_s=%r（gate 内**无**逐决策计时序列）'
        % (sm.get('max_decide_s'), sm.get('mean_solve_time_s')))
    if isinstance(sm.get('max_decide_s'), (int, float)) and tl:
        tier = 'VERIFIED' if sm['max_decide_s'] <= tl else 'MISMATCH'
        put('max_decide_within_time_limit', tier,
            'max_decide_s=%.4f vs time_limit=%s ⇒ %s'
            % (sm['max_decide_s'], tl,
               '未见越限' if sm['max_decide_s'] <= tl else '**越限**'))
    else:
        put('max_decide_within_time_limit', 'NOT_VERIFIABLE', '缺少 max_decide_s 或 time_limit')

    # ③ 超时量级：先**识别** `per_day` 的归属，再与该臂对账（**不得**臆断归属）
    pd = blk.get('per_day')
    pd_arm, pd_how, pd_util = _identify_per_day_arm(blk)
    put('per_day_arm_identified',
        'VERIFIED' if pd_arm else 'NOT_VERIFIABLE',
        ('per_day 归属 = %r（%s）' % (pd_arm, pd_how)) if pd_arm else
        ('**无法确定 per_day 属于哪个臂**（%s）⇒ 不得据此与该臂的超时对账；'
         '这是归档结构隐患：`per_day` 是单列表、与多臂 `arms` 并列，'
         '文件里**没有**任何字段标明归属' % pd_how))
    target = arm if pd_arm in (None, 'cond_hist') else (arms.get(pd_arm) or {})
    if isinstance(pd, list) and pd and 'timeouts' in (pd[0] or {}):
        tot = sum(int(d.get('timeouts') or 0) for d in pd)
        days_with = sum(1 for d in pd if (d.get('timeouts') or 0) > 0)
        mean = tot / float(len(pd))
        mt = target.get('mean_timeouts')
        td = target.get('timeout_days')
        if pd_arm is None:
            put('timeout_magnitude_selfconsistent', 'NOT_VERIFIABLE',
                'per_day 合计=%d、均值=%.4f、有超时天数=%d；'
                '但**归属未定** ⇒ 不与任何臂对账' % (tot, mean, days_with))
        else:
            ok_mean = (mt is not None and abs(mean - float(mt)) <= 1e-9)
            ok_days = (td is None or int(td) == days_with)
            put('timeout_magnitude_selfconsistent',
                'VERIFIED' if (ok_mean and ok_days) else 'MISMATCH',
                'per_day（=臂 %s）逐日 timeouts 合计=%d、均值=%.4f（该臂 mean_timeouts=%r）、'
                '有超时天数=%d（timeout_days=%r）'
                % (pd_arm, tot, mean, mt, days_with, td))
        put('per_decision_timeout_events_present', 'NOT_VERIFIABLE',
            'gate 只有**逐日** timeouts 计数（且该列表最多只覆盖一个臂），'
            '**没有**逐决策事件记录 ⇒ 无法证明每次超时都是独立可核验的硬超时')
    else:
        put('timeout_magnitude_selfconsistent', 'NOT_VERIFIABLE', '缺少 per_day.timeouts')

    # ④ fallback 语义
    put('fallback_semantics_recorded',
        'VERIFIED' if (sm.get('reject_reasons') or sm.get('n_fail') is not None)
        else 'NOT_VERIFIABLE',
        'solver_meta 有 n_fail=%r、reject_reasons=%r；'
        '**但 gate 里没有** fallback_applied / fallback_kind 这类回退语义字段'
        % (sm.get('n_fail'), sm.get('reject_reasons')))

    # ⑤ booking_horizon 独立配置证据
    put('booking_horizon_in_gate',
        'VERIFIED' if 'booking_horizon' in cfg else 'NOT_VERIFIABLE',
        'config 含 booking_horizon=%r ⇒ %s'
        % (cfg.get('booking_horizon'),
           '可核验' if 'booking_horizon' in cfg else 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE'))

    # 协议配置项缺席清单
    missing = [k for k in PROTOCOL_CONFIG_KEYS if k not in cfg]
    put('strict_online_config_completeness',
        'VERIFIED' if not missing else 'NOT_VERIFIABLE',
        'gate.config 缺席：%s' % (missing or '无'))

    # ⑥ 身份标注：method 列是否标识了被比较对象
    put('arm_label_identifies_provider',
        'MISMATCH' if cfg.get('provider') and cfg.get('provider') != rec.get('arm')
        else 'VERIFIED',
        '矩阵 arm/method=%r，而 gate.config.provider=%r ⇒ %s'
        % (rec.get('arm'), cfg.get('provider'),
           'arm 列是**场景抽样臂名**、不是被比较的 provider；'
           'provider 只能从 gate_path/config 恢复'
           if cfg.get('provider') != rec.get('arm') else '一致'))
    return {'gate_path': rec['gate_path'], 'provider': cfg.get('provider'),
            'checks': checks,
            'tiers': dict(collections.Counter(v['tier'] for v in checks.values()))}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='主档 cond_hist 六条只读审查')
    ap.add_argument('--out', default=None)
    ap.add_argument('--matrix', default=None)
    a = ap.parse_args(argv)
    found = find_main_tier_cond_hist(matrix=a.matrix)
    rows = [review_record(r) for r in found['records']]
    allt = collections.Counter()
    for r in rows:
        for k, v in r['tiers'].items():
            allt[k] += v
    rep = {'label': 'P1A-comparison-main-tier-cond-hist-readonly-review',
           'source': found,
           'per_record': rows,
           'tier_totals': dict(allt),
           'limits': [
               '只读；仅本地已有资料；未访问服务器；未修改任何 gate',
               '**未**重新计算效用、**未**重新判定旧结果',
               'gate 只有逐日超时计数，**没有**逐决策事件 ⇒ '
               '"真实硬超时"不可证明，只能表述为"**报告**超时量级"',
           ],
           'note': ('tier 语义：VERIFIED=该项可用现有归档机械核验；'
                    'NOT_VERIFIABLE=现有归档不足以核验（**不等于**不存在/未发生）；'
                    'MISMATCH=现有记录之间**互相矛盾**')}
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        with io.open(a.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % a.out)
    print(json.dumps({'n_records': found['n_records'],
                      'tier_totals': rep['tier_totals'],
                      'providers': [r['provider'] for r in rows]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
