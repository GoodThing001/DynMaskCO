#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 逐臂**独立**资格矩阵（只读审计；不改任何 gate / 冻结树）。

为什么必须"独立"：原运行器自己写的 `formal_adjudication` 只有一个条件 ——

    run_a1_external_accept.py:404
        blk[key]['formal_adjudication'] = bool(pairing['identity_verified'])

即"身份合格 ⇒ 正式合格"。但科研协议要求

    Q_formal = Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_method

本工具**不采信**运行器的汇总字段，而是从 gate 内的**原始逐日记录 `per_day`** 重新计算
`days / finite / hard / crash / 退化`，再与运行器自报值并排显示，**分歧显式列出**。

七个判据（与用户 P0 清单一一对应）：
  Q_identity  配对身份 `_pairing_identity.identity_verified`
  Q_days      40 个**唯一** day_id、无缺失（从 per_day 的 `i` 重算）
  Q_finite    全部效用有限（从 per_day 重算）
  Q_hard      全部逐日 `hard_feasible=True`（从 per_day 重算）
  Q_no_crash  无 `execution_crash`（从 per_day 的 failures 重算）
  Q_nondegen  非"全拒单"退化（`sum(n_served) > 0`）—— PyVRP 历史故障正是这个指纹
  Q_method    方法健康：`n_solves>0` ∧ `n_fail==0` ∧ 参与度证据在位

另报（不否决但要随表引用）：`mean_timeouts`、`fallback_rate`、`max_decide_s`。

用法（**扩展根**下运行；输出 JSON 供下载）：
    cd C-VRP_Cold-chainVehicleRoutingProblem
    python ../CC_Compare/tools/audit_qualification_matrix.py \
        --out ../CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_2026-10-14.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))          # <ROOT>/CC_Compare/tools
ROOT = os.path.dirname(os.path.dirname(HERE))              # <ROOT>
EXT = os.path.join(ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
PARTICIPATION = os.path.join(ROOT, 'CC_Compare', 'results', 'a1_compare_table', 'participation')

RULE_METHODS = {'myopic', 'valuee20', 'valuee30'}
SOLVERS = {'ortools', 'pyvrp'}
# v2 正式批次臂目录 + 旧正式批次（`_f`）+ a05e 求解器臂：与采集器同口径
SCAN = [
    ('a1_ext_formal2', 'results/a1_ext_formal2/*/gate.json'),
    ('a1_ext_a05e', 'results/a1_ext_a05e/*/gate.json'),
    ('a1_ext_extra', 'results/a1_ext_extra/*_f/gate.json'),
]
EXPECTED_DAYS = 40
# 在一个 p_c 块里，除对比方法臂外还有**内部基线臂**（C1 的 cond_hist 等）。
# 2026-10-14 实测踩到：不区分就会把 4 个臂各当成一个"方法"⇒ 每个 gate 多出 3 行、统计翻倍。
# 基线臂**照样审计**（它们的资格影响配对是否成立），但用 `role='baseline'` 标出，
# 且**不计入**方法资格通过率。
BASELINE_ARMS = {'cond_hist', 'uncond_hist', 'explicit_feat'}
# 交付表里 **Ours 两行**的 gate（与 render_compare_table.OURS_TABLE_SPECS 同规则选出）。
# 2026-10-14 补：矩阵原先只覆盖 gate 结构的对比臂，Ours 侧是空白 —— 而"Ours 是否合格"
# 恰恰是公平性主张的前提。这两个 gate 的 schema 与外部臂同构（results[seed][p_c]）。
OURS_GATES = [
    ('ours_delivered', 'A3（掩码重构·预训练权重）',
     'results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json'),
    ('ours_delivered', 'I3（决策感知）',
     'results/i3_online_recovery_1009_v1/online_*/gate.json'),
]


def sha256_file(p):
    try:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:                                        # noqa: BLE001
        return None


def method_key(name):
    """臂目录名 → 规范方法 key（必须剥掉档位后缀，否则规则臂会被当成"学习型方法"）。

    2026-10-14 实测踩到：`rule_myopic_0` 只去掉 `rule_` 得到 `myopic_0`，
    而 `RULE_METHODS` 里是 `myopic` ⇒ 规则臂被误判为"缺参与度产物"并计入方法资格失败率。
    这与 README §13.3 缺陷 J 是同一类错误：**判据对某些臂结构性不适用，却被算成失败**。
    """
    n = name
    for pre in ('rule_',):
        if n.startswith(pre):
            n = n[len(pre):]
    # 剥档位后缀：_0 / _51015 / _102030（以及可能出现的 _f）
    for suf in ('_102030', '_51015', '_0', '_f'):
        if n.endswith(suf):
            n = n[:-len(suf)]
            break
    for k in ('valuee20', 'valuee30', 'valuee2', 'valuee3'):
        if n.startswith(k):
            return k
    for k in ('routefinder', 'cada', 'mvmoe', 'pomo', 'attention', 'symnco', 'omnivrp',
              'sgbs', 'deepaco', 'lih', 'mapt', 'coenrich', 'rrnco',
              'myopic'):
        if n.startswith(k):
            return k
    return n


def participant_evidence(mk, role=None):
    """参与度证据是否落盘（只有**外部学习型方法**才有这一要求）。

    ⚠️ 2026-10-14 修复：内部基线臂（cond_hist / uncond_hist / explicit_feat）**不是对比方法**，
    它们没有"模型参与度证据"这一概念；不豁免就会给每个 gate 的 3 个基线行都判
    `Q_method_participation=False` ⇒ 审计里凭空多出大批假失败（与缺陷 J 同类）。
    """
    if role == 'baseline' or mk in RULE_METHODS or mk in SOLVERS:
        return None                                  # 不适用
    base = mk
    p = os.path.join(PARTICIPATION, base + '.json')
    return {'required': True, 'path': os.path.relpath(p, ROOT).replace('\\', '/'),
            'present': os.path.isfile(p)}


def audit_block(gate, blk, arm_name, expected_days=EXPECTED_DAYS):
    """从**原始逐日记录**独立重算资格，并与运行器自报值对照。"""
    # 两种 schema（2026-10-14 实测）：
    #   * 外部臂 gate：`per_day` 是 **list**（本块只属于这一个方法）
    #   * 主模型 gate：`per_day` 是 **dict{臂名: list}**（同一块里含 maskco/cond_hist/...）
    pd_raw = blk.get('per_day')
    if isinstance(pd_raw, dict):
        pd = pd_raw.get(arm_name) or []
    else:
        pd = pd_raw or []
    rep = (blk.get('arms') or {}).get(arm_name) or {}
    sm = blk.get('solver_meta') or {}
    if isinstance(sm, dict) and arm_name in sm and isinstance(sm[arm_name], dict):
        sm = sm[arm_name]            # 主模型 gate 的 solver_meta 也可能按臂分键

    ids = [int(d.get('i')) for d in pd if d.get('i') is not None]
    uniq = sorted(set(ids))
    utils = [d.get('utility') for d in pd]
    hard = [d.get('hard_feasible') for d in pd]
    crashes = [d for d in pd if 'execution_crash' in (d.get('failures') or [])]
    served = [d.get('n_served') for d in pd if isinstance(d.get('n_served'), int)]
    fin = [u for u in utils if isinstance(u, (int, float))]

    q_days = (len(uniq) == expected_days and len(ids) == expected_days
              and uniq == list(range(min(uniq), min(uniq) + expected_days)) if uniq else False)
    q_finite = bool(utils) and len(fin) == len(utils)
    q_hard = bool(hard) and all(h is True for h in hard)
    q_crash = len(crashes) == 0
    q_nondegen = bool(served) and sum(served) > 0
    n_solves, n_fail = sm.get('n_solves'), sm.get('n_fail')

    rec = {
        'independent': {
            'n_records': len(pd),
            'n_unique_days': len(uniq),
            'day_id_span': [uniq[0], uniq[-1]] if uniq else None,
            'n_finite': len(fin),
            'n_hard_true': sum(1 for h in hard if h is True),
            'n_execution_crash': len(crashes),
            'sum_n_served': sum(served) if served else None,
            'mean_utility': (sum(fin) / len(fin)) if fin else None,
        },
        'runner_reported': {
            'n_days': rep.get('n_days'), 'expected_n': rep.get('expected_n'),
            'finite_n': rep.get('finite_n'), 'missing_day_ids': rep.get('missing_day_ids'),
            'day_ids_complete': rep.get('day_ids_complete'),
            'hard_feasible_rate': rep.get('hard_feasible_rate'),
            'adjudicable': rep.get('adjudicable'),
            'utility_all_days_mean': (rep.get('utility_all_days') or {}).get('mean'),
            'mean_timeouts': rep.get('mean_timeouts'),
            'fail_counts': rep.get('fail_counts'),
        },
        'health': {
            'n_solves': n_solves, 'n_fail': n_fail,
            'fallback_rate': (float(n_fail) / float(n_solves)) if (n_solves and n_fail is not None) else None,
            'max_decide_s': sm.get('max_decide_s'),
            'reject_reasons': sm.get('reject_reasons'),
        },
        'Q': {
            'Q_days': q_days, 'Q_finite': q_finite, 'Q_hard': q_hard,
            'Q_no_crash': q_crash, 'Q_nondegen': q_nondegen,
            'Q_method_health': (bool(n_solves) and (n_fail == 0)),
        },
    }
    return rec


def adjudicate(rec, identity_ok, participation, mk, health_na=False):
    """合取判定。**关键**：结构性不适用的判据记 `None` 并附理由，**不是** False。

    否则就会重演缺陷 J 那一类错误 —— 「判据对某些臂永远不可能成立」，于是报告里永远有
    一批假失败，读表的人无法区分「真问题」和「设计如此」。

    `health_na=True` 用于**主模型 gate**：它的 driver 不在该块产出 `solver_meta`
    （口径与外部臂不同）⇒ 方法健康项记为不适用，而**超时仍照实上报**（`runner_reported.mean_timeouts`）。
    """
    q = {k: rec['Q'][k] for k in ('Q_days', 'Q_finite', 'Q_hard', 'Q_no_crash',
                                  'Q_nondegen')}
    why = {}
    is_rule = mk in RULE_METHODS

    if is_rule:
        q['Q_identity'] = None
        why['Q_identity'] = ('纯规则臂：无学习权重，gate 结构性没有 _pairing_identity'
                             '（by design，README §13.3 缺陷 J）')
        q['Q_method_health'] = None
        why['Q_method_health'] = ('纯规则臂：run_a1_downstream_ablation.py 不产出 '
                                  'solver_meta(n_solves/n_fail)，无外部求解器可体检')
        # λ=3 全拒单是**已登记的预期行为**（README §2 表注），标记出来而不是当故障
        q['Q_nondegen_expected_by_design'] = (mk == 'valuee30')
    else:
        q['Q_identity'] = bool(identity_ok)
        if health_na and (rec['health'] or {}).get('n_solves') is None:
            q['Q_method_health'] = None
            why['Q_method_health'] = ('主模型 gate 不在该块产出 solver_meta（driver 口径不同）'
                                      '⇒ 方法健康项不适用；超时/天仍照实上报')
        else:
            q['Q_method_health'] = rec['Q']['Q_method_health']

    if participation and participation.get('required'):
        q['Q_method_participation'] = bool(participation.get('present'))
    else:
        q['Q_method_participation'] = None
        why['Q_method_participation'] = '规则臂/经典求解器：无学习权重 ⇒ 不要求参与度产物'

    veto = ['Q_identity', 'Q_days', 'Q_finite', 'Q_hard', 'Q_no_crash',
            'Q_nondegen', 'Q_method_health', 'Q_method_participation']
    q['Q_formal_core'] = all(q[k] is not False for k in veto)
    to = (rec['runner_reported'] or {}).get('mean_timeouts')
    q['Q_timeout_clean'] = (to == 0) if to is not None else None
    q['Q_formal_strict'] = bool(q['Q_formal_core'] and q['Q_timeout_clean'] is not False)
    rec['Q'] = q
    rec['Q_reasons_not_applicable'] = why
    rec['Q_vetoes'] = [k for k in veto if q[k] is False]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default='results')
    args = ap.parse_args()

    rows, notes = [], []
    # ---- 交付表里的 Ours 两行（role='ours'；不要求 participation，那是外部方法才有的概念）----
    for group, label, pat in OURS_GATES:
        for path in sorted(glob.glob(os.path.join(EXT, *pat.split('/')))):
            try:
                g = json.load(open(path, encoding='utf-8'))
            except Exception as e:                            # noqa: BLE001
                notes.append({'gate': path, 'error': 'LOAD_FAIL %s' % e})
                continue
            for yr, yb in (g.get('results') or {}).items():
                if not isinstance(yb, dict):
                    continue
                ident = yb.get('_identity') or {}
                pair = yb.get('_pairing_identity') or {}
                for pn, blk in yb.items():
                    if pn.startswith('_') or not isinstance(blk, dict):
                        continue
                    for an in (blk.get('arms') or {}):
                        r = audit_block(g, blk, an)
                        r = adjudicate(r, pair.get('identity_verified'), None, an,
                                       health_na=True)
                        row = {
                            'group': group, 'arm_dir': label, 'method': an, 'role': 'ours',
                            'p_c': pn, 'arm': an, 'gate_path': path,
                            'identity_verified': pair.get('identity_verified'),
                            'identity_problems': pair.get('problems'),
                            'source_stable': ident.get('source_stable'),
                            'identity_has_checkpoint_sha256': any(
                                'ckpt' in k or 'checkpoint' in k or 'model' in k or 'weight' in k
                                for k in (ident or {})),
                            'identity_keys': sorted((ident or {}).keys()),
                            'participation': None,
                        }
                        row.update({k: r[k] for k in
                                    ('independent', 'runner_reported', 'health', 'Q',
                                     'Q_reasons_not_applicable', 'Q_vetoes') if k in r})
                        rows.append(row)
    for group, pat in SCAN:
        for path in sorted(glob.glob(os.path.join(EXT, *pat.split('/')))):
            name = os.path.basename(os.path.dirname(path))
            try:
                g = json.load(open(path, encoding='utf-8'))
            except Exception as e:                            # noqa: BLE001
                notes.append({'gate': path, 'error': 'LOAD_FAIL %s' % e})
                continue
            cfg = g.get('config') or {}
            mk = method_key(cfg.get('provider') or cfg.get('solver') or name)
            ident = {}
            for yr, yb in (g.get('results') or {}).items():
                if isinstance(yb, dict) and '_identity' in yb:
                    ident = yb['_identity']
                for pn, blk in (yb or {}).items():
                    if pn.startswith('_') or not isinstance(blk, dict):
                        continue
                    arms = blk.get('arms') or {}
                    pair = yb.get('_pairing_identity') or {}
                    for an in arms:
                        role = 'baseline' if an in BASELINE_ARMS else 'method'
                        mk_arm = an if role == 'baseline' else mk
                        r = audit_block(g, blk, an)
                        part = participant_evidence(mk_arm, role)
                        r = adjudicate(r, pair.get('identity_verified'), part, mk_arm)
                        rows.append({
                            'group': group, 'arm_dir': name, 'method': mk_arm, 'role': role,
                            'p_c': pn, 'arm': an, 'gate_path': path,
                            'provider': cfg.get('provider'), 'solver': cfg.get('solver'),
                            'provider_ckpt': cfg.get('provider_ckpt'),
                            'provider_device': cfg.get('provider_device'),
                            'baseline_from': cfg.get('baseline_from'),
                            'time_limit': cfg.get('time_limit'),
                            'workers': cfg.get('workers'),
                            'identity_verified': pair.get('identity_verified'),
                            'identity_problems': pair.get('problems'),
                            'source_stable': ident.get('source_stable'),
                            'identity_has_checkpoint_sha256': any(
                                'ckpt' in k or 'checkpoint' in k or 'weight' in k
                                for k in (ident or {})),
                            'identity_keys': sorted((ident or {}).keys()),
                            'participation': part,
                        }, )
                        rows[-1].update({k: r[k] for k in
                                         ('independent', 'runner_reported', 'health', 'Q',
                                          'Q_reasons_not_applicable', 'Q_vetoes')
                                         if k in r})

    out = {
        'generated_from': {'ext_root': EXT, 'expected_days': EXPECTED_DAYS,
                           'scanned': [g for g, _ in SCAN]},
        'criteria': {
            'Q_formal_core': 'Q_identity ∧ Q_days ∧ Q_finite ∧ Q_hard ∧ Q_no_crash '
                             '∧ Q_nondegen ∧ Q_method_health',
            'Q_formal_strict': 'Q_formal_core ∧ Q_timeout_clean',
            'note': '本文件为**独立复算**：days/finite/hard/crash/退化 一律从 gate 内原始 '
                    'per_day 重算，不采信运行器自报的 formal_adjudication。',
        },
        'runner_formal_adjudication_is_identity_only': {
            'claim': 'run_a1_external_accept.py:404 formal_adjudication = bool(identity_verified)',
            'verified': True,
        },
        'arms': rows,
        'notes': notes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    meth = [r for r in rows if r.get('role') == 'method']
    base = [r for r in rows if r.get('role') == 'baseline']
    ours = [r for r in rows if r.get('role') == 'ours']
    ok = sum(1 for r in meth if r['Q']['Q_formal_core'])
    strict = sum(1 for r in meth if r['Q']['Q_formal_strict'])
    print('扫描：方法臂 %d + 内部基线臂 %d + **交付 Ours 行 %d**（共 %d 个 gate）'
          % (len(meth), len(base), len(ours), len({r['gate_path'] for r in rows})))
    print('Q_formal_core   通过 = %d / %d（对比方法臂）' % (ok, len(meth)))
    print('Q_formal_strict 通过 = %d / %d（strict 额外要求 超时/天 == 0）' % (strict, len(meth)))
    if ours:
        print('--- 交付 Ours 行（独立复算；只列 maskco / reconstruction_control / warmstart，'
              '基线臂另计）---')
        for r in sorted(ours, key=lambda x: (x['arm_dir'], x['p_c'])):
            if r['arm'] not in ('maskco', 'reconstruction_control', 'warmstart'):
                continue
            print('   %-26s %-16s arm=%-22s Q_core=%-5s vetoes=%-22s 超时/天=%-8s 身份=%s'
                  % (r['arm_dir'], r['p_c'], r['arm'], r['Q']['Q_formal_core'],
                     ','.join(r['Q_vetoes']) or '—',
                     r['runner_reported']['mean_timeouts'], r['identity_verified']))
        print('   ⚠️ 若 Ours 行 `超时/天` 显著大于 0，则它**同样**受 CPU 争用影响 ——')
        print('      而它是本表的主模型结果行，引用时必须与本表其他行一样带超时限定。')
    import collections
    for gname in sorted({r['group'] for r in meth}):
        sub = [r for r in meth if r['group'] == gname]
        print('  [%s] core 通过 %d/%d' % (gname,
                                          sum(1 for r in sub if r['Q']['Q_formal_core']),
                                          len(sub)))
    for r in [x for x in meth if not x['Q']['Q_formal_core']][:20]:
        failed = [k for k, v in r['Q'].items() if v is False]
        print('  ✗ %-22s %-16s %-14s 未过: %s' % (r['arm_dir'], r['p_c'], r['method'],
                                                  ','.join(failed)))
    print('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
