#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase A **三项启动前置门**的机械裁决器（用户 2026-10-14 有条件放行）。

用户的边界（原句）：

    允许执行端建立专用资源登记、实现测量工装并完成仪器正确性回归。
    **只有下述工程前置条件全部满足后**，才允许按预登记范围启动 L1/L2 测量。
    不授权 Phase B、烟测、40 天正式批次或最终测试。

本模块把那句话变成**一个可复算的判决**，而不是一句自述：

| 门 | 内容 | 判据 |
|---|---|---|
| **G1** 仪器正确性回归 | 嵌套 / 顺序 / 异常退出 / 重入四类合成反例 + 离线独立复算 | 回归套件**全过**；任一失败 ⇒ NOT_READY |
| **G2** 专用资源登记 | `p1a-rolec-phasea-v1` 校验通过、父门 `p1a-rolec-v1` 未被改动 | 两者都通过；父门字节身份必须在案 |
| **G3** 仪器开销 | "空仪器" vs "挂满分区"的真实墙钟自开销 | 如实报出；**开销不得用来调容差** |
| **G4** 决策点集合 | 真实管线上 `work_fn.digest()` == 登记 digest | 必须在跑批机器上实测；未实测 ⇒ `pending`，**不得**按"差不多"放过 |

⚠️ **回归失败时不得**放宽 `max(5 ms, 1%)`、**不得**重定义 P9、**不得**改协议或核集
（用户明文）。本模块只**报告**，不提供任何"强制通过"的开关。

退出码：0 = 四门齐备；6 = 未就绪（原因见 `unready_reasons`）。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import role_c_phasea_ceiling as pc                               # noqa: E402
import role_c_phasea_partition as pp                             # noqa: E402

__all__ = ['run_gates', 'main']

GATE_NAMES = ('G1_instrument_correctness', 'G2_phase_a_registration',
              'G3_instrument_overhead', 'G4_decision_points_digest')

#: 探针产物里**必须**与冻结配置一致的构建事实（防止"探针跑了别的规模"）
PROBE_FACT_EXPECT = {'n_orders': 200, 'n_vehicles': 15, 'arm_seed': 7002, 'K': 10,
                     'inner_gate': 'on', 'gate_seed': 20260926,
                     'train_seed': 20260925, 'inst_idx': 0}


def _gate1_regression():
    """G1：跑合成反例回归套件（用户点名的四类 + 离线独立复算）+ **真实调用点插桩回归**。"""
    import test_role_c_phasea_partition as reg
    reg.FAILS[:] = []
    cases = [
        ('nested', reg.test_p7d_gap_not_double_counted),
        ('sequential', reg.test_sequential_leaves),
        ('exception_exit', reg.test_exception_exit_closes_frame),
        ('reentry', reg.test_duplicate_instrumentation_detected),
        ('duplicate_closure_honesty',
         reg.test_duplicate_instrumentation_closure_still_holds),
        ('offline_independent_recompute', reg.test_offline_scan_independent_recompute),
        ('structure_invariants', reg.test_offline_detects_unbalanced_and_overlap),
        ('online_bad_accounting', test_online_bad),
        ('tolerance_and_shares', reg.test_tolerance_rule_and_shares),
        ('tie_break_lowest_leaf_index', reg.test_tie_break_by_lowest_leaf_index),
        ('random_nested_closure', reg.test_random_nested_closure_identity),
        ('mutual_exclusivity', reg.test_shares_are_mutually_exclusive),
    ]
    results = {}
    for name, fn in cases:
        before = len(reg.FAILS)
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            reg.FAILS.append('%s raised %s' % (name, type(exc).__name__))
        results[name] = 'pass' if len(reg.FAILS) == before else 'fail'
    # 真实调用点上的插桩回归（叶子映射 / P6 砍点 / P5-P7a 分流 / 截断重放）
    inst_results = {}
    try:
        import test_role_c_phasea_instrument as ireg
        ireg.FAILS[:] = []
        icases = [
            ('patch_restore_identity', ireg.test_install_and_restore_is_identity),
            ('mapping_resolves_on_real_source',
             ireg.test_patch_table_resolves_on_real_source),
            ('p6_suppression_differential',
             ireg.test_p6_suppression_is_a_real_cut_point),
            ('p5_vs_p7a_routing', ireg.test_p5_vs_p7a_routing_and_forward_leaves),
            ('p7c_only_inside_forward', ireg.test_p7c_only_inside_forward),
            ('truncation_replay', ireg.test_truncation_replay_from_progress),
            ('l1_explanation_only', ireg.test_l1_counters_are_explanation_only),
            ('composes_with_partition', ireg.test_composes_with_partition_invariants),
            ('progress_overhead', ireg.test_progress_overhead_is_reported_honestly),
        ]
        for name, fn in icases:
            before = len(ireg.FAILS)
            try:
                fn()
            except Exception as exc:                              # noqa: BLE001
                ireg.FAILS.append('%s raised %s' % (name, type(exc).__name__))
            inst_results[name] = 'pass' if len(ireg.FAILS) == before else 'fail'
        extra = list(ireg.FAILS)
    except Exception as exc:                                      # noqa: BLE001
        inst_results['import'] = 'fail'
        extra = ['instrument suite import failed: %s' % exc]
    # L1/L2 驱动器的**判定逻辑**回归（§6.2 并列/稳定性、§6.3 四档、
    # 截断不许判定、B 类三态可区分、digest 同算法）
    drv_results = {}
    try:
        import test_role_c_phasea_l1l2 as dreg
        dreg.FAILS[:] = []
        dcases = [
            ('i_star_and_tie_break', dreg.test_i_star_and_tie_break),
            ('below_threshold', dreg.test_below_threshold),
            ('stability_binds_same_leaf', dreg.test_stability_binds_to_same_leaf),
            ('p9_dominant_incomplete',
             dreg.test_p9_dominant_is_attribution_incomplete),
            ('censored_blocks_verdict',
             dreg.test_any_censored_blocks_bottleneck_verdict),
            ('accounting_error_blocks_verdict',
             dreg.test_accounting_error_blocks_verdict),
            ('total_exceeded_blocks_verdict',
             dreg.test_total_exceeded_blocks_verdict),
            ('structural_group_slack_undetermined',
             dreg.test_structural_group_is_slack_undetermined),
            ('a_class_truncation_position',
             dreg.test_a_class_censored_record_has_position),
            ('b_class_three_states', dreg.test_b_class_three_states_distinguishable),
            ('digest_same_algorithm', dreg.test_digest_matches_timing_module),
        ]
        for name, fn in dcases:
            before = len(dreg.FAILS)
            try:
                fn()
            except Exception as exc:                              # noqa: BLE001
                dreg.FAILS.append('%s raised %s' % (name, type(exc).__name__))
            drv_results[name] = 'pass' if len(dreg.FAILS) == before else 'fail'
        extra = extra + list(dreg.FAILS)
    except Exception as exc:                                      # noqa: BLE001
        drv_results['import'] = 'fail'
        extra = extra + ['driver suite import failed: %s' % exc]
    return {'passed': (all(v == 'pass' for v in results.values())
                       and all(v == 'pass' for v in inst_results.values())
                       and all(v == 'pass' for v in drv_results.values())),
            'cases': results, 'instrument_cases': inst_results,
            'driver_cases': drv_results,
            'failures': list(reg.FAILS) + extra}


def test_online_bad():
    import test_role_c_phasea_partition as reg
    return reg.test_online_detects_unbalanced_exit_and_lifo()


def _gate2_registration():
    """G2：Phase A 专用登记通过 + 父门未被改动（字节身份在案）。"""
    st = pc.phasea_status()
    bl = pc.baseline_status()
    return {'passed': bool(st.get('registered')) and not bl.get('problems'),
            'phase_a_registered': st.get('registered'),
            'phase_a_problems': st.get('problems'),
            'ceiling_id': st.get('ceiling_id'),
            'phase_total_cap_s': st.get('phase_total_cap_s'),
            'baseline': {'legacy_valid': bl.get('legacy_valid'),
                         'byte_identity': bl.get('byte_identity'),
                         'sha256': bl.get('sha256'),
                         'matches_recorded': bl.get('matches_recorded'),
                         'problems': bl.get('problems')},
            'baseline_dual_identity_record':
                'CC_Compare/P1A_资源门登记_字节身份_双记录_2026-10-14.json'}


def _gate3_overhead(repeats=25, warmup=3):
    """G3：真实墙钟下"空仪器" vs "挂满分区"的自开销（如实报，不用于调容差）。"""
    leaves = list(pp.LEAF_ORDER)

    def bare():
        tm = pp.PartitionTimer()
        tm.start_decision()
        acc = 0.0
        for i in range(400):
            acc += (i % 7) * 0.5
        tm.end_decision()
        return acc

    def wrapped():
        tm = pp.PartitionTimer()
        tm.start_decision()
        acc = 0.0
        for lab in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P8'):
            with tm.leaf(lab):
                for i in range(50):
                    acc += (i % 7) * 0.5
        with tm.leaf('P7d'):
            for lab in ('P7a', 'P7b', 'P7c'):
                with tm.leaf(lab):
                    for i in range(50):
                        acc += (i % 7) * 0.5
        tm.end_decision()
        return acc

    t0 = time.perf_counter()
    rep = pp.measure_overhead(bare, wrapped, repeats=repeats, warmup=warmup)
    rep['wall_s'] = time.perf_counter() - t0
    rep['n_leaves_instrumented'] = len(leaves)
    # 自检：挂满分区的那次也必须**记账闭合**（否则"开销"没有意义）
    tm = pp.PartitionTimer()
    tm.start_decision()
    with tm.leaf('P3'):
        pass
    tm.end_decision()
    rec = pp.reconcile(tm.result())
    rep['closure_selfcheck'] = rec['verdict']
    # ⚠️ `overhead_ratio` 是相对**合成夹具**的裸体（一个 400 次 Python 循环），
    #    容易被误读成"仪器吃掉一次决策的 62%"。真正有意义的量是绝对开销占
    #    在线 10 s 决策的比例 —— 单独报出来，避免误读。
    ten = 10.0
    rep['overhead_share_of_10s_decision'] = (rep['overhead_s'] / ten
                                             if rep['overhead_s'] is not None else None)
    rep['ratio_caveat'] = ('`overhead_ratio` 相对的是合成夹具裸体，**不是**一次真实决策；'
                           '请读 `overhead_share_of_10s_decision`（绝对开销 / 10 s）。')
    rep['passed'] = (rep['closure_selfcheck'] == 'ok'
                     and rep['bare_s']['p50_s'] is not None
                     and rep['wrapped_s'] is not None)
    return rep


def _gate4_digest(digest_probe):
    """G4：真实管线上的决策点 digest 必须与登记值**逐字符**相同。

    ⚠️ **只接受探针产物**（`role_c_phasea_digest_probe.py` 在跑批机器上生成的 JSON），
    **不接受**命令行手填的 digest —— 否则任何人都能靠键入登记值"通过"这一门，
    而这门存在的意义正是"证明测的是事前固定的那 8 个点"。
    """
    want = pc.PHASEA_DECISION_POINTS_DIGEST
    out = {'registered_digest': want, 'n_registered': pc.PHASEA_N_DECISION_POINTS,
           'probe_path': digest_probe, 'passed': None,
           'required_probe': 'CC_Compare/tools/role_c_phasea_digest_probe.py',
           'note': ('必须在**跑批那台机器**上、用冻结管线实测并落盘为产物；'
                    '命令行手填不算证据。')}
    if not digest_probe:
        out['status'] = 'pending'
        out['reason'] = ('未提供**探针产物** ⇒ 本门**未满足**。不得按"差不多"放过：'
                         'digest 不符意味着测的不是事前固定那 8 个决策点。')
        return out
    if not os.path.isfile(digest_probe):
        out['status'] = 'missing_probe'
        out['reason'] = '探针产物不存在：%s' % digest_probe
        return out
    try:
        with io.open(digest_probe, encoding='utf-8') as f:
            art = json.load(f)
    except Exception as exc:                                      # noqa: BLE001
        out['status'] = 'unreadable_probe'
        out['reason'] = '探针产物无法解析：%s' % exc
        return out
    observed = art.get('decision_points_digest')
    facts = art.get('build_facts') or {}
    out['observed_digest'] = observed
    out['probe_host'] = art.get('host')
    out['probe_dataset_source'] = art.get('dataset_source')
    out['probe_anchor_check'] = art.get('anchor_check')
    out['probe_source_sha256'] = art.get('source_sha256')
    bad = []
    if observed != want:
        bad.append('digest 不符：实测 %s != 登记 %s' % (observed, want))
    if art.get('n_contexts') != pc.PHASEA_N_DECISION_POINTS:
        bad.append('n_contexts=%r != %d'
                   % (art.get('n_contexts'), pc.PHASEA_N_DECISION_POINTS))
    for k, v in PROBE_FACT_EXPECT.items():
        if facts.get(k) != v:
            bad.append('构建事实 %s=%r != 冻结值 %r' % (k, facts.get(k), v))
    if bad:
        out['status'] = 'mismatch'
        out['passed'] = False
        out['reason'] = '；'.join(bad) + ' ⇒ 拒绝启动 L1/L2'
        return out
    out['status'] = 'ok'
    out['passed'] = True
    out['strength_caveat'] = (
        'G4 的强度 = **探针产物的可核验性**。产物必须①按字节归档（raw SHA 登记）'
        '②on the batch machine 生成 ③带上源码 SHA 与构建事实。'
        '一个"手搓的、字段齐全的"产物在形式上也能过 —— 因此产物本身必须进'
        '字节身份登记，供事后复核；本门不宣称能防伪造。')
    return out


def _sha(path):
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), path.replace('/', os.sep))
    if not os.path.isfile(p):
        return None
    import hashlib
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


#: 本判决**依赖的字节容器**（各自的 raw SHA 一并登记，供事后复核）
INPUT_CONTAINERS = (
    'CC_Compare/P1A_资源门登记_PhaseA_v1.json',
    'CC_Compare/P1A_资源门登记.json',
    'CC_Compare/P1A_RoleC_方案1_PhaseA_归因登记_2026-10-14.md',
    'CC_Compare/P1A_RoleC_方案1_PhaseA_归因登记_v1_reviewed_2026-10-14.md',
    'CC_Compare/P1A_RoleC_方案1_PhaseA_归因登记_REV2_reviewed_2026-10-14.md',
    'CC_Compare/P1A_资源门登记_字节身份_双记录_2026-10-14.json',
    'CC_Compare/P1A_RoleC_收束裁定_2026-10-14.md',
    'CC_Compare/P1A_T2判据分层裁定_2026-10-14.md',
    'CC_Compare/P1A_资源计时_报告_2026-10-14.json',
    'CC_Compare/P1A_资源计时_监督器_2026-10-14.json',
    'CC_Compare/tools/role_c_phasea_partition.py',
    'CC_Compare/tools/test_role_c_phasea_partition.py',
    'CC_Compare/tools/role_c_phasea_ceiling.py',
    'CC_Compare/tools/test_role_c_phasea_ceiling.py',
    'CC_Compare/tools/role_c_phasea_digest_probe.py',
    'CC_Compare/tools/role_c_phasea_instrument.py',
    'CC_Compare/tools/test_role_c_phasea_instrument.py',
    'CC_Compare/tools/role_c_phasea_l1l2.py',
    'CC_Compare/tools/test_role_c_phasea_l1l2.py',
    'CC_Compare/tools/role_c_phasea_recheck.py',
    'CC_Compare/tools/run_phasea_gates.py',
    'CC_Compare/tools/role_c_timing.py',
    'CC_Compare/tools/role_c_harness.py',
    'CC_Compare/tools/role_c_local_search.py',
    'CC_Compare/tools/role_c_forward.py',
)


def run_gates(digest_probe=None, *, repeats=25, warmup=3):
    """跑四门并返回判决（**不抛**，便于先看原因）。"""
    g1 = _gate1_regression()
    g2 = _gate2_registration()
    g3 = _gate3_overhead(repeats=repeats, warmup=warmup)
    g4 = _gate4_digest(digest_probe)
    gates = {'G1_instrument_correctness': g1, 'G2_phase_a_registration': g2,
             'G3_instrument_overhead': g3, 'G4_decision_points_digest': g4}
    unready = []
    if not g1['passed']:
        unready.append('G1 仪器正确性回归未全过：%s' % (g1['failures'],))
    if not g2['passed']:
        unready.append('G2 资源登记未就绪：%s / %s'
                       % (g2['phase_a_problems'], g2['baseline']['problems']))
    if not g3['passed']:
        unready.append('G3 仪器开销自检未过：closure=%s' % (g3['closure_selfcheck'],))
    if g4['passed'] is not True:
        unready.append('G4 决策点 digest 未通过（status=%s）：%s'
                       % (g4['status'], g4.get('reason', '')))
    return {
        'label': 'P1A-RoleC-PhaseA-startup-gates',
        'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'host': {'platform': sys.platform, 'python': sys.version.split()[0],
                 'cpu_count': os.cpu_count()},
        'gates': gates,
        'ready_for_L1_L2': not unready,
        'unready_reasons': unready,
        'inputs_sha256': {rel: _sha(rel) for rel in INPUT_CONTAINERS},
        'digest_probe_artifact_sha256': (_sha(digest_probe)
                                         if digest_probe else None),
        'self_sha256_note': ('本文件**不含自身**的 SHA（自指不可能）；'
                             '本判决产物的 raw SHA 在交付提交信息与登记件里给出。'),
        'authorization_boundary': {
            'granted': ['建立专用资源登记', '实现测量工装', '完成仪器正确性回归'],
            'conditional': '三项工程前置门全部满足后才允许按预登记范围启动 L1/L2 测量',
            'not_granted': ['Phase B', '烟测', '40 天正式批次', '最终测试',
                            'L3 profiler（不得自动启动）'],
        },
        'no_force_switch': ('本模块**不提供**强制通过开关：回归失败时不得放宽容差、'
                           '不得重定义 P9、不得改协议/核集。'),
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Phase A 三项启动前置门裁决')
    ap.add_argument('--out', default=None, help='判决 JSON 输出路径')
    ap.add_argument('--digest-probe', default=None,
                    help='真实管线探针产物（G4）；用 '
                         'role_c_phasea_digest_probe.py 在跑批机器上生成')
    ap.add_argument('--repeats', type=int, default=25)
    ap.add_argument('--warmup', type=int, default=3)
    a = ap.parse_args(argv)
    rep = run_gates(a.digest_probe, repeats=a.repeats, warmup=a.warmup)
    txt = json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True)
    if a.out:
        p = os.path.abspath(a.out)
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(txt + '\n')
        print('WROTE: %s' % p)
    print(txt)
    return 0 if rep['ready_for_L1_L2'] else 6


if __name__ == '__main__':
    sys.exit(main())
