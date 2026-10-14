"""Record goal round 1: the pickup A-v1 headroom upper bound and the selection gap."""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
B = TREE / 'pickup/a1_budget_reachability_bound_root_v1'
REPORT = EXT / 'docs/当前规划/决策与审计/取货A-v1预算可达上界与选择缺口_2026-10-13.md'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    R = json.loads((B / 'RESULT.json').read_text(encoding='utf-8'))
    A = json.loads((B / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    S = json.loads((B / 'SENSITIVITY_ADDENDUM.json').read_text(encoding='utf-8'))
    assert R['failed'] == 0 and A['failed'] == 0
    v = 'tw_and_horizon'
    br = A['bracket'][v]
    mix = S['mix']

    evidence = dict(
        stage='pickup_a1_budget_reachability_bound',
        directory=B.relative_to(ROOT).as_posix(),
        registration_sha256=sha(B / 'REGISTRATION.json'),
        producer=dict(checks=R['checks_count'], failed=R['failed']),
        audit=dict(checks=A['checks_count'], failed=A['failed']),
        dataset_sha256=R['dataset_sha256'], contract_sha256=R['contract_sha256'],
        tier=R['tier'], budget_B=R['budget_B'],
        rigorous_upper_bound=dict(variant=v, mean=br['mean_upper_bound'], min=br['min_upper_bound'],
                                  max=br['max_upper_bound']),
        strongest_recorded_control=dict(arm='explicit_feat', mean_utility=br['explicit_feat']['mean_utility'],
                                        mean_gap=br['explicit_feat']['mean_gap'],
                                        min_gap=br['explicit_feat']['min_gap']),
        other_controls={k: dict(mean_utility=br[k]['mean_utility'], mean_gap=br[k]['mean_gap'],
                                min_gap=br[k]['min_gap']) for k in ('uncond_hist', 'cond_hist')},
        sensitivity=[dict(share=s['share_of_precool'], rigorous=s['rigorous_bound'],
                          mean_upper_bound=s['mean_upper_bound'],
                          gap_vs_cond_hist=s['cond_hist']['mean_gap'],
                          gap_vs_explicit_feat=s['explicit_feat']['mean_gap'])
                     for s in S['sensitivity'].values()],
        falsification_oracle=dict(arm_days_checked=3 * 40,
                                  utility_above_bound=0, served_above_reachable=0,
                                  revenue_above_reachable=0, vehicle_hours_over_capacity=0,
                                  note='all recorded arm-days respect every relaxation, so no relaxation was falsified'),
        day0_class_mix=dict(available=mix['available_class_counts_day0'],
                            relaxation_selected=mix['lp_selected_class_counts_day0'],
                            observed_service_rate={k: v0.get('service_rate')
                                                   for k, v0 in S['observed_mix'].items()}),
        flags=dict(formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                   scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                   stable_online_gain=False),
        evidence_hashes={}
    )
    for n in ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json', 'AUDIT_RUN_EXIT.json',
              'SENSITIVITY_ADDENDUM.json', 'SOURCE_START.json', 'SOURCE_END.json', 'AUDIT_SOURCE_START.json',
              'AUDIT_SOURCE_END.json', 'run.py', 'audit.py', 'prepare.py', 'supervise.py',
              'sensitivity_addendum.py'):
        evidence['evidence_hashes'][(B / n).relative_to(ROOT).as_posix()] = sha(B / n)

    out = TREE / 'goal_round_20261013_r1'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'PICKUP_A1_HEADROOM_BOUND_EVIDENCE.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

    text = """# 取货 A-v1：预算可达上界与选择缺口

2026-10-13，持续目标第 1 轮。本阶段是**分析**：不跑任何臂、不训练、不加载模型、不生成新数据。它回答一个此前一直空着的问题——**当前 A-v1 合同下，最强记录在案的非学习对照之上到底还有多少效用空间**。

## 为什么这个问题此前没有被回答

项目一直用"clairvoyant"当作参照，但那是**贪心 value/energy 启发式**，而最强的规则臂反而超过它（旧面板 uncond_dynamic 1161.7 > clairvoyant 1151.6）。一个能被规则打败的启发式**不构成上界**，因此"无头腔"这个结论在 A-v1（接拒 + 能耗预算）设定下从未有过界来支撑。

## 方法：单边松弛（只可能高估，因此结果是上界）

- **R1 可服务性**：任何排程中订单 i 的最早服务开始时间 ≥ max(直线行驶时间, 揭示时刻)，且必须在时间窗内完成；直线距离是任何真实行驶距离的下界。
- **R2 能耗**：服务 i 必然至少计入其预冷能耗 q·0.15·(25−target_c) 加上非负的门开恢复项，因此被服务集合满足 Σe_lower ≤ B=710.6548。
- **R3 车辆时间**：服务 i 至少消耗一次往返行驶加装载服务时间，因此 Στ ≤ 15×16 车辆小时。
- 日效用 = revenue − fuel − reject_loss ≤ revenue，所以对各约束做分数背包（0/1 背包的上界），取 R2、R3 及其最小值即为日效用上界。

**冻结身份**：门数据由 `generate_dataset(40,200,20260926)` 经 `add_v2_initial_quality` 重建，`data_sha256=8767a03d…`、`meta_sha256=72a805fd…`、contract `e25a12104b…` 与记录逐字节相同；对照逐日行取自 `results/a1_c1rr2_reveal_40/gate.json`（SHA 已封存）。本阶段刻意只依赖**未改动**的模块（run_exp_reserve / run_identity / coldchain_evaluator_a1 / coldchain_contract / coldchain_state）并直接重建合同，以避开该批次与当前 `scenario_saa.py`、`run_a1_step2_gate.py` 的版本差。

## 结果（生产 29 项 0 失败、独立审计 49 项 0 失败）

时间窗+返回视界松弛，40 天：

| 量 | 值 |
|---|---|
| 严格上界均值 | **2544.30**（最小 2176.19，最大 2715.33） |
| explicit_feat（最强记录臂） | 1742.83，间隙 **801.47**（最小 628.71） |
| uncond_hist | 1712.05，间隙 832.25 |
| cond_hist | 1703.74，间隙 840.56 |

**可伪证性检验全部通过**：3 臂 × 40 天共 120 个臂-日，没有任何一天出现"效用高于上界 / 服务数超可达集 / 收入超可达收入 / 车辆时间超容量"，因此没有一条松弛被证伪。

## 缺口定位：不是算术，是**选择**

- 车辆时间松弛**完全不紧**（其上界等于可达订单的总收入），因此间隙全部由能耗松弛产生。
- 第 0 天可用类别 0/1/2 = 37/56/101 单；松弛选择的组合是 37.0 / 45.0 / 38.9。
- 而记录在案的对照实际服务率是类 0 **72%**、类 1 **56%**、类 2 **26%**（第 0 天服务 87–94 单、收入 1730–1840）。
- 按每单位预冷能耗的收入（类 0 ≈ 9.52/q、类 1 ≈ 6.35/q、类 2 ≈ 4.65/q），类 0 最划算、类 2 最差。**对照却把预算花在了类 2 上，同时把约 10 单类 0 和约 24 单类 1 留在门外。**

机制解释：接拒是**在线**的，早揭示的高能耗订单会先花掉预算，而对照（含 SAA）没有按"每 kWh 收入"预留预算。

## 敏感性（明确标注：只有 share=1.00 是上界）

把每单能耗下界按观测的预冷占比放大，其余行**只是敏感性、不是界**（"总能耗 ≥ 预冷/占比"逐单并不成立）：

| 预冷占比 | 上界均值 | 严格 | 间隙 vs cond_hist | 间隙 vs explicit_feat |
|---|---|---|---|---|
| 1.00 | 2554.31 | **是** | 850.56 | 811.47 |
| 0.90 | 2379.96 | 否 | 676.22 | 637.13 |
| 0.80 | 2195.56 | 否 | 491.82 | 452.73 |
| 0.72 | 2038.34 | 否 | 334.60 | 295.51 |
| 0.60 | 1786.02 | 否 | 82.27 | 43.18 |

即：即使按记录在案的 0.72 预冷占比做现实核算，**仍有约 300 的间隙**；只有把每单真实能耗放大到预冷的 1/0.60 才把间隙压到噪声水平。

## 结论（诚实口径）

1. **A-v1 存在未被排除的效用空间**：严格上界比最强记录对照高 **801**（≈46%），现实能耗核算下仍有约 **300**。此前"无可捕获头腔"的结论是旧设定（无接拒、无能耗预算）的产物，**不能外推到 A-v1**。
2. 这**不是**头腔存在的证明，而是"头腔未被排除"的证明；上界本身不能证明可达。
3. 缺口已被定位到**预算按每 kWh 收入预留**这个具体机制上，而不是路由或算术。
4. 因此第一个有明确目标量的正向前景出现了：**预算感知的类别优先接拒规则**，目标量 ≤801（现实核算下 ≈300）。

## 下一有界步骤

- 用**未改动**模块实现一个预算感知接拒规则（按 REV/预冷能耗降序预留预算），在**预登记的有限面板**（例如冻结种子的前 5 天）上与记录在案的同日对照行做逐日配对比较；面板与判据先登记，不按结果挑天。
- 两条收紧上界的推导也已命名：把 `coldchain_state.transition_segment` 的整段能耗（预冷+门开恢复+直行最低制冷）做成逐单严格下界；以及在松弛中保留时间窗×车队的联合耦合。
- 旧 FAIL、封印原件与最终种子 20260927/28 未触碰；第2族仍封闭、第3族未启动；稳定可归因核心在线增益仍未建立。
"""
    REPORT.write_text(text, encoding='utf-8', newline='\n')

    note = ('\n> 2026-10-13 持续目标第1轮（取货）：新增 A-v1 预算可达**严格上界**分析（生产29项0失败、独立审计49项0失败，'
            '数据/合同身份逐字节复现，120 个臂-日可伪证性检验全部通过）。时间窗+返回视界松弛下 40 天上界均值 **2544.30**，'
            '最强记录对照 explicit_feat 1742.83，间隙 **801.47**；现实预冷占比 0.72 的敏感性仍有 ≈300 间隙。'
            '缺口定位为**选择效应**：类 0/1/2 的每单位预冷能耗收入为 9.52/6.35/4.65，而对照只服务 72%/56%/26%，'
            '把预算花在最不划算的类 2 上并留下约 10 单类 0、24 单类 1（第0天）。结论是 A-v1 **存在未被排除的效用空间**，'
            '旧"无头腔"结论不能外推到 A-v1；下一步是实现预算感知类别优先接拒规则并在预登记有限面板上与同日对照配对。'
            '见[取货A-v1预算可达上界与选择缺口](docs/当前规划/决策与审计/取货A-v1预算可达上界与选择缺口_2026-10-13.md)。'
            '稳定可归因核心在线增益仍未建立；第2族封闭、第3族未启，最终种子未触碰。\n')

    entries = [EXT / '项目当前状态.md',
               EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
               EXT / '01_取货线/README.md',
               EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_goal_round_20261013_r1'
    backup.mkdir(exist_ok=False)
    identities = []
    rel_report = 'docs/当前规划/决策与审计/' + REPORT.name
    for p in entries:
        raw = p.read_bytes()
        q = backup / p.relative_to(EXT)
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open('xb') as f:
            f.write(raw)
        if p.suffix == '.json':
            d = json.loads(raw.decode('utf-8-sig'))
            d['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            d['pickup']['a1_headroom_bound_20261013'] = evidence
            d['pickup']['next'] = ('实现预算感知类别优先接拒规则，在预登记有限面板上与记录在案同日对照逐日配对；'
                                   '不按结果挑天，不重跑旧批。')
            d['latest_goal_round_20261013_r1'] = dict(
                round=1, pickup_evidence=(out / 'PICKUP_A1_HEADROOM_BOUND_EVIDENCE.json').relative_to(ROOT).as_posix(),
                report=rel_report, stable_online_gain=False)
            new = json.dumps(d, ensure_ascii=False, indent=2) + '\n'
        else:
            local = note
            if p != EXT / '项目当前状态.md':
                local = note.replace(rel_report, os.path.relpath(REPORT, p.parent).replace('\\', '/'))
            new = local + raw.decode('utf-8-sig')
        assert p.read_bytes() == raw
        p.write_text(new, encoding='utf-8', newline='\n')
        identities.append({'path': p.relative_to(EXT).as_posix(), 'before_sha256': sha(q), 'after_sha256': sha(p)})
    (backup / 'UPDATE_IDENTITY.json').write_text(
        json.dumps({'files': identities, 'report_sha256': sha(REPORT)}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8', newline='\n')

    tp = TREE / 'dual_track_optimization_20261013_root_v1/.arbor/tree.json'
    t = json.loads(tp.read_text(encoding='utf-8'))
    now = int(datetime.datetime.now().timestamp())
    t['nodes']['n15'] = dict(
        id='n15', parent='n14', depth=5, status='executed',
        hypothesis=('Pickup: a one-sided relaxation can bound the A-v1 daily utility from above and decide whether any '
                    'headroom exists above the strongest recorded control'),
        insight=('The bound is valid and all 120 recorded arm-days respect it, yet it sits 801 above the strongest control '
                 '(about 335 under the recorded pre-cooling split). The gap is a selection effect: controls spend the '
                 'energy budget on the worst value-per-kWh class while leaving cheap classes unserved, so A-v1 headroom '
                 'is not excluded and the old no-headroom conclusion does not transfer to A-v1.'),
        metadata=dict(dev_score=float(A['checks_count']), test_score=None,
                      result=('Producer29/0, audit49/0. Frozen dataset/contract reproduced byte-exactly. tw_and_horizon '
                              'mean upper bound 2544.30 (min 2176.19) vs explicit_feat 1742.83 -> gap 801.47 (min 628.71); '
                              'sensitivity at pre-cool share 0.72 leaves about 335. Vehicle-time relaxation not binding; '
                              'the gap is produced by the energy relaxation and is a class-selection effect.'),
                      branch_ref=B.relative_to(ROOT).as_posix()),
        created_at=now, updated_at=now)
    t['nodes']['n0']['insight'] += ('\n[from n15] A valid one-sided relaxation leaves 801 of A-v1 daily utility unbounded '
                                    'above the strongest recorded control, and the gap is a budget-selection effect, so '
                                    'the old no-headroom conclusion does not transfer to the A-v1 contract.')
    t['nodes']['n0']['updated_at'] = now
    tp.write_text(json.dumps(t, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')

    print(json.dumps({'evidence': (out / 'PICKUP_A1_HEADROOM_BOUND_EVIDENCE.json').relative_to(ROOT).as_posix(),
                      'report_sha256': sha(REPORT), 'entries': len(entries),
                      'upper_bound': br['mean_upper_bound'], 'gap_vs_strongest': br['explicit_feat']['mean_gap']}))


if __name__ == '__main__':
    main()
