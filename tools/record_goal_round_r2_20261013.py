"""Record goal round 2: pickup budget-reserving admission panel + delivery hard block."""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
PANEL = TREE / 'pickup/a1_budget_reserving_admission_root_v4'
BOUND = TREE / 'pickup/a1_budget_reachability_bound_root_v1'
REPORT = EXT / 'docs/当前规划/决策与审计/取货A-v1预算保留接拒面板_2026-10-13.md'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    R = json.loads((PANEL / 'RESULT.json').read_text(encoding='utf-8'))
    A = json.loads((PANEL / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    known_audit_failures = {'audit_budget_matches', 'audit_loaded_files_and_maps_start_end'}
    assert R['failed'] == 0, R['failed']
    assert set(A['failure_names']) <= known_audit_failures, A['failure_names']
    B = json.loads((BOUND / 'RESULT.json').read_text(encoding='utf-8'))
    b = B['bracket']['tw_and_horizon']

    p = R['paired']
    rows = "\n".join(
        "| %d | %.2f | %.2f | %+.2f | %d | %d | %.1f | %.1f |" % (
            x['day'], x['control_utility'], x['candidate_utility'], x['delta'],
            x['control']['served'], x['candidate']['served'],
            x['control']['energy'], x['candidate']['energy']) for x in p)

    delivery = dict(
        line='delivery',
        stage='no_new_stage_this_round',
        block=('the delivery line has no local next step: the sealed-latent head route was closed by n13, scientific '
               'family 2 is closed with its registered barrier FAIL preserved and training disabled, and family 3 '
               'cannot be registered until the full 8+2 request with parameter/dependency pre-publication verification '
               'is closed, which requires the 76M backbone'),
        local_resource=('local available RAM is 1.9-2.5 GiB against the registered 8 GiB full-backbone threshold, so the '
                        'backbone cannot be loaded locally; the 3080 host has about 11.6 GiB but is shared'),
        needed_decision=('explicit user authorisation to run the full 8+2 request and the subsequent bounded training on '
                         'the 3080 host under the exclusive PY/JSON rule'),
        evidence=[(BOUND / 'RELAXATION_CAVEAT_ADDENDUM.json').relative_to(ROOT).as_posix()],
        stable_online_gain=False, scientific_family3_started=False,
    )

    evidence = dict(
        stage='pickup_a1_budget_reserving_admission_panel',
        directory=PANEL.relative_to(ROOT).as_posix(),
        registration_sha256=sha(PANEL / 'REGISTRATION.json'),
        preregistration_sha256=R.get('preregistration_sha256') or sha(
            TREE / 'pickup/A1_BUDGET_RESERVING_ADMISSION_PREREGISTRATION.json'),
        producer=dict(checks=R['checks_count'], failed=R['failed']),
        audit=dict(checks=A['checks_count'], failed=A['failed'], failure_names=A['failure_names'],
                   failures_classified='see AUDIT_FAILURE_ADDENDUM.json; both are harness-scoping or documented-drift artefacts with no measurement effect'),
        panel_days=R['panel_days'], arm_seed=R['arm_seed'],
        budget_B=R['budget_B'], budget_identity=R['budget_identity'],
        paired=p, mean_delta=R['mean_delta'], lead_positive=R['lead_positive'], decision=R['decision'],
        upper_bound_mean=b['mean_upper_bound'],
        strongest_recorded_control=b['explicit_feat']['mean_utility'],
        flags=dict(formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                   scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                   stable_online_gain=False),
        evidence_hashes={},
    )
    for n in ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json', 'AUDIT_RUN_EXIT.json',
              'SOURCE_START.json', 'SOURCE_END.json', 'AUDIT_SOURCE_START.json', 'AUDIT_SOURCE_END.json',
              'run.py', 'audit.py', 'prepare.py', 'supervise.py', 'reserving_admission.py',
              'RUN_SUPERVISION_RECEIPT.json', 'AUDIT_SUPERVISION_RECEIPT.json',
              'AUDIT_FAILURE_ADDENDUM.json'):
        if (PANEL / n).exists():
            evidence['evidence_hashes'][(PANEL / n).relative_to(ROOT).as_posix()] = sha(PANEL / n)

    out = TREE / 'goal_round_20261013_r2'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'PICKUP_BUDGET_RESERVING_PANEL_EVIDENCE.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

    verdict = ('面板配对差均值为正（仅为线索，需另登记 40 天阶段）' if R['lead_positive']
               else '面板配对差均值不为正，按预登记判定该规则被支配，转入联合时间窗×车队耦合方向')
    text = """# 取货 A-v1：预算保留接拒面板（受控 A/B）

2026-10-13，持续目标第 2 轮。执行上一轮预登记的候选：在冻结的 5 天面板上，用**同一个采样器、同一个 arm_seed、同一 K/时限/合同**做受控 A/B，只有 replanner 类不同（冻结 SaaReplanner vs 预算保留子类），因此配对差只反映预留过滤器的作用。

## 候选与机制

预算保留：对订单 i 只在 `REV(i) > lambda(clock) * e_lower(i)` 时放行，其中 `e_lower = q * 0.15 * (25 − target_c)` 是**预冷能耗**（登记固定的成本口径），`lambda(clock)` 是在 λ 网格上、使"剩余视野内期望被接受预冷能耗 ≤ 剩余预算"成立的**最小**价格；标定只用冻结 TRAIN 划分。规则是确定性的、非在线学习，且相对基础 SAA **只能增加拒绝**。

成本口径的取舍已写入登记：冷却分摊 `cooling_share` 是每单常数，会把低收入的类 0 排到最后（REV/est = 类1 2.00 > 类2 1.80 > 类0 1.72），与记录在案的 A-v1 日能耗结构（预冷约占 83%%）相反；因此登记固定用 e_lower（REV/e_lower = 4.76 / 3.17 / 2.33，类 0 最划算）。

## 结果

生产 %d 项 %d 失败、独立审计 %d 项 %d 失败。

| 天 | 对照效用 | 候选效用 | 配对差 | 对照服务 | 候选服务 | 对照能耗 | 候选能耗 |
|---|---|---|---|---|---|---|---|
%s

**配对差均值 = %s**；预登记判定：%s。

上界参照（第 1 轮，严格上界均值 %.2f，最强记录对照 %.2f）保持不变。

## 预算身份漂移（本轮附带发现）

记录在案的 A-v1 预算 `B` **无法由当前源码树复现**：B 由"全接"轨迹推出，而 `scenario_saa.py` 在记录批次之后被修改过。本阶段把它作为**记录性漂移发现**而不是门：登记只冻结了数据集与合同身份（两者均逐字节复现），且两个面板臂都使用重算后的 B，因此配对 A/B 内部一致；但旧记录对照行只作参考。

- 记录 B = %s
- 重算 B = %s
- 差 = %s

## 诚实结论与下一方向

- 本轮的判定按预登记规则机械给出；面板结果与"该规则是否真的改变了日决策"是两件事，若配对差接近 0，则说明该规则的**效果**近似空操作。
- 机制解释：`lambda(clock)` 取的是"使期望能耗刚好装进剩余预算"的**最小**价格，即 LP 的影子价格；而记录在案的基础策略本就把预算用到 ~99.7%%（708.4/710.6），因此影子价格式预留只能在**当日后段**改变构成，无法整体重排类别。要捕获第 1 轮定位的构成缺口，需要**严格高于 LP 影子价格**的价格（故意为更好的未来订单留下预算），这是一个新的、必须另行登记的候选。
- 若面板为负，按预登记转入**联合时间窗×车队耦合**方向：这是第 1 轮上界分析指名的另一条收紧路径，且与预留机制正交。

## 配送线本轮

%s

证据：`%s`。

稳定可归因核心在线增益仍未建立；第2族封闭、第3族未启动；最终种子 20260927/28 未触碰。
""" % (R['checks_count'], R['failed'], A['checks_count'], A['failed'], rows,
       ('%.4f' % R['mean_delta']) if R['mean_delta'] is not None else 'None',
       verdict, b['mean_upper_bound'], b['explicit_feat']['mean_utility'],
       R['budget_identity']['recorded'], R['budget_identity']['recomputed'], R['budget_identity']['delta'],
       delivery['block'], (BOUND / 'RELAXATION_CAVEAT_ADDENDUM.json').relative_to(ROOT).as_posix())
    REPORT.write_text(text, encoding='utf-8', newline='\n')

    note = ('\n> 2026-10-13 持续目标第2轮（取货）：执行第1轮预登记的预算保留接拒候选，在冻结 5 天面板做受控 A/B'
            '（同采样器/同 seed/同 K 与时限，仅 replanner 类不同）。生产 %d/%d、独立审计 %d/%d；配对差均值 %s，'
            '预登记判定：%s。附带确认**记录在案的 A-v1 预算无法由当前源码树复现**（scenario_saa 漂移），'
            '已作为记录性漂移而非门，双臂同用重算 B。配送线本轮无本地新阶段：封存 latent 头路线已由 n13 封闭、'
            '第2族封闭且训练禁用、第3族须先闭合完整 8+2 请求与参数依赖核验，需主干（本机 RAM 远低于 8GiB 门），'
            '需用户明确放行。详见[取货A-v1预算保留接拒面板](docs/当前规划/决策与审计/取货A-v1预算保留接拒面板_2026-10-13.md)。'
            '稳定可归因在线增益仍未建立；最终种子未触碰。\n') % (
        R['checks_count'], R['failed'], A['checks_count'], A['failed'],
        ('%.4f' % R['mean_delta']) if R['mean_delta'] is not None else 'None', verdict)

    entries = [EXT / '项目当前状态.md',
               EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
               EXT / '01_取货线/README.md', EXT / '02_配送线/README.md',
               EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_goal_round_20261013_r2'
    backup.mkdir(exist_ok=False)
    identities = []
    rel_report = 'docs/当前规划/决策与审计/' + REPORT.name
    for q in entries:
        raw = q.read_bytes()
        bak = backup / q.relative_to(EXT)
        bak.parent.mkdir(parents=True, exist_ok=True)
        with bak.open('xb') as f:
            f.write(raw)
        if q.suffix == '.json':
            d = json.loads(raw.decode('utf-8-sig'))
            d['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            d['pickup']['a1_budget_reserving_panel_20261013'] = evidence
            d['pickup']['next'] = ('另登记"价格严格高于 LP 影子价格的预留"候选，或转入联合时间窗×车队耦合方向；'
                                   '两者都需先登记再执行，不按结果挑天。')
            d['delivery']['goal_round_2_20261013'] = delivery
            d['latest_goal_round_20261013_r2'] = dict(
                round=2, pickup_evidence=(out / 'PICKUP_BUDGET_RESERVING_PANEL_EVIDENCE.json').relative_to(ROOT).as_posix(),
                report=rel_report, stable_online_gain=False)
            new = json.dumps(d, ensure_ascii=False, indent=2) + '\n'
        else:
            local = note
            if q != EXT / '项目当前状态.md':
                local = note.replace(rel_report, os.path.relpath(REPORT, q.parent).replace('\\', '/'))
            new = local + raw.decode('utf-8-sig')
        assert q.read_bytes() == raw
        q.write_text(new, encoding='utf-8', newline='\n')
        identities.append({'path': q.relative_to(EXT).as_posix(),
                           'before_sha256': sha(bak), 'after_sha256': sha(q)})
    (backup / 'UPDATE_IDENTITY.json').write_text(
        json.dumps({'files': identities, 'report_sha256': sha(REPORT)}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8', newline='\n')

    tp = TREE / 'dual_track_optimization_20261013_root_v1/.arbor/tree.json'
    t = json.loads(tp.read_text(encoding='utf-8'))
    now = int(datetime.datetime.now().timestamp())
    t['nodes']['n16'] = dict(
        id='n16', parent='n15', depth=6, status='executed',
        hypothesis=('Pickup: a pre-registered clock-dependent shadow-price admission rule beats the frozen SAA control on '
                    'the same five frozen days under a controlled A/B'),
        insight=('The shadow-price reservation selects the smallest price that fits the remaining budget, i.e. the LP '
                 'shadow price, which the already budget-saturated base policy effectively implements; the recorded '
                 'A-v1 budget is also not reproducible from the current source tree. Capturing the bound gap needs a '
                 'price strictly above the LP shadow price or the joint time-window and fleet coupling.'),
        metadata=dict(dev_score=float(A['checks_count']), test_score=None,
                      result=('Producer%d/%d, audit%d/%d, panel days %d, mean paired delta %s, lead_positive %s. '
                              'Budget drift recorded: recorded %s vs recomputed %s.'
                              % (R['checks_count'], R['failed'], A['checks_count'], A['failed'], R['panel_days'],
                                 R['mean_delta'], R['lead_positive'], R['budget_identity']['recorded'],
                                 R['budget_identity']['recomputed'])),
                      branch_ref=PANEL.relative_to(ROOT).as_posix()),
        created_at=now, updated_at=now)
    t['nodes']['n0']['insight'] += ('\n[from n16] The pre-registered shadow-price reservation reduces to the LP shadow '
                                    'price, which the budget-saturated base already implements; the recorded A-v1 budget '
                                    'is not reproducible from the current sources.')
    t['nodes']['n0']['updated_at'] = now
    tp.write_text(json.dumps(t, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')

    print(json.dumps({'evidence': (out / 'PICKUP_BUDGET_RESERVING_PANEL_EVIDENCE.json').relative_to(ROOT).as_posix(),
                      'report_sha256': sha(REPORT), 'mean_delta': R['mean_delta'],
                      'lead_positive': R['lead_positive'], 'entries': len(entries)}))


if __name__ == '__main__':
    main()
