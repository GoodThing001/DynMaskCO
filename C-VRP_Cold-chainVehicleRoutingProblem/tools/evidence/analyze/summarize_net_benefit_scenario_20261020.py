"""Independently verify and report the bounded numerical-hypothesis pilot."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT, WORKSPACE_ROOT

E = EXTENSION_ROOT
A = E/'comparisons/static_coldchain/adapters'
R = E/'results/static_delivery/net_benefit_scenario_pilot_20261020'
D = E/'docs/static_delivery_research_20261020/net_benefit_scenario'
sys.path.insert(0, str(A))
from net_benefit_audit import audit
from net_benefit_model import config, load_case, structural_audit


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def avg(rows, field):
    return statistics.mean(r['report_at_common_center_rho'][field] for r in rows)


def main():
    raw, c = read(R/'results.json'), config()
    result = copy.deepcopy(raw)
    rho_names = {v['rho']: v['name'] for v in c['tomato']['rho_sensitivity']}
    # The raw run's record() loop shadowed only the setting-name argument.
    # Its scalar optimization_rho and checked target/route/money fields are intact.
    for row in result['rows']:
        row['optimization_setting'] = 'shared_cost_warm_start' if row['seed'] is None else rho_names[row['optimization_rho']]
    result['metadata_correction'] = {'raw_result': 'results.json', 'raw_sha256': sha(R/'results.json'), 'field_corrected': 'rows[*].optimization_setting only', 'reason': 'record() sensitivity loop shadowed setting-name argument; reconstruct unambiguously from saved optimization_rho and seed', 'routes_amounts_seeds_and_optimization_rho_changed': False, 'rerun_of_search': False}
    write(R/'results_metadata_corrected.json', result)
    write(R/'METADATA_CORRECTION.json', result['metadata_correction'])
    D.mkdir(parents=True, exist_ok=True)
    assert c == read(R/'scenario_config.json')
    for path, digest in result['config_and_code_sha256'].items():
        snapshot = R/'source_at_run'/Path(path).name
        assert snapshot.exists() and sha(snapshot) == digest, path
        if Path(path).name != 'run_net_benefit_scenario.py':
            assert sha(E/path) == digest, path
    author = E/'data/classic_static_sources_20261020/VRPTWQ_Solomon'
    provenance = read(author/'SNAPSHOT_PROVENANCE.json')
    assert all(sha(author/p['path']) == p['sha256'] for p in provenance['files'])
    with zipfile.ZipFile(author/'vrptwq_instances.zip') as z:
        members = [p for p in z.infolist() if not p.is_dir()]
        assert len(members) == 56
        assert all((author/'instances'/p.filename).read_bytes() == z.read(p) for p in members)
    upstream = E/'comparisons/static_coldchain/upstream/ALNS'
    pins = read(upstream/'SNAPSHOT_PROVENANCE.json')
    assert all(sha(upstream/p['path']) == p['sha256'] for p in pins['files'])
    originals = read(R/'all56_structural_audit.json')
    assert len(originals) == 56
    for row in originals:
        assert row == structural_audit(load_case(row['case'], c), c)
    conflicts = [r for r in originals if not r['passes_necessary_bounds']]
    assert len(conflicts) == 9
    # Verify the predeclared first-passing-family selection, independent of gains.
    selected = []
    for prefix in ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']:
        selected.append(next(r['case'] for r in originals if r['case'].startswith(prefix) and r['passes_necessary_bounds']))
    assert selected == ['c105', 'c205', 'r101', 'r201', 'rc101', 'rc201']
    assert sorted({r['case'] for r in result['rows']}) == sorted(selected)
    assert not result['warm_start_failures']
    assert len(result['rows']) == 78  # six common initial states plus72 ALNS runs
    objectives = []
    for row in result['rows']:
        ins = load_case(row['case'], c)
        assert ins['sha256'] == row['source_sha256']
        new = audit(ins['nodes'], row['routes'], c, c['tomato']['rho_center'], result['max_vehicles'])
        old = row['report_at_common_center_rho']
        assert new == old and new['feasible_routing']
        assert not new['tomato_marketability_certified'] and not new['mullet_profit_computed']
        assert math.isclose(new['net_benefit_scenario_EUR']/4.5, new['normalized_net_benefit'])
        target = audit(ins['nodes'], row['routes'], c, row['optimization_rho'], result['max_vehicles'])
        assert math.isclose(target['net_benefit_scenario_EUR'], row['optimization_target_net_benefit_EUR_independently_checked'], abs_tol=1e-7)
        for sensitivity in row['fixed_routes_sensitivity_not_reoptimization']:
            check = audit(ins['nodes'], row['routes'], c, sensitivity['rho'], result['max_vehicles'])
            assert all(sensitivity[k] == v for k, v in check.items() if k != 'tours')
        objectives.append(new['net_benefit_scenario_EUR'])
    for case in selected:
        rows = [r for r in result['rows'] if r['case'] == case]
        assert len(rows) == 13
        for setting in c['tomato']['rho_sensitivity']:
            relevant = [r for r in rows if r['optimization_setting'] == setting['name']]
            assert sorted(r['seed'] for r in relevant) == sorted(result['seeds_predeclared'])
    test = A/'test_net_benefit_scenario.py'
    spec = importlib.util.spec_from_file_location('net_benefit_semantic_tests', test)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logs = io.StringIO()
    tests = unittest.TextTestRunner(stream=logs, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    (D/'tests.log').write_text(logs.getvalue(), encoding='utf-8')
    assert tests.wasSuccessful() and tests.testsRun == 10
    native = list(A.glob('*net_benefit*.py'))+[Path(__file__).resolve()]
    for p in native:
        ast.parse(p.read_text(encoding='utf-8'))
    assert not list(A.rglob('__init__.py'))
    old = read(E/'results/static_delivery/quality_evidence_demo_final_20261020.json')
    assert old['config_sha256'] == sha(E/'configs/static_delivery/product_evidence_v1.json')
    assert old['script_sha256'] == sha(A/'quality_evidence_demo.py')
    legacy = subprocess.run(['git', 'diff', '--numstat', '--', 'MASKCO_code', 'CC_Compare'], cwd=WORKSPACE_ROOT, capture_output=True, text=True, check=True)
    assert not legacy.stdout.strip()
    paired = []
    summaries = []
    for case in selected:
        rows = [r for r in result['rows'] if r['case'] == case]
        controls = {r['seed']: r for r in rows if r['method'] == 'ALNS_cost_only_control'}
        center = [r for r in rows if r['optimization_setting'] == 'li_midpoint_ratio']
        differences = [r['report_at_common_center_rho']['net_benefit_scenario_EUR']-controls[r['seed']]['report_at_common_center_rho']['net_benefit_scenario_EUR'] for r in center]
        for seed, diff in zip([r['seed'] for r in center], differences):
            paired.append({'case': case, 'seed': seed, 'center_hypothesis_minus_cost_control_EUR': diff})
        summaries.append({'case': case, 'center_profit_mean_EUR': avg(center, 'net_benefit_scenario_EUR'), 'cost_control_profit_mean_EUR': avg(list(controls.values()), 'net_benefit_scenario_EUR'), 'paired_delta_mean_EUR': statistics.mean(differences), 'paired_delta_min_EUR': min(differences), 'paired_delta_max_EUR': max(differences), 'center_distance_mean_km': avg(center, 'distance_km'), 'cost_control_distance_mean_km': avg(list(controls.values()), 'distance_km'), 'center_vehicles': [r['report_at_common_center_rho']['vehicles_used'] for r in center]})
    rho_summary = []
    for setting in c['tomato']['rho_sensitivity']:
        for case in selected:
            group = [r for r in result['rows'] if r['case'] == case and r['optimization_setting'] == setting['name']]
            vals = [r['optimization_target_net_benefit_EUR_independently_checked'] for r in group]
            rho_summary.append({'case': case, 'rho': setting['rho'], 'setting': setting['name'], 'reoptimized_net_benefit_mean_EUR_at_own_rho': statistics.mean(vals), 'min_EUR': min(vals), 'max_EUR': max(vals)})
    write(D/'summary.json', {'common_center_comparison': summaries, 'paired_seeds': paired, 'reoptimized_sensitivity': rho_summary, 'different_rho_profit_levels_are_different_assumed_price_models': True, 'confidence_intervals_or_significance_claimed': False})
    checks = {'status': 'SCENARIO_NET_BENEFIT_RUNS_AND_ORIGINAL_INPUTS_INDEPENDENTLY_VERIFIED', 'raw_results_sha256': sha(R/'results.json'), 'metadata_corrected_results_sha256': sha(R/'results_metadata_corrected.json'), 'metadata_correction': result['metadata_correction'], 'run_source_snapshots_SHA_verified': 4, 'original_instances_verified_against_author_zip': 56, 'author_source_files_SHA_verified': len(provenance['files']), 'ALNS_upstream_files_SHA_verified': len(pins['files']), 'structurally_conflicting_instances': len(conflicts), 'routing_and_accounting_recomputed': len(result['rows']), 'ALNS_runs': 72, 'common_OR_Tools_warm_starts': 6, 'paired_center_vs_control_seeds': len(paired), 'each_route_set_cross_evaluated_rho_count': 4, 'new_semantic_checks_passed': tests.testsRun, 'native_syntax_files': len(native), 'old_quality_inputs_unchanged': True, 'legacy_upstream_diff_empty': True, 'customers_generated': False, 'profits_are_scenario_only': True, 'TA_to_value_hypothesis_empirically_verified': False, 'tomato_marketability_threshold_configured': False, 'mullet_profit_computed': False, 'MaskCO_gain_claimed': False, 'formal_statistics_claimed': False, 'git_commit_or_push_by_this_round': False}
    write(D/'verification.json', checks)
    lines = [
        '# 最大单次配送净效益：文献数值情景初步结果', '',
        '2026-10-20。本轮从调研推进到实际计算：独立净效益评价器、固定上游ALNS适配、6个完整100客户公开实例、4种价值设定×3种子共72次ALNS搜索，以及6个共同OR-Tools初解。**这是明确假设下的数值研究，不是实测番茄利润预测，不是MaskCO效果结论。**', '',
        '## 模型与来源', '',
        '`max Π = Σ_i m_i × 4.50 × [rho+(1-rho)r_i] − 3.60Σ_i m_i − D − 0.0864Σ_v duration_v`，其中r_i为交付完成时的相对TA保留率。质量kg、分钟、km和EUR分别核算，按一次配送，不年化。', '',
        '- 樱桃番茄4.50 EUR/kg售价、采购为售价80%来自[Buisman与Rohmer2023](https://doi.org/10.1016/j.samod.2023.100020)。3.60是推导的论文假设值，非当前采购报价。',
        '- r_i的温度动力学来自[Foods12(1):6](https://doi.org/10.3390/foods12010006)。全程4°C是理想恒温情景；全体原kg需求解释为单产品番茄是新情景假设，不是原数据商品标签。',
        '- **TA到价格的线性映射是本项目明确提出、尚未实证验证的假设。** 仅参考[Li2018博士论文](https://www.biblio.univ-evry.fr/theses/2018/2018SACLE036.pdf)Table6.1的分级价格形状。rho=0.2、5.5/15、0.7分别取文献端点比与中点比；rho=1为无折价控制。它们不是已拟合的番茄价格参数。',
        '- 容量20000kg、速度60km/h、距离费1 EUR/km及壁面热学/费用系数来自[作者公开基准README](https://github.com/SelinHulagu/vrptwq-solomon-instances)。按稳态壁面热漏λS/L×(20−4)=1728W、原费用系数5e−5 EUR/(W·min)，推导0.0864 EUR/车分钟。**这是简化热负荷费用代理，未模拟开门、太阳辐射、热惯性，也不是实测压缩机电功率或当前电价。未套用18m³机组。**',
        '- 每辆使用中的车在仓库时窗起点出发；等待、服务和空载返仓均计入该车费用。每车独立时钟，客户恰好服务一次，容量/客户开始服务时窗/返仓时限/K≤25为硬约束。K=25是显式数值实验上限，不宣称来自作者车队。固定既有人员等共同费用从排序中省略，净效益采用增量运输口径。',
        '- **番茄TA可售阈值仍未配置，因此这里只确认路由约束可行，不能称食品交付品质/安全完整合格。** 鲻鱼只做相同路线的单产品品质反事实：源图示初始约9分、4°C零阶下降和5分感官阈值来自[Fishes9:480](https://doi.org/10.3390/fishes9120480)，没有给它分配货币价值或混合商品需求。', '',
        '## 原始数据检查与选择', '',
        f'56份原文件逐件与作者ZIP字节比对通过。按60km/h、90分钟服务、仓库0—1440分钟及欧氏直达距离检查，**{len(conflicts)}例存在单客户即无法满足的必要条件冲突**，其余47例仅通过必要条件，不表示已证明整体可行。不能由此认定作者论文出错，结论限定于README与本轮明确运行域。', '',
        '| 原例 | 单客户冲突ID |', '|---|---|'
    ]
    for row in conflicts:
        lines.append('| '+row['case'].upper()+' | '+', '.join(str(v['customer']) for v in row['impossible_customers'])+' |')
    lines += ['', '例如C201的87、90、96号客户即使单独直送，最早返仓仍超过1440分钟；C101的13、57、81、98号客户直达已经晚于最晚服务开始时间。每项证据含精确下界，见[56例检查原始记录](../../../results/static_delivery/net_benefit_scenario_pilot_20261020/all56_structural_audit.json)。原文件未修改，未移动客户或延长时窗。', '',
        '预先选择规则为六类各取首个通过必要条件的原例：C105、C205、R101、R201、RC101、RC201。这个选择在看到任何算法收益之前完成；C101/C201保留跳过记录。每例仍是原100客户，无新造或裁剪实例。', '',
        '## 相同价值模型下的对照', '',
        '每例共同OR-Tools初解预算5秒；两个主要ALNS臂各用5秒、相同3个预先声明种子，均由同一初解开始。以下统一在中心rho=5.5/15下独立重算，比较的是“按净效益搜索”与“按距离＋制冷费用搜索”的**目标消融**，不是ALNS对OR-Tools的公平总预算竞赛。', '',
        '| 实例 | 净效益搜索均值 EUR | 成本控制均值 EUR | 配对差值均值 EUR | 3种子差值范围 EUR | 净效益臂平均km | 成本臂平均km |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for row in summaries:
        lines.append(f"| {row['case'].upper()} | {row['center_profit_mean_EUR']:.2f} | {row['cost_control_profit_mean_EUR']:.2f} | {row['paired_delta_mean_EUR']:+.2f} | [{row['paired_delta_min_EUR']:+.2f}, {row['paired_delta_max_EUR']:+.2f}] | {row['center_distance_mean_km']:.2f} | {row['cost_control_distance_mean_km']:.2f} |")
    positive = sum(r['center_hypothesis_minus_cost_control_EUR'] > 1e-7 for r in paired)
    negative = sum(r['center_hypothesis_minus_cost_control_EUR'] < -1e-7 for r in paired)
    lines += ['', f'18个配对结果中，正差{positive}、负差{negative}、近零{18-positive-negative}。正负结果全部保留；3个种子、短预算和共同初解不足以建立正式统计优势。情景净效益本身为正也不证明算法创造了业务收益，因为收入/采购差额来自所选经济设定。距离增加而净效益提高在该目标下可以发生，须同时看收入折价和制冷费用。', '',
        '## 价值设定敏感性', '',
        '四个rho均重新执行3种子搜索，另对每份返回路线做四设定交叉评价。下表是每种设定在**自身价格函数**下的均值，不应把不同列利润水平之差称作算法提升。', '',
        '| 实例 | rho=0.2 | rho=5.5/15 | rho=0.7 | rho=1无折价 |', '|---|---:|---:|---:|---:|'
    ]
    for case in selected:
        values = [next(r['reoptimized_net_benefit_mean_EUR_at_own_rho'] for r in rho_summary if r['case'] == case and r['setting'] == s['name']) for s in c['tomato']['rho_sensitivity']]
        lines.append('| '+case.upper()+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    fish_min = min(r['report_at_common_center_rho']['minimum_mullet_counterfactual_sensory_score_approx'] for r in result['rows'])
    ta_min = min(r['report_at_common_center_rho']['minimum_tomato_TA_retention'] for r in result['rows'])
    lines += ['', f'全部78份路线返回结果经独立重算路由可行；最低TA保留率{ta_min:.6f}，同路线鲻鱼反事实最低约{fish_min:.6f}分，高于源感官阈值5。后者是参考初值和中心速率的预测，不包含拟合不确定性，也不是微生物安全认证。', '',
        'rho=1时收入与采购为常数，最大净效益与最小决策成本代数等价，这一控制已检查。统一将每个货币项除以4.50 EUR可得到虚拟归一化价值，排序不变；没有通过改变单位提高收益。', '',
        '## 实现、核验与下一步', '',
        '- [配置与假设合同](../../../configs/static_delivery/net_benefit_scenario_v1.json)、[运行源码](../../../comparisons/static_coldchain/adapters/run_net_benefit_scenario.py)、[独立评价器](../../../comparisons/static_coldchain/adapters/net_benefit_audit.py)。使用固定作者ALNS7.0.0源码，原105个成员SHA保持；自定义破坏/修复算子在扩展目录。',
        '- [元数据更正后的完整结果](../../../results/static_delivery/net_benefit_scenario_pilot_20261020/results_metadata_corrected.json)、[原始结果](../../../results/static_delivery/net_benefit_scenario_pilot_20261020/results.json)、[更正说明](../../../results/static_delivery/net_benefit_scenario_pilot_20261020/METADATA_CORRECTION.json)、[逐种子和敏感性摘要](summary.json)、[核验](verification.json)、[10项语义检查](tests.log)。原记录的设定名称被局部循环变量覆盖，仅按原存rho与种子无歧义恢复名称，未改路线/金额/种子/优化参数；运行源码按原SHA保存在source_at_run，当前源码已修复命名，未重复搜索。检查包含独立净效益重算、等待/服务/返仓、覆盖/容量/时窗、无损耗重复扣除、固定价格控制、统一归一化、原数据冲突及完整100客户上4客户两路槽的有限穷举一致性；该穷举是受限邻域检查，不是全问题最优性证明。',
        '- 当前新阶段已实现“可审计的情景净效益计算”，此前的“尚未实现任何净效益求解”仅适用于上一阶段。旧v1品质配置、30项变化量结果、12份原Solomon经典解和RRP历史结论保留。',
        '- 下一步优先验证/替换TA—价值假设与番茄交付接受指标，并完善同车热负荷模型，再确定正式共同协议和MaskCO适配。当前结果可向导师展示建模及计算管线已运行，不能包装成完整食品品质合同或论文主实验方法优势。', '',
        '复跑须指定新输出路径：', '', '```powershell', 'python -B -X utf8 comparisons/static_coldchain/adapters/run_net_benefit_scenario.py --out results/static_delivery/net_benefit_scenario_NEW --max-vehicles 25 --warm-seconds 5 --search-seconds 5', '```', ''
    ]
    (D/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
