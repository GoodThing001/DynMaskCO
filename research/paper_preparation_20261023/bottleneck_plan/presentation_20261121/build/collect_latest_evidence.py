import csv
import json
from collections import defaultdict
from statistics import fmean, median
from pathlib import Path

OUT = Path(__file__).resolve().parent
B = OUT.parents[1]
U = B / 'unified_external_20261120'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        result = list(csv.DictReader(f))
    for r in result:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return result

campaigns = {c: read(U / 'results' / c / 'analysis.json')
             for c in ['benchmark01', 'prospective01', 'retail01']}
three = read(U / 'results/three_sources.json')
sources = [('VRPPDL-p100', '新荷比配送日', 'prospective01'),
           ('XTW-p100', '新XTW几何', 'prospective01'),
           ('RWVRP-p100', '新商超城市', 'retail01')]
primary = {}
for key, label, campaign in sources:
    p = campaigns[campaign]['panels'][key]
    primary[key] = dict(label=label, campaign=campaign, cases=p['cases'],
                       units=p['geometries'], scores=p['scores'],
                       rank16=p['all_ranks'], rank13=p['external_plus_candidate_ranks'],
                       strongest_external=p['strongest_external'], contrasts=p['contrasts'],
                       criterion=three['primary'][key],
                       source_file=str(U/'results'/campaign/'analysis.json'))
methods = campaigns['prospective01']['methods']
main_table = [dict(method=m, values={s: dict(U=primary[s]['scores'][m],
                  rank16=primary[s]['rank16'][m], rank13=primary[s]['rank13'].get(m))
                  for s in primary}) for m in methods]
forests = [dict(source=s, label=p['label'], units=p['units'],
               strongest=dict(method=p['strongest_external'], **p['contrasts'][p['strongest_external']]),
               heat=dict(method='ORDER_SPLIT_D', **p['contrasts']['ORDER_SPLIT_D']))
           for s, p in primary.items()]
data = dict(updated='2026-11-21',
    scope=dict(unique_cases=62, methods=16, external_methods=12, internal_ablations=3,
               search_seeds=3, endpoints=2976, audited_routes=81251,
               existing_retested_cases=30, prospective_primary_cases=26,
               gh_same_parent_stress_cases=6, prospective_source_top3=3,
               prospective_source_total=3, prospective_unit_top3=26,
               prospective_unit_total=26, prospective_unit_first=19,
               future_sealed_cases=32, online_budget_s=60, physical_cores_per_endpoint=1),
    primary=primary, main_table=main_table, forest_by_source=forests,
    equal_source=three['equal_source'], criteria=three['criteria'],
    cities=rows(U/'results/retail01/cities.csv'),
    campaigns=campaigns,
    audit={c:read(U/'results'/c/'independent_audit.json') for c in campaigns},
    source_files={c:str(U/'results'/c/'RESULTS.md') for c in campaigns})

OLD = B / 'presentation_20261120/build'
venues = read(OLD/'venue_ranks.json')
recent = read(OLD/'recent5_sources.json')
method_meta = {}
for m in venues['method_venues']:
    if m['method'] not in campaigns['benchmark01']['external_methods']:
        continue
    item = dict(m)
    v = next((v for v in venues['venues'] if m['method'] in v.get('methods', [])), {})
    item['grade_display'] = m.get('ranking_display', v.get('display', '未核实'))
    if m['method'] in ('ALNS', 'VNS'):
        item['grade_display'] = '中科院工程技术2区（2025）*'
        item['grade_evidence'] = '公开逐刊资料一致；官方原表未直接核实'
    item['venue_evidence'] = v
    method_meta[m['method']] = item
for m in recent['methods']:
    key = 'PyVRP' if m['method'] == 'PyVRP-HGS' else m['method']
    method_meta[key] = dict(m, method=key)
data['method_metadata'] = method_meta
data['grade_policy'] = dict(conference='CCF 2026', journal='中科院2025升级版大类',
    journal_star='2区和3区带*为已核对公开逐刊记录；官方逐刊原表未直接取得',
    unverified=['NDS-Econ: TMLR', 'RouteFinder: TMLR'],
    source_files=[str(OLD/'venue_ranks.json'), str(OLD/'recent5_sources.json')])

freight = read(B/'maskco_perishable_external_20261119/freight_source_inventory.json')
data['datasets'] = {
    'training_and_existing_sources': read(OLD/'dataset_sources.json')['datasets'],
    'VRPPDL-p100': dict(label='荷比新配送日', units=10, role='前瞻主来源：同运营商跨订单日',
        full_name='Vehicle Routing Problem with Pickups, Deliveries, and Scheduled Linehauls',
        authors=['Casper H. Bazelmans','Siem L. ter Braake','Albert H. Schrotenboer','Rolf N. van Lieshout','Tom Van Woensel'],
        venue='European Journal of Operational Research', year=2026, volume='330(3)',pages='995–1008',
        early_online='2025-08-23', doi='10.1016/j.ejor.2025.07.070',
        source_url='https://github.com/siemterbraake/vrppdl-algorithms-instances',
        publication_url='https://research.tue.nl/en/publications/the-vehicle-routing-problem-with-pickups-deliveries-and-scheduled/',
        archive_commit=freight['archive_commit'], countries={'BE':5,'NL':5},
        selection='每国MO/TU/WE/TH/FR编号1，原顺序首100个可用配送客户',
        identity='一般零担货运；同运营商；位置经作者匿名扰动；统一参数冷链投影',
        max_overlap_with_sealed_days=0,max_projection_relative_error_percent=0.01509,
        admission=str(U/'new_panel/准入.md')),
    'XTW-p100': dict(label='新XTW几何',units=8,role='前瞻主来源：跨客户几何',
        source_url='https://galgos.inf.puc-rio.br/cvrplib/en/instances',
        selection=['XTW-n716','XTW-n749','XTW-n766','XTW-n801','XTW-n819','XTW-n876','XTW-n936','XTW-n979'],
        identity='官方原件保留原仓和前100客户；与历史、封存及彼此最多重叠10个客户位置',
        registered_or_sealed_originals=83,max_customer_overlap=10,admission=str(U/'new_panel/准入.md')),
    'RWVRP-p100': dict(label='商超八城',units=8,role='前瞻主来源：跨城市门店布局',
        institution='University of Zagreb, FER–LARES',dataset_name='RWVRP Benchmark / VRP Benchmark Dataset',
        paper_title='A Multi-City Benchmark for Vehicle Routing in Real Urban Environments',
        venue='IEEE International Conference on Intelligent Transportation Systems (ITSC)',year=2026,
        presenter='Nikica Perić',authors=None,doi=None,
        bibliographic_note='研究组官方新闻确认论文题目、报告者及会议；完整作者表与DOI未核实',
        paper_source='https://www.lares.fer.hr/lares/about_lares/news?%40=33uop',
        source_url='https://github.com/unizg-fer-lares/RWVRP_benchmark/tree/2c3a976f2212923db31ac86aea11483930c64b14',
        original_cities=16,original_variant_instances=144,selected_cities=[r['city'] for r in data['cities']],
        identity='OpenStreetMap真实门店；需求和时窗由作者建模；统一直线公里、速度和冷链费用',
        selection='Medium和Large整档8城，各原编号前100客户；同质静态基础变体',
        original_capacity_pallets=18,project_capacity_kg=20000,depot_operation_minutes_each_end=30,
        max_projection_relative_error_percent=0.00151, max_customer_overlap=0,
        attribution='© OpenStreetMap contributors',admission=str(U/'retail_source/准入.md')),
    'GH-p100-block2': dict(label='GH同母例新客户块',cases=6,units=5,role='压力测试，单列',
        source_url='https://www.sintef.no/projectweb/top/vrptw/',
        selection='既有母实例的第101–200客户，6类各编号3',
        identity='同母客户集合内跨客户块；不是独立新母几何，也不进入3前瞻来源等权',
        originals_screened=300,previous_parent_geometries=20,eligible_new_parent_geometries=0,
        admission=str(U/'new_panel/准入.md'))}

data['rank_coverage'] = read(U/'results/prospective01/rank_coverage.json')
data['rank_coverage']['RWVRP-p100'] = dict(units=8,top3_13=8,top3_16=8,first_13=4,
    strongest_external='ALNS',controllable_loss_reduction_percent=0.11)
data['retail_heterogeneity'] = dict(wins_against_ALNS=4,losses_against_ALNS=4,
    sf_bay_delta=next(r['delta_panel_best'] for r in data['cities'] if r['city']=='SF Bay Area'),
    leave_out_sf_mean=fmean(r['delta_panel_best'] for r in data['cities'] if r['city']!='SF Bay Area'))

data['origin'] = read(U/'origin_effects.json')
data['origin']['role'] = '已测开发实例：8例/7几何×2种子×4臂；另存26旧例跨批桥接，分别报告'
origin_rows = rows(U/'results/origin01/paired.csv')
data['origin']['direct']['scores'] = {}
for s in sorted({r['source'] for r in origin_rows}):
    data['origin']['direct']['scores'][s] = {}
    for m in ['MASK','SLOT','ORDER_SPLIT','ORDER_SPLIT_D']:
        bygeo = defaultdict(list)
        for r in origin_rows:
            if r['source']==s: bygeo[r['geometry']].append(r[m])
        data['origin']['direct']['scores'][s][m] = fmean(fmean(v) for v in bygeo.values())

data['trajectory'] = read(U/'results/trajectory01/analysis.json')
trajectory_groups = defaultdict(list)
for r in rows(U/'results/trajectory01/paired_seeds.csv'):
    trajectory_groups[(r['source'],r['control'],r['readout_s'])].append(r['paired_delta_U'])
data['trajectory']['curves'] = [dict(source=s,control=c,time_s=t,mean=fmean(v))
    for (s,c,t),v in sorted(trajectory_groups.items())]
data['trajectory']['role'] = '既有来源各3代表几何、10种子；60秒运行的实际接受轨迹，非新前瞻验证'

startup = B/'maskco_followup_dev_20261120/startup'
service_rows = rows(startup/'results/service01/paired_service.csv')
data['startup'] = dict(role='冻结训练例计时和3既有开发例的首次请求复核，主排名仍是原锁版',
    compile_reuse=read(startup/'results/startup02/summary.json'),service_rows=service_rows,
    service_summary={},source=str(startup/'RESULTS.md'))
for method in ['ORIGINAL','CACHED','LEAN_DISTANCE','ALNS']:
    selected=[r for r in service_rows if r['service']==method]
    valid=[r for r in selected if isinstance(r['U_after_60s_from_entry'],(int,float))]
    data['startup']['service_summary'][method]=dict(cells=len(selected), delivered=len(valid),
        entry_to_search_median_s=median(r['entry_to_search_s'] for r in selected))
data['startup']['service_summary']['CACHED']['delta_ALNS']=37.33
data['startup']['service_summary']['CACHED']['delta_LEAN_DISTANCE']=-3.53

data['bounds'] = dict(source=str(U/'bounds/RESULTS.md'),
    all_method_source_gaps=rows(U/'bounds/source_gaps.csv'),case_bounds=rows(U/'bounds/case_bounds.csv'),
    exact_small=read(U/'bounds/exact_small.json'),
    definition='允许子圈的乐观最小费用指派松弛；当前U至上界的距离为潜在改进的上限，不是已知最优误差',
    best_feasible_definition='同例16法3种子中最高已实现可行U',
    existing_XTW_scope='bounds中既有XTW合并13个不同案例；主表为10例，低重叠另列10例且共享7例',
    positive_average_cases=58,negative_average_cases=4,
    proven_negative=dict(case='xtw_n716_p100',nominal_value_U=4135.29,loss_lower_bound_U=10007.53,net_benefit_upper_bound_U=-5872.24))
data['bounds']['prospective_maskco']=[r for r in data['bounds']['all_method_source_gaps']
    if r['method']=='ORDER_SPLIT' and r['batch'] in ('prospective01','retail01')]

scenario_rows=rows(U/'economic_sensitivity/cost_scenarios.csv')
scenario_groups=defaultdict(list)
for r in scenario_rows: scenario_groups[(r['panel'],r['control'])].append(r)
data['sensitivity']=dict(source=str(U/'economic_sensitivity/RESULTS.md'),
    role='旧sync01四法360个终点固定路线和时刻重新计价，非三前瞻来源重优化',
    all_scenarios=scenario_rows,summary=[])
for (panel,control),rr in sorted(scenario_groups.items()):
    base=next(r['mean_delta_U'] for r in rr if all(r[k]==1 for k in
        ['quality_price_factor','distance_price_factor','cold_price_factor']))
    corners=[r['mean_delta_U'] for r in rr if all(r[k] in (.8,1.2) for k in
        ['quality_price_factor','distance_price_factor','cold_price_factor'])]
    data['sensitivity']['summary'].append(dict(panel=panel,control=control,baseline_delta_U=base,
        min_corner_delta_U=min(corners),max_corner_delta_U=max(corners)))

data['interpretation_notes']=[
    '62去重案例=30已测复测+26前瞻主案例+6同母例GH压力案例；三前瞻来源为10配送日+8新XTW几何+8零售城市。',
    '三前瞻来源平均外部排名均1/13，加入内部距离等三参照后为2/16、1/16、1/16；26/26前三、19/26第一是逐单位13法排名。',
    '热图稳定门槛是均值>1e-6、严格多数单位改善、至少2/3种子正；新XTW通过但95%描述区间仍跨零。',
    '各来源最强外部法由实测来源均值选择；森林图区间是描述性配对区间，未对选择最强对照作多重选择修正。',
    '三来源等权热图12.11 U区间跨零；ALNS75.37 U区间为正。完整方法优势与热图独立增量分别解释。',
    'RWVRP对ALNS4胜4负；去掉旧金山湾区后的7城均值-0.62 U是描述性影响检查，不重定义主面板。',
    '荷比订单日同运营商，客户坐标经匿名扰动；RWVRP是真实门店位置和作者建模订单，不是实测冷链经济收益。',
    'CaR/DeepACO/POMO/AM沿用项目冷链训练权重；RF/CaDA/MVMoE/RRNCO直接迁移作者权重；训练投入未统一。',
    '十种子和原版四臂属于已测几何补充；敏感性为旧sync01固定方案；缓存首次调用为开发结果，均不合并进2976主终点。',
    '原FLEET_PRICE的32例继续封存；此次26前瞻例并非开封这32例。']
(OUT/'latest_evidence.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'file':str(OUT/'latest_evidence.json'),'methods':len(main_table),'primary':list(primary),'cities':len(data['cities'])}, ensure_ascii=False))
