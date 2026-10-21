"""Assemble an inspectable evidence packet from fetched sources and actual runs."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT

BASE=EXTENSION_ROOT/'docs/static_delivery_research_20261020'
COMPARE=EXTENSION_ROOT/'comparisons/static_coldchain'
DATA=EXTENSION_ROOT/'data/classic_static_sources_20261020'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

SOURCES=[
 {'id':'T1','title':'Analysis of Nutrients and Volatile Compounds in Cherry Tomatoes Stored at Different Temperatures','doi':'10.3390/foods12010006','year':2023,'url':'https://www.mdpi.com/2304-8158/12/1/6','access':'publisher PDF and EuropePMC XML saved; Table2 p10 visually checked','supports':'first-order TA and Arrhenius central parameters','does_not_support':'saleable mass, price function, exact batch initial quality or marketability threshold'},
 {'id':'T2','title':'Modelling colour and firmness changes of tomatoes (Lycopersicon esculentum L.) fruit during storage at different temperatures','doi':'10.1016/j.jfoodeng.2012.08.024','year':2013,'url':'https://doi.org/10.1016/j.jfoodeng.2012.08.024','access':'publisher abstract and author institution poster; final journal tables not acquired','supports':'firmness/colour modelling candidate','does_not_support':'copying poster coefficients as final journal coefficients; N to N/mm threshold conversion'},
 {'id':'T3','title':'Modelling the acceptance period of truss tomato batches','doi':'10.1016/j.postharvbio.2007.03.012','year':2007,'url':'https://research.wur.nl/en/publications/modelling-the-acceptance-period-of-truss-tomato-batches/','access':'author institutional abstract','supports':'consumer acceptance depends on firmness, colour, harvest maturity and intended consumption time','does_not_support':'universal TA-to-saleability threshold'},
 {'id':'T4','title':'Estimating Implicit Marginal Prices of Quality Characteristics of Tomatoes','doi':'10.1017/S0081305200025139','year':1985,'url':'https://doi.org/10.1017/S0081305200025139','access':'publisher abstract; AgEcon direct PDF downloads returned non-PDF, not claimed as reviewed','supports':'hedonic pricing is a published way to estimate quality value','does_not_support':'current Chinese prices or cherry-tomato TA-to-money regression coefficients'},
 {'id':'T5','title':'The value of time and temperature history information for the distribution of perishables','doi':'10.1016/j.ejor.2023.03.006','year':2023,'url':'https://www.sciencedirect.com/science/article/pii/S037722172300200X','access':'publisher abstract and university metadata; university PDF download403','supports':'vine-ripened tomato time/temperature history can inform economic transport decisions','does_not_support':'unread numeric price or quality coefficients; this is mode choice rather than our multi-stop fleet model'},
 {'id':'F1','title':'Shelf Life Study of Chilled Mullet (Mugil cephalus): Histamine Formation and Quality Degradation at Constant and Dynamic Storage Conditions','doi':'10.3390/fishes9120480','year':2024,'url':'https://www.mdpi.com/2410-3888/9/12/480','access':'publisher PDF saved; p11 text and Table6 p12 visually checked','supports':'zero-order overall sensory score decline; 4C reference0.452 score/day, Ea81.4kJ/mol; sensory threshold5','does_not_support':'initial score9, mass loss fraction, price, or replacing safety evaluation by a sensory score'},
 {'id':'F2','title':'The Welfare Value of Freshness: A Hedonic Price Analysis in the Retail Seafood Market in Japan','doi':'10.1086/727493','year':2024,'url':'https://www.journals.uchicago.edu/doi/10.1086/727493','access':'publisher abstract; fulltext paywalled/direct PDF403','supports':'freshness-price effect depends on species and market; horse mackerel premium but no significant sardine premium','does_not_support':'transfer of a horse-mackerel bioimpedance price coefficient to mullet sensory score'},
 {'id':'F3','title':'The Impact of Disclosing Freshness on Sensory Evaluation and Willingness to Pay for Fresh Seafood','doi':'10.1111/1750-3841.70764','year':2026,'url':'https://pubmed.ncbi.nlm.nih.gov/41454604/','access':'PubMed abstract; 2025 online/2026 issue','supports':'red sea-bream sashimi WTP depends on disclosure, counterexample to universal monotonic freshness pricing','does_not_support':'a mullet price curve'},
 {'id':'V1','title':'CO2跨临界系统在冷藏车上的应用研究','doi':'10.13380/j.ltpl.2023.01.009','year':2023,'url':'https://dwwl.ustc.edu.cn/ch/reader/create_pdf.aspx?file_no=202301009','access':'official seven-page PDF extracted by web tool; direct download403 and screenshot cache miss; no local PDF claimed','supports':'same transport-condition Q0=1.75kW and COP1.66 at outside30C/cabin0C/80Hz/9MPa; P includes compressor and fan; P=1.0542kW is derived','does_not_support':'whole-route duty cycle, door heat, all-temperature constant power, 20-ton payload mapping or electricity tariff'},
 {'id':'V2','title':'Experimental analysis of refrigerated truck thermal behaviour','doi':None,'year':2014,'url':'https://repository.up.ac.za/items/63f40fa1-f2d2-4765-89d3-3fa9977df180','access':'HEFAT2014 institutional PDF saved; Table4 p7 visually checked','supports':'cooling power differs from fuel thermal input and COP_f','does_not_support':'using15kW engine rating as electric refrigeration load or transplanting -20C results into chilled operation'},
 {'id':'V3','title':'Extending and Solving the Refrigerated Routing Problem','doi':'10.3390/en13236214','year':2020,'url':'https://doi.org/10.3390/en13236214','access':'existing publisher PDF and original dataset SHA retained; current v3 formulas independently audited','supports':'traction and refrigeration cost baseline with documented original physical units','does_not_support':'tomato/mullet pricing, fresh temperature setpoints or original multi-vehicle data'},
 {'id':'D1','title':'Solomon VRPTW 100-customer benchmark, SINTEF mirror','doi':'10.1287/opre.35.2.254','year':1987,'url':'https://www.sintef.no/projectweb/top/vrptw/100-customers/','access':'original backup ZIP56 files and BKS HTML saved','supports':'classic routing benchmark, hierarchical vehicle-count/distance objective and double-precision evaluation','does_not_support':'kg, km, minutes, product labels or cold-chain profit without sourced mapping'},
 {'id':'D2','title':'An Exact Algorithm for the Vehicle Routing Problem with Time Windows and Perishable Products','doi':'10.1287/trsc.2025.0262','year':2026,'url':'https://research.vu.nl/en/publications/an-exact-algorithm-for-the-vehicle-routing-problem-with-time-wind/','access':'author institutional abstract and author GitHub dataset/README saved; paper PDF403','supports':'author-provided56 transformed Solomon cases with explicit km/kg/min and declared quality/refrigeration benchmark parameters','does_not_support':'calling generic quality400 a tomato/mullet measurement, reproducing unread exact-solver formulas or adding sales prices'},
 {'id':'M1','title':'ALNS: a Python implementation of the adaptive large neighbourhood search metaheuristic','doi':'10.21105/joss.05028','year':2023,'url':'https://joss.theoj.org/papers/10.21105/joss.05028','access':'publisher paper page and official source snapshot','supports':'problem-agnostic objective and operator framework suitable for a new explicit net-benefit adapter','does_not_support':'claiming a downloaded library already implements our cold-chain model'},
 {'id':'M2','title':'PyVRP: a high-performance VRP solver package','doi':'10.1287/ijoc.2023.0055','year':2024,'url':'https://github.com/PyVRP/PyVRP','access':'author source README, pinned snapshot and installed0.14.0 wheel used','supports':'strong traditional CVRP/VRPTW routing baseline','does_not_support':'arbitrary route-dependent food deterioration objective via static arc costs'},
 {'id':'M3','title':'RouteFinder: Towards Foundation Models for Vehicle Routing Problems','doi':None,'year':2025,'url':'https://openreview.net/forum?id=QzGLoaOPiY','access':'official author repository and frozen original snapshot','supports':'static multi-variant VRP neural baseline candidate','does_not_support':'unchanged pretrained reward for new net-benefit objective'},
 {'id':'M4','title':'DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization','doi':None,'year':2023,'url':'https://github.com/henry-yeh/DeepACO','access':'official author repository and source snapshot','supports':'neural-guided ant search candidate with CVRP implementation','does_not_support':'unmodified time-window/quality/temperature feasibility'},
 {'id':'M5','title':'Towards Efficient Constraint Handling in Neural Solvers for Routing Problems','doi':None,'year':2026,'url':'https://openreview.net/forum?id=raDFGuQxvD','access':'author CaR source and README, ICLR2026 link','supports':'new construct-and-refine constrained-routing baseline candidate','does_not_support':'already learning quality constraints or food values'},
 {'id':'M6','title':'Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood','doi':'10.1016/j.cor.2021.105643','year':2022,'url':'https://github.com/vidalt/HGS-CVRP','access':'author source README and source snapshot','supports':'classic CVRP routing control','does_not_support':'native VRPTW or arbitrary cold-chain reward'},
 {'id':'M7','title':'Google OR-Tools Vehicle Routing Problem with Time Windows','doi':None,'year':None,'url':'https://developers.google.com/optimization/routing/vrptw','access':'official documentation,9.14 source snapshot and9.14.6206 wheel used','supports':'capacity/time dimension baseline and guided local search','does_not_support':'putting an arbitrary route-history quality objective into a static two-index arc callback'},
]

METHODS=[
 ('PyVRP','PyVRP/PyVRP','retain_routing_only','已运行经典VRPTW；净效益需要真正修改搜索评价，不能仅事后重评分'),
 ('OR-Tools','google/or-tools','retain_routing_only','已运行经典VRPTW；新目标优先自定义ALNS，非简单静态弧成本回调'),
 ('RRNCO','ai4co/real-routing-nco','retain_neural_candidate','矩阵输入有价值；静态重写桥接、单位和目标；旧在线限制不自动适用于静态'),
 ('CaDA','CIAM-Group/CaDA','retain_neural_candidate','保留约束感知构造；需新增品质、目标与静态协议适配'),
 ('RouteFinder','ai4co/routefinder','priority_neural_candidate','原生静态多变体路由；新净效益须适配并同目标训练'),
 ('MVMoE','RoyalSkye/Routing-MVMoE','secondary_neural_candidate','多任务结构可用；原奖励与新品质约束不同'),
 ('POMO','yd-kwon/POMO','secondary_routing_control','经典CVRP控制；不原生覆盖新品质与制冷目标'),
 ('AttentionModel','wouterkool/attention-learn-to-route','secondary_routing_control','CVRP等原任务控制；不引入未授权可选接单'),
 ('DeepACO','henry-yeh/DeepACO','priority_neural_search_candidate','保留蚁群与神经启发式；重写完整路线评价和约束后才能进净效益主表'),
 ('SGBS','yd-kwon/SGBS','secondary_search_candidate','搜索增强可复用；不是独立冷链模型，需同目标评分与预算'),
 ('Sym-NCO','alstn12088/Sym-NCO','secondary_routing_control','对称假设须核验；时变/有向路线和新品质目标不能直接沿用'),
 ('Omni-VRP','RoyalSkye/Omni-VRP','secondary_neural_candidate','保留跨分布路由控制；不把其原目标成绩当新任务结果'),
 ('Learning-to-Delegate','mit-wu-lab/learning-to-delegate','secondary_improvement_candidate','可复用子问题选择思想；子问题须计入全路线品质影响且核验资产许可'),
 ('Learn-Improvement-Heuristics','WXY1427/Learn-Improvement-Heuristics-for-Routing','secondary_improvement_candidate','可复用改进策略；需改目标/可行性并核验许可'),
 ('PIP-constraint','jieyibi/PIP-constraint','narrow_task_constraint_reference','已下载主要TSPTW/TSPDL实验非当前车队主表；不把框架理论判为永久不能扩展'),
 ('CO-enriched-ML','tumBAIS/euro-meets-neurips-2022','static_components_only','旧动态接拒策略不作当前主表；原静态求解部件可独立核验'),
 ('MAPT','Beihang-BIGSCity/MAPT','outside_current_task','动态多车取送货与本轮静态配送不同；暂不以旧适配直接比较'),
 ('ALNS','N-Wouda/ALNS','priority_new_net_benefit_baseline','新增通用框架；可显式用负净效益作最小化值、不可行解拒绝；适配尚未完成'),
 ('HGS-CVRP','vidalt/HGS-CVRP','new_cvrp_control','新增经典CVRP源；本轮主VRPTW用PyVRP，不能把HGS直接当原生TW解法'),
 ('CaR','jieyibi/CaR-constraint','new_constraint_candidate','新增ICLR2026源码；原生CVRP/TSPTW/TSPDL/VRPBLTW/SOP，食品品质与温度需自定义'),
]

def main():
    write(BASE/'sources.json',SOURCES)
    methods=[{'method':n,'official_repo':'https://github.com/'+r,'status':s,'assessment':why,
              'new_snapshot_saved':(COMPARE/'upstream'/n/'SNAPSHOT_PROVENANCE.json').exists(),
              'legacy_dynamic_adapter_reused':False,'net_benefit_ready':False} for n,r,s,why in METHODS]
    write(COMPARE/'method_eligibility.json',methods)
    pins=[]
    for folder in sorted((COMPARE/'upstream').iterdir()):
        provenance=folder/'SNAPSHOT_PROVENANCE.json'
        if not provenance.exists(): continue
        r=json.loads(provenance.read_text(encoding='utf-8'))
        verified=all(sha(folder/f['path'])==f['sha256'] for f in r['files'])
        pins.append({k:r[k] for k in ('name','repo','commit','url','archive_sha256','archive_bytes','license_files')}|{'files':len(r['files']),'all_member_sha256_verified':verified,'provenance_sha256':sha(provenance)})
    write(COMPARE/'SOURCE_PINS.json',pins)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(COMPARE/'.runtime/packages')])}
    write(COMPARE/'RUNTIME_VERSIONS.json',{'scope':'local --target directory; global environment not modified','packages':packages})
    def rows(path):
        out=[]
        for p in sorted(path.rglob('*.txt')):
            table=[]
            for line in p.read_text().splitlines():
                bits=line.split()
                if len(bits)==7:
                    try: table.append(list(map(float,bits)))
                    except ValueError: pass
            out.append({'file':str(p.relative_to(DATA)),'sha256':sha(p),'nodes':len(table),
                        'customers':len(table)-1,'total_demand':sum(r[3] for r in table),
                        'maximum_customer_demand':max(r[3] for r in table),
                        'depot_due':table[0][5]})
        return out
    dataset={'source_date':'2026-10-20','no_instances_generated':True,
             'original_solomon':{'count':56,'units':'original abstract benchmark units; not assumed physical','rows':rows(DATA/'solomon_100/instances')},
             'author_vrptwq':{'count':56,'units':'README explicitly km/kg/min; capacity20000kg, speed60km/h','rows':rows(DATA/'VRPTWQ_Solomon/instances'),
                            'commit':json.loads((DATA/'VRPTWQ_Solomon/SNAPSHOT_PROVENANCE.json').read_text())['commit'],
                            'boundary':'generic numerical quality model, no product/customer food labels or sale prices; no dataset license file declared'}}
    write(DATA/'DATASET_AUDIT.json',dataset)
    results=json.loads((EXTENSION_ROOT/'results/static_delivery/classic_solomon_preliminary_v2_20261020/results.json').read_text())
    summary=[]
    for r in results['rows']:
        ref=r['reference']; report=r['report']; gap=r['distance_gap_pct']
        note='车数不同，不比较距离gap' if gap is None else ('与两位小数BKS一致' if abs(report['distance']-ref['distance'])<.005 else f'{gap:.3f}%')
        summary.append(f"| {r['case'].upper()} | {r['method']} | {report['vehicles_used']}/{ref['vehicles']} | {report['distance']:.2f} | {note} |")
    table='| 实例 | 方法 | 实际车数/BKS车数 | 原单位距离 | 距离对照 |\n|---|---|---:|---:|---|\n'+'\n'.join(summary)
    (BASE/'initial_results.md').write_text('''# 可向导师展示的初步结果（2026-10-20）

本轮交付为经典路由验证与文献品质计算。它不是番茄/鲻鱼最大净效益结果，也未比较MaskCO优劣。

原版Solomon C1/C2/R1/R2/RC1/RC2六类各第一例，均保持100客户原文件，无截取、生成、坐标反造或温区标签。OR-Tools9.14.6206和PyVRP0.14.0每例各10秒搜索上限，构模与搜索总时间另报。PyVRP种子20261020；OR-Tools未声称使用等价随机种子。单次先导试验不构成统计显著性结论。容量/车数/时间窗为原文件值。

'''+table+'''

12/12解由原始双精度欧氏距离、客户覆盖、载量、服务开始时间窗和返仓时间独立重算可行。内部采用1e6整数缩放，距离四舍五入、行驶时长向上取整；并非声称内部使用精确实数。M=(客户数+可用车数)×最大缩放弧长+1仅编码“先最少车、后最短距离”，不是车辆费用。PyVRP为此数字尺度设置罚参数上限1e7，并检查保守int64界。

只在车数相同时比较距离。BKS是SINTEF页面两位小数参考值；C101/C201等微小负gap来自显示舍入，不表示刷新BKS。RC101页面标签为rc101b，官方已指出1696.94涉及舍入，当前显示1696.95，本轮沿用1696.95；b是详细解署名，不另改实例。R201的PyVRP距离虽短但多用一车，不能据此称优于BKS。

首轮结果在classic_solomon_preliminary_20261020保留：PyVRP路线导出误用了旧visits()接口，6项失败属于适配器错误。修复为0.14 ScheduledActivity客户端索引后，完整重跑写入v2新目录；未用失败数据做性能裁决。修复后源SHA与结果绑定。

品质计算：quality_evidence_demo_final_20261020.json提供30项恒温中心参数计算。番茄只算TA相对保留；鲻鱼只算感官评分下降，初值未填9，未算可售量或利润。CO2机组0°C/外30°C/80Hz/9MPa的输入功率由同工况1.75/1.66推导为1.0542169kW；连续运行8h为8.4337349kWh是参考计算，不是道路实测或循环占空比模型。

9项针对检查通过，包括12个实际解独立复核、重复漏客拒绝、容量拒绝、逆序时间窗拒绝、车数不同不报距离gap、源文件数、评分day→h和缺定价证据拒绝利润。

完整路线/时刻/源SHA/软件版本见../../results/static_delivery/classic_solomon_preliminary_v2_20261020/results.json。参数和缺项见../../configs/static_delivery/product_evidence_v1.json。
''',encoding='utf-8')
    (COMPARE/'方法适用性.md').write_text('# 静态冷链对比方法重新分类\n\n评估20项（原17项+新增ALNS/HGS/CaR）。判断依据是当前下载版本的原任务和接口；旧动态适配资格及旧成绩均不沿用。不按结果好坏剔除方法。所有方法的新净效益适配均未完成。\n\n| 方法 | 当前分类 | 理由 |\n|---|---|---|\n'+'\n'.join(f'| [{n}](https://github.com/{r}) | {s} | {why} |' for n,r,s,why in METHODS)+'\n\n优先次序：经典路由用OR-Tools/PyVRP；净效益主传统基线优先ALNS+共同独立评价器，小例再用显式数学规划校验；学习方法优先MaskCO/RouteFinder/DeepACO，CaR作为约束处理候选。仅将距离模型产生的路线事后计算净效益，应标为“距离策略迁移对照”，不能冒称同目标训练的方法。原始预训练、同架构随机和新目标训练结果必须区分。\n',encoding='utf-8')
    bib=[]
    for s in SOURCES:
        fields=[f'  title = {{{s["title"]}}}',f'  url = {{{s["url"]}}}']
        if s['doi']: fields.append(f'  doi = {{{s["doi"]}}}')
        if s['year']: fields.append(f'  year = {{{s["year"]}}}')
        bib.append('@misc{'+s['id']+',\n'+',\n'.join(fields)+'\n}')
    (BASE/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
    ledger={'strategy':'targeted question-specific primary evidence packet, not systematic review or full 60-reference manuscript packet',
            'skill':'research-lookup; Parallel CLI absent, existing web search used; no credentials queried or external CLI installed',
            'question_groups':['tomato nutrient/firmness kinetics vs acceptance vs hedonic value','chilled mullet sensory kinetics and species-specific fish WTP','refrigerated input power vs cooling capacity; matching operating conditions','Solomon original and author VRPTWQ units','native solver tasks vs new net-benefit objective'],
            'queries':['tomato quality price willingness to pay firmness shelf life kinetic model cold chain routing profit','fish freshness price sensory quality cold chain deterioration value routing','Solomon vehicle routing time windows benchmark SINTEF100 customers download','An Exact Algorithm for the Vehicle Routing Problem with Time Windows and Perishable Products','mullet freshness price quality hedonic','refrigerated vehicle experimental chilled4C electrical power consumption','CO2跨临界系统在冷藏车上的应用研究'],
            'source_count':len(SOURCES),'excluded':['2027-volume transmodal seafood pricing article, not used as a current numeric parameter source','secondary summaries not used for numeric parameter values'],
            'failures_preserved':'acquisition manifests record GitHub REST rate limit, publisher403, non-PDF challenge responses and missing PDFs; public git/codeload used for code pins',
            'not_done':['not all cited full texts acquired','no empirical commodity price-quality calibration found for the selected pair','no full cold-chain profit optimization','no new neural training or claimed method benefit']}
    write(BASE/'search_ledger.json',ledger)
    print(json.dumps({'sources':len(SOURCES),'methods':len(METHODS),'snapshots':len(pins),'all_snapshot_SHA_verified':all(p['all_member_sha256_verified'] for p in pins),'actual_routes_feasible':sum(r['report']['feasible'] for r in results['rows']),'datasets':[len(dataset['original_solomon']['rows']),len(dataset['author_vrptwq']['rows'])]},ensure_ascii=True))

if __name__=='__main__': main()
