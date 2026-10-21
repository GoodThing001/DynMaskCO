"""Save source-quoted case tables and check research provenance; no generated instances."""
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT

ROOT = EXTENSION_ROOT
DOC = ROOT/'docs/static_delivery_research_20261020/dataset_product_review'
CASE = ROOT/'data/published_cases_20261020/QDVRP_2021'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    source = ROOT/'docs/static_delivery_research_20261020/sources/quality_driven_2021.pdf'
    dist = [[0,91,6,134,82,117,74,192], [91,0,91,75,134,43,20,155], [6,91,0,134,84,117,75,194], [134,75,134,0,168,46,94,90], [82,134,84,168,0,158,140,180], [117,43,117,46,158,0,59,117], [74,20,75,94,140,59,0,174], [192,155,194,90,180,117,174,0]]
    demand = [[432,432,216], [2160,648,648], [648,216,432], [792,403,432], [1080,432,648], [979,86,259], [864,432,432]]
    speeds = [[None,45.9,50.7,51.4,63.4,65.9,50.5,46.7], [59.1,None,42.9,48.1,60.2,65.3,44.5,46.9], [64.2,60.3,None,60.2,66.4,63.6,59.1,50.5], [61.0,64.7,44.2,None,67.9,58.9,51.3,52.3], [41.8,49.7,50.0,50.4,None,65.5,48.2,48.1], [50.7,52.1,46.8,47.8,42.3,None,42.5,50.2], [62.5,63.1,44.9,65.3,65.2,61.1,None,50.7], [62.8,56.8,59.7,59.8,59.3,59.2,60.9,None]]
    assert len(dist)==len(speeds)==8 and all(len(x)==8 for x in dist+speeds)
    assert all(dist[i][j]==dist[j][i] for i in range(8) for j in range(8))
    assert all(dist[i][i]==0 and speeds[i][i] is None for i in range(8))
    assert len(demand)==7 and all(len(x)==3 and all(y>0 for y in x) for x in demand)
    assert speeds[0][1]!=speeds[1][0]  # original directed travel speed, not symmetrized
    case = {
        'status': 'PUBLISHED_TABLE_TRANSCRIPTION_NOT_SYNTHETIC_CUSTOMER_INSTANCE_NOT_FULL_REPLICATION',
        'source_doi': '10.1016/j.ijpe.2020.107849',
        'source_url': 'https://transsonic.nl/documents/Stellingwerf_EtAl_IJPE_2021_The_Quality_Driven_Vehicle_Routing_Problem.pdf',
        'source_pdf_sha256': sha(source),
        'source_locations': {'distance':'Table2 p6', 'demand_and_kinetics':'Table3 p6', 'speed':'Table14 p11', 'base_assumptions':'section4.1 p7'},
        'node_ids':[0,1,2,3,4,5,6,7], 'depot_id':0,
        'customer_ids':[1,2,3,4,5,6,7],
        'product_ids':['Product1','Product2','Product3'],
        'product_identity':'fresh cut vegetables; table does not assign named species/quality attribute to each numbered product; do not rename as tomato or fish',
        'distance_km':dist, 'demand_kg_by_customer_then_product':demand,
        'speed_km_per_h_directed':speeds, 'speed_diagonal': 'not printed / not used; null, not fabricated zero-speed observation',
        'source_kinetics': {'source_symbol':'k0 in reference-temperature-corrected Arrhenius Eq2,8,9; NOT preexponential in an unshifted Arrhenius equation', 'reference_rate_per_s':[3.08e-6,3.84e-6,1.96e-6], 'Ea_J_per_mol':[77900,76304,88521], 'reference_temperature_K_as_paper':275, 'reaction_order_as_paper':0},
        'source_base_assumptions': {'setpoint_K':275, 'ambient_K':293, 'unloading_s_per_kg':0.8, 'cooling_s_per_kg':0.4, 'available_vehicles':3, 'curb_weight_kg':10000, 'payload_capacity_kg':30000},
        'not_filled': {'coordinates':None, 'customer_time_windows':None, 'sales_prices':None, 'purchase_prices':None, 'empirical_quality_value_mapping':None},
        'limits': ['case study, not classic multi-instance benchmark', 'source uses rounded Kelvin conventions; do not silently replace275K by275.15K in a claimed reproduction', 'kinetic parameter product/attribute identity and complete initial-quality/cost constants require follow-up; no full food model or profit claimed'],
        'customers_generated':False, 'route_or_profit_computed':False,
    }
    save(CASE/'tables_2_3_14.json',case)
    examples = {
      'status':'PUBLISHED_NUMERICAL_SETTINGS_FOR_PROTOCOL_DESIGN_NOT_ACTIVE_EXPERIMENT_CONFIG',
      'zhu_2021': {'doi':'10.3390/su13148110', 'location':'section5.1 p13-14, Table2 p14', 'product':'generic fresh agricultural products; not a calibrated named species', 'dataset':'R108 first50 customers; demand and capacity multiplied by10 per author', 'price_yuan_per_kg':12, 'eta1_as_printed':0.005, 'eta2_as_printed':0.01, 'eta_time_unit_printed_in_table':None, 'speed_km_per_h':40, 'fixed_vehicle_yuan':200, 'fuel_price_yuan_per_L':5.41, 'empty_fuel_L_per_km':0.18, 'full_fuel_L_per_km':0.41, 'moving_refrigeration_yuan_per_h':5, 'unloading_refrigeration_yuan_per_h':12, 'scope':'author numerical setting, not measured generic-food physics or current market prices'},
      'el_raoui_2021': {'doi':'10.3390/electronics10162018', 'location':'section7.1 Table4 p14; cost Eq1-8 p6-7', 'product':'generic perishable food', 'dataset':'100-customer adaptedRC101 application; section6 usesR101,C104,RC107 subsets', 'price_EUR_per_abstract_unit':20, 'delta1_as_printed':0.002, 'delta2_as_printed':0.003, 'refrigeration_closed_EUR_per_unit_as_printed':0.03, 'refrigeration_open_EUR_per_unit_as_printed':0.04, 'fixed_vehicle_EUR':25, 'distance_cost_EUR_per_km_as_paper':3, 'time_and_refrigeration_unit_issue':'Table4 says EUR/unit, not explicitly EUR/min or kWh tariff; Eq3-4 require cost per time. Any benchmark-time interpretation must be declared, not reported as measured power.', 'not_adopted_as_quality_threshold':'alpha0.8 in Table4 is NOT established as a food-quality rejection threshold by this review', 'comparison_scope_issue':'section6.2 Eq35 CPLEX/GVNS comparison omits damage costC4; cannot use its results as a full quality-value baseline'},
      'gharehyakheh_2020': {'doi':'10.3390/su12166668', 'location':'section5.1 Table3 p15', 'product':'generic perishable food; shelf-life sensitivity study', 'dataset':'Solomon25 and100customer instances', 'shelf_life_h':2880, 'vehicle_speed_km_per_h':15, 'fixed_vehicle_dollars':1000, 'distance_cost_dollars_per_km':1.5, 'capacity_kg_as_printed':200, 'scope':'do not identify this120-day numerical shelf life as chilled mullet or ready-to-eat salad shelf life'},
      'zhao_2020': {'doi':'10.1371/journal.pone.0235950', 'location':'section4.1 and Table2', 'product':'fresh milk numerical application; Solomon data, not observed enterprise order logs', 'dataset':'C,R,RC;100customers,1depot; demand unit10kg;200units=2000kgcapacity', 'horizon_min':1020, 'fixed_vehicle_yuan':150, 'fuel_yuan_per_L':6, 'baseline_setpoint_C':1, 'baseline_refrigeration_L_per_h':2.4, 'baseline_refrigeration_context':'paper section2.5 numerical/experimental-simulation statement at outside25C and inside10C; not a verified road-duty dataset or current truck measurement', 'scope':'paper microbial zero-growth below0C is a model assumption; not universal milk safety evidence'},
      'normalization': {'identity':'divide every monetary term by the same positive monetary reference; customers, costs, loss and revenue must share source identity and unit convention', 'example_source':'el_raoui_2021', 'reference_EUR':20, 'price_value_units_per_demand_unit':1, 'fixed_vehicle_value_units':1.25, 'distance_coefficient_value_units_per_paper_distance_unit':0.15, 'closed_time_coefficient_value_units_per_paper_time_unit':0.0015, 'open_time_coefficient_value_units_per_paper_time_unit':0.002, 'physical_unit_mapping_filled':False, 'is_new_empirical_calibration':False},
      'objective_deduction': {'constant_reference_value':'B=P*sum(q_i), if every original order is fixed and mandatory', 'net_benefit_reference_loss_accounting':'Pi=B-C_transport-C_refrigeration-C_incremental_fixed-L_value', 'equivalence':'maxPi is equivalent to minimizing the sum of those costs including value loss; if L_value=0 and sale revenue is constant, it reduces to ordinary cost minimization', 'not_double_count':'do not subtract both an identical revenue markdown and a duplicate deterioration loss', 'real_food_profit_mapping_ready':False},
      'new_routing_or_training':False
    }
    save(DOC/'parameter_examples.json',examples)
    downloaded = []
    for file in ['acquisition_manifest.json','additional_acquisition_manifest.json']:
        for row in json.loads((DOC/file).read_text(encoding='utf-8')):
            if row['status']=='DOWNLOADED':
                p=ROOT/row['path']; assert sha(p)==row['sha256']; downloaded.append({'path':row['path'],'sha256':sha(p)})
    xml=ET.parse(DOC/'sources/plos_milk_2020.xml')
    title=' '.join(xml.find('.//article-title').itertext())
    assert 'Optimization of transportation routing' in title
    table=next(x for x in xml.findall('.//table-wrap') if x.attrib.get('id')=='pone.0235950.t002')
    table_text=' '.join(table.itertext())
    assert '150' in table_text and '6' in table_text
    assert [sum(row[p] for row in demand) for p in range(3)]==[6955,2649,3067]
    assert sum(sum(x) for x in demand)==12671
    normal=examples['normalization']
    assert normal['fixed_vehicle_value_units']==25/20 and normal['distance_coefficient_value_units_per_paper_distance_unit']==3/20
    images=[]
    for name in ['quality_driven_2021_p6.png','quality_driven_2021_p11.png','freshness_investment_2021_p14.png','customer_centric_2021_p12.png','customer_centric_2021_p14.png','temperature_shelf_life_2020_p15.png']:
        p=DOC/'page_checks'/name; assert p.is_file(); images.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    preserved = {'configs/static_delivery/product_evidence_v1.json':'3dd8c10768781145fe3a084dbce2c2e80cb7e13351de89d17ef6cab646b53947','results/static_delivery/net_benefit_scenario_pilot_20261020/results.json':'02f3b2ee2f67547501afed1ef3f8514cfe567992062294cd676d94830d39391b'}
    for p,h in preserved.items(): assert sha(ROOT/p)==h,p
    matrix=json.loads((DOC/'evidence_matrix.json').read_text(encoding='utf-8'))
    studies=matrix['studies']
    assert len(studies)==11 and len({s['doi'].lower() for s in studies})==11
    bib=[]
    for s in studies:
        bib.append('@article{'+s['id']+',\n  author = {'+s['authors'].replace(';',' and')+'},\n  title = {'+s['title']+'},\n  year = {'+str(s['year'])+'},\n  doi = {'+s['doi']+'},\n  url = {'+s['primary_url']+'}\n}')
    (DOC/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
    local_links=0
    for p in [DOC/'README.md',CASE/'README.md',ROOT/'项目当前状态.md',ROOT/'docs/文档索引与版本状态.md',ROOT/'docs/static_delivery_research_20261020/README.md',ROOT/'docs/static_delivery_research_20261020/导师汇报摘要.md']:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'): continue
            resolved=(p.parent/target.split('#')[0]).resolve()
            assert resolved.exists() or resolved==(DOC/'verification.json').resolve(),(str(p),target)
            local_links+=1
    diff=subprocess.run(['git','diff','--numstat','HEAD','--','MASKCO_code','CC_Compare'],cwd=ROOT.parent,capture_output=True,text=True,check=True)
    assert not diff.stdout.strip()
    save(DOC/'verification.json', {'status':'PUBLISHED_TABLES_AND_PROVENANCE_CHECKED_NOT_NEW_PROFIT_EXPERIMENT','routing_primary_studies':len(studies),'evidence_matrix_sha256':sha(DOC/'evidence_matrix.json'),'new_downloads_SHA_verified':downloaded,'visually_reviewed_original_pages':images,'published_case':{'nodes':8,'customers':7,'products':3,'demand_total_kg':sum(sum(x) for x in demand),'product_totals_kg':[6955,2649,3067],'distance_table_symmetric':True,'speed_table_asymmetric':True,'table_file_sha256':sha(CASE/'tables_2_3_14.json')},'parameter_examples_sha256':sha(DOC/'parameter_examples.json'),'local_links_checked':local_links,'legacy_upstream_diff_empty':True,'old_inputs_and_results_SHA_unchanged':preserved,'new_instances_generated':False,'new_routing_or_training':False,'real_food_profit_mapping_ready':False})
    print(json.dumps({'routing_primary_studies':len(studies),'downloaded_originals_checked':len(downloaded),'reviewed_pages':len(images),'local_links_checked':local_links,'published_case_total_demand_kg':sum(sum(x) for x in demand),'profit_or_routes_computed':False}))


if __name__ == '__main__':
    main()
