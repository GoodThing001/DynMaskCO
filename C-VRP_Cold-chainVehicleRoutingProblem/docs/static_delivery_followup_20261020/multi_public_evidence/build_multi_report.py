"""Record source-backed candidate decisions without constructing customer instances."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
EXT=HERE.parents[2]
downloads=json.loads((HERE/'download_manifest.json').read_text(encoding='utf-8'))
for item in downloads:
 if item.get('downloaded'):
  assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256']
archive=zipfile.ZipFile(HERE/'vrprep_christofides_1979_set_m.zip')
members=[]
for name in archive.namelist():
 raw=archive.read(name)
 entry={'member':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if name.endswith('.xml'):
  root=ET.fromstring(raw)
  entry.update(nodes=len(root.findall('network/nodes/node')),customers=len(root.findall('requests/request')),capacity_native=float(root.findtext('fleet/vehicle_profile/capacity')),quantity_units='not declared as physical units in the XML',coordinate_units='not declared as km in the XML',decimals=root.findtext('network/decimals'))
 members.append(entry)
 if name in ('M-n101-k10.xml','readme.txt'):
  target=HERE/('vrprep_'+name)
  if target.exists(): assert target.read_bytes()==raw
  else: target.write_bytes(raw)
rrp=[]
for path in sorted((EXT/'data/public_sources_20261020/refrigerated_routing_rrp/Datasets').rglob('*.json')):
 raw=path.read_bytes();d=json.loads(raw)
 rrp.append({'file':str(path.relative_to(EXT)).replace('\\','/'),'sha256':hashlib.sha256(raw).hexdigest(),'sum_demands_pallets':sum(d['demands'])})
assert len(rrp)==30 and max(x['sum_demands_pallets'] for x in rrp)==33
candidates=[
 {'id':'original_rrp_split_extension','title':'Ceschia et al. 2020 RRP, unchanged published orders with an explicit multi-vehicle method extension','doi':'10.3390/en13236214','source':'https://bitbucket.org/sceschia/the-refrigerated-routing-problem/','source_commit':'f5c5317a5ecd8fe64df779e79bd0397904bc0f9e','native_multi_vehicle':False,'data_status':'existing 30 original JSON instances and 30 original reference TXT files; no generated orders','demand_unit':'pallet','capacity_pallets':33,'decision':'recommended_first_extension','caveat':'Every original total demand is at most one truck capacity; maximum fleet size is an experiment setting and capacity does not force multiple routes. No fixed charge is sourced or added.'},
 {'id':'chen_2023','title':'Green vehicle routing using mixed fleets for cold chain distribution','doi':'10.1016/j.eswa.2023.120979','source':'https://pure.tue.nl/ws/portalfiles/portal/309283747/1-s2.0-S0957417423014811-main.pdf','native_multi_vehicle':True,'data_status':'article accessible, original empirical customer data not shareable by authors','demand_unit':'nonnegative quantity; per-customer original records unavailable','vehicle_capacity':{'EV':1.22,'GDV':1.29,'unit':'tons','locator':'Table 4, PDF page 9 / printed page 8'},'source_locator':'Section 5.1 PDF page 8; Data availability PDF page 15 / printed page 14','decision':'not_an_open_order_dataset','caveat':'Explicit statement that authors do not have permission to share data. Do not recreate the Shanghai customer instance.'},
 {'id':'thammasat_opensolver','title':'Optimization of vehicle routing with time windows for cold chain logistics using OpenSolver','source':'https://ethesisarchive.library.tu.ac.th/thesis/2024/TU_2024_6722040620_20924_30485.pdf','native_multi_vehicle':True,'data_status':'PDF accessible through web extraction; local original download failed TLS; no full original matrix or workbook obtained','demand_unit':'pallet; table 3.3','capacity_unit':'pallet; section 3.3.2','capacity_numeric_value':None,'source_locator':'Section 3.2.1 PDF page 18 / printed page 7; Tables 3.2-3.4 PDF pages 19-20; Appendix A PDF pages 44-45','decision':'incomplete_candidate_do_not_enable','caveat':'Original is explicitly an academic synthetic dataset. Place names and plotted locations do not justify inventing coordinates. Appendix describes generating distance/time matrices with OSRM; no generator run and no new matrix built.'},
 {'id':'escobar_2022','title':'A granular tabu search for the refrigerated vehicle routing problem with homogeneous fleet','doi':'10.5267/j.ijiec.2021.6.001','source':'https://www.growingscience.com/ijiec/Vol13/IJIEC_2021_16.pdf','native_multi_vehicle':True,'data_status':'publisher PDF obtained; adapted classical CVRP instances, not original commercial cold-chain orders','demand_unit':'physical mass/pallet mapping not established','capacity_example':{'instance':'M-n101-k10','Q':200,'K':10,'unit':'benchmark native capacity units; no physical unit in table 3'},'source_locator':'Section 4.2 printed page 145; Tables 3 and 4 printed page 146 / PDF page 12; equation 9 printed page 141','decision':'method_precedent_only_until_unit_and_adaptation_gaps_close','caveat':'Table 4 diesel calorific field prints W/kg, while equation 9 defines its fuel conversion as J/gal. Truck engine prints 146.7 cc. Preserve these observations; do not silently correct units or transfer numbers to RRP.'},
 {'id':'vrprep_set_m','title':'Christofides et al. 1979 Set M in VRP-REP','source':'https://www.vrp-rep.org/datasets/item/2014-0003.html','raw_file':'https://www.vrp-rep.org/datasets/download/christofides-et-al-1979-set-m.zip','native_multi_vehicle':True,'native_cold_chain':False,'data_status':'original public repository archive obtained with five XML instances and readme','demand_unit':'native quantity units; physical identity unspecified','capacity_native':200,'source_locator':'XML fleet/vehicle_profile/capacity and requests/request/quantity; archive readme','decision':'eligible_for_original_CVRP_routing_checks_only','caveat':'No 600 kg/pallet conversion, no demand scaling to 33 pallets, and no coordinate-to-km assumption. Repository readme explicitly cautions about rounding ambiguity.'},
]
methods=[
 {'title':'Prins 2004 giant-tour Split','doi':'10.1016/S0305-0548(03)00158-8','source':'https://www.sciencedirect.com/science/article/pii/S0305054803001588','review_status':'publisher abstract and scope; no local full text','support':'Partition a fixed customer permutation into routes; optimality is subject to that sequence.'},
 {'title':'Vidal 2016 Split technical note','doi':'10.1016/j.cor.2015.11.012','source':'https://arxiv.org/abs/1508.02759','paper_file':'vidal_split_1508.02759.pdf','code':'https://github.com/vidalt/Split-Library','support':'CVRP linear Split and O(nm) limited-fleet extension require the proved cost structure; generic Bellman limited-fleet labels remain applicable to separately evaluated feasible segment costs.','rrp_boundary':'Time-varying climate, traction load and continuous waiting do not inherit the CVRP linear-time property automatically. Exact decoding applies only to the fixed order and supplied segment costs, not all permutations or continuous schedules.'},
]
result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_count':len(candidates),'candidates':candidates,'methods':methods,'downloads':downloads,'zip_member_provenance':members,'rrp_instance_demand_audit':rrp,'generated_customer_instances':False,'author_generators_run':False,'primary_code_modified':False,'recommendation':'Use unchanged RRP orders, source 33-pallet homogeneous vehicles, original RRP physics, explicit maximum fleet size; call it a multi-vehicle extension, not an author native multi-vehicle benchmark reproduction.','remaining_gaps':['No verified complete native multi-vehicle public cold-chain dataset with original customer files, explicit demand/capacity physical mapping and full source physics was obtained among these bounded candidates.','Thai raw PDF download failed with TLS UNEXPECTED_EOF_WHILE_READING; web text was reviewed, capacity numeric value and full matrix/workbook remain unverified.','Escobar adapted raw cold-chain instance package was not obtained and printed parameter dimensions need resolution.','No missing D1-n9 reference wait is resolved by this research.'],'backend_note':'parallel-cli absent; used available web search/open and direct named-source downloads. No CLI installed and no credentials queried.'}
result['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.is_file()}
(HERE.parent/'multi_public_data.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'candidates':len(candidates),'original_rrp_instances_verified':len(rrp),'downloads_saved':sum(bool(x.get('downloaded')) for x in downloads),'xml_members_verified':len([x for x in members if x['member'].endswith('.xml')])}))
