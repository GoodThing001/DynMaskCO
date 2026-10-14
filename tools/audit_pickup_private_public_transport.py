"""Offline independent field projection check; never calls public export/transport."""
import hashlib,json,sys,math
from pathlib import Path
from dataclasses import is_dataclass
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'private_public_transport_root_v1'
for folder in ('evaluation','simulation','coldchain','correction_engineering'):sys.path.insert(0,str(E/'scripts'/folder))
import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
from audit_io import decode
from recourse_snapshot import snapshot_state_hash
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa) for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def dec(x):return decode(x,CLASSES)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 reg=json.loads((OUT/'REGISTRATION.json').read_text(encoding='utf-8'));result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'))
 raw=P/'complete_history_energy_qa_root_v1/complete_ledger/FULLRAW.json';trace=P/'complete_history_energy_qa_root_v1/complete_ledger/cond_hist_trace.jsonl'
 bound={str(p):sha(p) for p in (OUT/'REGISTRATION.json',OUT/'RESULT.json',raw,trace,Path(__file__))}
 with trace.open(encoding='utf-8') as f:private=dec(json.loads(next(f))['payload'])['dataset']
 original=dec(json.loads(raw.read_text(encoding='utf-8')))['admissions'];checks=[]
 def ck(n,v):checks.append({'name':n,'pass':bool(v)})
 ck('original_registered_sources',sha(raw)==reg['fixture_sha256'] and sha(trace)==reg['private_trace_sha256'])
 ck('producer_48_pass',result['checks_count']==48 and result['failed']==0)
 ck('numpy_identity',np.__version__==reg['numpy'])
 for record in result['records']:
  i,v=record['index'],record['variant'];prefix=str((i,v));packet=dec(record['packet']);old=original[i]['public_request'];snap=packet['snapshot'];clock=old['snapshot']['clock']
  known=[o for o in range(1,private['demands'].shape[1]) if private['reveal_time'][0,o]<=clock+1e-6];n=max(known,default=0)+1
  ck(prefix+':known_exact',list(snap['customer_universe'])==known)
  ck(prefix+':width_from_observed_ID',snap['num_nodes']==n)
  for field,value in packet['dataset'].items():
   source=private[field];expected=np.zeros((1,n)+source.shape[2:],dtype=source.dtype);expected[0,0]=source[0,0]
   for order in known:expected[0,order]=source[0,order]
   ck(prefix+':field:'+field,np.array_equal(value,expected,equal_nan=True) and value.dtype==source.dtype)
  expected_visible=np.zeros(n,bool);expected_visible[[0]+known]=True
  ck(prefix+':visibility',np.array_equal(snap['visible_mask'],expected_visible))
  ck(prefix+':served_slice',np.array_equal(snap['served_mask'],old['snapshot']['served_mask'][:n]))
  ck(prefix+':snapshot_hash',snap['state_hash']==snapshot_state_hash(snap))
  for field in ('clock','event_id','reveal_idx','replan_count','num_vehicles','replanner_state'):
   ck(prefix+':state:'+field,snap[field]==old['snapshot'][field])
  for field in ('vehicle_node','vehicle_ready','vehicle_load','committed_next','committed_arrive','committed_finish','coldchain_updated_time'):
   ck(prefix+':physical:'+field,np.array_equal(snap[field],old['snapshot'][field],equal_nan=True))
  t=record['transport'];ck(prefix+':component_timing',t['started']<t['finished']<t['deadline'])
  for path,h in record['source_start'].items():ck(prefix+':actual_source:'+path,sha(path)==h==record['source_end'][path])
 ck('six_unique_records',{(r['index'],r['variant']) for r in result['records']}=={(i,v) for i in (0,9,13) for v in (0,1)})
 for fault in result['faults']:
  t=fault['response'];ck('fault:'+str(fault['fault'])+':stage_evidence',fault['target'] in t['stages'])
  ck('fault:'+str(fault['fault'])+':no_complete_payload',t['kind']=='cancelled' and 'bytes' not in t and t['worker_reaped'])
 for p,h in bound.items():ck('unchanged:'+p,sha(p)==h)
 out={'checks':checks,'checks_count':len(checks),'failed':sum(not c['pass'] for c in checks),'input_hashes':bound,
      'normal_walls':[r['transport']['finished']-r['transport']['started'] for r in result['records']],
      'fault_walls':[f['response']['finished']-f['response']['started'] for f in result['faults']],
      'cancel_reap_may_finish_after_component_deadline':True,'independent_public_fields_only':True,
      'not_full_physical_replay':True,'restored_fixture_not_new_live_day':True,'whole_live_request_qualified':False,'stable_online_gain':False}
 with (OUT/'INDEPENDENT_PUBLIC_FIELD_ACCEPTANCE.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps({'checks':out['checks_count'],'failed':out['failed'],'normal_walls':out['normal_walls'],'fault_walls':out['fault_walls']}))
 return int(out['failed']>0)
if __name__=='__main__':raise SystemExit(main())
