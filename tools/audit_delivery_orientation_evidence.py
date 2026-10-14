"""Independent raw-event arithmetic for four observed reversal pairs; no solver."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010'
BASE=R/'delivery/family2_orientation_action_diagnosis_root_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 scope=json.loads((BASE/'ENGINEERING_SCOPE.json').read_text(encoding='utf-8'))
 result=json.loads((BASE/'RESULT.json').read_text(encoding='utf-8'))
 pairs=json.loads((BASE/'OBSERVED_PAIRS.json').read_text(encoding='utf-8'))
 bound={str(p):sha(p) for p in [BASE/'ENGINEERING_SCOPE.json',BASE/'RESULT.json',BASE/'OBSERVED_PAIRS.json',Path(__file__)]}
 checks=[]
 def ck(n,v):checks.append({'name':n,'pass':bool(v)})
 ck('producer_pass',result['failed']==0 and result['checks_count']==83)
 ck('four_unique_selected_keys',len(pairs)==4 and {(p['instance_id'],p['step']) for p in pairs}=={(18,3),(29,2),(51,1),(59,1)})
 for path,h in scope['inputs'].items():ck('original_identity:'+path,sha(path)==h)
 summaries=[]
 for row in pairs:
  key=str((row['instance_id'],row['step']));t=row['teacher'];o=row['oracle'];changes=[]
  ck(key+':same_order_vehicle_ownership',all(set(a)==set(b) for a,b in zip(t['routes'],o['routes'])))
  for v,(a,b) in enumerate(zip(t['routes'],o['routes'])):
   if a!=b:changes.append(v);ck(key+':reversed:'+str(v),a[::-1]==b)
  ck(key+':exactly_one_direction_change',len(changes)==1)
  totals={}
  for label,block in [('teacher',t),('oracle',o)]:
   report=block['report'];events=[e for trace in report['traces'] for e in trace['events']]
   distance=math.fsum(e.get('distance_km',0.) for e in events)
   energy=math.fsum(e.get('energy_kwh',0.)+e.get('preconditioning_energy_kwh',0.)+e.get('door_recovery_kwh',0.) for e in events)
   quality=math.fsum(d['quality_loss_value'] for d in report['delivered'])
   totals[label]={'distance':distance,'energy':energy,'quality':quality,'cost':distance+.1*energy+quality}
   for term,value in [('distance_km',distance),('energy_kwh',energy),('quality_loss_value',quality),('cost',totals[label]['cost'])]:
    ck(key+':'+label+':event_sum:'+term,abs(value-report[term])<1e-12)
   for trace in report['traces']:
    v=trace['vehicle'];sequence=[e['order'] for e in trace['events'] if e['kind']=='deliver']
    ck(key+':'+label+':execution_order:'+str(v),sequence==block['routes'][v])
  delta={k:totals['oracle'][k]-totals['teacher'][k] for k in totals['teacher']}
  ck(key+':distance_same',abs(delta['distance'])<1e-12)
  ck(key+':nonzero_coldchain_cost',abs(delta['cost'])>1e-8)
  summaries.append({'key':[row['instance_id'],row['step']],'changed_vehicle':changes[0],'event_arithmetic_oracle_minus_teacher':delta})
 for p,h in bound.items():ck('unchanged:'+p,sha(p)==h)
 out={'checks':checks,'checks_count':len(checks),'failed':sum(not c['pass'] for c in checks),'bound_files':bound,
      'summaries':summaries,'new_physical_replay':False,'new_scientific_family':False,'learned_decoder_qualified':False,
      'registered_family2_still_FAIL':True,'stable_online_gain':False,'interpretation':'four fixed aliases isolated to same-slot route reversal; record arithmetic only'}
 with (BASE/'INDEPENDENT_EVENT_ARITHMETIC.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps({'checks':out['checks_count'],'failed':out['failed']}))
 return int(out['failed']>0)
if __name__=='__main__':raise SystemExit(main())
