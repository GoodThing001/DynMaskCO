"""Independent old-record tensor indexing/targets. No producer/binding/chain/physics import."""
import os,sys
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,hashlib,ctypes
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,psutil
from pickup_loaded_closure_v1 import capture,sha
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';O=D/'action_tensor_binding_root_v1';CHAIN=D/'directed_action_chain_root_v1/ACTION_CHAINS.jsonl';OLD=D/'family2_remaining_action_diagnosis_root_v1/OBSERVED_ACTION_RECORDS.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def own_hash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name']));capture();identities={str(p):sha(p) for p in list(O.iterdir())+[CHAIN,OLD,Path(__file__)] if p.is_file()};save('AUDIT_REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs=identities,scope='Independent exact int/dtype/order/mask/tie/raw9 float32 cast, temporary anchors and full hypothetical route subsequence teacher compatibility for every stored candidate. No new binding/producer/action chain/route_ok/matrices/C0/model or solver reference; old feasibility flags transferred, not recomputed.'))
    begin=capture(identities);save('AUDIT_SOURCE_START.json',begin);checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    observed={(x['instance_id'],x['step']):x for x in read(OLD)};arrays=np.load(O/'TENSORS.npz',allow_pickle=False);contexts=[json.loads(line) for line in (O/'CONTEXTS.jsonl').open(encoding='utf-8')];idx=offset=groups=0;rounding=0.;label_positive=0;eligible_positive=0
    for name,dtype in [('actions',np.int32),('node_features',np.float32),('eligible',np.bool_),('distance_tie',np.float64),('labels',np.float32),('scores',np.float32),('state_offsets',np.int64),('customer_offsets',np.int64)]:ck('arraydtype:'+name,arrays[name].dtype==dtype)
    # npz entries fetched once; no per-row repeated decompression/materialization.
    packed={k:arrays[k] for k in arrays.files};arrays.close()
    with CHAIN.open(encoding='utf-8') as stream:
        for line in stream:
            x=json.loads(line);req=observed[tuple(x['key'])];data=req['request']['data'];teacher=req['teacher_routes']
            for step,z in enumerate(x['trace']):
                rows=[r for c in z['customers'] for r in c['rows']];n=len(rows);sl=slice(offset,offset+n);identity=np.asarray([r[:5] for r in rows],dtype=np.int32);expected_features=[];expected_labels=[];anchors=True
                for i,v,p,a,b in identity.tolist():
                    route=z['before'][v];anchors=anchors and i in z['remaining'] and a==([0]+route)[p] and b==(route+[0])[p]
                    hypothetical=route[:p]+[i]+route[p:];target=teacher[v];cursor=0;compatible=True
                    for node in hypothetical:
                        while cursor<len(target) and target[cursor]!=node:cursor+=1
                        if cursor==len(target):compatible=False;break
                        cursor+=1
                    expected_labels.append(float(compatible))
                    expected_features.append([[data['coords'][j][0],data['coords'][j][1],data['demands'][j],data['temp_class'][j],data['reveal_time'][j],data['ready_time'][j],data['tw_start'][j],data['tw_end'][j],data['service_time'][j]] for j in (i,a,b)])
                raw=np.asarray(expected_features,dtype=np.float64);cast=raw.astype(np.float32);labels=np.asarray(expected_labels,dtype=np.float32);eligible=np.asarray([r[5] and r[6] for r in rows],dtype=np.bool_);ties=np.asarray([r[8] for r in rows],dtype=np.float64);rounding=max(rounding,float(np.max(np.abs(raw-cast.astype(np.float64)))));label_positive+=int(labels.sum());eligible_positive+=int((labels.astype(bool)&eligible).sum());c=contexts[idx]
                ck(str(idx)+':all_action_order_and_temporary_anchors',anchors and np.array_equal(packed['actions'][sl],identity))
                ck(str(idx)+':all_raw9_features_exact_registered_cast',np.array_equal(packed['node_features'][sl],cast))
                ck(str(idx)+':all_directed_slot_target_and_score',np.array_equal(packed['labels'][sl],labels) and np.array_equal(packed['scores'][sl],labels) and np.array_equal(labels,np.asarray([r[7] for r in rows],dtype=np.float32)))
                ck(str(idx)+':all_saved_feasibility_and_ties',np.array_equal(packed['eligible'][sl],eligible) and np.array_equal(packed['distance_tie'][sl],ties))
                ck(str(idx)+':context_action_identity',c['context_sha256']==own_hash(dict(key=x['key'],step=step,before=z['before'],remaining=z['remaining'],data=data)) and c['action_sha256']==own_hash(dict(actions=identity.tolist(),eligible=eligible.tolist(),distance_hex=[float(v).hex() for v in ties])) and c['offset']==offset and c['size']==n and c['source']=='observed_ideal_fixture' and not c['learned_output_qualified'])
                ck(str(idx)+':state_offsets',packed['state_offsets'][idx]==offset and packed['state_offsets'][idx+1]==offset+n)
                for customer in z['customers']:
                    ck(str(idx)+':customer_group:'+str(customer['customer']),packed['customer_offsets'][groups]==offset and packed['customer_offsets'][groups+1]==offset+len(customer['rows']))
                    offset+=len(customer['rows']);groups+=1
                idx+=1
    ck('all750_370500_9750_preserved',idx==len(contexts)==750 and offset==370500 and groups==9750 and len(packed['actions'])==offset and len(packed['state_offsets'])==751 and len(packed['customer_offsets'])==9751)
    b=read(O/'BOUNDARY_REJECTIONS.json');ck('nine_actual_consumer_boundary_rejections',b['count']==9 and len(set(b['names']))==9 and b['numeric_fixture_not_actual_model'])
    result=read(O/'RESULT.json');ck('no_model_physics_budget_science_upgrade',not any(result[k] for k in ('learned_output_qualified','full_C0_qualified','hard_request_budget_qualified','new_scientific_family','stable_online_gain')))
    for p,h in read(O/'REGISTRATION.json')['inputs'].items():ck('registered:'+p,sha(Path(p))==h)
    for p,h in identities.items():ck('audit_input:'+p,sha(Path(p))==h)
    end=capture(identities);ck('audit_loaded_closure',begin['files']==end['files'] and begin['module_files']==end['module_files']);save('AUDIT_SOURCE_END.json',end)
    for p,h in begin['files'].items():ck('audit_loaded:'+p,sha(Path(p))==h)
    for p,h in read(O/'SOURCE_START.json')['files'].items():ck('production_actual_loaded:'+p,sha(Path(p))==h)
    final=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),states=idx,actions=offset,customer_groups=groups,positive_targets=label_positive,eligible_positive_targets=eligible_positive,max_float64_to_float32_feature_rounding=rounding,actual_loaded_files=len(begin['files']),module_maps=len(begin['module_files']),independent_no_new_binding_reference=True,saved_feasibility_flags_not_recomputed=True,learned_output_qualified=False,new_scientific_family=False,stable_online_gain=False);save('INDEPENDENT_ACTION_TENSOR_BINDING.json',final);print(json.dumps({k:v for k,v in final.items() if k!='checks'}));return int(final['failed']>0)
if __name__=='__main__':raise SystemExit(main())
