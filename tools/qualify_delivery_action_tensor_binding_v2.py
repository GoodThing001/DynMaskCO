"""Register then encode only saved old chains; no solver/reconstruction/model call."""
import os,sys
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,hashlib,ctypes,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,psutil,io,zipfile
from delivery_action_tensor_binding_v1 import batch,targets,score_packet,consume,digest
from pickup_loaded_closure_v1 import capture,sha
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';D=R/'delivery';O=D/'action_tensor_binding_root_v2';CHAIN=D/'directed_action_chain_root_v1/ACTION_CHAINS.jsonl';OLD=D/'family2_remaining_action_diagnosis_root_v1/OBSERVED_ACTION_RECORDS.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
old_result=D/'action_tensor_binding_root_v1/RESULT.json';old_begin=D/'action_tensor_binding_root_v1/SOURCE_START.json';old_end=D/'action_tensor_binding_root_v1/SOURCE_END.json'
def main():
    if sys.argv[1:]==['--register']:
        O.mkdir(exist_ok=False);inputs=[ROOT/'tools/recover_delivery_action_tensor_binding_v2.py',old_result,old_begin,old_end,CHAIN,OLD,D/'directed_action_chain_root_v1/REGISTRATION.json',Path(__file__),ROOT/'tools/delivery_action_tensor_binding_v1.py',ROOT/'tools/audit_delivery_action_tensor_binding_v2.py',ROOT/'tools/pickup_loaded_closure_v1.py',Path(sys.executable)]
        save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs={str(p):sha(p) for p in inputs},scope='All750 saved temporary states/all370500 saved action rows only. Encode int32 customer/vehicle/position/predecessor/successor, float32 raw9 observed node fields for customer/predecessor/successor, bool joint retained+route eligibility, float64 saved distance tie, float32 teacher slot/directed-order compatibility targets in separate label function. SHA binds request key/step/entire temporary skeleton/remaining/data and action order/mask/tiehex. Consume finite float32 score packet only if source/context/order match; source remains observed ideal or unqualified numeric fixture. Reject wrong anchors/context/order/shape/nonfinite/claim learned eligibility. No original or new action-chain/route_ok/C0/solver/model/gradient/optimizer calls. No new science, no control qualification, no throughput/full10s claim.',expected_states=750,expected_actions=370500,node_features=9,integer_dtype='int32',feature_score_label_dtype='float32',distance_tie_dtype='float64',learned_output_qualified=False,new_scientific_family=False,stable_online_gain=False));print('{"registered":true}');return 0
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name']));warm=io.BytesIO();np.savez(warm,sample=np.zeros((1,3,9),dtype=np.float32));warm.seek(0);sample=np.load(warm,allow_pickle=False);sample['sample'];sample.close();capture();reg=read(O/'REGISTRATION.json');assert all(sha(Path(p))==h for p,h in reg['inputs'].items());observed={(x['instance_id'],x['step']):x for x in read(OLD)}
    begin=capture(reg['inputs']);save('SOURCE_START.json',begin);checks=[];arrays={k:[] for k in ('actions','node_features','eligible','distance_tie','labels','scores')};offset=0;states=0;boundaries=[0];group_boundaries=[0];cases=0
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    with (O/'CONTEXTS.jsonl').open('x',encoding='utf-8') as stream,CHAIN.open(encoding='utf-8') as src:
        for line in src:
            x=json.loads(line);req=observed[tuple(x['key'])];cases+=1
            for step,z in enumerate(x['trace']):
                rows=[r for c in z['customers'] for r in c['rows']];b=batch(req['request']['data'],x['key'],step,z['before'],z['remaining'],rows);y=targets(z['before'],b['actions'],req['teacher_routes']);packet=score_packet(b,y,'observed_ideal_fixture');values=consume(b,packet)
                ck(str(states)+':saved_labels_exact',np.array_equal(y,np.asarray([r[7] for r in rows],dtype=np.float32)))
                ck(str(states)+':score_vector_exact',np.array_equal(values,y))
                ck(str(states)+':dtypes_shapes',b['actions'].dtype==np.int32 and b['node_features'].shape==(len(rows),3,9) and b['node_features'].dtype==np.float32 and b['eligible'].dtype==np.bool_ and b['distance_tie'].dtype==np.float64)
                for k in arrays:arrays[k].append(y.copy() if k=='labels' else values if k=='scores' else b[k])
                sizes=[len(c['rows']) for c in z['customers']]
                for size in sizes:group_boundaries.append(group_boundaries[-1]+size)
                stream.write(json.dumps(dict(key=x['key'],step=step,offset=offset,size=len(rows),group_sizes=sizes,context_sha256=b['context_sha256'],action_sha256=b['action_sha256'],source='observed_ideal_fixture',learned_output_qualified=False))+'\n');offset+=len(rows);boundaries.append(offset);states+=1
                if states==1:
                    faults=[]
                    def reject(name,f):
                        try:f()
                        except (ValueError,IndexError):faults.append(name);return
                        raise AssertionError('boundary not rejected:'+name)
                    bad=[list(r) for r in rows];bad[0][3]=50 if bad[0][3]!=50 else 49
                    reject('wrong_temporary_predecessor',lambda:batch(req['request']['data'],x['key'],step,z['before'],z['remaining'],bad))
                    other=batch(req['request']['data'],x['key'],step+1,z['before'],z['remaining'],rows);reject('stale_step',lambda:consume(other,packet))
                    other=batch(req['request']['data'],x['key'],step,z['before'],z['remaining'],list(reversed(rows)));reject('reordered_actions',lambda:consume(other,packet))
                    for field,value in [('context_sha256','0'*64),('action_sha256','0'*64),('learned_output_qualified',True),('declared_source','trained_model')]:
                        badp=dict(packet);badp[field]=value;reject('packet_'+field,lambda p=badp:consume(b,p))
                    reject('wrong_shape',lambda:score_packet(b,y[:-1],'observed_ideal_fixture'))
                    badvalues=y.copy();badvalues[0]=np.nan;reject('nonfinite',lambda:score_packet(b,badvalues,'observed_ideal_fixture'))
                    numeric=np.linspace(-1.,1.,len(rows),dtype=np.float32);ck('unqualified_numeric_passthrough',np.array_equal(consume(b,score_packet(b,numeric,'unqualified_numeric_fixture')),numeric));save('BOUNDARY_REJECTIONS.json',dict(names=faults,count=len(faults),numeric_fixture_not_actual_model=True))
    with (O/'TENSORS.npz').open('xb') as f:np.savez(f,**{k:np.concatenate(v) for k,v in arrays.items()},state_offsets=np.asarray(boundaries,dtype=np.int64),customer_offsets=np.asarray(group_boundaries,dtype=np.int64))
    ck('all_fixed30_750_370500',cases==30 and states==750 and offset==370500);end=capture(reg['inputs']);save('SOURCE_END.json',end);ck('loaded_identity',begin['files']==end['files'] and begin['module_files']==end['module_files']);ck('input_bytes',all(sha(Path(p))==h for p,h in reg['inputs'].items()))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),cases=cases,states=states,action_rows=offset,customer_groups=len(group_boundaries)-1,tensor_bytes=(O/'TENSORS.npz').stat().st_size,tensor_sha256=sha(O/'TENSORS.npz'),loaded_files=len(begin['files']),module_maps=len(begin['module_files']),learned_output_qualified=False,full_C0_qualified=False,hard_request_budget_qualified=False,new_scientific_family=False,stable_online_gain=False)
    save('RESULT.json',result);save('RUN_EXIT.json',dict(exit_code=int(result['failed']>0),pid=os.getpid()));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));return int(result['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        if O.exists():save('FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
