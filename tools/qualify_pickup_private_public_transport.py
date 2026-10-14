"""Restored private-environment public projection and cancellable transfer QA.

No admission/proposal/SAA/install; no claim of live simulator ownership integration.
"""
import base64,copy,hashlib,json,os,struct,sys,time
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import is_dataclass
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup'
OUT=P/'private_public_transport_root_v1'
FULL=P/'complete_history_energy_qa_root_v1/complete_ledger/FULLRAW.json'
TRACE=P/'complete_history_energy_qa_root_v1/complete_ledger/cond_hist_trace.jsonl'
for f in (P,P/'actual_masked_fleet_engineering_v1',P/'actual_masked_fleet_followup_v1',P/'shadow_energy_correction_root_v1'):sys.path.insert(0,str(f))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import public_fleet as pf
pf.bootstrap(E/'scripts')
from pickup_interruptible_public_transport import PublicTransport,HEADER,CONTROL,MAX_PAYLOAD
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def codec():
 import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
 from audit_io import encode,decode
 classes={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa)
          for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
 return encode,lambda x:decode(x,classes)
def state(index,variant):
 import numpy as np
 from strict_online_env import StrictOnlineEnv
 from recourse_snapshot import restore_recourse_snapshot
 from run_exp_energy_c0 import C0ReserveReplanner
 _,decode=codec();row=decode(json.loads(FULL.read_text(encoding='utf-8')))['admissions'][index];old=row['public_request']
 with TRACE.open(encoding='utf-8') as f:private=decode(json.loads(next(f))['payload'])['dataset']
 private=copy.deepcopy(private);clock=old['snapshot']['clock'];n=private['demands'].shape[1]
 hidden=[o for o in range(1,n) if private['reveal_time'][0,o]>clock+1e-6]
 if variant:
  for name,a in private.items():
   if isinstance(a,np.ndarray) and a.ndim>=2 and a.shape[1]==n:
    extra=np.repeat(a[:,-1:],4,axis=1);private[name]=np.concatenate([a,extra],axis=1)
  private['reveal_time'][0,n:]=999.;private['coords'][0,hidden]=.987;private['demands'][0,hidden]=1
  private['tw_start'][0,hidden]=7.;private['tw_end'][0,hidden]=8.;private['temp_class'][0,hidden]=0
 rp=C0ReserveReplanner(old['budget'],old['capacity'],old['booking_horizon'],old['contract'],0.)
 rp.restore_state(old['snapshot']['replanner_state'])
 env=StrictOnlineEnv(private,old['capacity'],old['tw_speed'],old['snapshot']['num_vehicles'],rp,old['contract'],old['booking_horizon'])
 env.replan_count=old['snapshot']['replan_count']
 fleet,traces,served=restore_recourse_snapshot(old['snapshot'])
 served=np.pad(served,(0,env.num_nodes-len(served)),constant_values=False)
 return row,env,fleet,traces,served
def worker():
 import numpy as np,psutil
 from multiprocessing.shared_memory import SharedMemory
 import public_state
 if np.__version__!='1.26.4':raise RuntimeError('Pinned original NumPy required')
 psutil.Process().cpu_affinity([7]);memory=SharedMemory(name=sys.argv[2]);variant=int(sys.argv[3])
 # Old observed restored environments: declared test setup, not online day preload.
 states={i:state(i,variant) for i in (0,9,13)};encode,_=codec()
 sources=pf.source_hashes(E/'scripts')
 for p in (Path(__file__),Path(__file__).with_name('pickup_interruptible_public_transport.py'),P/'actual_masked_fleet_followup_v1/public_state.py',Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__),FULL,TRACE):sources[str(p)]=sha(p)
 stream=sys.stdout.buffer
 def send(kind,payload):stream.write(HEADER.pack(kind,len(payload))+payload);stream.flush()
 send(0,json.dumps({'sources':sources,'numpy':np.__version__,'cpu':psutil.Process().cpu_affinity()}).encode())
 while not struct.unpack_from('<Q',memory.buf,0)[0]:time.sleep(.001)
 generation,index,deadline,fault=CONTROL.unpack_from(memory.buf,0)
 row,env,fleet,traces,served=states[index];old=row['public_request'];clock=old['snapshot']['clock']
 def phase(name,code):
  send(1,name.encode('ascii'))
  if fault==code:time.sleep(2.)
  if time.perf_counter()>=deadline:raise TimeoutError(name)
 try:
  phase('private_projection',1)
  packet=public_state.export_public(env,0,clock,old['snapshot']['event_id'],old['snapshot']['reveal_idx'],fleet,traces,served,list(range(1,env.num_nodes)))
  phase('public_encoding',2)
  data=json.dumps(encode(packet),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
  phase('opaque_transfer',3)
  if fault==4:
   stream.write(HEADER.pack(2,len(data))+data[:128]);stream.flush();send_delay=True;time.sleep(2.);stream.write(data[128:]);stream.flush()
  elif fault==5:stream.write(HEADER.pack(2,MAX_PAYLOAD+1));stream.flush()
  else:send(2,data)
  # End source check recorded separately, outside timed component result.
  send(3,json.dumps({'source_end':{p:sha(p) for p in sources}}).encode())
 except Exception as exc:send(3,json.dumps({'error':repr(exc)}).encode())
 finally:memory.close()
def main():
 import numpy as np,psutil
 if np.__version__!='1.26.4':raise RuntimeError('Original NumPy environment required')
 psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False);encode,decode=codec()
 reg={'created_utc':datetime.now(timezone.utc).isoformat(),'fixture_sha256':sha(FULL),'private_trace_sha256':sha(TRACE),
      'program_sha256':sha(__file__),'transport_sha256':sha(Path(__file__).with_name('pickup_interruptible_public_transport.py')),
      'indices':[0,9,13],'selection':'old first, mutable-width and outcome-selected rejection engineering fixtures',
      'normal_cap_s':10.,'fault_cap_s':.2,'max_payload_bytes':MAX_PAYLOAD,'cpu':[7],'numpy':'1.26.4',
      'setup':'restored old private environments before request, no projection until shared deadline',
      'parent_bulk_decode_included':False,'admission_SAA_install_included':False,'production_live_ownership_integrated':False,
      'new_scientific_data':False,'whole_live_request_qualified':False,'stable_online_gain':False}
 save('REGISTRATION.json',reg);records=[];checks=[]
 def ck(n,v):checks.append({'name':n,'pass':bool(v)})
 for index in reg['indices']:
  outputs=[]
  for variant in (0,1):
   worker=PublicTransport(Path(__file__),variant)
   try:
    ready=json.loads(worker.ready[1]);response=worker.ask(index,10.)
    ck(str((index,variant))+':complete_frame',response['kind']=='opaque_public_packet')
    if response['kind']!='opaque_public_packet':raise RuntimeError(response)
    ck(str((index,variant))+':within_budget',response['finished']<response['deadline'])
    ck(str((index,variant))+':actual_stage_markers',response['stages']==['private_projection','public_encoding','opaque_transfer'])
    _,end=worker.read_frame(time.perf_counter()+5);end=json.loads(end)
    ck(str((index,variant))+':actual_source_end',end['source_end']==ready['sources'])
    ck(str((index,variant))+':runtime_cpu',ready['cpu']==[7] and ready['numpy']=='1.26.4')
    payload=decode(json.loads(response.pop('bytes')));outputs.append(payload)
    records.append({'index':index,'variant':variant,'transport':response,'packet':encode(payload),'source_start':ready['sources'],'source_end':end['source_end']})
   finally:worker.close()
  ck(str(index)+':hidden_attributes_and_cardinality_invariance',pf.digest(outputs[0])==pf.digest(outputs[1]))
  ck(str(index)+':public_known_universe',set(outputs[0]['snapshot']['customer_universe'])==set(outputs[0]['snapshot']['replanner_state']['accepted'])|set(outputs[0]['snapshot']['replanner_state']['rejected']))
 faults=[]
 for fault,target in [(1,'private_projection'),(2,'public_encoding'),(3,'opaque_transfer'),(4,'opaque_transfer'),(5,'opaque_transfer')]:
  worker=PublicTransport(Path(__file__))
  try:
   response=worker.ask(9,.2,fault)
   ck('fault:'+str(fault)+':target_reached',target in response['stages'])
   ck('fault:'+str(fault)+':cancelled_reaped',response['kind']=='cancelled' and response['worker_reaped'])
   if fault==4:ck('fault:4:partial_frame_observed',response['partial_bytes_discarded']>=128)
   faults.append({'fault':fault,'target':target,'response':response})
  finally:worker.close()
 ck('registration_sources_end',sha(FULL)==reg['fixture_sha256'] and sha(TRACE)==reg['private_trace_sha256'] and sha(__file__)==reg['program_sha256'] and sha(Path(__file__).with_name('pickup_interruptible_public_transport.py'))==reg['transport_sha256'])
 result={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'records':records,'faults':faults,
         'restored_private_projection_transport_only':True,'parent_bulk_decode_included':False,'whole_live_request_qualified':False,'stable_online_gain':False}
 save('RESULT.json',result);save('RUN_EXIT.json',{'exit_code':int(result['failed']>0)})
 print(json.dumps({'checks':len(checks),'failed':result['failed'],'records':len(records),'faults':len(faults)}));return int(result['failed']>0)
if __name__=='__main__':
 if '--worker' in sys.argv:worker()
 else:raise SystemExit(main())
