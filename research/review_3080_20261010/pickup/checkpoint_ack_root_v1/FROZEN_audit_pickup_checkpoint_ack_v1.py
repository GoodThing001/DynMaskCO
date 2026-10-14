"""Independent original physical step and exact stored checkpoint audit.

Does not call new checkpoint worker/restore/publication routines.
"""
import copy,hashlib,json,pickle,sys,time
from pathlib import Path
from dataclasses import is_dataclass,fields
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_observed_day_extension_v1 as observed
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
from pickup_loaded_closure_v1 import capture,sha
OUT=old.P/'checkpoint_ack_root_v1';SOURCE=old.P/'resident_two_events_root_v1/RESIDENT_RESULT.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    text=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(text)
def independent_encode(v):
    if is_dataclass(v):return {'__dataclass__':type(v).__module__+'.'+type(v).__name__,'fields':{f.name:independent_encode(getattr(v,f.name)) for f in fields(v)}}
    if isinstance(v,np.ndarray) or isinstance(v,np.generic):return {'__array__':str(v.dtype),'shape':list(v.shape),'values':independent_encode(v.tolist())}
    if isinstance(v,float) and not np.isfinite(v):return {'__float__':repr(v)}
    if isinstance(v,dict):return {'__mapping__':[[independent_encode(k),independent_encode(x)] for k,x in v.items()]}
    if isinstance(v,(tuple,list,set,frozenset)):return {'__sequence__':type(v).__name__,'values':[independent_encode(x) for x in (sorted(v) if isinstance(v,(set,frozenset)) else v)]}
    return v
def digest(v):
    # public_fleet digest uses standard original canonical encoded audit tree.
    encoded=independent_encode(v)
    return hashlib.sha256(json.dumps(encoded,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4';assert not (OUT/'INDEPENDENT_CHECKPOINT_ACK.json').exists()
    reg=read(OUT/'REGISTRATION.json');result=read(OUT/'RESULT.json');assert result['failed']==0 and read(OUT/'RUN_EXIT.json')['exit_code']==0
    paths=[p for p in OUT.iterdir() if p.is_file()]+[SOURCE,Path(__file__)]
    identity={str(p):sha(p) for p in paths};save('AUDIT_REGISTRATION.json',dict(inputs=identity,scope='direct original physical advance at fixed next natural boundary; independent scalar-bit/array/dataclass fields, lineage/payload and timing/source identities; no admission or solver calls'))
    _,decode=old.codecs();initial=decode(read(SOURCE))['final'];template,dataset=observed.setup();rp=C0ReserveReplanner(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.);rp.restore_state(copy.deepcopy(initial['state']))
    env=StrictOnlineEnv(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();clock=initial['clock'];times=[float(env.tw_end[0].max())]
    times += [float(env.reveal_time[0,j]) for j in range(1,env.num_nodes) if env.demands[0,j]>0 and env.reveal_time[0,j]>clock+1e-6]
    if env.booking_horizon>clock+1e-6:times.append(env.booking_horizon)
    for v in fleet:
        if v.status=='committed' and v.committed_finish is not None:times.append(v.committed_finish)
        elif v.status=='returning' and v.return_finish is not None:times.append(v.return_finish)
    next_clock=min(times);assert next_clock>clock
    # Fully initialize audit primitives before its own loaded workload closure.
    digest(initial);capture();begin=capture(identity);save('AUDIT_SOURCE_START.json',begin)
    env.event_id+=1;env._advance_fleet(0,next_clock,fleet,traces,served)
    expected=dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=next_clock,event_id=env.event_id,replan_count=env.replan_count)
    checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    def exact(n,a,b):
        if isinstance(a,np.ndarray) or isinstance(a,np.generic):
            ck(n+':array_bits',type(a) is type(b) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes());return
        if is_dataclass(a):
            ck(n+':class',type(a) is type(b))
            for f in fields(a):exact(n+'.'+f.name,getattr(a,f.name),getattr(b,f.name))
        elif isinstance(a,dict):
            ck(n+':keys',list(a)==list(b))
            for k,v in a.items():exact(n+'.'+str(k),v,b[k])
        elif isinstance(a,(list,tuple)):
            ck(n+':sequence',type(a) is type(b) and len(a)==len(b))
            for i,(x,y) in enumerate(zip(a,b)):exact(n+'.'+str(i),x,y)
        elif isinstance(a,float):ck(n+':float_hex',type(a) is type(b) and a.hex()==b.hex())
        else:ck(n+':value',type(a) is type(b) and a==b)
    # Imported original hash uses the same canonical contract, but independently
    # coded encoder above must first match this fixed source, not assume it.
    baseline=digest(initial);target=digest(expected);ck('independent_canonical_source',baseline==old.pf.digest(initial));ck('original_one_next_event',env.event_id==initial['event_id']+1 and env.replan_count==initial['replan_count'])
    for case in range(3):
        n=str(case);h=read(OUT/('HOST_'+n+'.json'));a=decode(read(OUT/('PHYSICAL_ATTEMPT_'+n+'.json')));exact(n+':actual_physical_fields',expected,a['checkpoint']['state'])
        ck(n+':exact_digest_lineage',a['before']==baseline and a['after']==target and a['checkpoint']['base_digest']==baseline and a['checkpoint']['epoch']==1)
        payload=pickle.dumps(a['checkpoint'],protocol=5);ck(n+':payload_identity',len(payload)==a['payload_size'] and hashlib.sha256(payload).hexdigest()==a['payload_sha256'])
        ck(n+':physical_before_fault_and_reap',h['started']<=a['started']<=a['physical_finished']<h['finished']<h['started']+10. and h['reaped'] and h['stage']==(1 if case==1 else 2))
        start=read(OUT/('SOURCE_START_'+n+'.json'));end=read(OUT/('SOURCE_END_'+n+'.json'))
        for p,v in start['files'].items():ck(n+':source:'+p,end['files'].get(p)==v==sha(p))
        ck(n+':actual_loaded_module_map',start['module_files']==end['module_files'] and set(start['files'])==set(end['files']))
        if case==1:ck(n+':unpublished_old_authority',h['ticket'] is None and h['chosen']==baseline and not h['ack_present'])
        else:
            r=decode(read(OUT/('RECOVERY_'+n+'.json')));rh=read(OUT/('RECOVERY_HOST_'+n+'.json'));exact(n+':restored_private_fields',expected,r['restored'])
            ck(n+':committed_ticket',h['ticket']['base']==baseline and h['ticket']['digest']==target==h['chosen']==r['digest']==rh['digest'] and h['chosen_at']<h['cutoff'])
            ck(n+':restored_payload',r['payload_sha256']==h['ticket']['payload_sha256']==a['payload_sha256'])
            ck(n+':duplicate_no_advance',r['advance_calls']==0 and r['duplicate_digest']==target)
            ck(n+':lost_ACK_detected',h['ack_present']==(case==0))
            ck(n+':recovery_timed_reap',rh['started']<=r['started']<=r['finished']<=rh['finished']<rh['started']+10. and rh['exit_code']==0)
            s=read(OUT/('RECOVERY_SOURCE_START_'+n+'.json'));e=read(OUT/('RECOVERY_SOURCE_END_'+n+'.json'))
            for p,v in s['files'].items():ck(n+':recovery_source:'+p,e['files'].get(p)==v==sha(p))
            ck(n+':recovery_module_map',s['module_files']==e['module_files'] and set(s['files'])==set(e['files']))
    ck('all_registered_sources',all(sha(p)==v for p,v in reg['sources'].items()))
    ck('all_audit_inputs_unchanged',all(sha(p)==v for p,v in identity.items()))
    for p in psutil.process_iter(['name','cmdline']):
        if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name=='qualify_pickup_checkpoint_ack_v1.py' for x in p.info['cmdline'] or []):ck('owned_process_gone:'+str(p.pid),False)
    end=capture(identity);ck('audit_loaded_closure',begin['files']==end['files'] and begin['module_files']==end['module_files']);save('AUDIT_SOURCE_END.json',end)
    failed=[c['name'] for c in checks if not c['pass_']]
    save('INDEPENDENT_CHECKPOINT_ACK.json',dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,source_sha256=sha(Path(__file__)),registration_sha256=sha(OUT/'REGISTRATION.json'),result_sha256=sha(OUT/'RESULT.json'),initial_digest=baseline,final_digest=target,clock=float(next_clock),event_id=int(env.event_id),service_count=sum(len(t.services) for t in traces),shared_original_physics=True,full_live_request_qualified=False,external_irreversible_effects_qualified=False));print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,clock=float(next_clock),event_id=int(env.event_id))));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
