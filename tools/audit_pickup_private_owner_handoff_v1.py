"""Independent prepared-state physical replay and direct public fields, no SAA."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_private_public_transport_v2 as fixture
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
ROOT=old.ROOT;P=old.P;OUT=P/'private_owner_handoff_root_v1';SOURCE=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);encode,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');actual=decode(read(OUT/'OWNER_RESULT_0.json'));checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    with (SOURCE/'REQUESTS_STARTED.jsonl').open(encoding='utf-8') as f:request=decode(json.loads(next(f)))
    with (SOURCE/'PREPARED_RESPONSES.jsonl').open(encoding='utf-8') as f:response=decode(json.loads(next(f)))['response']
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    initial=decode(read(PREFIX/'RESULT.json'))['final'];packet=request['packet'];reference={}
    class Stop(Exception):pass
    class Policy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):reference.update(clock=clock,visible=visible);raise Stop()
    rp=Policy(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.);rp.restore_state(copy.deepcopy(initial['state']))
    env=StrictOnlineEnv(copy.deepcopy(dataset),packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();clock=float(initial['clock']);env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];customers=[j for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0))
    try:env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()))
    except Stop:pass
    else:raise AssertionError('registered reveal absent')
    clock=reference['clock']
    def snapshot():return dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=clock,event_id=env.event_id,replan_count=env.replan_count)
    baseline=old.pf.digest(snapshot());ck('independent_actual_private_before_identity',baseline==actual['baseline'] and clock==packet['snapshot']['clock'] and env.event_id==packet['snapshot']['event_id'])
    known=reference['visible'];projected=actual['public_packet'];ck('original_sealed_public_digest',old.pf.digest(projected)==old.pf.digest(packet)==actual['public_digest'])
    for key,value in projected['dataset'].items():
        expected=np.zeros_like(value);expected[0,0]=dataset[key][0,0]
        for o in known:expected[0,o]=dataset[key][0,o]
        ck('direct_observed_field:'+key,np.array_equal(value,expected))
    ck('actual_pending_not_placeholder',request['order'] in projected['admission_input']['pending'] and request['order'] not in set(rp._accepted)|set(rp._rejected))
    compiled=response['compiled_state'];state=dict(version='c0-reserve-v1',budget=compiled.budget,mode=compiled.mode,accepted=sorted(compiled.accepted),rejected=sorted(compiled.rejected),plan={str(k):list(v) for k,v in compiled.plan.items()},status_inst=compiled.status_inst)
    independent_compiled=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();ck('original_compiled_sha',independent_compiled==response['compiled_state_sha256']==actual['compiled_sha256'])
    rp.restore_state(copy.deepcopy(state));env._plan_and_commit(0,clock,fleet,traces,served,customers)
    for k,v in snapshot().items():ck('original_actual_plan_commit:'+k,old.pf.digest(v)==old.pf.digest(actual['after_commit'][k]))
    next_clock=float(env.tw_end[0].max());remaining=[t for t in reveals if t>clock+1e-6]
    if remaining:next_clock=min(next_clock,remaining[0])
    if env.booking_horizon>clock+1e-6:next_clock=min(next_clock,env.booking_horizon)
    for v in fleet:
        if v.status=='committed' and v.committed_finish is not None:next_clock=min(next_clock,v.committed_finish)
        elif v.status=='returning' and v.return_finish is not None:next_clock=min(next_clock,v.return_finish)
    before=old.pf.digest(snapshot());clock=next_clock;env.event_id+=1;env._advance_fleet(0,clock,fleet,traces,served)
    ck('original_natural_physical_clock',clock==actual['after_primary_advance']['clock'] and before==actual['primary_advance_before'])
    for k,v in snapshot().items():ck('original_primary_physical_execution:'+k,old.pf.digest(v)==old.pf.digest(actual['after_primary_advance'][k]))
    ck('actual_primary_physical_digest',old.pf.digest(snapshot())==actual['primary_advance_digest'])
    ck('prepared_only_no_new_strategy',reg['prepared_response_reused'] and not reg['new_admission_or_SAA'] and not reg['new_scientific_data'] and not actual['new_admission_or_SAA'])
    for p,h in reg['sources'].items():ck('registered:'+p,sha(p)==h)
    names={'qualify_pickup_private_owner_handoff_v1.py'}
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(arg).name in names for arg in p.info['cmdline'] or [])]
    ck('actual_owned_processes_gone',not active)
    for r in raw['records']:
        n=str(r['case']);ck(n+':independent_old_owner_unchanged',r['baseline']==r['old_owner_after']==baseline);ck(n+':actual_timing_and_reap',r['finished']-r['started']==r['wall_s'] and r['wall_s']<r['cap_s'] and r['reaped']);ck(n+':predeclared_budget',r['cap_s']==(.4 if r['fault'] else 10.) and r['reserve_s']==(.15 if r['fault'] else 2.));ck(n+':fixed_host_work',not r['timed_parent_public_projection_or_pickle']);begin=read(OUT/('SOURCE_START_'+n+'.json'));ck(n+':actual_loaded_pid',begin['pid']==r['pid']==r['ready']['pid'])
        for p,h in begin['files'].items():ck(n+':loaded:'+p,sha(p)==h)
        if r['fault']:ck(n+':cancel_actual_stage_without_owner_promotion',r['stage_reached']==r['fault'] and r['error']=='cancelled' and not r['actual_process_owner_selected'] and r['prepared'] is None and not (OUT/('OWNER_RESULT_'+n+'.json')).exists())
        else:
            end=read(OUT/'SOURCE_END_0.json');ck(n+':normal_actual_owner_and_progress',r['actual_process_owner_selected'] and r['primary_advance']==actual['primary_advance_digest'] and r['prepared']==independent_compiled and r['started']<=r['promotion']<r['finished']);ck(n+':normal_loaded_file_closure',begin['files']==end['files'] and begin['module_files']==end['module_files'] and actual['loaded_closure_qualified'])
    ck('full_request_scientific_qualification_remains_false',not raw['full_live_request_qualified'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['stable_online_gain'])
    failed=[c['name'] for c in checks if not c['pass_']];result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,registration_sha256=sha(OUT/'REGISTRATION.json'),producer_sha256=sha(OUT/'RESULT.json'),independent_public_fields_and_original_physical_replay=True,new_SAA_or_admission_reexecution=False,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_PRIVATE_OWNER.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed)));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
