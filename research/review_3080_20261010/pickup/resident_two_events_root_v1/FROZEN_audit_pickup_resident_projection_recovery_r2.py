"""Narrow stored-state/AST recovery of private-versus-public suffix assumption."""
import ast,copy,json
from pathlib import Path
import numpy as np,psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from strict_online_env import StrictOnlineEnv
import inspect
ROOT=old.ROOT;P=old.P;OUT=P/'resident_two_events_root_v1';SOURCE=P/'endogenous_owner_transaction_root_v2/OWNER_RESULT_0.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=decode(read(OUT/'RESIDENT_RESULT.json'));original=read(OUT/'INDEPENDENT_RESIDENT_TWO_EVENTS.json');checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    ck('original_single_named_failure_preserved',original['checks_count']==1216 and original['failed']==1 and original['failure_names']==['new:1:original_private_admission_before'])
    initial=decode(read(SOURCE))['after_primary_advance'];first=raw['records'][0];public=first['packet']['snapshot']['replanner_state'];private=first['before']
    ck('original_private_state_exact_from_sealed_start',old.pf.digest(initial['state'])==old.pf.digest(private))
    for k in private:
        if k!='plan':ck('private_and_public_nonplan:'+k,old.pf.digest(private[k])==old.pf.digest(public[k]))
    snap=first['packet']['snapshot'];served=set(np.flatnonzero(snap['served_mask']));plans={};removed={}
    for vid in range(snap['num_vehicles']):
        route=private['plan'][str(vid)];suffix=[int(o) for o in route if o not in served];removed[str(vid)]=[int(o) for o in route if o in served];cn=int(snap['committed_next'][vid])
        if snap['vehicle_status'][vid]=='committed' and cn>0:suffix=[cn]+[o for o in suffix if o!=cn]
        plans[str(vid)]=suffix;ck('independent_current_suffix:'+str(vid),suffix==public['plan'][str(vid)]==first['packet']['plan'][str(vid)])
    ck('only_just_served52_removed',removed=={'0':[52],'1':[],'2':[]} and 52 in served)
    events=[e for e in raw['events'] if e['sequence']==1];ck('single_new_reveal_physical_step_before_first_admission',len(events)==1 and events[0]['event_id']==initial['event_id']+1==first['event_id'] and events[0]['clock']==first['clock'])
    newly=set(np.flatnonzero(events[0]['served']))-set(np.flatnonzero(initial['served']));ck('original_stored_physical_service52',newly=={52} and any(s.node==52 for t in events[0]['traces'] for s in t.services))
    tree=ast.parse(Path(inspect.getfile(StrictOnlineEnv)).read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StrictOnlineEnv');advance=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_advance_fleet');loop=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_run_loop')
    ck('original_advance_fleet_has_no_replanner_access',not [n for n in ast.walk(advance) if isinstance(n,ast.Attribute) and n.attr=='replanner'])
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('_advance_fleet','on_reveal','_plan_and_commit')]
    ck('original_loop_reveal_entry_precedes_plan',next(n.lineno for n in calls if n.func.attr=='_advance_fleet')<next(n.lineno for n in calls if n.func.attr=='on_reveal')<next(n.lineno for n in calls if n.func.attr=='_plan_and_commit'))
    ck('stored_second_private_continuity_exact',raw['records'][1]['resident_before']==first['resident_after'] and old.pf.digest(raw['records'][1]['before'])==old.pf.digest(first['after']))
    reg=read(OUT/'REGISTRATION.json')
    for p,h in reg['sources'].items():ck('unchanged_registered:'+p,old.sha(p)==h)
    names={'qualify_pickup_resident_two_events_v1.py','audit_pickup_resident_two_events_v1.py'}
    ck('original_owned_processes_gone',not [p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name in names for a in p.info['cmdline'] or [])])
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),failure_names=[c['name'] for c in checks if not c['pass_']],original_audit_sha256=old.sha(OUT/'INDEPENDENT_RESIDENT_TWO_EVENTS.json'),producer_sha256=old.sha(OUT/'RESULT.json'),script_sha256=old.sha(__file__),private_public_suffix_distinction_only=True,removed_completed_orders=removed,no_original_admission_solver_SAA_or_event_reexecution=True,original_1216_1FAIL_unchanged=True,qualified_composition='original1215 successful checks plus this narrow private/public suffix evidence',full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=result['checks_count'],failed=result['failed'],failures=result['failure_names'],removed=removed)));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
