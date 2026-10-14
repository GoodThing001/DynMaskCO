"""Projection-only engineering: no decisions, SAA, model or plan installation."""
import copy,json,sys,hashlib,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import qualify_pickup_private_public_transport_v2 as fixture
import pickup_undecided_public_v1 as pending
import public_state as old_projector
import public_fleet as pf
import numpy as np
ROOT=fixture.ROOT;E=fixture.E;P=fixture.P
OUT=P/'undecided_public_input_root_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def private_digest(env,fleet,traces,served):
    return pf.digest(dict(dataset=env.dataset,state=env.replanner.export_state(),fleet=fleet,
                          traces=traces,served=served,replan_count=env.replan_count,force_suffix=env.force_suffix))
def derived(index,variant):
    row,env,fleet,traces,served=fixture.state(index,variant)
    o=int(row['order']);rp=env.replanner
    if o not in rp._rejected or o in rp._accepted:raise ValueError('declared old current-order placeholder fixture')
    rp._rejected.remove(o)  # Declared derived fixture, never a live admission transition.
    return row,env,fleet,traces,served,o
def live_initial(variant):
    """Observe first actual reveal-created pending orders; stop before decisions."""
    from strict_online_env import StrictOnlineEnv
    from run_exp_energy_c0 import C0ReserveReplanner
    row,oldenv,_,_,_=fixture.state(0,variant);old=row['public_request']
    class CaptureStop(Exception):pass
    captured={}
    class ObservePolicy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            self._reset_if_new(i)
            _,_,actual_fleet,traces,actual_served=env.observed_context
            if actual_fleet is not fleet or actual_served is not served:raise RuntimeError('live context identity')
            known=sorted(int(o) for o in visible if env.demands[i,o]>0)
            o=sorted(known,key=lambda c:(env.reveal_time[i,c],c))[0]
            before=private_digest(env,fleet,traces,served);t0=time.perf_counter()
            packet=pending.export_undecided(env,i,clock,env.event_id,env.observed_reveal_index,fleet,traces,served,range(1,env.num_nodes),o)
            captured.update(packet=packet,wall_s=time.perf_counter()-t0,before=before,
                            after=private_digest(env,fleet,traces,served),order=o,
                            callsite='StrictOnlineEnv._run_loop -> on_reveal before policy decision and _plan_and_commit',
                            fleet_created_by_run=True,admission_calls=0,decision_calls=0,
                            initial_plan_calls=self.observed_initial_plans)
            raise CaptureStop()
        def plan(self,*args,**kwargs):
            self.observed_initial_plans+=1
            return super().plan(*args,**kwargs)
    rp=ObservePolicy(old['budget'],old['capacity'],old['booking_horizon'],old['contract'],0.)
    rp._reset_if_new(0);rp.observed_initial_plans=0
    class Observe(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served)
            self.observed_context=(i,clock,fleet,traces,served)
            # Only reveal times <= the current clock; no private future count.
            self.observed_reveal_index=len(set(float(self.reveal_time[i,o]) for o in range(1,self.num_nodes)
                if self.demands[i,o]>0 and 0<self.reveal_time[i,o]<=clock+1e-6))
            return value
    env=Observe(copy.deepcopy(oldenv.dataset),old['capacity'],old['tw_speed'],old['snapshot']['num_vehicles'],rp,old['contract'],old['booking_horizon'])
    try:env.run(0)
    except CaptureStop:pass
    else:raise RuntimeError('registered first-reveal capture did not stop')
    return captured
def main():
    import psutil
    psutil.Process().cpu_affinity([7])
    if np.__version__!='1.26.4':raise RuntimeError('pinned original NumPy required')
    OUT.mkdir(exist_ok=False);encode,_=fixture.codec()
    save('ENGINEERING_RECOVERY.json',dict(reason='v1 expected visible orders at t=0 but actual old day has none; observe first actual reveal before decision',old_failure_sha256=sha(P/'undecided_public_input_root_v1/FAILURE.json'),old_run_exit_sha256=sha(P/'undecided_public_input_root_v1/RUN_EXIT.json'),old_source_sha256=sha(Path(__file__).with_name('qualify_pickup_undecided_public_v1.py')),new_scientific_data=False))
    inputs=[fixture.FULL,fixture.TRACE,P/'actual_masked_fleet_followup_v1/public_state.py',P/'public_fleet.py',
            E/'scripts/simulation/strict_online_env.py',E/'scripts/simulation/recourse_snapshot.py',
            E/'scripts/evaluation/run_exp_energy_c0.py',Path(fixture.__file__),Path(pending.__file__),
            Path(__file__),Path(__file__).with_name('audit_pickup_undecided_public_v2.py'),
            Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__)]
    identities={str(p):sha(p) for p in inputs}
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=identities,
        derived_indices=[0,9,13],variants=[0,1],actual_first_reveal_only=True,
        current_order_semantics='observed pending, absent from both accepted/rejected; no placeholder mutation by projector',
        derived_fixture_rule='remove only current-order rejection placeholder before projection; not a new live midcourse event',
        real_initial_rule='actual run-created fleet, first reveal callback, first sorted visible order; initial empty planning already occurred; stop before any admission decision',
        hidden_variant_rule='same previously declared hidden attributes and four added future orders',
        old_decided_schema_unchanged=True,new_scientific_data=False,
        admission_scoring_installation_qualified=False,full_live_request_qualified=False,
        full_loaded_dependency_closure_qualified=False,stable_online_gain=False))
    checks=[];records=[];boundaries=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    def rejected(call):
        try:call()
        except ValueError as exc:return str(exc)
        return None
    try:
        for index in (0,9,13):
            pair=[]
            for variant in (0,1):
                row,env,fleet,traces,served,o=derived(index,variant);old=row['public_request'];snap=old['snapshot']
                before=private_digest(env,fleet,traces,served);t0=time.perf_counter()
                packet=pending.export_undecided(env,0,snap['clock'],snap['event_id'],snap['reveal_idx'],fleet,traces,served,range(1,env.num_nodes),o)
                wall=time.perf_counter()-t0;after=private_digest(env,fleet,traces,served)
                tag=str((index,variant));a=set(packet['snapshot']['replanner_state']['accepted']);r=set(packet['snapshot']['replanner_state']['rejected'])
                ck(tag+':private_state_unchanged',before==after)
                ck(tag+':current_pending_not_rejected',o not in a|r and o in packet['admission_input']['pending'])
                ck(tag+':old_accepted_unchanged',a==set(snap['replanner_state']['accepted']))
                ck(tag+':old_rejected_minus_current_only',r==set(snap['replanner_state']['rejected'])-{o})
                ck(tag+':ownership_unchanged',packet['ownership']==old['ownership'] and packet['plan']==old['plan'])
                ck(tag+':explicit_schema_rejects_old_consumer',rejected(lambda:pf.validate_public(packet)) is not None)
                ck(tag+':old_exporter_still_rejects_pending',rejected(lambda:old_projector.export_public(env,0,snap['clock'],snap['event_id'],snap['reveal_idx'],fleet,traces,served,range(1,env.num_nodes))) is not None)
                ck(tag+':old_export_rejection_nonmutating',private_digest(env,fleet,traces,served)==before)
                pair.append(packet);records.append(dict(kind='derived_old_state',index=index,variant=variant,order=o,packet=encode(packet),before=before,after=after,wall_s=wall))
            ck(str(index)+':future_attribute_cardinality_invariant',pf.digest(pair[0])==pf.digest(pair[1]))
        initial=[]
        for variant in (0,1):
            got=live_initial(variant);packet=got.pop('packet');st=packet['snapshot']['replanner_state']
            ck('initial:'+str(variant)+':actual_run_hook',got['fleet_created_by_run'] and got['before']==got['after'])
            ck('initial:'+str(variant)+':no_decisions',not st['accepted'] and not st['rejected'])
            ck('initial:'+str(variant)+':all_visible_pending',set(packet['admission_input']['pending'])==set(packet['snapshot']['customer_universe']))
            ck('initial:'+str(variant)+':no_service',all(x==0 for x in packet['snapshot']['services_len']) and not packet['ownership'])
            initial.append(packet);records.append(dict(kind='actual_first_reveal_hook',variant=variant,packet=encode(packet),**got))
        ck('initial:future_attribute_cardinality_invariant',pf.digest(initial[0])==pf.digest(initial[1]))
        base=initial[0];o=base['admission_input']['current_order']
        mutations={
            'pending_marked_rejected':lambda x:x['snapshot']['replanner_state']['rejected'].append(o),
            'missing_pending':lambda x:x['admission_input']['pending'].remove(o),
            'duplicate_pending':lambda x:x['admission_input']['pending'].append(o),
            'already_decided_current':lambda x:x['admission_input'].update(current_order=0),
            'boolean_current':lambda x:x['admission_input'].update(current_order=True),
            'private_channel':lambda x:x['dataset'].update(private_future_count=81),
            'pending_in_plan':lambda x:x['plan']['0'].append(o),
            'unrevealed_attribute':lambda x:x['dataset']['reveal_time'].__setitem__((0,o),999.),
            'state_digest_changed':lambda x:x['snapshot'].update(replan_count=999)}
        for name,mutate in mutations.items():
            trial=copy.deepcopy(base);mutate(trial);error=rejected(lambda:pending.validate_undecided(trial))
            ck('boundary:'+name,error is not None);boundaries.append(dict(name=name,error=error))
        ck('registered_input_and_source_bytes_end',{p:sha(p) for p in identities}==identities)
        result=dict(checks=checks,checks_count=len(checks),failed=sum(not x['pass_'] for x in checks),records=records,boundaries=boundaries,
                    sources_start=identities,sources_end={p:sha(p) for p in identities},
                    undecided_projection_component_only=True,actual_first_reveal_input_hook=True,
                    continuous_midcourse_events=False,admission_scoring_installation=False,full_live_request_qualified=False,stable_online_gain=False)
        save('RESULT.json',result);save('RUN_EXIT.json',dict(exit_code=int(result['failed']>0)))
        print(json.dumps(dict(checks=len(checks),failed=result['failed'],records=len(records),boundaries=len(boundaries))))
        return int(result['failed']>0)
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc(),checks=checks,records=records));save('RUN_EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':raise SystemExit(main())
