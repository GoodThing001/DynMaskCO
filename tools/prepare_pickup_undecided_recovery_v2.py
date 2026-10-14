"""Preserve failed v1 and prepare first actual reveal observation in new v2."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools'
old=(T/'qualify_pickup_undecided_public_v1.py').read_text(encoding='utf-8')
start=old.index('def live_initial(variant):');end=old.index('def main():',start)
replacement='''def live_initial(variant):
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
'''
new=old[:start]+replacement+old[end:]
new=new.replace("undecided_public_input_root_v1","undecided_public_input_root_v2").replace("audit_pickup_undecided_public_v1.py","audit_pickup_undecided_public_v2.py")
new=new.replace("actual_initial_event_only=True","actual_first_reveal_only=True")
new=new.replace("actual run-created fleet, first sorted visible order, stop before any policy or vehicle execution","actual run-created fleet, first reveal callback, first sorted visible order; initial empty planning already occurred; stop before any admission decision")
new=new.replace("kind='actual_initial_hook'","kind='actual_first_reveal_hook'")
new=new.replace("actual_initial_input_hook=True","actual_first_reveal_input_hook=True")
new=new.replace("actual_midcourse_event=False","continuous_midcourse_events=False")
new=new.replace("actual_initial_hook=True","actual_first_reveal_hook=True")
new=new.replace("OUT.mkdir(exist_ok=False);encode,_=fixture.codec()","OUT.mkdir(exist_ok=False);encode,_=fixture.codec()\n    save('ENGINEERING_RECOVERY.json',dict(reason='v1 expected visible orders at t=0 but actual old day has none; observe first actual reveal before decision',old_failure_sha256=sha(P/'undecided_public_input_root_v1/FAILURE.json'),old_run_exit_sha256=sha(P/'undecided_public_input_root_v1/RUN_EXIT.json'),old_source_sha256=sha(Path(__file__).with_name('qualify_pickup_undecided_public_v1.py')),new_scientific_data=False))")
with (T/'qualify_pickup_undecided_public_v2.py').open('x',encoding='utf-8') as f:f.write(new)
audit=(T/'audit_pickup_undecided_public_v1.py').read_text(encoding='utf-8').replace('undecided_public_input_root_v1','undecided_public_input_root_v2')
audit=audit.replace("record['plan_calls']==0","record['decision_calls']==0")
audit=audit.replace("s['clock']==0 and s['event_id']==0 and s['replan_count']==0","s['clock']==min(float(env.reveal_time[0,c]) for c in range(1,env.num_nodes) if env.demands[0,c]>0) and s['event_id']>=1")
audit=audit.replace('actual_initial_hook_only=True','actual_first_reveal_hook_only=True')
with (T/'audit_pickup_undecided_public_v2.py').open('x',encoding='utf-8') as f:f.write(audit)
print('prepared new v2; original v1 unchanged')
