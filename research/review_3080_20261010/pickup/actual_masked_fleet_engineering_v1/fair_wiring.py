"""Uniform causal per-admission wrappers; no edits to protected simulator/source.

Rolling remains the existing nonlearning admission+relocate/swap heuristic, not
an oracle. Exact observed physical certificates guard its proposed live commit.
"""
import copy
import hashlib
import json
from numbers import Real
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import public_fleet as pf

def history_hash(history):
    h=hashlib.sha256()
    for times,classes in history:
        for a in (times,classes):
            a=np.ascontiguousarray(a);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def make_trace_env_class(scripts):
    pf.bootstrap(scripts)
    from strict_online_env import StrictOnlineEnv
    class TraceAwareStrictEnv(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            result=super()._advance_fleet(i,clock,fleet,traces,served)
            self._public_physical_context=(int(i),float(clock),fleet,traces,served)
            return result
    return TraceAwareStrictEnv

def uniform_per_admission(base_class,scripts):
    """Every admission gets10s from entry, including certified commit.

    Scoped synchronous legacy baseline calls are checked after completion and
    rolled back if late. They can overrun wall computation; this guard is a
    submission cap, not a hard process-kill guarantee for legacy rolling/SAA.
    Masked actor uses separate warmed subprocess within this SAME remaining cap.
    """
    pf.bootstrap(scripts)
    from run_exp_energy_c0 import certify_plan
    class UniformAdmission(base_class):
        def __init__(self,*args,**kwargs):
            limit=kwargs.pop('uniform_limit',10.)
            if isinstance(limit,bool) or not isinstance(limit,Real):raise ValueError('numeric admission cap')
            self.uniform_limit=float(limit)
            if not np.isfinite(self.uniform_limit) or not 0<self.uniform_limit<=10:raise ValueError('finite admission cap <=10')
            super().__init__(*args,**kwargs)
            self.admission_audit=[];self._uniform_timing=True
        def _saa_decide(self,env,i,clock,fleet,served,o,scenarios,t0):
            # Base sampling occurred after the wrapper start, so all sampling is counted.
            if len(scenarios)!=10:raise ValueError('complete registered K10 scenario set required')
            self._uniform_scenario_calls=0
            return super()._saa_decide(env,i,clock,fleet,served,o,scenarios,self._uniform_admission_started)
        def _sim_scenario(self,*args,**kwargs):
            value=super()._sim_scenario(*args,**kwargs)
            self._uniform_scenario_calls+=1
            return value
        def on_reveal(self,env,i,clock,fleet,served,visible):
            self._reset_if_new(i)
            reserved=env.get_reserved_customers(fleet)
            new=sorted([int(o) for o in visible if not served[o] and int(o) not in reserved
                        and int(o) not in self._accepted and int(o) not in self._rejected],
                       key=lambda o:(float(env.reveal_time[i,o]),o))
            for o in new:
                t0=time.perf_counter();deadline=t0+self.uniform_limit
                self._uniform_admission_started=t0;self._uniform_admission_clock=float(clock)
                saved=copy.deepcopy(self.export_state());old_limit=getattr(self,'time_limit',None)
                if old_limit is not None:self.time_limit=self.uniform_limit
                reason=None;cert_ok=False;cert_kind='original known C0 certify_plan'
                self._uniform_scenario_calls=0
                try:
                    super().on_reveal(env,i,clock,fleet,served,[o])
                    if time.perf_counter()>deadline:raise TimeoutError('legacy admission returned after deadline')
                    if hasattr(self,'sampler') and o in self._accepted and self._uniform_scenario_calls!=20:
                        raise ValueError('accepted SAA action lacks all20 accept/reject branch completions')
                    speed=env.coldchain_contract.units.speed_kmph/env.coldchain_contract.units.distance_km_per_unit
                    if abs(speed-env.tw_speed)>1e-12:raise ValueError('contract/env speed mismatch')
                    cert_ok,_=certify_plan(env,i,clock,fleet,served,self._plan,self.contract,self.budget)
                    if not cert_ok:raise ValueError('known C0 plan not certified')
                    # The complete A1 certificate uses only observed state. Pending
                    # other simultaneous revealed requests are constant temporary
                    # rejections for the certificate, then restored before next order.
                    if clock>0:
                        ctx=getattr(env,'_public_physical_context',None)
                        if ctx is None or ctx[0]!=int(i) or ctx[1]!=float(clock):raise ValueError('exact physical context missing')
                        truly_rejected=set(self._rejected)
                        known={int(x) for x in np.flatnonzero(env.demands[i]>0) if env.reveal_time[i,x]<=clock+1e-6}
                        self._rejected.update(known-self._accepted-self._rejected)
                        try:
                            packet=pf.export_public(env,i,clock,env.event_id,0,fleet,ctx[3],served,sorted(known))
                            cert,_=pf.certificate(packet,packet['plan']);cert_ok=cert['qualified'] is True
                            cert_kind='observed A1/C0 all-fleet immediate-close'
                        finally:self._rejected=truly_rejected
                        if not cert_ok:raise ValueError('actual public physical certificate rejected')
                    if time.perf_counter()>deadline:raise TimeoutError('certification returned after deadline')
                except (ValueError,TimeoutError) as exc:
                    reason=str(exc);self.restore_state(saved);self._rejected.add(o)
                    cert_ok=False
                finally:
                    if old_limit is not None:self.time_limit=old_limit
                self.admission_audit.append(dict(instance=int(i),event_id=int(env.event_id),clock=float(clock),order=o,
                    started=t0,deadline=deadline,wall_seconds=time.perf_counter()-t0,accepted=o in self._accepted,
                    fallback=reason is not None,reason=reason,actual_certified=cert_ok,certificate_kind=cert_kind,
                    completed_scenario_branch_calls=self._uniform_scenario_calls,
                    per_admission_scope=True,all_in_submission_cap=self.uniform_limit))
    return UniformAdmission

def build_matched_replanner(name,scripts,history,contract,budget,cooling_share,capacity=50.,booking_horizon=16.,k=10,sampler=None,arm_seed=7001,uniform_limit=10.):
    pf.bootstrap(scripts)
    if history is None or not len(history):raise ValueError('explicit same nonempty history required')
    if type(k) is not int or k!=10:raise ValueError('registered history/scenario K10 required')
    if isinstance(cooling_share,bool) or not isinstance(cooling_share,Real) or not np.isfinite(cooling_share):raise ValueError('explicit finite cooling_share')
    contract.validate()
    from rolling_optimizer import RollingOptimizeReplanner
    from run_exp_rule_control import UncondDynamicReplanner
    from scenario_saa import SaaReplanner
    base={'strong_rolling':RollingOptimizeReplanner,'own_uncond_dynamic':UncondDynamicReplanner,'saa':SaaReplanner}.get(name)
    if base is None:raise ValueError('unknown baseline')
    common=dict(budget=budget,capacity=capacity,booking_horizon=booking_horizon,contract=contract,cooling_share=cooling_share,hist=history,k=k)
    if name=='strong_rolling':common.update(reject_penalty={0:5.,1:10.,2:15.},time_limit=uniform_limit)
    if name=='saa':
        if sampler is None:raise ValueError('explicit common future sampler required')
        common.update(sampler=sampler,reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=uniform_limit,arm_seed=arm_seed,anytime_vote=False)
    rp=uniform_per_admission(base,scripts)(**common,uniform_limit=uniform_limit)
    rp.fair_identity=dict(history_sha256=history_hash(history),history_days=len(history),k=k,
        cooling_share=float(cooling_share),contract_hash=contract.contract_hash,
        admission_order='reveal_time then stable public order ID',scope='10s EACH admission from entry; certified commit included',
        legacy_compute_kill_qualified=False,baseline_objective='existing nonlearning rolling proxy plus actual physical commit certificate' if name=='strong_rolling' else name)
    return rp
