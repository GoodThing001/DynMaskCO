"""CTRL-01 synthetic acceptance; no real root execution, labels or training."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experimental'))
import p2_pos01c_control as ctrl
from test_p2_pos01_reference import saa_synthetic_dataset
from test_p2_pos01c_gap import synthetic_view


def make_case(data=None,spec=None):
    np,t,r,_,_=ctrl.imports()
    if data is None:
        data,spec=saa_synthetic_dataset()
    p=ctrl.myopic_class()(spec)
    e=t.SaaPhaseEnv(data,50.,1.5,num_vehicles=2,replanner=p,
                   coldchain_contract=spec.contract,booking_horizon=spec.booking_horizon)
    return p,e


def current_plan(p,e):
    from strict_online_env import VehicleState
    vs=[VehicleState(0),VehicleState(1)]
    served=np.zeros(51,dtype=bool);served[0]=True
    p.plan(e,0,.05,vs,served,[],replan_ids={0,1})
    return vs,served


def test_exact_shared_insert_and_both_checks_are_inherited():
    _,t,_,_,_=ctrl.imports()
    p,e=make_case()
    assert p._try_insert_certified.__func__ is t.PublicSaaReplanner._try_insert_certified
    assert p.plan.__func__ is t.PublicSaaReplanner.plan
    assert p._try_insert.__func__ is t.PublicSaaReplanner._try_insert
    assert p._route_feasible.__func__ is t.PublicSaaReplanner._route_feasible


def test_actual_natural_engine_no_saa_votes_forecast_draws_or_mask_repair(monkeypatch):
    _,t,_,Proposal,_=ctrl.imports()
    p,e=make_case()
    def forbidden(*a,**k):
        raise AssertionError('forbidden future vote/repair/label path')
    p.sampler.sample=forbidden;p._saa_decide=forbidden
    for name in ('prepare_mask_actions','make_future_tapes','run_branch','evaluate_trace_a1'):
        monkeypatch.setattr(t,name,forbidden)
    events=[];sampler=Proposal(p.spec.history,10)
    def hook(engine,view,context):
        events.append(ctrl.opp.observe(engine,view,context,p,sampler,['SYNTH0','SYNTH1']))
    e.p2_capture=hook
    traces,served=e.run(0)
    assert len(events)==e.replan_count>2
    assert all(x['P0_reference_check']['ok'] for x in events)
    assert p.draw_audit==[] and p.admission_count==8
    assert {s.node for tr in traces for s in tr.services}==p._accepted
    assert all(tr.final_coldchain_state.closed for tr in traces if tr.dispatch_time is not None)


def test_hidden_source_attributes_and_reveal_times_do_not_change_current_plan():
    data,spec=saa_synthetic_dataset()
    changed={k:v.copy() for k,v in data.items()}
    hidden=changed['reveal_time'][0]>.05+1e-6;hidden[0]=False
    changed['coords'][0,hidden]=999.
    changed['demands'][0,hidden]=49.
    changed['temp_class'][0,hidden]=2
    changed['reveal_time'][0,hidden]=.49
    a,ea=make_case(data,spec);b,eb=make_case(changed,spec)
    va,_=current_plan(a,ea);vb,_=current_plan(b,eb)
    assert a._plan==b._plan and a._accepted==b._accepted and a._rejected==b._rejected
    assert [v.mutable_suffix for v in va]==[v.mutable_suffix for v in vb]
    assert [d['reason'] for d in a.decisions]==[d['reason'] for d in b.decisions]


def test_failed_common_check_rolls_back_every_order_without_partial_acceptance():
    p,e=make_case()
    def bad_insert(proxy,inst,o,*args):
        p._plan={0:[o]}
        return False
    p._try_insert_certified=bad_insert
    current_plan(p,e)
    assert p._plan=={} and p._accepted==set() and len(p._rejected)==6
    assert all(d['reason']=='SHARED_INSERT_OR_CERT_REJECT' for d in p.decisions)


def test_event_deadline_post_insert_rejection_is_atomic(monkeypatch):
    p,e=make_case()
    clock={'after':False}
    def fake_time():
        return 7. if clock['after'] else 0.
    def slow_insert(proxy,inst,o,*args):
        p._plan={0:[o]};clock['after']=True
        return True
    p._try_insert_certified=slow_insert
    monkeypatch.setattr(ctrl.time,'perf_counter',fake_time)
    current_plan(p,e)
    assert not p._accepted and p._plan=={} and len(p._rejected)==6
    assert p.decisions[0]['reason']=='EVENT_TIMEOUT_AFTER_INSERT'
    assert p.timeouts==6


def test_exact_capacity_rejected_by_real_shared_conservative_gate():
    data,spec=saa_synthetic_dataset()
    data=copy.deepcopy(data)
    data['demands'][0,:]=0.;data['demands'][0,1]=50.
    p,e=make_case(data,spec)
    current_plan(p,e)
    assert p._accepted==set() and p._rejected=={1} and p._plan=={}


def test_occupancy_distinguishes_provisional_and_legal_mutable_domain():
    from dataclasses import replace
    view=synthetic_view()
    full=ctrl.occupancy(view)
    protected=ctrl.occupancy(replace(view,protected=(0,1)))
    assert full['total_mutable_customers']==6
    assert full['provisional_suffix_lengths']==protected['provisional_suffix_lengths']
    assert protected['total_mutable_customers']==0


def test_incomplete_day_not_eligible_for_complete_root_comparison():
    summaries=ctrl.summaries({},[{'status':'INCOMPLETE'},{'status':'NOT_ATTEMPTED'}])
    assert all(s['rates']['p_joint'] is None for s in summaries)
    assert all(not s['complete_root_day'] for s in summaries)
