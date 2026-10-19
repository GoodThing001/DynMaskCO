"""Synthetic observer acceptance; no real SOURCE runs, models or label calls."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experimental'))
import p2_pos01c_opportunity as opp
from p2_pos01c_gap import CapacityConditionedProposal
from test_p2_pos01c_gap import synthetic_view, history


@pytest.mark.parametrize('n',[2,6])
def test_raw_proposal_expansion_matches_real_pool_without_mutation(n):
    np,t,r,_,_=opp.imports()
    view=synthetic_view(n=n)
    before=r.identity(view)
    row=opp.raw_mask_proposals(view,r.plans_from_view(view))
    assert row['masks']==r.mask_pool(view,r.plans_from_view(view),seed=0)
    assert row['distinct']==(1 if n==2 else len(row['masks']))
    assert row['nonempty_raw']-row['distinct']==row['duplicates']
    assert r.identity(view)==before


def test_post_booking_natural_phase_has_empty_history_suffix_without_clock_clamp():
    view=replace(synthetic_view(),clock=17.)
    h=CapacityConditionedProposal(history())
    audit=opp.support(view,h)
    assert audit['suffix_counts']==[0]*12
    assert len(audit['eligible'])==10
    assert audit['free_slots']==44


def test_unchanged_real_saa_synthetic_path_all_natural_phases_and_no_forbidden_calls(monkeypatch):
    from test_p2_pos01_reference import saa_synthetic_dataset
    np,t,r,Proposal,_=opp.imports()
    def forbidden(*a,**k):
        raise AssertionError('forbidden reconstruction/continuation/evaluation call')
    for name in ('prepare_mask_actions','make_future_tapes','run_branch','evaluate_trace_a1'):
        monkeypatch.setattr(t,name,forbidden)
    source,spec=saa_synthetic_dataset()
    policy=t.PublicSaaReplanner(spec)
    env=t.SaaPhaseEnv(source,50.,1.5,num_vehicles=2,replanner=policy,
        coldchain_contract=spec.contract,booking_horizon=spec.booking_horizon)
    h=Proposal(spec.history,10)
    events=[]
    def hook(engine,view,context):
        before=r.identity([view,policy.export_state()])
        events.append(opp.observe(engine,view,context,policy,h,['SYNTHETIC0','SYNTHETIC1']))
        assert r.identity([view,policy.export_state()])==before
    env.p2_capture=hook
    traces,served=env.run(0)
    assert len(events)==env.replan_count>2
    assert any(e['masks']['distinct']==0 for e in events)
    assert any(e['multi'] for e in events)
    assert all(e['immutable_observer'] for e in events)
    assert {s.node for tr in traces for s in tr.services}==policy._accepted


def row(distinct,supported,p0=True):
    return dict(status='COMPLETE',masks={'distinct':distinct},multi=distinct>=2,
        supported=supported,joint=distinct>=2 and supported,
        reference_usable=distinct>=2 and supported and p0,P0_reference_check={'ok':p0},
        future_support={'status':'REFERENCE_SUPPORT_PASS' if supported else 'NO_SUPPORT'})


def test_denominators_include_zero_masks_reference_rejects_and_unfinished_attempts():
    rows={(0,0):row(0,False),(0,1):row(1,True),(0,2):row(2,True,False),
          (1,0):row(0,True),(1,1):{'status':'UNFINISHED'}}
    summary=opp.summaries(rows,[{'status':'COMPLETE'},{'status':'INCOMPLETE'}])
    a,b=summary
    assert a['observed_denominator']==3
    assert a['rates']=={'p_multi':1/3,'p_support':2/3,'p_joint':1/3}
    assert a['counts']['reference_usable_joint']==0
    assert a['counts']['P0_reference_rejections']==1
    assert b['observed_denominator']==2 and b['counts']['incomplete']==1
    assert all(v is None for v in b['rates'].values())
    assert b['unobserved_future_phase_count']=='UNKNOWN_NOT_ZERO'


def test_absent_day_is_not_a_zero_opportunity_day():
    summaries=opp.summaries({},[{'status':'NOT_ATTEMPTED'},{'status':'NOT_ATTEMPTED'}])
    assert all(s['rates']['p_joint'] is None for s in summaries)
    assert all(s['observed_denominator']==0 and not s['complete_root_day'] for s in summaries)
