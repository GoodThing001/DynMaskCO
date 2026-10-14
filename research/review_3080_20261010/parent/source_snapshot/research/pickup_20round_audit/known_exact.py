"""Exact bounded initial-idle, already-revealed pickup subproblem, engineering only.

Input is a finite public packet, never a complete hidden day. A-v1 utility is
unchanged; energy and quality participate through actual C0 hard feasibility.
This is exact over declared finite sequences/subsets, not an online/global oracle.
"""
import argparse
from dataclasses import is_dataclass
import hashlib
from itertools import permutations
import json
from numbers import Real
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts=Path(scripts).resolve()
    for sub in ('simulation','evaluation','coldchain','correction_engineering'):
        sys.path.insert(0,str(scripts/sub))
    return scripts


def numeric(v):
    if isinstance(v,bool) or not isinstance(v,Real) or not np.isfinite(v):
        raise ValueError('finite exact numeric public field required')
    return float(v)


def validate(packet):
    if set(packet) != {'clock','depot','depot_deadline','orders','accepted','baseline','budget'}:
        raise ValueError('only declared public fields allowed; no hidden-future dataset')
    if numeric(packet['clock']) != 0:
        raise ValueError('current exact implementation only initial idle depot at clock0')
    if len(packet['depot']) != 2:
        raise ValueError('depot must have2 coordinates')
    for x in packet['depot']: numeric(x)
    if numeric(packet['budget']) <= 0 or numeric(packet['depot_deadline']) <= 0:
        raise ValueError('positive common energy budget/depot deadline required')
    orders=packet['orders']
    if not isinstance(orders,list) or len(orders)>5:
        raise ValueError('finite exact problem is limited to5 visible orders')
    ids=[]
    for o in orders:
        if set(o) != {'id','reveal','x','y','demand','temp_class','tw_start','tw_end','service_time'}:
            raise ValueError('unknown or missing public order fields')
        if type(o['id']) is not int or o['id']<=0 or type(o['temp_class']) is not int or o['temp_class'] not in (0,1,2):
            raise ValueError('exact ID/class required')
        for k in ('reveal','x','y','demand','tw_start','tw_end','service_time'): numeric(o[k])
        if o['reveal']<0 or o['reveal']>packet['clock']:
            raise ValueError('future order is forbidden')
        if o['demand']<=0 or o['service_time']<0 or o['tw_start']<0 or o['tw_end']<o['tw_start']:
            raise ValueError('invalid visible order constraint')
        ids.append(o['id'])
    if len(set(ids))!=len(ids): raise ValueError('duplicate visible ID')
    for k in ('accepted','baseline'):
        if not isinstance(packet[k],list) or any(type(i) is not int for i in packet[k]) or len(set(packet[k]))!=len(packet[k]):
            raise ValueError('exact unique order IDs required')
        if not set(packet[k]) <= set(ids): raise ValueError('unknown accepted/baseline order')
    if not set(packet['accepted'])<=set(packet['baseline']): raise ValueError('baseline violates accepted commitment')
    return ids


def dataset_from_public(packet,contract):
    validate(packet)
    from coldchain_evaluator_a1 import add_v2_initial_quality
    orders=packet['orders'];n=len(orders)+1
    coords=[packet['depot']]+[[o['x'],o['y']] for o in orders]
    ds=dict(coords=np.array([coords],np.float32),demands=np.array([[0]+[o['demand'] for o in orders]],np.float32),
        temp_class=np.array([[0]+[o['temp_class'] for o in orders]],np.int32),
        tw_start=np.array([[0]+[o['tw_start'] for o in orders]],np.float32),
        tw_end=np.array([[packet['depot_deadline']]+[o['tw_end'] for o in orders]],np.float32),
        reveal_time=np.zeros((1,n),np.float32),
        service_time=np.array([[0]+[o['service_time'] for o in orders]],np.float32))
    return add_v2_initial_quality(ds,contract)


def sources(scripts):
    here=Path(__file__).resolve().parent
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values())
           if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
            if p.is_relative_to(scripts) or p.is_relative_to(here)}


def solve(packet,scripts,out,seconds=10.0):
    started=time.perf_counter();scripts=bootstrap(scripts)
    from scenario_saa import make_c0_contract_v2
    from run_exp_reserve import BOOKING_HORIZON,SPEED_KMH,KM_PER_UNIT
    from strict_online_env import StrictOnlineEnv,Replanner
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from audit_io import Journal,encode,decode
    from pickup_trace_runner import replay
    from run_identity import dataset_identity
    ids=validate(packet);numeric(seconds)
    if seconds<=0 or seconds>10: raise ValueError('all-in exact solve cap must be <=10s')
    contract=make_c0_contract_v2();ds=dataset_from_public(packet,contract)
    local={oid:i+1 for i,oid in enumerate(ids)};mandatory=set(packet['accepted'])
    deadline=started+seconds
    class FixedVisibleRoute(Replanner):
        def __init__(self,seq): self.seq=[local[i] for i in seq]
        def plan(self,env,inst_idx,clock,vehicles,served_mask,visible_ids,replan_ids=None):
            # Refuse future/private queries: the schedule names only public packet IDs.
            for v in vehicles:
                if v.status in ('idle','ready'):
                    v.mutable_suffix=[i for i in self.seq if not served_mask[i]]+[0]
    def evaluate(seq):
        rp=FixedVisibleRoute(seq)
        env=StrictOnlineEnv(ds,capacity=50,tw_speed=SPEED_KMH/KM_PER_UNIT,num_vehicles=1,
            replanner=rp,coldchain_contract=contract,booking_horizon=BOOKING_HORIZON)
        traces,_=env.run(0)
        accept={local[i] for i in seq};reject=set(range(1,len(ids)+1))-accept
        ev=evaluate_trace_a1(traces,{k:v[0] for k,v in ds.items()},contract,accept,reject,
                              packet['budget'],{0:5.,1:10.,2:15.})
        return ev,traces,accept,reject
    source_start=sources(scripts)
    base,base_trace,base_accept,base_reject=evaluate(packet['baseline'])
    rows=[];best=None;complete=True
    for length in range(len(ids)+1):
        for seq in permutations(ids,length):
            if not mandatory<=set(seq):continue
            if time.perf_counter()>deadline: complete=False;break
            ev,tr,ac,rj=evaluate(seq)
            rows.append(dict(sequence=list(seq),evaluation=ev))
            if ev['hard_feasible'] is True and (best is None or ev['utility']>best[1]['utility']+1e-10):
                best=(seq,ev,tr,ac,rj)
        if not complete:break
    # No exact/upper-bound claim if enumeration or certification crosses deadline.
    wall=time.perf_counter()-started
    complete=complete and wall<=seconds
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def save_trace(filename,traces,accepted,rejected,evaluation):
        journal=Journal(out/filename)
        journal.write('run_start',dataset=ds,contract=contract,budget=packet['budget'],
            penalties={0:5.,1:10.,2:15.})
        journal.write('source_start',sha256=source_start)
        journal.write('day_end',day_id=0,traces=traces,accepted=accepted,rejected=rejected,evaluation=evaluation)
        journal.write('source_end',sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_start})
        journal.write('run_end',days=1,formal_eligible=False)
        journal.close()
    save_trace('baseline_trace.jsonl',base_trace,base_accept,base_reject,base)
    if best is not None:save_trace('best_trace.jsonl',best[2],best[3],best[4],best[1])
    independent=replay(out/'best_trace.jsonl') if best is not None else None
    result=dict(schema='pickup-known-exact-engineering-v1',scope='initial idle one-vehicle currently revealed finite packet only',
        exact_within_declared_subproblem=complete,enumerated=len(rows),wall_seconds=wall,budget=packet['budget'],
        contract_hash=contract.contract_hash,dataset_sha256=dataset_identity(ds),source_start=source_start,
        baseline_sequence=packet['baseline'],baseline_evaluation=base,
        best_sequence=list(best[0]) if best else None,best_evaluation=best[1] if best else None,
        utility_gain=(best[1]['utility']-base['utility']) if best and base['hard_feasible'] else None,
        independent_replay=independent,baseline_independent_replay=replay(out/'baseline_trace.jsonl'),
        formal_online_gain_established=False,global_optimality_claim=False,
        physical_calibration_claim=False,public_packet=packet,all_candidates=rows)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def fixtures(budget):
    def order(i,x,y,tw=0,c=0):
        return dict(id=i,reveal=0.,x=x,y=y,demand=2.,temp_class=c,tw_start=tw,tw_end=20.,service_time=.05)
    common=dict(clock=0.,depot=[0.,0.],depot_deadline=22.,accepted=[],budget=budget)
    return [dict(common,orders=[order(1,.9,.9,c=1),order(2,.1,.1,c=0),order(3,.9,.1,c=2)],baseline=[1,2,3]),
            dict(common,accepted=[1,2],orders=[order(1,.1,0.,tw=4),order(2,2.,0.)],baseline=[1,2])]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=solve(json.loads(a.input.read_text(encoding='utf-8')),a.scripts,a.out)
    print(json.dumps({k:r[k] for k in ('exact_within_declared_subproblem','enumerated','utility_gain','global_optimality_claim')}))
