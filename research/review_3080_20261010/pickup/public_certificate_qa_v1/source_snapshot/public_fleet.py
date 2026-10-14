"""Public-only two-vehicle reallocation certificate. No future J feeds the actor.

The private resumable snapshot belongs to the evaluator; the actor receives a
public projection containing observed physics and revealed orders only.
"""
import argparse
import copy
import hashlib
import itertools
import json
from numbers import Real
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np


def bootstrap(scripts):
    scripts=Path(scripts).resolve()
    for sub in ('evaluation','simulation','coldchain','correction_engineering'):
        sys.path.insert(0,str(scripts/sub))
    return scripts


def digest(value):
    from audit_io import encode
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes(scripts):
    here=Path(__file__).resolve().parent
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values())
           if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
    paths.add(Path(__file__).resolve())
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
            if p.is_relative_to(scripts) or p.is_relative_to(here)}


def export_public(env,i,clock,event,reveal,fleet,traces,served,universe):
    """Construct an actor input without unknown attributes/counts or full-day IDs.

    Original visible IDs are retained as public identities. Array width derives
    only from their maximum, not from the total hidden order count.
    """
    from recourse_snapshot import capture_recourse_snapshot,snapshot_state_hash
    rp=env.replanner
    known=[int(o) for o in universe if env.reveal_time[i,o]<=clock+1e-6]
    visible=set(known)
    if not visible or not float(clock)>0: raise ValueError('nonzero observed decision clock required')
    n=max(known)+1
    ds={}
    # Only explicit observed data fields; do not forward derived private matrices.
    fields=('coords','demands','temp_class','tw_start','tw_end','service_time','reveal_time','initial_quality')
    for key in fields:
        if key not in env.dataset: continue
        value=env.dataset[key]
        shape=(1,n)+value.shape[2:]
        dst=np.zeros(shape,dtype=value.dtype)
        dst[0,0]=value[i,0]
        for o in known: dst[0,o]=value[i,o]
        ds[key]=dst
    # capture records no unknown feature arrays, then explicitly project masks
    # and universe. Replanner state has only previously accepted/rejected IDs.
    snap=capture_recourse_snapshot(env,i,clock,event,reveal,fleet,traces,served,known)
    snap['instance_id']=0
    snap['num_nodes']=n
    snap['served_mask']=np.asarray(served[:n],bool).copy()
    snap['visible_mask']=np.zeros(n,bool);snap['visible_mask'][[0]+known]=True
    snap['customer_universe']=np.asarray(known,np.int32)
    snap['force_suffix']=[]
    state=snap['replanner_state']
    if state is None or state.get('version')!='c0-reserve-v1':
        raise ValueError('supported accepted-ownership state required')
    state['status_inst']=0
    for key in ('accepted','rejected'):
        if not set(state[key])<=visible: raise ValueError('private or unknown policy ID')
    if set(state['accepted'])|set(state['rejected']) != visible:
        raise ValueError('every revealed order must already have admission decision')
    plans={}
    ownership={}
    for v in fleet:
        route=[int(o) for o in rp._plan.get(v.vehicle_id,[]) if not served[o]]
        if v.status=='committed' and v.committed_next not in (None,0):
            nxt=int(v.committed_next)
            route=[nxt]+[o for o in route if o!=nxt]
        if not set(route)<=visible: raise ValueError('unknown suffix forbidden')
        for o in route:
            if o in ownership and ownership[o]!=v.vehicle_id: raise ValueError('duplicate ownership')
            ownership[o]=v.vehicle_id
        plans[str(v.vehicle_id)]=route
    outstanding=set(state['accepted'])-set(np.flatnonzero(snap['served_mask']))
    if set(ownership)!=outstanding: raise ValueError('accepted outstanding orders must have exact owner')
    state['plan']=plans
    snap['state_hash']=snapshot_state_hash(snap)
    physical=[]
    for v in fleet:
        origin=np.asarray(env.coords[i,v.current_node],float)
        target=v.committed_next if v.status in ('committed','returning') else None
        if target is not None:
            arrival=v.committed_arrive if v.status=='committed' else v.return_finish
            duration=float(arrival)-float(v.ready_time)
            progress=1. if duration<=0 else float(np.clip((clock-v.ready_time)/duration,0.,1.))
            pos=origin+progress*(np.asarray(env.coords[i,target],float)-origin)
        else:pos=origin
        physical.append(dict(vehicle_id=int(v.vehicle_id),position_xy=pos.tolist(),
            position_semantics='constant-speed interpolation of immutable committed leg',
            energy_kwh=0. if v.coldchain_state is None else float(v.coldchain_state.cumulative_energy_kwh)))
    return dict(schema='pickup-public-midcourse-v1',dataset=ds,snapshot=snap,
        plan=plans,ownership={str(o):v for o,v in ownership.items()},
        observed_vehicle_physics=physical,
        budget=float(rp.budget),capacity=float(env.capacity),tw_speed=float(env.tw_speed),
        booking_horizon=float(env.booking_horizon),contract=env.coldchain_contract,
        contract_hash=env.coldchain_contract.contract_hash,
        weight_hashes={},weight_scope='no learned model in this engineering actor',
        penalties={0:5.,1:10.,2:15.})


def select_vehicle(packet):
    snap=packet['snapshot'];served=snap['served_mask']
    for vid,status in enumerate(snap['vehicle_status']):
        if status not in ('idle','ready','committed'): continue
        cn=int(snap['committed_next'][vid]) if status=='committed' else None
        tail=[o for o in packet['plan'][str(vid)] if not served[o] and o!=cn]
        if 2<=len(tail)<=5: return vid,tail
    return None,[]


def evaluate_known(packet,plan):
    """Actual observed cargo/thermal state, all fleet energy, known-only closing.

    Forced completion immediately returns after the known accepted routes. This
    is a declared surrogate, not a prediction of future SAA's waiting/admissions.
    """
    from strict_online_env import StrictOnlineEnv,Replanner
    from recourse_snapshot import clone_snapshot,snapshot_state_hash
    from coldchain_evaluator_a1 import evaluate_trace_a1
    contract=packet['contract'];contract.validate()
    if contract.contract_hash!=packet['contract_hash']: raise ValueError('contract identity mismatch')
    speed=contract.units.speed_kmph/contract.units.distance_km_per_unit
    if abs(speed-packet['tw_speed'])>1e-12:raise ValueError('C0 unit speed mismatches physical env')
    snap=clone_snapshot(packet['snapshot'])
    class KnownOwned(Replanner):
        def plan(self,env,inst_idx,clock,vehicles,served_mask,visible_ids,replan_ids=None):
            for v in vehicles:
                if v.status in ('idle','ready') and (replan_ids is None or v.vehicle_id in replan_ids):
                    tail=[o for o in plan[str(v.vehicle_id)] if not served_mask[o]]
                    v.mutable_suffix=tail+[0]
    # Install pending known routes without changing the committed destination.
    width=max([len(v) for v in plan.values()]+[1])+1
    tails=np.full((snap['num_vehicles'],width),-1,np.int32);lens=[]
    for vid,status in enumerate(snap['vehicle_status']):
        cn=int(snap['committed_next'][vid]) if status=='committed' else None
        tail=[o for o in plan[str(vid)] if not snap['served_mask'][o] and o!=cn]
        if status in ('returning','closed'): tail=[]
        if status=='committed': tail=tail+[0]
        tails[vid,:len(tail)]=tail;lens.append(len(tail))
    snap['mutable_suffix']=tails;snap['mutable_suffix_len']=np.array(lens,np.int32)
    snap['state_hash']=snapshot_state_hash(snap)
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],
        snap['num_vehicles'],KnownOwned(),contract,packet['booking_horizon'])
    tr,_=env.run_resumed(snap)
    state=snap['replanner_state']
    ev=evaluate_trace_a1(tr,{k:v[0] for k,v in packet['dataset'].items()},contract,
        set(state['accepted']),set(state['rejected']),packet['budget'],packet['penalties'])
    return ev,tr


def validate_public(packet):
    snap=packet['snapshot'];ds=packet['dataset']
    known=set(int(o) for o in snap['customer_universe'])
    if set(ds)-{'coords','demands','temp_class','tw_start','tw_end','service_time','reveal_time','initial_quality'}:
        raise ValueError('private/unknown dataset channel')
    if set(np.flatnonzero(ds['demands'][0]>0))!=known:raise ValueError('private or missing order')
    if any(ds['reveal_time'][0,o]>snap['clock']+1e-6 for o in known):raise ValueError('unrevealed order')
    expected=np.zeros(snap['num_nodes'],bool);expected[[0]+sorted(known)]=True
    if not np.array_equal(expected,snap['visible_mask']):raise ValueError('visibility mismatch')
    state=snap['replanner_state']
    if set(state['accepted'])|set(state['rejected'])!=known:raise ValueError('incomplete public admissions')
    if set(state['accepted'])&set(state['rejected']):raise ValueError('overlapping public admissions')
    if packet['contract'].contract_hash!=packet['contract_hash']:raise ValueError('contract changed')
    if abs(packet['contract'].units.speed_kmph/packet['contract'].units.distance_km_per_unit-packet['tw_speed'])>1e-12:
        raise ValueError('C0 unit speed mismatches physical env')


def family(packet,max_orders=5):
    """Deterministic small two-vehicle action family, not a global fleet oracle.

    Pool is first <=5 mutable owned known IDs among the first two open vehicles.
    Non-pool accepted orders stay on their owner in their original relative order;
    each assigned pool is appended after them. Incumbent is explicitly included.
    """
    snap=packet['snapshot'];open_ids=[v for v,s in enumerate(snap['vehicle_status'])
        if s in ('idle','ready','committed')]
    if len(open_ids)<2:return [],[],[]
    # Prefer a nonempty owner, then the earliest other open vehicle.
    owners=[v for v in open_ids if packet['plan'][str(v)]]
    if not owners:return [],[],[]
    vids=[owners[0],next(v for v in open_ids if v!=owners[0])]
    pool=[]
    for vid in vids:
        cn=int(snap['committed_next'][vid]) if snap['vehicle_status'][vid]=='committed' else None
        pool.extend(o for o in packet['plan'][str(vid)] if o!=cn and not snap['served_mask'][o])
    pool=pool[:max_orders]
    return vids,pool,[v for v in packet['plan'][str(vids[0])] if v not in pool]


def certificate(packet,plan):
    snap=packet['snapshot'];accepted=set(snap['replanner_state']['accepted'])
    outstanding=accepted-set(np.flatnonzero(snap['served_mask']))
    flat=[o for route in plan.values() for o in route]
    if len(flat)!=len(set(flat)) or set(flat)!=outstanding:raise ValueError('accepted service commitment violated')
    if set(plan)!=set(packet['plan']):raise ValueError('fleet changed')
    for vid,status in enumerate(snap['vehicle_status']):
        cn=int(snap['committed_next'][vid])
        if status=='committed' and cn>0 and (not plan[str(vid)] or plan[str(vid)][0]!=cn):
            raise ValueError('immutable committed owner/destination violated')
        if status in ('returning','closed') and plan[str(vid)]!=packet['plan'][str(vid)]:
            raise ValueError('unavailable vehicle reassigned')
    ev,tr=evaluate_known(packet,plan)
    return dict(qualified=ev['hard_feasible'] is True,evaluation=ev,
        all_fleet_energy_accounted=True,accepted_unchanged=True,committed_unchanged=True),tr


def solve_public(packet,scripts):
    bootstrap(scripts)
    from audit_io import encode
    validate_public(packet)
    vids,pool,_=family(packet)
    if len(pool)<2:return dict(qualified=False,reason='no eligible two-vehicle known pool',packet_sha=digest(packet))
    baseline_cert,_=certificate(packet,packet['plan']);baseline=baseline_cert['evaluation']
    source_start=source_hashes(Path(scripts).resolve())
    best=copy.deepcopy(packet['plan']);best_ev=baseline;rows=[]
    for perm in itertools.permutations(pool):
        for cut in range(len(pool)+1):
            plan=copy.deepcopy(packet['plan'])
            for vid,assigned in zip(vids,(perm[:cut],perm[cut:])):
                plan[str(vid)]=[o for o in plan[str(vid)] if o not in pool]+list(assigned)
            try:
                cert,_=certificate(packet,plan);ev=cert['evaluation']
                rows.append(dict(plan=plan,certificate=cert))
                if cert['qualified'] is True and (best_ev['hard_feasible'] is not True or
                        ev['utility']>best_ev['utility']+1e-10): best,best_ev=plan,ev
            except ValueError as exc: rows.append(dict(plan=plan,failure=str(exc)))
    source_end={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_start}
    if source_start!=source_end: raise ValueError('runtime source changed')
    return dict(qualified=best_ev['hard_feasible'] is True and baseline['hard_feasible'] is True,
        packet_sha=digest(packet),vehicle_ids=vids,pool=pool,selected_plan=best,baseline_known=baseline,
        selected_known=best_ev,candidates=rows,source_start=source_start,source_end=source_end,
        exact_scope='declared two-vehicle pool ordered-partitions; incumbent included; fixed admissions/nonpool order',
        committed_next_frozen=True,formal_online_gain_established=False)


def bounded_select(packet,scripts,seconds=10.0,started=None):
    """Subprocess wall deadline includes process startup, scoring, and validation.

    Result received after the submission deadline is rejected. OS termination
    cleanup may add wall latency; no late route is applied to the parent state.
    """
    t0=time.perf_counter() if started is None else started
    if isinstance(seconds,bool) or not isinstance(seconds,Real) or not np.isfinite(seconds) or not 0<seconds<=10:
        raise ValueError('finite all-in budget <=10 seconds required')
    if not isinstance(t0,Real) or not np.isfinite(t0):raise ValueError('finite monotonic start required')
    deadline=t0+seconds;baseline=copy.deepcopy(packet['plan'])
    result=dict(qualified=False,selected_plan=baseline,timeout=False,fallback=True)
    with tempfile.TemporaryDirectory(prefix='pickup-public-branch-') as tmp:
        inp=Path(tmp)/'public.pkl';out=Path(tmp)/'decision.json'
        inp.write_bytes(pickle.dumps(packet,protocol=4))
        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
            '--scripts',str(scripts),'--input',str(inp),'--output',str(out)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        remaining=deadline-time.perf_counter()
        try:
            stdout,stderr=proc.communicate(timeout=max(0,remaining))
            if proc.returncode!=0: result['failure']=stderr.decode(errors='replace')[-2000:]
            elif out.exists():
                candidate=json.loads(out.read_text())
                identity=candidate.get('packet_sha')==digest(packet)
                sources_start=candidate.get('source_start',{})
                sources_ok=bool(sources_start) and sources_start==candidate.get('source_end') and all(
                    Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                    for p,h in sources_start.items())
                # Strict bool qualification, and deadline checked after scoring/certification.
                if candidate.get('qualified') is True and identity and sources_ok and time.perf_counter()<=deadline:
                    result=candidate;result.update(fallback=False,timeout=False)
                else: result['reason']='qualification, identity, or deadline failed'
        except subprocess.TimeoutExpired:
            proc.kill();proc.communicate();result['timeout']=True
        if time.perf_counter()>deadline:
            result=dict(qualified=False,selected_plan=baseline,timeout=True,fallback=True)
    result['wall_seconds']=time.perf_counter()-t0
    result['deadline_monotonic']=deadline
    return result


def install_in_live(rp,packet,result):
    """After selection and before ordinary plan(): only update the owned order.

    Caller must export a fresh packet from unchanged live state and compare its
    hash before calling; committed legs, admissions and vehicle ownership persist.
    """
    deadline=result.get('deadline_monotonic')
    if not isinstance(deadline,Real) or not np.isfinite(deadline) or time.perf_counter()>deadline:return False
    if result.get('qualified') is not True or result.get('packet_sha')!=digest(packet):return False
    selected=result['selected_plan'];old=packet['plan']
    if set(selected)!=set(old):raise ValueError('fleet membership changed')
    for vid in old:
        if int(vid) not in result['vehicle_ids'] and selected[vid]!=old[vid]:raise ValueError('other vehicle route changed')
        status=packet['snapshot']['vehicle_status'][int(vid)]
        cn=int(packet['snapshot']['committed_next'][int(vid)])
        if status=='committed' and cn>0 and selected[vid][0]!=cn:raise ValueError('committed destination changed')
    cert,_=certificate(packet,selected)
    if cert['qualified'] is not True or time.perf_counter()>deadline:return False
    rp._plan={int(k):list(v) for k,v in selected.items()}
    return True


def intervention_hook(scripts,log):
    """Attach to env.snapshot_hook for one first eligible online decision.

    Existing hook may be chained by the caller. This is not a trained MaskCO actor.
    """
    done={'value':False}
    def hook(env,i,clock,event,reveal,fleet,traces,served,universe):
        if done['value'] or clock<=0:return
        if not any(v.status in ('idle','ready') and v.needs_replan for v in fleet):return
        rp=env.replanner
        if not getattr(rp,'_suffix_timing_instrumented',False):
            raise ValueError('instrument SAA timing with shared_budget_replanner before attaching hook')
        # Share the existing last admission's all-in budget at a reveal event.
        # Route-only decisions have their own original planning-event budget.
        t0=(rp._suffix_decision_started if getattr(rp,'_suffix_decision_clock',None)==float(clock)
            else time.perf_counter())
        try:packet=export_public(env,i,clock,event,reveal,fleet,traces,served,universe)
        except ValueError:return
        if len(family(packet)[1])<2:return
        done['value']=True
        result=bounded_select(packet,scripts,started=t0)
        fresh=export_public(env,i,clock,event,reveal,fleet,traces,served,universe)
        if digest(fresh)!=digest(packet) or time.perf_counter()-t0>10:
            result.update(qualified=False,fallback=True)
        installed=install_in_live(env.replanner,packet,result)
        log.append(dict(clock=clock,event=event,public_packet=packet,selection=result,installed=installed))
    return hook


def shared_budget_replanner(base_class):
    """Instrument rather than reset the existing admission-decision timer.

    Subclass SaaReplanner (or an I3 SAA subclass) without altering its scenario
    law, weights, admissions, or K. The suffix uses only remaining time from the
    final same-clock admission; previously processed orders retain their existing
    per-order budgets. This scope must be declared in the paired online protocol.
    """
    class SharedBudget(base_class):
        _suffix_timing_instrumented=True
        def on_reveal(self,env,i,clock,fleet,served,visible):
            self._suffix_decision_started=time.perf_counter()
            self._suffix_decision_clock=float(clock)
            return super().on_reveal(env,i,clock,fleet,served,visible)
        def _saa_decide(self,env,i,clock,fleet,served,order,scenarios,t0):
            self._suffix_decision_started=float(t0)
            self._suffix_decision_clock=float(clock)
            return super()._saa_decide(env,i,clock,fleet,served,order,scenarios,t0)
    return SharedBudget


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true')
    ap.add_argument('--scripts',required=True);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    bootstrap(a.scripts)
    packet=pickle.loads(a.input.read_bytes())
    r=solve_public(packet,a.scripts)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
