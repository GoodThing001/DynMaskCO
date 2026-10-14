"""Pre-registered full-day/all-vehicle diagnostic of four already observed days.

No model, no mask generator, no numeric improvement claim. Every snapshot hook
event and every vehicle is recorded; no min-clock filter or capture early return.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import collections
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts=Path(scripts).resolve()
    for folder in ('evaluation','simulation','coldchain','correction_engineering'):
        sys.path.insert(0,str(scripts/folder))
    return scripts


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(scripts):
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values()) if
        getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
    paths.add(Path(__file__).resolve())
    return {str(p):sha(p) for p in sorted(paths) if p.is_relative_to(scripts) or p==Path(__file__).resolve()}


def run(scripts,out):
    started=time.perf_counter();scripts=bootstrap(scripts);out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    from coldchain_contract import ColdChainContractV2
    from coldchain_evaluator_a1 import add_v2_initial_quality,evaluate_trace_a1
    from run_exp_energy_c0 import BOOKING_HORIZON,KM_PER_UNIT,SPEED_KMH,generate_dataset
    from run_exp_rule_control import UncondDynamicReplanner
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import capture_recourse_snapshot
    from audit_io import Journal,encode
    from run_identity import dataset_identity
    from pickup_trace_runner import replay
    contract=ColdChainContractV2();contract.validate()
    protocol=dict(protocol_id='P-FULL-DAY-ALL-FLEET-CENSUS-DIAGNOSTIC-v1',
        registration_actual=datetime.now(timezone.utc).isoformat(),purpose='diagnose already observed StageA days; no fresh gain inference',
        data_seed=2026120201,history_seed=2026120202,instances=4,orders=200,vehicles=15,
        B=710.6548152249378,capacity=50.,cooling_share=3.7125478804082377,
        contract_constructor='ColdChainContractV2() as original StageA; not silently corrected',
        contract_hash=contract.contract_hash,per_event='every original snapshot hook, initial clock0 through terminal',
        per_vehicle='all15 at each event; exact day/event/vehicle keys; no early return',
        histograms=['mutable visible suffix','accepted owned unserved tail excluding committed','fleet jointly owned tail'],
        protected_final_seeds=[20260927,20260928],no_model=True,no_new_gain_claim=True)
    manifest=sources(scripts)
    # Genuine diagnostic registration before regenerating the already viewed days.
    (out/'DIAGNOSTIC_REGISTRATION.json').write_text(json.dumps(dict(protocol=protocol,source_start=manifest),indent=2),encoding='utf-8')
    train=generate_dataset(1,200,2026120202)
    ds=add_v2_initial_quality(generate_dataset(4,200,2026120201),contract)
    dem=train['demands'][0];mask=dem[1:]>0
    hist=[(train['reveal_time'][0][1:][mask],train['temp_class'][0][1:][mask].astype(int))]
    journal=Journal(out/'full_day_all_fleet_trace.jsonl')
    journal.write('run_start',dataset=ds,contract=contract,budget=protocol['B'],penalties={0:5.,1:10.,2:15.},
        diagnostic_only=True,registration_sha256=sha(out/'DIAGNOSTIC_REGISTRATION.json'))
    journal.write('source_start',sha256=manifest)
    mutable_hist=collections.Counter();owned_hist=collections.Counter();joint_hist=collections.Counter()
    eligible_mutable=collections.Counter();eligible_owned=collections.Counter()
    records=[];keys=set();day_rows=[];first_public_capture=None
    try:
        for day in range(4):
            rp=UncondDynamicReplanner(protocol['B'],50.,BOOKING_HORIZON,contract,protocol['cooling_share'],hist,k=10)
            env=StrictOnlineEnv(ds,50.,SPEED_KMH/KM_PER_UNIT,15,rp,contract,BOOKING_HORIZON)
            count=[0]
            def hook(e,i,clock,event,reveal,fleet,traces,served,universe):
                nonlocal first_public_capture
                count[0]+=1;union=set();rows=[]
                replan_trigger=any(v.status in ('idle','ready') and v.needs_replan for v in fleet)
                for v in fleet:
                    key=(day,int(event),int(v.vehicle_id))
                    if key in keys:raise ValueError('duplicate day/event/vehicle key')
                    keys.add(key)
                    committed=v.committed_next if v.status=='committed' else None
                    mutable=[int(o) for o in v.mutable_suffix if o>0 and not served[o] and e.reveal_time[i,o]<=clock+1e-6]
                    owned=[int(o) for o in rp._plan.get(v.vehicle_id,[]) if not served[o] and o!=committed
                           and e.reveal_time[i,o]<=clock+1e-6]
                    if len(owned)!=len(set(owned)):raise ValueError('duplicate owned suffix')
                    if union & set(owned):raise ValueError('cross-vehicle duplicate ownership')
                    union.update(owned)
                    row=dict(day=day,event_id=int(event),vehicle_id=int(v.vehicle_id),clock=float(clock),
                        status=v.status,needs_replan=bool(v.needs_replan),replan_reason=v.replan_reason,
                        original_replan_trigger=replan_trigger,committed_next=committed,
                        mutable_visible_unserved=mutable,accepted_owned_unserved=owned,
                        mutable_n=len(mutable),owned_n=len(owned),
                        cargo_order_ids=[] if v.coldchain_state is None else [x.order_id for x in v.coldchain_state.cargo_manifest],
                        energy_kwh=0. if v.coldchain_state is None else v.coldchain_state.cumulative_energy_kwh)
                    rows.append(row);records.append(row)
                    mutable_hist[len(mutable)]+=1;owned_hist[len(owned)]+=1
                    if replan_trigger and v.status in ('idle','ready','committed'):
                        eligible_mutable[len(mutable)]+=1;eligible_owned[len(owned)]+=1
                joint_hist[len(union)]+=1
                journal.write('census_event',day_id=day,event_id=int(event),clock=float(clock),
                    fleet=rows,joint_owned_n=len(union),all_vehicle_energy=sum(r['energy_kwh'] for r in rows))
                # Save one evaluator state deterministically, without filtering census rows.
                if first_public_capture is None and clock>0 and replan_trigger and len(union)>=2:
                    first_public_capture=dict(snapshot=capture_recourse_snapshot(e,i,clock,event,reveal,
                        fleet,traces,served,universe),dataset=ds,day=day)
            env.snapshot_hook=hook
            tr,served=env.run(day)
            ev=evaluate_trace_a1(tr,{k:v[day] for k,v in ds.items()},contract,rp._accepted,rp._rejected,
                protocol['B'],{0:5.,1:10.,2:15.})
            journal.write('day_end',day_id=day,traces=tr,accepted=set(rp._accepted),rejected=set(rp._rejected),evaluation=ev)
            day_rows.append(dict(day=day,events=count[0],vehicle_records=count[0]*15,evaluation=ev))
        end={p:sha(p) for p in manifest}
        if end!=manifest:raise ValueError('source changed')
        journal.write('source_end',sha256=end);journal.write('run_end',days=4,formal_eligible=False)
    finally:journal.close()
    # Full raw evaluator capture stays private; later actor must re-project it.
    if first_public_capture is not None:
        (out/'private_first_capture.json').write_text(json.dumps(encode(first_public_capture),indent=2),encoding='utf-8')
    def hist_json(c):return {str(k):v for k,v in sorted(c.items())}
    result=dict(protocol=protocol,protocol_sha256=sha(out/'DIAGNOSTIC_REGISTRATION.json'),
        dataset_sha256=dataset_identity(ds),history_dataset_sha256=dataset_identity(train),
        source_start=manifest,source_end=end,days=day_rows,events=sum(r['events'] for r in day_rows),
        vehicle_records=len(records),coverage_verified=len(records)==sum(r['events'] for r in day_rows)*15,
        mutable_histogram=hist_json(mutable_hist),owned_histogram=hist_json(owned_hist),
        original_trigger_mutable_histogram=hist_json(eligible_mutable),original_trigger_owned_histogram=hist_json(eligible_owned),
        joint_owned_histogram=hist_json(joint_hist),max_mutable=max(mutable_hist),max_owned=max(owned_hist),max_joint_owned=max(joint_hist),
        observed_vehicle_mutable_ge12=sum(v for k,v in mutable_hist.items() if k>=12),
        observed_joint_owned_ge12=sum(v for k,v in joint_hist.items() if k>=12),
        no_clock_filter=True,no_capture_early_return=True,
        contract_env_speed_consistent=abs(contract.units.speed_kmph/contract.units.distance_km_per_unit-env.tw_speed)<1e-12,
        contract_unit_warning='Original StageA bare C0-v2 units retained for reproducibility; consistency flag is not silently waived.',
        independent_terminal_replay=replay(out/'full_day_all_fleet_trace.jsonl'),wall_seconds=time.perf_counter()-started,
        stable_online_gain_established=False,pickup_impossibility_proven=False,maskco_attribution=False,
        interpretation='observed finite cohort opportunity sizes only; not distribution support or reward threshold proof')
    (out/'per_event_vehicle.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scripts',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();r=run(a.scripts,a.out)
    print(json.dumps({k:r[k] for k in ('events','vehicle_records','coverage_verified','max_mutable','max_owned','max_joint_owned','observed_joint_owned_ge12','contract_env_speed_consistent','wall_seconds')}))
