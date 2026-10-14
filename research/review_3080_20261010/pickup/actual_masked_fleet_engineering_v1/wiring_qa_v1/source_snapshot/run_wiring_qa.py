"""Fresh engineering registration, reuse archived day, no new gain inference.

Concrete full-day controller entry point. Optional actual masked hook loads before
the day on caller's registered shapes, then shares the admission deadline.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import fair_wiring as fw
import public_fleet as pf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    scripts=pf.bootstrap(a.scripts);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from scenario_saa import make_c0_contract_v2
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from audit_io import decode,encode,Journal
    from pickup_trace_runner import replay
    from resident_actor import ResidentActor,live_hook
    trace=Path(a.archive_trace)
    contract=make_c0_contract_v2()
    # Entirely fixed synthetic history fixture; not extracted from private future.
    history=[(np.linspace(.5,15.5,30),np.asarray([(j+d)%3 for j in range(30)],dtype=int)) for d in range(3)]
    source=pf.source_hashes(scripts)
    reg=dict(id='P-UNIFORM-HISTORY-ADMISSION-WIRING-ENGINEERING-v1',registered=datetime.now(timezone.utc).isoformat(),
        archive_trace_sha256=sha(trace),known_archived_day_seed=202610101801,new_days=0,
        history_fixture='three fixed synthetic30order days; NOT formal I3 historical200day pool',history_sha256=fw.history_hash(history),
        budget=710.6548152249378,capacity=50.,vehicles=3,booking_horizon=16.,cooling_share=0.,
        contract_hash=contract.contract_hash,tw_speed=contract.units.speed_kmph/contract.units.distance_km_per_unit,
        admission_order='reveal_time then stable public ID',per_admission_submission_seconds=10.,legacy_process_kill_qualified=False,
        arms=['own_uncond_dynamic','strong_rolling'],actor_mode=a.mode,actor_shapes=a.shapes,actor_interventions=1 if a.mode else 0,
        all_in_remaining_admission_budget=True,private_future_used_by_actor=False,protected_final=[20260927,20260928],
        source_start=source,formal_eligible=False)
    (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    row=json.loads(trace.open(encoding='utf-8').readline())
    ds=decode(next(v for k,v in row['payload']['__mapping__'] if k=='dataset'))
    if len(ds['coords'])!=1:raise ValueError('one registered archived day')
    env_class=fw.make_trace_env_class(scripts);results=[]
    actor=None
    if a.mode:
        if not a.checkpoint or not a.upstream:raise ValueError('actual model paths required')
        actor=ResidentActor(scripts,a.upstream,a.checkpoint,a.mode,list(map(int,a.shapes.split(','))))
        (out/'ACTUAL_MODEL_READY.json').write_text(json.dumps(actor.ready,indent=2),encoding='utf-8')
    try:
        for name in reg['arms']:
            rp=fw.build_matched_replanner(name,scripts,history,contract,reg['budget'],reg['cooling_share'])
            env=env_class(ds,50.,reg['tw_speed'],3,rp,contract,16.)
            proposals=[]
            if actor and name=='own_uncond_dynamic':env.snapshot_hook=live_hook(actor,proposals,max_interventions=1)
            journal=Journal(out/(name+'_trace.jsonl'))
            journal.write('run_start',dataset=ds,contract=contract,budget=reg['budget'],penalties={0:5.,1:10.,2:15.},registration_sha256=sha(out/'REGISTRATION.json'),formal_eligible=False)
            journal.write('source_start',sha256=source)
            try:
                traces,served=env.run(0)
                ev=evaluate_trace_a1(traces,{k:v[0] for k,v in ds.items()},contract,rp._accepted,rp._rejected,reg['budget'],{0:5.,1:10.,2:15.})
                journal.write('day_end',day_id=0,traces=traces,served=served,accepted=set(rp._accepted),rejected=set(rp._rejected),evaluation=ev)
                end={p:sha(p) for p in source}
                if end!=source:raise ValueError('source changed during QA')
                journal.write('source_end',sha256=end);journal.write('run_end',formal_eligible=False)
            finally:journal.close()
            results.append(dict(arm=name,evaluation=ev,identity=rp.fair_identity,admissions=rp.admission_audit,proposals=proposals,
                                independent_replay=replay(out/(name+'_trace.jsonl'))))
    finally:
        if actor:actor.close()
    result=dict(registration=reg,rows=results,matching_history_verified=len({r['identity']['history_sha256'] for r in results})==1,
                full_online_fresh_gain_established=False,engineering_only=True)
    (out/'result.json').write_text(json.dumps(encode(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(engineering_only=True,matching_history_verified=result['matching_history_verified'],rows=len(results))),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--archive-trace',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['pretrained','random','discard']);p.add_argument('--checkpoint');p.add_argument('--upstream');p.add_argument('--shapes',default='4')
    run(p.parse_args())
