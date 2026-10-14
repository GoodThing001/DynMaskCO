"""Registered bounded engineering day, public worker, complete full-J raw replay."""
import argparse,copy,hashlib,json,pickle,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import hard_admission as ha
from hard_admission import pf,ps,fw,wire

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    scripts=pf.bootstrap(a.scripts);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from scenario_saa import make_c0_contract_v2
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from audit_io import Journal,encode,decode
    from pickup_trace_runner import replay
    contract=make_c0_contract_v2()
    history=[(np.linspace(.5,15.5,30),np.asarray([(j+d)%3 for j in range(30)],int)) for d in range(3)]
    names=['own_uncond_dynamic','strong_rolling','cond_report','own_pretrained']
    reg=dict(id='P-HARD-BASE-ACTUAL-FULLDAY-ENGINEERING-v1',registered=datetime.now(timezone.utc).isoformat(),
        archive_trace_sha256=sha(a.archive_trace),already_observed_seed=202610101801,new_days=0,orders=80,vehicles=3,
        synthetic_history_sha256=fw.history_hash(history),hist_days=3,K_history=10,no_formal_I3_history_pool_claim=True,
        arms=names,contract_hash=contract.contract_hash,budget=710.6548152249378,cooling_share=0.,
        registered_model_shapes=list(range(3,14)),fixed_model_checkpoint_sha256=sha(a.checkpoint),maskseed=202610101903,
        per_admission_hard_worker_seconds=10.,overall_subprocess_limit=900,clock0_full_A1_certificate=True,
        future_mutation='one worker admission repeated at same public packet; hidden dataset values replaced before projection',
        no_terminal_future_J_selection=True,protected_final=[20260927,20260928],formal_gain_established=False,
        original_base_scope='actual original UncondDynamic per admission with identical history; only public env handoff+hard cap+actual full A1 commit gate',
        conditional_report='public demand>0 reveal prefix counts adapter; diagnostic report arm only',source_start=pf.source_hashes(scripts))
    (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    source_snapshot=out/'source_snapshot';source_snapshot.mkdir()
    for p in Path(__file__).resolve().parent.glob('*.py'):(source_snapshot/p.name).write_bytes(p.read_bytes())
    start=json.loads(Path(a.archive_trace).open(encoding='utf-8').readline())
    ds=decode(next(v for k,v in start['payload']['__mapping__'] if k=='dataset'))
    if len(ds['coords'])!=1:raise ValueError('one archived engineering day required')
    rows=[];workers=[];model=None;env_class=ha.environment_class(scripts)
    from dtype_probe import run as compare_dtype
    dtype_result=compare_dtype(scripts,a.upstream,a.checkpoint,Path(__file__).resolve().parent.parent/'public_certificate_qa_v2/public_packet.pkl',out/'dtype_probe')
    common=dict(budget=reg['budget'],capacity=50.,booking_horizon=16.,contract=contract,cooling_share=0.,history=history)
    try:
        model=wire.ResidentActor(scripts,a.upstream,a.checkpoint,'pretrained',reg['registered_model_shapes'])
        (out/'ACTUAL_MODEL_READY.json').write_text(json.dumps(model.ready,indent=2),encoding='utf-8')
        for name in names:
            config=dict(common,name='own_uncond_dynamic' if name=='own_pretrained' else name)
            worker=ha.HardAdmission(scripts,config);workers.append(worker);logs=[];events=[];mutation=[]
            rp=ha.controller(scripts,config,worker,model if name=='own_pretrained' else None,logs)
            env=env_class(ds,50.,1.5,3,rp,contract,16.)
            def hook(e,i,clock,event,reveal,fleet,traces,served,universe):
                known=[o for o in universe if e.reveal_time[i,o]<=clock+1e-6]
                pending=set(known)-rp._accepted-rp._rejected;rej=set(rp._rejected);rp._rejected.update(pending)
                try:packet=ps.export_public(e,i,clock,event,reveal,fleet,traces,served,known)
                finally:rp._rejected=rej
                cert,raw=pf.certificate(packet,packet['plan'])
                events.append(dict(event_id=event,clock=float(clock),public_packet=packet,known_certificate=cert))
                # Mutate only private future features; export ignores unknown values.
                if not mutation and clock>0 and len(logs)>0:
                    hidden=[x for x in universe if e.reveal_time[i,x]>clock+1e-6]
                    altered=copy.deepcopy(e.dataset)
                    for k in ('coords','demands','temp_class','tw_start','tw_end','service_time','initial_quality'):altered[k][i,hidden]=0
                    proxy=copy.copy(e);proxy.dataset=altered
                    for k in ('coords','demands','tw_start','tw_end','service_time','reveal_time'):setattr(proxy,k,altered[k])
                    rp._rejected.update(pending)
                    try:packet2=ps.export_public(proxy,i,clock,event,reveal,fleet,traces,served,known)
                    finally:rp._rejected=rej
                    same=pf.digest(packet)==pf.digest(packet2)
                    prior=logs[0]['public_request'];order=logs[0]['order']
                    one=worker.select(prior,order,time.perf_counter());two=worker.select(copy.deepcopy(prior),order,time.perf_counter())
                    mutation.append(dict(hidden_orders_mutated=len(hidden),packet_invariant=same,
                        admission_choice_invariant=one.get('state')==two.get('state'),both_qualified=one.get('qualified') is True and two.get('qualified') is True,
                        note='selection repeats same resulting public packet; no fullfuture rollout in actor'))
            env.snapshot_hook=hook
            journal=Journal(out/(name+'_trace.jsonl'));sources=pf.source_hashes(scripts)
            journal.write('run_start',dataset=ds,contract=contract,budget=reg['budget'],penalties={0:5.,1:10.,2:15.},registration_sha256=sha(out/'REGISTRATION.json'),formal_eligible=False)
            journal.write('source_start',sha256=sources)
            try:
                traces,served=env.run(0)
                ev=evaluate_trace_a1(traces,{k:v[0] for k,v in ds.items()},contract,rp._accepted,rp._rejected,reg['budget'],{0:5.,1:10.,2:15.})
                journal.write('day_end',day_id=0,traces=traces,served=served,accepted=set(rp._accepted),rejected=set(rp._rejected),evaluation=ev)
                end={p:sha(p) for p in sources}
                if end!=sources:raise ValueError('source changed')
                journal.write('source_end',sha256=end);journal.write('run_end',formal_eligible=False)
            finally:journal.close();worker.close()
            row=dict(arm=name,history_sha256=fw.history_hash(history),base_ready=worker.ready,evaluation=ev,admissions=logs,events=events,
                future_mutation=mutation,independent_replay=replay(out/(name+'_trace.jsonl')))
            (out/(name+'_fullraw.json')).write_text(json.dumps(encode(row),allow_nan=False),encoding='utf-8');rows.append(row)
        # Actual process kill regression of original baseline worker.
        first=rows[0]['admissions'][0];config=dict(common,name='strong_rolling')
        timeout_worker=ha.HardAdmission(scripts,config);workers.append(timeout_worker)
        timeout=timeout_worker.select(first['public_request'],first['order'],time.perf_counter(),.001,sleep_qa=True)
        result=dict(registration=reg,rows=[dict(arm=r['arm'],evaluation=r['evaluation'],admissions=len(r['admissions']),
                      events=len(r['events']),clock0_certified=r['events'][0]['clock']==0 and r['events'][0]['known_certificate']['qualified'],
                      fullraw_sha256=sha(out/(r['arm']+'_fullraw.json')),replay=r['independent_replay'],future_mutation=r['future_mutation'],
                      model_proposals=sum(x['masked_proposal'] is not None for x in r['admissions']),
                      model_installed=sum(x['masked_installed'] for x in r['admissions']),
                      admission_timeouts=sum(x['base_response'].get('timeout') is True for x in r['admissions'])) for r in rows],
                    hard_base_timeout_regression=timeout,formal_gain_established=False,engineering_only=True)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(complete=True,rows=len(rows),actual_model_full_day_proposals=result['rows'][-1]['model_proposals'],engineering_only=True)),flush=True)
    finally:
        for worker in workers:worker.close()
        if model:model.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--archive-trace',required=True);p.add_argument('--out',required=True);run(p.parse_args())
