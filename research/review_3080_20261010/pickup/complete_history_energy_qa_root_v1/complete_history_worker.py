"""Actual CondHistoricalSampler+SaaReplanner report; not the C0kNN proxy arm."""
import argparse,copy,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,'D:\\PyCharm_\\MASKCO-Main\\research\\review_3080_20261010\\pickup\\actual_masked_fleet_followup_v1')
sys.path.insert(0,'D:\\PyCharm_\\MASKCO-Main\\research\\review_3080_20261010\\pickup\\shadow_energy_correction_root_v1')
import hard_admission as ha
from hard_admission import pf,ps,fw,wire
original_base=ha.base

def conditional_base(config,scripts):
    if config['name']!='cond_hist':return original_base(config,scripts)
    pf.bootstrap(scripts)
    from scenario_saa import SaaReplanner,CondHistoricalSampler,ScenarioOrder
    import pickle
    from energy_complete_saa import EnergyCompleteSAA
    if not hasattr(conditional_base,'common_prior'):
        with Path(config['history_pickle']).open('rb') as f:
            conditional_base.common_prior=pickle.load(f)
    historic=conditional_base.common_prior['history']
    class Observed(EnergyCompleteSAA if config['ledger_corrected'] else SaaReplanner):
        def _saa_decide(self,*args,**kwargs):
            self.completed_calls=0
            return super()._saa_decide(*args,**kwargs)
        def _sim_scenario(self,*args,**kwargs):
            out=super()._sim_scenario(*args,**kwargs);self.completed_calls+=1;return out
        def on_reveal(self,env,i,clock,fleet,served,visible):
            super().on_reveal(env,i,clock,fleet,served,visible)
            if any(o in self._accepted for o in visible) and getattr(self,'completed_calls',0)!=20:
                raise ValueError('accepted conditional SAA lacks K10 complete paired branches')
    common={k:config[k] for k in ('budget','capacity','booking_horizon','contract','cooling_share')}
    return Observed(**common,hist=config['history'],k=10,sampler=CondHistoricalSampler(historic,n_neighbors=10),
        reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,
        shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
ha.base=conditional_base

class ConditionalWorker(ha.HardAdmission):
    # Same bounded public channel; choose this explicit module for child adapter.
    def __init__(self,scripts,config):
        import subprocess,queue,threading,os
        self.queue=queue.Queue();self.stopped=False;self.errors=[]
        self.p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--scripts',str(scripts)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        def read():
            for line in self.p.stdout:self.queue.put(json.loads(line))
            self.queue.put(dict(kind='closed'))
        def err():
            for line in self.p.stderr:self.errors.append(line)
        threading.Thread(target=read,daemon=True).start();threading.Thread(target=err,daemon=True).start()
        self.p.stdin.write(json.dumps(dict(config=wire.wire(config)))+'\n');self.p.stdin.flush()
        try:self.ready=self.queue.get(timeout=30.)
        except queue.Empty:self.close();raise TimeoutError('conditional setup bounded')
        if self.ready.get('kind')!='ready':self.close();raise RuntimeError(str(self.errors[-3:]))

def run(a):
    pf.bootstrap(a.scripts);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from scenario_saa import make_c0_contract_v2
    from audit_io import encode,decode,Journal
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from pickup_trace_runner import replay
    contract=make_c0_contract_v2()
    import pickle
    with Path(a.history_pickle).open('rb') as f: prior=pickle.load(f)
    history=prior['history_times_classes']
    config=dict(name='cond_hist',budget=710.6548152249378,capacity=50.,booking_horizon=16.,contract=contract,cooling_share=0.,history=history,history_pickle=a.history_pickle,ledger_corrected=a.corrected)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    reg=dict(id='P-COMMON200-SHADOW-LEDGER-ENGINEERING-root-v1',registered=datetime.now(timezone.utc).isoformat(),
        archive_trace_sha256=sha(a.archive_trace),same_archived_seed=202610101801,new_days=0,history_sha256=fw.history_hash(history),
        future_history='exact original I3 historical seed20260925 all200 days; independently regenerated and qualified',
        S3=False,S3_provider_active=False,actual_provider='scenario_saa.CondHistoricalSampler(n_neighbors10)+SaaReplanner K10',
        shared_deadline_seconds=10,paired_branch_count=20,partial_vote=False,arm_seed=7001,history_pickle_sha256=sha(a.history_pickle),ledger_corrected=a.corrected,adapter_sha256=sha(__file__),source_start=pf.source_hashes(Path(a.scripts).resolve()),
        protects_final=[20260927,20260928],engineering_only=True,formal_gain=False)
    (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    data=json.loads(Path(a.archive_trace).open(encoding='utf-8').readline())
    ds=decode(next(v for k,v in data['payload']['__mapping__'] if k=='dataset'))
    worker=ConditionalWorker(a.scripts,config);logs=[];events=[];rp=ha.controller(a.scripts,config,worker,None,logs)
    env=ha.environment_class(a.scripts)(ds,50.,1.5,3,rp,contract,16.)
    def hook(e,i,clock,event,reveal,fleet,traces,served,universe):
        known=[x for x in universe if e.reveal_time[i,x]<=clock+1e-6];rej=set(rp._rejected);rp._rejected.update(set(known)-rp._accepted-rp._rejected)
        try:packet=ps.export_public(e,i,clock,event,reveal,fleet,traces,served,known)
        finally:rp._rejected=rej
        cert,_=pf.certificate(packet,packet['plan']);events.append(dict(clock=clock,event_id=event,known_certificate=cert))
    env.snapshot_hook=hook;journal=Journal(out/'cond_hist_trace.jsonl');sources=pf.source_hashes(Path(a.scripts).resolve())
    journal.write('run_start',dataset=ds,contract=contract,budget=config['budget'],penalties={0:5.,1:10.,2:15.},formal_eligible=False)
    journal.write('source_start',sha256=sources)
    try:
        tr,served=env.run(0);ev=evaluate_trace_a1(tr,{k:v[0] for k,v in ds.items()},contract,rp._accepted,rp._rejected,config['budget'],{0:5.,1:10.,2:15.})
        journal.write('day_end',day_id=0,traces=tr,served=served,accepted=set(rp._accepted),rejected=set(rp._rejected),evaluation=ev)
        end={p:sha(p) for p in sources}
        if end!=sources:raise ValueError('source changed')
        journal.write('source_end',sha256=end);journal.write('run_end',formal_eligible=False)
    finally:worker.close();journal.close()
    result=dict(registration=reg,evaluation=ev,admissions=logs,events=events,actual_provider=reg['actual_provider'],S3=False,
                independent_terminal_replay=replay(out/'cond_hist_trace.jsonl'),formal_gain=False)
    (out/'FULLRAW.json').write_text(json.dumps(encode(result),allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(cond_hist_actual_report_completed=True,hard=ev['hard_feasible'],formal_gain=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--history-pickle');p.add_argument('--corrected',action='store_true');p.add_argument('--archive-trace');p.add_argument('--out');a=p.parse_args()
    if a.worker:ha.serve(a)
    else:run(a)
