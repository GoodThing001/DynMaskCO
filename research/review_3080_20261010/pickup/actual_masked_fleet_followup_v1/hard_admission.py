"""Public-state resident admission workers and shared hard request deadlines."""
import argparse,copy,json,math,os,pickle,queue,subprocess,sys,threading,time
from numbers import Real
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'actual_masked_fleet_engineering_v1'))
import public_fleet as pf
import public_state as ps
import resident_actor as wire
import fair_wiring as fw

def base(config,scripts):
    pf.bootstrap(scripts)
    from rolling_optimizer import RollingOptimizeReplanner
    from run_exp_rule_control import UncondDynamicReplanner
    from run_exp_energy_c0 import C0ReserveReplanner
    common={k:config[k] for k in ('budget','capacity','booking_horizon','contract','cooling_share')}
    common.update(hist=config['history'],k=10)
    if config['name']=='strong_rolling':return RollingOptimizeReplanner(**common,reject_penalty={0:5.,1:10.,2:15.},time_limit=10.)
    if config['name']=='own_uncond_dynamic':return UncondDynamicReplanner(**common)
    if config['name']=='cond_report':
        class PublicConditional(C0ReserveReplanner):
            def _lookahead_lambda(self,env,i,clock,o):
                # Original kNN counts include depot/padding. In a public projection
                # hidden slots are zero; count only actual revealed demand>0.
                t=min(float(clock),3.);mask=(env.reveal_time[i]<=t)&(env.demands[i]>0)
                feat=np.asarray([np.sum(env.dataset['temp_class'][i][mask]==c) for c in range(3)])
                hfeat=np.asarray([[np.sum(hc[ht<=t]==c) for c in range(3)] for ht,hc in self.hist])
                sd=hfeat.std(0)+1e-6;kn=np.argsort(np.sum(((hfeat-feat)/sd)**2,axis=1))[:self.k]
                from run_exp_energy_c0 import c0_marginal_energy,_marginal_lambda
                from coldchain_evaluator_a1 import REV
                remaining=self.budget-sum(self._est_energy(env,i,x) for x in self._accepted)
                vals=[]
                for h in kn:
                    ht,hc=self.hist[h];f=ht>clock
                    vals.append(_marginal_lambda(np.asarray([REV[int(c)] for c in hc[f]]),
                        np.asarray([c0_marginal_energy(int(c),2.,self.cooling_share,self.contract) for c in hc[f]]),remaining))
                return float(np.mean(vals))
        return PublicConditional(**common,mode='knn')
    raise ValueError('unknown registered base')

def canonical(packet,plan):
    out={str(v):[int(o) for o in plan.get(v,plan.get(str(v),[])) if not packet['snapshot']['served_mask'][o]] for v in range(packet['snapshot']['num_vehicles'])}
    for v,s in enumerate(packet['snapshot']['vehicle_status']):
        cn=int(packet['snapshot']['committed_next'][v])
        if s=='committed' and cn>0:out[str(v)]=[cn]+[o for o in out[str(v)] if o!=cn]
    return out

def decide(config,packet,o,scripts,deadline):
    pf.validate_public(packet)
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot
    rp=base(config,scripts);state=copy.deepcopy(packet['snapshot']['replanner_state'])
    state['rejected']=set(state['rejected'])-{o};rp.restore_state(state)
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    fleet,traces,served=restore_recourse_snapshot(packet['snapshot']);clock=float(packet['snapshot']['clock'])
    if hasattr(rp,'time_limit'):rp.time_limit=max(0.,deadline-time.perf_counter())
    rp.on_reveal(env,0,clock,fleet,served,[o])
    state=rp.export_state();plan=canonical(packet,state['plan']);state['plan']=plan
    trial=copy.deepcopy(packet);trial['plan']=plan;trial['snapshot']['replanner_state']=state
    cert,raw=pf.certificate(trial,plan)
    return dict(qualified=cert['qualified'] is True and time.perf_counter()<=deadline,state=state,plan=plan,
                packet_sha=pf.digest(packet),certificate=cert,known_raw=raw,deadline_monotonic=deadline)

def serve(a):
    pf.bootstrap(a.scripts)
    config=wire.unwire(json.loads(sys.stdin.readline())['config'])
    base(config,a.scripts)
    # Seal all causal/physical dependency imports before accepting requests.
    from recourse_snapshot import restore_recourse_snapshot
    from coldchain_evaluator_a1 import evaluate_trace_a1
    manifest=pf.source_hashes(Path(a.scripts).resolve())
    print(json.dumps(dict(kind='ready',source_start=manifest,history_sha256=fw.history_hash(config['history']))),flush=True)
    for line in sys.stdin:
        request=wire.unwire(json.loads(line)['request']);packet=request['packet'];deadline=request['deadline']
        try:
            if request.get('sleep_qa'):time.sleep(1.)
            result=decide(config,packet,request['order'],a.scripts,deadline)
            end={p:wire.sha(p) for p in manifest};result['source_end']=end
            if end!=manifest or time.perf_counter()>deadline:result['qualified']=False
        except (ValueError,TimeoutError) as exc:result=dict(qualified=False,reason=str(exc))
        print(json.dumps(dict(kind='response',result=wire.wire(result))),flush=True)

class HardAdmission:
    def __init__(self,scripts,config):
        self.queue=queue.Queue();self.stopped=False;self.errors=[]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        self.p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--scripts',str(scripts)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=env)
        def read():
            for line in self.p.stdout:self.queue.put(json.loads(line))
            self.queue.put(dict(kind='closed'))
        def err():
            for line in self.p.stderr:self.errors.append(line)
        threading.Thread(target=read,daemon=True).start();threading.Thread(target=err,daemon=True).start()
        self.p.stdin.write(json.dumps(dict(config=wire.wire(config)))+'\n');self.p.stdin.flush()
        try:self.ready=self.queue.get(timeout=30.)
        except queue.Empty:self.close();raise TimeoutError('bounded base setup')
        if self.ready.get('kind')!='ready':self.close();raise RuntimeError('base setup failed: '+str(self.errors[-3:]))
    def select(self,packet,order,started,limit=10.,sleep_qa=False):
        if isinstance(started,bool) or not isinstance(started,Real) or not math.isfinite(started) or started>time.perf_counter():raise ValueError('elapsed finite start')
        if isinstance(limit,bool) or not isinstance(limit,Real) or not math.isfinite(limit) or not 0<limit<=10:raise ValueError('finite <=10 cap')
        deadline=started+limit;fallback=dict(qualified=False,fallback=True,timeout=False,deadline_monotonic=deadline)
        if self.stopped or time.perf_counter()>=deadline:return fallback
        self.p.stdin.write(json.dumps(dict(request=wire.wire(dict(packet=packet,order=order,deadline=deadline,sleep_qa=sleep_qa))))+'\n');self.p.stdin.flush()
        try:response=self.queue.get(timeout=max(0.,deadline-time.perf_counter()))
        except queue.Empty:self.close();fallback['timeout']=True;fallback['worker_reaped']=self.p.poll() is not None;return fallback
        if response.get('kind')!='response':self.close();return fallback
        r=wire.unwire(response['result'])
        if r.get('qualified') is not True or r.get('packet_sha')!=pf.digest(packet) or r.get('source_end')!=self.ready['source_start'] or time.perf_counter()>deadline:return fallback
        return r
    def close(self):
        self.stopped=True
        if self.p.poll() is None:self.p.kill();self.p.wait(timeout=5)

def environment_class(scripts):
    pf.bootstrap(scripts)
    from strict_online_env import StrictOnlineEnv
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            r=super()._advance_fleet(i,clock,fleet,traces,served);self.public_context=(i,clock,fleet,traces,served);return r
        def _maybe_snapshot(self,i,clock,reveal,fleet,traces,served,universe):
            self.public_context=(i,clock,fleet,traces,served)
            if clock==0:self.replanner.on_reveal(self,i,clock,fleet,served,[o for o in universe if self.reveal_time[i,o]<=clock+1e-6])
            return super()._maybe_snapshot(i,clock,reveal,fleet,traces,served,universe)
    return Env

def controller(scripts,config,worker,actor,log):
    pf.bootstrap(scripts)
    from run_exp_energy_c0 import C0ReserveReplanner
    class Controller(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):
            self._reset_if_new(i);reserved=env.get_reserved_customers(fleet)
            new=sorted([int(o) for o in visible if not served[o] and int(o) not in reserved and int(o) not in self._accepted and int(o) not in self._rejected],key=lambda o:(env.reveal_time[i,o],o))
            for o in new:
                t0=time.perf_counter();deadline=t0+10.;saved=copy.deepcopy(self.export_state())
                known=[int(x) for x in np.flatnonzero(env.demands[i]>0) if env.reveal_time[i,x]<=clock+1e-6]
                pending=set(known)-self._accepted-self._rejected
                self._rejected.update(pending)
                packet=ps.export_public(env,i,clock,env.event_id,0,fleet,env.public_context[3],served,known)
                self.restore_state(saved)
                selection=worker.select(packet,o,t0);fallback=selection.get('qualified') is not True
                if not fallback:
                    state=copy.deepcopy(selection['state'])
                    state['status_inst']=i
                    state['rejected']=(set(state['rejected'])-pending)|set(saved['rejected'])|({o} if o in state['rejected'] else set())
                    self.restore_state(state)
                else:self._rejected.add(o)
                proposals=None;installed=False
                if actor and not fallback and time.perf_counter()<deadline:
                    rej=set(self._rejected);self._rejected.update(set(known)-self._accepted-self._rejected)
                    try:
                        current=ps.export_public(env,i,clock,env.event_id,0,fleet,env.public_context[3],served,known)
                        if len(pf.family(current)[1])>=2:
                            proposals=actor.select(current,t0)
                            installed=pf.install_in_live(self,current,proposals)
                    finally:self._rejected=rej
                if time.perf_counter()>deadline:
                    self.restore_state(saved);self._rejected.add(o);fallback=True;installed=False
                log.append(dict(event_id=int(env.event_id),clock=float(clock),order=o,public_request=packet,base_response=selection,
                    started=t0,deadline=deadline,wall_seconds=time.perf_counter()-t0,accepted=o in self._accepted,
                    fallback=fallback,masked_proposal=proposals,masked_installed=installed,
                    source_history_sha256=fw.history_hash(config['history']),all_in_hard_worker=True))
    return Controller(config['budget'],config['capacity'],config['booking_horizon'],config['contract'],config['cooling_share'],hist=config['history'],k=10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--scripts',required=True);serve(p.parse_args())
