"""Observed admission + actual route proposal + common SAA + joint publication QA."""
import argparse,copy,hashlib,json,os,pickle,queue,subprocess,sys,threading,time
from pathlib import Path
from dataclasses import is_dataclass,replace
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup'
OUT=P/'joint_policy_transaction_root_v2';INPUT=P/'complete_history_energy_qa_root_v1/complete_ledger/FULLRAW.json'
HISTORY=P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl'
for folder in (P,P/'actual_masked_fleet_engineering_v1',P/'actual_masked_fleet_followup_v1',P/'common_candidate_saa_root_v2',
               P/'shadow_energy_correction_root_v1',P/'atomic_plan_installation_root_v2',OUT):sys.path.insert(0,str(folder))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import public_fleet as pf
pf.bootstrap(E/'scripts')
import resident_actor as wire
from joint_install import compile_state,state_dict,digest,transactional_class,JointPolicyOwner

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def codecs():
    import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
    from audit_io import encode,decode
    classes={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa)
             for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
    # Store compiled bundle as standard state dictionaries, not arbitrary class JSON.
    classes['joint_install.PolicyState']=__import__('joint_install').PolicyState
    return encode,lambda x:decode(x,classes)
def fixtures():
    _,decode=codecs();raw=decode(json.loads(INPUT.read_text(encoding='utf-8')))
    rows=raw['admissions'];indices=[0,1,next(j for j,r in enumerate(rows) if len(pf.family(r['public_request'])[1])>=2)]
    return [(j,rows[j]) for j in sorted(set(indices))]

def worker():
    import numpy as np,psutil
    if np.__version__!='1.26.4':raise RuntimeError('Pinned original public/admission runtime required')
    psutil.Process().cpu_affinity([7])
    import hard_admission as hd,public_state as ps
    from common_score import score_common
    from energy_complete_saa import EnergyCompleteSAA
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
    from rolling_optimizer import RollingOptimizeReplanner
    with HISTORY.open('rb') as f:history=pickle.load(f)
    # Prime actual imports before capture, no observed requests processed in setup.
    manifest=pf.source_hashes(E/'scripts')
    for p in (Path(__file__),OUT/'joint_install.py',HISTORY,Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__)):
        manifest[str(p)]=sha(p)
    print(json.dumps({'kind':'ready','source_start':manifest,'numpy':np.__version__,'python':sys.version,'cpu':psutil.Process().cpu_affinity()}),flush=True)
    for line in sys.stdin:
        q=wire.unwire(json.loads(line)['request']);deadline=q['deadline'];steps={}
        def phase(name):
            if q.get('delay')==name:time.sleep(1.)
            if time.perf_counter()>=deadline:raise TimeoutError(name+' exhausted shared deadline')
            steps[name]=time.perf_counter()
        try:
            old=q['packet'];config={'name':'own_uncond_dynamic','budget':old['budget'],'capacity':old['capacity'],
                'booking_horizon':old['booking_horizon'],'contract':old['contract'],'cooling_share':0.,'history':history['history_times_classes']}
            phase('projection')
            rp=hd.base(config,E/'scripts');rp.restore_state(old['snapshot']['replanner_state'])
            env=StrictOnlineEnv(old['dataset'],old['capacity'],old['tw_speed'],old['snapshot']['num_vehicles'],rp,old['contract'],old['booking_horizon'])
            fleet,traces,served=restore_recourse_snapshot(old['snapshot']);clock=old['snapshot']['clock']
            projected=ps.export_public(env,0,clock,old['snapshot']['event_id'],0,fleet,traces,served,list(old['snapshot']['customer_universe']))
            phase('admission')
            base=hd.decide(config,projected,q['order'],E/'scripts',deadline)
            if base['qualified'] is not True:raise ValueError('Actual admission branch not certified')
            current=copy.deepcopy(projected);current['plan']=base['plan'];current['snapshot']['replanner_state']=copy.deepcopy(base['state'])
            current['snapshot']['replanner_state']['status_inst']=old['snapshot']['replanner_state']['status_inst']
            current['ownership']={str(o):int(v) for v,r in current['plan'].items() for o in r}
            prior_snapshot_hash=projected['snapshot']['state_hash']
            current['snapshot']['state_hash']=snapshot_state_hash(current['snapshot'])
            phase('proposal')
            proposal=copy.deepcopy(current['plan'])
            if q['mode']=='strong_rolling':
                rolling=RollingOptimizeReplanner(budget=current['budget'],capacity=current['capacity'],booking_horizon=current['booking_horizon'],
                    contract=current['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,reject_penalty=current['penalties'],time_limit=max(0.,deadline-time.perf_counter()))
                rolling.restore_state(copy.deepcopy(current['snapshot']['replanner_state']))
                candidate_env=StrictOnlineEnv(current['dataset'],current['capacity'],current['tw_speed'],current['snapshot']['num_vehicles'],rolling,current['contract'],current['booking_horizon'])
                plans,accepted,rejected,proxy=rolling._optimize(candidate_env,0,clock,fleet,served,set())
                if accepted or rejected:raise ValueError('Fixed-admissions route proposal changed ownership policy')
                proposal=hd.canonical(current,plans)
            elif q['mode']!='own_incumbent':raise ValueError('No unregistered proposer')
            phase('scoring')
            scores=score_common(current,{'own_incumbent':current['plan'],'candidate':proposal},history,deadline,seed=202610111301)
            own,candidate=scores['candidate_scores']['own_incumbent'],scores['candidate_scores']['candidate']
            if not own['qualified']:raise ValueError('Actual admitted incumbent not hard')
            accepted_route=candidate['qualified'] and candidate['mean_shadow_utility']>own['mean_shadow_utility']+1e-10
            selected=proposal if accepted_route else current['plan']
            phase('certification');certificate,raw=pf.certificate(current,selected)
            phase('compile');state=copy.deepcopy(base['state']);state['plan']=selected;state['status_inst']=old['snapshot']['replanner_state']['status_inst']
            compiled=compile_state(state);end={p:sha(p) for p in manifest};done=time.perf_counter()
            result={'kind':'response','request_id':q['request_id'],'packet_digest':pf.digest(old),'deadline':deadline,
                'certified':certificate['qualified'] is True and done<deadline,'source_qualified':end==manifest,'source_end':end,
                'compiled_state':compiled,'compiled_state_sha256':digest(state_dict(compiled)),'projected_packet':projected,
                'base_response':base,'scoring_packet':current,'proposal':proposal,'common_scores':scores,'selected_certificate':certificate,
                'accepted_route':accepted_route,'prior_snapshot_hash':prior_snapshot_hash,'derived_snapshot_hash':current['snapshot']['state_hash'],'worker_finished':done,'phase_started':steps,'live_extraction_included':False}
        except Exception as exc:result={'kind':'error','error':type(exc).__name__+':'+str(exc),'phase_started':steps}
        print(json.dumps({'kind':'response','result':wire.wire(result)}),flush=True)

class Worker:
    def __init__(self):
        self.q=queue.Queue();self.errors=[]
        self.p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        def read():
            for line in self.p.stdout:
                try:self.q.put(json.loads(line))
                except Exception:self.errors.append(line)
            self.q.put({'kind':'closed'})
        def err():
            for line in self.p.stderr:self.errors.append(line)
        threading.Thread(target=read,daemon=True).start();threading.Thread(target=err,daemon=True).start()
        try:self.ready=self.q.get(timeout=30)
        except queue.Empty:self.close();raise TimeoutError('bounded setup')
        if self.ready.get('kind')!='ready':self.close();raise RuntimeError('Setup: '+repr(self.errors))
    def ask(self,q):
        self.p.stdin.write(json.dumps({'request':wire.wire(q)})+'\n');self.p.stdin.flush()
        try:
            r=self.q.get(timeout=max(0.,q['deadline']-time.perf_counter()))
            return wire.unwire(r['result']) if r.get('kind')=='response' else r
        except queue.Empty:self.close();return {'kind':'timeout','worker_reaped':self.p.poll() is not None}
    def close(self):
        if self.p.poll() is None:self.p.kill()
        self.p.wait(timeout=5)

def main():
    import psutil,numpy as np
    if np.__version__!='1.26.4':raise RuntimeError('Original engineering environment required')
    psutil.Process().cpu_affinity([7]);encode,decode=codecs()
    from run_exp_energy_c0 import C0ReserveReplanner
    from recourse_snapshot import restore_recourse_snapshot
    cls=transactional_class(C0ReserveReplanner);observed=fixtures()
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Three already recorded admission requests; actual public-packet reprojection/admission/own or rolling proposal/common K10/worker certification and whole policy-pointer publication',
        'fixture_sha256':sha(INPUT),'history_sha256':sha(HISTORY),'source_sha256':sha(__file__),'installer_sha256':sha(OUT/'joint_install.py'),
        'fixture_indices':[j for j,r in observed],'selection':'first two recorded admissions and first pre-admission state with >=2 mutable orders; no outcome-based selection',
        'modes':['own_incumbent','strong_rolling'],'K':10,'common_scenario_seed':202610111301,'cpu':[7],'numpy':'1.26.4',
        'normal_cap_s':10.,'delay_fault_cap_s':.01,'new_scientific_data':False,'live_raw_public_extraction_included':False,
        'parent_serialization_is_interruptible':False,'all_formal_arms':False,'whole_live_request_qualified':False,'delays_are_outer_deadline_probes_not_proof_target_stage_reached':True,'stable_online_gain':False}
    write(OUT/'REGISTRATION.json',reg);records=[];faults=[];checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    def owner(packet):
        rp=cls(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.)
        rp.restore_state(packet['snapshot']['replanner_state']);fleet,_,_=restore_recourse_snapshot(packet['snapshot'])
        return JointPolicyOwner(rp,fleet,packet['snapshot']['clock'],packet['snapshot']['event_id'])
    w=Worker()
    try:
        ck('actual_worker_cpu_numpy',w.ready['cpu']==[7] and w.ready['numpy']=='1.26.4')
        for j,row in observed:
            for mode in reg['modes']:
                packet=row['public_request'];live=owner(packet);before=live.rp.export_state();old_pointer=live.rp._policy
                started=time.perf_counter();deadline=started+10.;rid=str((j,mode));token=live.begin(rid,row['order'],deadline,pf.digest(packet))
                response=w.ask({'packet':packet,'order':row['order'],'mode':mode,'request_id':rid,'deadline':deadline});got=time.perf_counter()
                ck(rid+':worker_result',response.get('kind')=='response')
                if response.get('kind')!='response':raise RuntimeError('Recorded worker failure '+repr(response))
                original=pf.certificate
                def prohibited(*a,**kw):raise RuntimeError('No parent full physical recertification')
                pf.certificate=prohibited
                try:installed,reason=live.commit(token,response)
                finally:pf.certificate=original
                done=time.perf_counter();after=live.rp.export_state()
                ck(rid+':joint_actual_pointer',installed and live.rp._policy is response['compiled_state'] and live.rp._policy is not old_pointer)
                ck(rid+':actual_export_matches_worker',after==state_dict(response['compiled_state']))
                ck(rid+':all_phases_shared_budget',started<=got<=done<deadline and set(response['phase_started'])=={'projection','admission','proposal','scoring','certification','compile'})
                ck(rid+':all_K10_branches',all(len(s['branch_values'])==10 for s in response['common_scores']['candidate_scores'].values() if s['qualified']))
                ck(rid+':worker_full_certificate',response['selected_certificate']['qualified'])
                ck(rid+':actual_policy_interfaces',live.rp._accepted==set(after['accepted']) and live.rp._rejected==set(after['rejected']) and live.rp._plan=={int(k):v for k,v in after['plan'].items()})
                ck(rid+':source_exact',response['source_end']==w.ready['source_start'])
                records.append({'fixture_index':j,'mode':mode,'order':row['order'],'packet':packet,'before':before,'after':after,
                    'response':response,'started':started,'received':got,'finished':done,'wall_s':done-started,'installed':installed,'reason':reason})
        template=records[-1]
        for fault in ('changed_admissions','changed_plan','changed_fleet','wrong_request','changed_prepared_state','changed_prior_acceptance','expired','replayed_token'):
            packet=template['packet'];live=owner(packet);token=live.begin('fault',template['order'],time.perf_counter()+10,pf.digest(packet))
            response=copy.deepcopy(template['response']);response.update(request_id='fault',deadline=token['deadline'])
            if fault=='changed_admissions':live.rp._rejected.add(99999)
            elif fault=='changed_plan':live.rp._plan[0].append(99999)
            elif fault=='changed_fleet':live.fleet[0].current_load+=1
            elif fault=='wrong_request':response['request_id']='wrong'
            elif fault=='changed_prepared_state':response['compiled_state'].plan[0].append(99999)
            elif fault=='changed_prior_acceptance':
                c=response['compiled_state'];response['compiled_state']=replace(c,accepted=set())
                response['compiled_state_sha256']=digest(state_dict(response['compiled_state']))
            elif fault=='expired':token['deadline']=time.perf_counter()-1
            elif fault=='replayed_token':live.cancel(token)
            pre=live.rp._policy;ok,reason=live.commit(token,response)
            ck(fault+':no_joint_publication',not ok and live.rp._policy is pre)
            faults.append({'kind':fault,'installed':ok,'reason':reason,'state_unchanged':live.rp._policy is pre})
    except Exception as exc:
        ck('normal_stage_completed',False);faults.append({'kind':'execution_failure','error':repr(exc),'stderr':w.errors})
    finally:w.close()
    for phase in ('projection','admission','proposal','scoring','certification','compile'):
        fresh=Worker();row=observed[-1][1];packet=row['public_request'];live=owner(packet);before=live.rp._policy
        started=time.perf_counter();token=live.begin('delay:'+phase,row['order'],started+.01,pf.digest(packet))
        try:
            r=fresh.ask({'packet':packet,'order':row['order'],'mode':'own_incumbent','request_id':token['request_id'],'deadline':token['deadline'],'delay':phase})
            installed,reason=live.commit(token,r)
        finally:fresh.close()
        ck(phase+':delayed_worker_no_publication',not installed and live.rp._policy is before and fresh.p.poll() is not None)
        faults.append({'kind':'delay:'+phase,'response':r,'installed':installed,'reason':reason,'worker_reaped':fresh.p.poll() is not None,'old_policy_unchanged':live.rp._policy is before})
    ck('all_six_normal_requests_retained',len(records)==len(observed)*2)
    ck('input_source_unchanged',sha(INPUT)==reg['fixture_sha256'] and sha(HISTORY)==reg['history_sha256'] and sha(__file__)==reg['source_sha256'] and sha(OUT/'joint_install.py')==reg['installer_sha256'])
    failed=sum(not c['pass'] for c in checks)
    result={'registration_sha256':sha(OUT/'REGISTRATION.json'),'ready':w.ready,'checks':checks,'checks_count':len(checks),'failed':failed,
        'records':records,'faults':faults,'new_science':False,'whole_live_10s_qualified':False,'stable_online_gain':False}
    write(OUT/'RESULT.json',encode(result))
    print(json.dumps({'checks':len(checks),'failed':failed,'normal_requests':len(records),'faults':len(faults),'walls':[r['wall_s'] for r in records]}))
    return int(failed>0)
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
