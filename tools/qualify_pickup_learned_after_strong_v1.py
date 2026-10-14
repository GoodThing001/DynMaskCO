"""Two real pretrained/random proposals after independent admission on one old live state; no research days.

Whole worker request is killed at the shared deadline. Public projection and live
installation remain explicitly outside this engineering qualification.
"""
import argparse,ast,copy,gc,hashlib,json,os,pickle,queue,subprocess,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'learned_after_strong_root_v1'
SCRIPTS=E/'scripts';CKPT=ROOT/'MASKCO_code/ckpts/cvrp100.ckpt'
MODES=('pretrained','random')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def boot():
    sys.path.insert(0,str(P));sys.path.insert(0,str(P/'actual_masked_fleet_engineering_v1'))
    sys.path.insert(0,str(P/'shadow_energy_correction_root_v1'));sys.path.insert(0,str(P/'common_candidate_saa_root_v2'))
    import public_fleet as pf;pf.bootstrap(SCRIPTS);return pf
def get_fixture():
    pf=boot()
    sys.path.insert(0,str(ROOT/'tools'))
    import qualify_pickup_joint_policy_transaction_root_v3 as old
    _,decode=old.codecs()
    source=P/'strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json'
    response=decode(json.loads(source.read_text(encoding='utf-8')))
    packet=response['scoring_packet']
    pf.validate_public(packet)
    if packet['admission_input']['pending'] or packet['admission_input']['current_order']!=77:
        raise ValueError('fixed post-independent-admission public state')
    if len(pf.family(packet)[1])<2:raise ValueError('actual model domain ineligible')
    return packet
def worker(mode):
    import psutil;psutil.Process().cpu_affinity([7])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_PLATFORMS']='cpu'
    pf=boot();import resident_actor as wire;import masked_fleet as mf
    from common_score import score_common
    from energy_complete_saa import EnergyCompleteSAA
    from scenario_saa import CondHistoricalSampler
    from hard_admission import canonical
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot
    from rolling_optimizer import RollingOptimizeReplanner
    with (P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl').open('rb') as f:history=pickle.load(f)
    provider=None
    if mode in ('pretrained','random','discard'):
        mf.bootstrap(SCRIPTS);from delivery_maskco import MaskCORouteProvider
        from flax import nnx
        import numpy as np
        provider=MaskCORouteProvider(CKPT,ROOT/'MASKCO_code',random_seed=42 if mode=='random' else None)
        for n in range(3,14):provider.prewarm(n,50.)
        h=hashlib.sha256();leaves=0
        for key,value in nnx.state(provider.model,nnx.Param).flat_state():
            h.update(str(key).encode());h.update(np.asarray(value.value).tobytes());leaves+=1
        provider.identity.update(actual_param_hash=h.hexdigest(),actual_param_leaves=leaves)
    from coldchain_evaluator_a1 import evaluate_trace_a1
    manifest=pf.source_hashes(SCRIPTS);manifest[str(Path(__file__).resolve())]=sha(__file__)
    initial={'kind':'ready','mode':mode,'source_start':manifest,'checkpoint_sha256':sha(CKPT),
      'history_sha256':sha(P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl'),'actual_cpu_affinity':psutil.Process().cpu_affinity(),
      'registered_synthetic_shapes':list(range(3,14)) if provider else [],'provider_identity':provider.identity if provider else None}
    print(json.dumps(initial),flush=True)
    for line in sys.stdin:
        q=wire.unwire(json.loads(line)['request']);packet=q['packet'];deadline=q['deadline'];started=time.perf_counter()
        result={'qualified':False,'mode':mode,'timeout':False,'selected_plan':packet['plan'],'deadline':deadline}
        try:
            if q.get('sleep_fault'):time.sleep(1.)
            pf.validate_public(packet)
            if provider:
                if len(pf.family(packet)[1])<2:
                    candidate=copy.deepcopy(packet['plan']);proposal={'kind':'ineligible_incumbent_fallback','reason':'fewer than two mutable orders; committed destination excluded','decoder_calls':0}
                else:
                    n=len(mf.topology(packet)[2])+1
                    if n not in range(3,14):raise ValueError('unwarmed observed graph; cannot exclude new setup')
                    _,proposal=mf.propose(provider,packet,SCRIPTS,mode=mode,seed=202610101903,mask_fraction=.3,guide_weight=1.,deadline=deadline)
                    candidate=proposal.get('raw_candidate',packet['plan'])
            elif mode=='strong_rolling':
                rp=RollingOptimizeReplanner(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],
                  contract=packet['contract'],cooling_share=0.,reject_penalty=packet['penalties'],time_limit=max(0.,deadline-time.perf_counter()),hist=history['history_times_classes'],k=10)
                rp.restore_state(copy.deepcopy(packet['snapshot']['replanner_state']))
                env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
                fleet,traces,served=restore_recourse_snapshot(packet['snapshot']);clock=packet['snapshot']['clock']
                plan,acc,rej,utility=rp._optimize(env,0,clock,fleet,served,set())
                candidate=canonical(packet,plan)
                proposal={'kind':'existing_rolling_relocate_swap_fixed_admissions','accepted_new':sorted(acc),'rejected_new':sorted(rej),'proxy_utility':utility}
            else:candidate=copy.deepcopy(packet['plan']);proposal={'kind':'actual_own_incumbent'}
            # Bypass the legacy immediate-close selector. Compare actual raw plans
            # with the same K10 future law and complete energy ledger.
            scores=score_common(packet,{'own_incumbent':packet['plan'],'actual_candidate':candidate},history,deadline,seed=202610111301)
            inc=scores['candidate_scores']['own_incumbent'];cand=scores['candidate_scores']['actual_candidate']
            if not inc['qualified']:raise ValueError('uncertified own incumbent')
            accept=cand['qualified'] and cand['mean_shadow_utility']>inc['mean_shadow_utility']+1e-10
            selected=candidate if accept else copy.deepcopy(packet['plan'])
            cert,_=pf.certificate(packet,selected)
            end={p:sha(p) for p in manifest};checkpoint_end=sha(CKPT)
            if time.perf_counter()>deadline:raise TimeoutError('whole request too late after certification/identity')
            result.update(qualified=cert['qualified'] is True and end==manifest and checkpoint_end==initial['checkpoint_sha256'],
              selected_plan=selected,raw_candidate=candidate,proposal=proposal,common_scores=scores,selected_certificate=cert,
              accepted_on_common_SAA=accept,packet_sha256=pf.digest(packet),source_end=end,checkpoint_end_sha256=checkpoint_end,
              worker_started=started,worker_completed=time.perf_counter(),actual_cpu_affinity=psutil.Process().cpu_affinity())
        except (TimeoutError,ValueError) as exc:result.update(reason=str(exc),timeout=isinstance(exc,TimeoutError))
        print(json.dumps({'kind':'response','result':wire.wire(result)}),flush=True)
def main():
    OUT.mkdir(exist_ok=False);pf=boot();sys.path.insert(0,str(P/'actual_masked_fleet_followup_v1'))
    import resident_actor as wire;import psutil
    psutil.Process().cpu_affinity([7])
    resource={'available_memory_gb':psutil.virtual_memory().available/2**30}
    if resource['available_memory_gb']<2:raise RuntimeError('not enough memory for isolated one-model setup')
    fixture_path=P/'strong_independent_owner_next_root_v2/RESPONSE_ATTEMPT_0.json'
    reg={'registered_utc':datetime.now(timezone.utc).isoformat(),'launcher_sha256':sha(__file__),
      'scope':'one actual post-independent-admission old public state, pretrained/random proposals only; not formally trained CE or fresh dynamic days',
      'fixture_sha256':sha(fixture_path),'fixture_selection':'fixed naturally revealed order77 after independent admission; no score-based selection','old_failed_fixture_preserved':True,'history_sha256':sha(P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl'),
      'checkpoint_sha256':sha(CKPT),'common_score_sha256':sha(P/'common_candidate_saa_root_v2/common_score.py'),
      'masked_fleet_sha256':sha(P/'actual_masked_fleet_engineering_v1/masked_fleet.py'),
      'energy_complete_saa_sha256':sha(P/'shadow_energy_correction_root_v1/energy_complete_saa.py'),
      'rolling_source_sha256':sha(SCRIPTS/'evaluation/rolling_optimizer.py'),
      'public_fleet_sha256':sha(P/'public_fleet.py'),'actual_cpu_affinity':[7],'K':10,'request_budget_s':10.,
      'synthetic_model_setup_shapes':list(range(3,14)),'setup_bound_s':120.,'arm_outer_bound_s':150.,
      'modes':list(MODES),'scenario_seed':202610111301,'mask_seed':202610101903,'mask_fraction':.3,'guide_weight':1.,
      'all_five_formal_deployment_qualified':False,'public_projection_live_installation_qualified':False,'research_new_days':0,'stable_online_gain':False}
    write(OUT/'REGISTRATION.json',reg);packet=get_fixture();records=[]
    for mode in MODES:
        q=queue.Queue();errors=[]
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker-mode',mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        def read_stream(proc=p,out=q):
            for line in proc.stdout:
                try:out.put(json.loads(line))
                except json.JSONDecodeError:errors.append(line)
            out.put({'kind':'closed'})
        def read_errors(proc=p,err=errors):
            for line in proc.stderr:err.append(line)
        threading.Thread(target=read_stream,daemon=True).start();threading.Thread(target=read_errors,daemon=True).start()
        row={'mode':mode,'worker_pid':p.pid,'setup_started':time.perf_counter()}
        try:
            ready=q.get(timeout=120.)
            if ready.get('kind')!='ready':raise RuntimeError('worker setup failed')
            row['ready']=ready;row['setup_wall_s']=time.perf_counter()-row['setup_started']
            t0=time.perf_counter();deadline=t0+10.
            p.stdin.write(json.dumps({'request':wire.wire({'packet':packet,'deadline':deadline})})+'\n');p.stdin.flush()
            try:answer=q.get(timeout=max(0.,deadline-time.perf_counter()))
            except queue.Empty:answer={'kind':'timeout'}
            received=time.perf_counter();row.update(started=t0,deadline=deadline,received=received,wall_s=received-t0)
            if answer.get('kind')=='response' and received<=deadline:
                row['actual']=wire.unwire(answer['result']);row['parent_on_time']=True
            else:row.update(actual={'qualified':False,'timeout':True,'selected_plan':packet['plan']},parent_on_time=False)
        except (queue.Empty,RuntimeError) as exc:row['setup_failure']=str(exc)
        finally:
            if p.poll() is None:p.kill()
            p.wait(timeout=10);row['worker_reaped']=p.poll() is not None;row['stderr']=errors
        records.append(row)
        from audit_io import encode
        write(OUT/(mode+'_RAW.json'),encode(row));print(json.dumps({'mode':mode,'qualified':row.get('actual',{}).get('qualified',False),'request_s':row.get('wall_s'),'setup_s':row.get('setup_wall_s'),'worker_reaped':row['worker_reaped']}),flush=True)
    from audit_io import encode
    write(OUT/'RESULT.json',encode({'public_packet':packet,'records':records,'registration_sha256':sha(OUT/'REGISTRATION.json'),
      'all_five_formal_deployment_qualified':False,'public_projection_live_installation_qualified':False,'stable_online_gain':False}))
    return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker-mode',choices=MODES);a=ap.parse_args()
    if a.worker_mode:
        sys.path.insert(0,str(P/'actual_masked_fleet_followup_v1'));worker(a.worker_mode)
    else:sys.exit(main())
