"""New isolated common K10 scorer QA. No actual five-arm deployment qualification."""
import ast,hashlib,json,os,pickle,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'common_candidate_saa_root_v2'
MODULE=r'''
import copy,math,time
import numpy as np
def score_common(packet,plans,history,deadline,seed=202610111301):
    import public_fleet as pf
    from scenario_saa import CondHistoricalSampler
    from energy_complete_saa import EnergyCompleteSAA
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot
    pf.validate_public(packet)
    rp=EnergyCompleteSAA(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],
      contract=packet['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,
      sampler=CondHistoricalSampler(history['history'],n_neighbors=10),reject_penalty={0:5.,1:10.,2:15.},
      K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
    state=copy.deepcopy(packet['snapshot']['replanner_state']);rp.restore_state(state)
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    fleet,traces,served=restore_recourse_snapshot(packet['snapshot']);clock=float(packet['snapshot']['clock'])
    rp._sync_id_map(env,0,clock)
    snapshot=rp._build_snapshot(env,0,clock,fleet,served)
    # Draw once from public prefix and original known historical pool. No candidate
    # resource, arm label or private current-day future affects the demand law.
    scenarios=rp.sampler.sample(snapshot,np.random.default_rng(seed),10)
    scenario_hash=pf.digest(scenarios)
    space,orders=rp._build_space(env,0,scenarios);index={id(o):space.n_real+j for j,o in enumerate(orders)}
    indices=[[index[id(o)] for o in scen] for scen in scenarios]
    result={}
    for label,plan in plans.items():
        if time.perf_counter()>=deadline:raise TimeoutError('No partial candidate comparison')
        certificate,_=pf.certificate(packet,plan)
        if certificate['qualified'] is not True:
            result[label]={'qualified':False,'certificate':certificate,'scenario_sha256':scenario_hash,'branch_values':[]};continue
        shadow=rp._sim_states(env,0,clock,fleet,served,{int(k):list(v) for k,v in plan.items()})
        values=[]
        for ids in indices:
            if time.perf_counter()>=deadline:raise TimeoutError('No partial K10 candidate acceptance')
            value=float(rp._sim_scenario(env,0,clock,shadow,ids,space))
            if not math.isfinite(value):raise ValueError('Nonfinite complete scenario')
            values.append(value)
        if time.perf_counter()>=deadline:raise TimeoutError('Common candidate certification/submission too late')
        result[label]={'qualified':True,'certificate':certificate,'scenario_sha256':scenario_hash,
          'branch_values':values,'mean_shadow_utility':float(np.mean(values)),'complete_scenarios':10}
    return {'scenario_sha256':scenario_hash,'scenarios':scenarios,'candidate_scores':result,'sampling_once':True,
      'shadow_scores_are_not_realized_online_utility':True,'all_five_actual_deployment_qualified':False}
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker():
    sys.path.insert(0,str(P));sys.path.insert(0,str(P/'actual_masked_fleet_followup_v1'))
    import public_fleet as pf;pf.bootstrap(E/'scripts')
    sys.path.insert(0,str(P/'shadow_energy_correction_root_v1'));sys.path.insert(0,str(OUT))
    from common_score import score_common
    from audit_io import decode,encode
    with (P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl').open('rb') as f:history=pickle.load(f)
    import coldchain_contract,coldchain_state,strict_online_env
    from dataclasses import is_dataclass
    classes={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env) for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
    raw=decode(json.loads((P/'complete_history_energy_qa_root_v1/complete_ledger/FULLRAW.json').read_text()),classes)
    packets=[r['public_request'] for r in raw['admissions'] if any(len(v)>=2 for v in r['public_request']['plan'].values())]
    if not packets:raise ValueError('No observed positive active suffix QA fixture')
    packet=packets[0]
    labels=['own_incumbent','pretrained','random','discard','strong_rolling']
    plans={label:packet['plan'] for label in labels};start=time.perf_counter()
    result=score_common(packet,plans,history,start+10.)
    first_wall=time.perf_counter()-start
    checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    reference=result['candidate_scores']['own_incumbent']
    for label in labels:
        score=result['candidate_scores'][label]
        ck(label+':same_plan_exact_shadow_values',score['branch_values']==reference['branch_values'])
        ck(label+':all10_complete_and_known_hard',score['qualified'] and score['complete_scenarios']==10)
        ck(label+':same_actual_draw_hash',score['scenario_sha256']==result['scenario_sha256'])
    for j,scenario in enumerate(result['scenarios']):
        ck('draw'+str(j)+':only_future_to_public_clock',all(o.reveal>packet['snapshot']['clock']+1e-6 for o in scenario))
    # Demand sampling uses orders+clock only. Changing candidate budget/plan must
    # not redraw, and repeated independent invocation with same public seed agrees.
    again=score_common(packet,{'same':packet['plan']},history,time.perf_counter()+10.)
    ck('repeat_exact_common_draws',again['scenario_sha256']==result['scenario_sha256'])
    ck('repeat_exact10_values',again['candidate_scores']['same']['branch_values']==reference['branch_values'])
    try:score_common(packet,plans,history,time.perf_counter()-1.)
    except TimeoutError:ck('expired_request_cannot_commit_partial',True)
    else:ck('expired_request_cannot_commit_partial',False)
    failed=sum(not x['pass'] for x in checks)
    result.update(checks=checks,checks_count=len(checks),failed=failed,scope='five identical plan labels only; actual model and rolling proposal integration NOT qualified',
      public_packet=packet,elapsed_first_call_s=first_wall,stable_online_gain=False)
    with (OUT/'RESULT.json').open('x') as f:json.dump(encode(result),f,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'actual_five_arm_deployment_qualified':False}));return int(failed>0)
def main():
    OUT.mkdir(exist_ok=False);ast.parse(MODULE);(OUT/'common_score.py').write_bytes(MODULE.encode())
    reg={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'known public fixture same-plan label scorer engineering; no new days/models/training',
      'common_score_sha256':sha(OUT/'common_score.py'),'launcher_sha256':sha(Path(__file__)),
      'history_sha256':sha(P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl'),
      'original_saa_sha256':sha(E/'scripts/evaluation/scenario_saa.py'),'energy_complete_saa_sha256':sha(P/'shadow_energy_correction_root_v1/energy_complete_saa.py'),
      'scenario_seed':202610111301,'K':10,'maximum_seconds':60,'all_five_actual_arms_qualified':False,'stable_online_gain':False}
    (OUT/'REGISTRATION.json').write_text(json.dumps(reg,indent=2))
    started=time.perf_counter()
    with (OUT/'stdout.log').open('xb') as stdout,(OUT/'stderr.log').open('xb') as stderr:
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'worker'],stdout=stdout,stderr=stderr,
          env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        try:code=p.wait(timeout=60)
        except subprocess.TimeoutExpired:p.kill();p.wait();code=124
    (OUT/'RUN_EXIT.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.perf_counter()-started,'worker_reaped':p.poll() is not None}))
    print((OUT/'stdout.log').read_text());return code
if __name__=='__main__':sys.exit(worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main())
