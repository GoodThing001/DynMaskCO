"""Recompute recorded common draws, means and public certificates; narrow scope."""
import hashlib,json,math,sys
from dataclasses import is_dataclass
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'common_candidate_saa_root_v2'
sys.path.insert(0,str(P));import public_fleet as pf
pf.bootstrap(E/'scripts')
import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
from audit_io import decode
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa)
 for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    reg=json.loads((OUT/'REGISTRATION.json').read_text());r=decode(json.loads((OUT/'RESULT.json').read_text()),CLASSES);checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('original_worker28_checks0_failure',r['checks_count']==28 and r['failed']==0 and all(x['pass'] for x in r['checks']))
    ck('bounded_outer60_exit0',json.loads((OUT/'RUN_EXIT.json').read_text())['exit_code']==0 and json.loads((OUT/'RUN_EXIT.json').read_text())['wall_seconds']<60)
    ck('exact_scorer_bytes',sha(OUT/'common_score.py')==reg['common_score_sha256'])
    ck('original_saa_unchanged',sha(E/'scripts/evaluation/scenario_saa.py')==reg['original_saa_sha256'])
    ck('correction_source_unchanged',sha(P/'shadow_energy_correction_root_v1/energy_complete_saa.py')==reg['energy_complete_saa_sha256'])
    ck('actual_history_unchanged',sha(P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl')==reg['history_sha256'])
    ck('sampling_exact10',len(r['scenarios'])==10 and pf.digest(r['scenarios'])==r['scenario_sha256'])
    ck('first_shared_call_within10',0<r['elapsed_first_call_s']<=10)
    cert,_=pf.certificate(r['public_packet'],r['public_packet']['plan'])
    reference=r['candidate_scores']['own_incumbent']
    for name,score in r['candidate_scores'].items():
        ck(name+':all_complete10',score['qualified'] and len(score['branch_values'])==score['complete_scenarios']==10)
        ck(name+':actual_same_draws',score['scenario_sha256']==pf.digest(r['scenarios']))
        ck(name+':mean_recomputed',math.isclose(float(np.mean(score['branch_values'])),score['mean_shadow_utility'],abs_tol=1e-10))
        ck(name+':identical_plan_identical_values',score['branch_values']==reference['branch_values'])
        ck(name+':exact_public_certificate',pf.digest(score['certificate'])==pf.digest(cert) and cert['qualified'])
    for j,scenario in enumerate(r['scenarios']):ck('causal_historical_draw'+str(j),all(order.reveal>r['public_packet']['snapshot']['clock']+1e-6 for order in scenario))
    result={'checks':checks,'checks_count':len(checks),'failed':sum(not c['pass'] for c in checks),
      'original_same_plan_worker_checks':28,'recorded_raw_sha256':sha(OUT/'RESULT.json'),
      'same_plan_labels_only':True,'all_five_actual_model_and_rolling_arms_qualified':False,
      'atomic_all_in_projection_certification_submission_qualified':False,'fresh_days':0,
      'independent_physical_implementation':False,'stable_online_gain':False,
      'limits':'This validates candidate scorer interface with identical plans. Does not validate actual candidates, matching admission, complete rejection logs or10s atomic end-to-end parent.'}
    with (OUT/'INDEPENDENT_ROOT_COMMON_SCORER_QA.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':result['failed'],'actual_five_arm_qualified':False}));return int(result['failed']>0)
if __name__=='__main__':sys.exit(main())
