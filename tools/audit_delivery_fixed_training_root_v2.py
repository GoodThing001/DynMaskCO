"""Independent fixed schedule arithmetic and optional actual saved tensor audit."""
import hashlib,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'research/review_3080_20261010/delivery'
B=D/'remote_finite_research_root_v2/training_downloaded'
S=D/'finite_training_research_root_v2'
T=D/'remote_finite_research_root_v2/teacher_downloaded'
V=D/'independent_training_tensors_root_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main(final=False):
    reg=load(S/'RESEARCH_REGISTRATION.json');result=load(B/'TRAINING_RESULT.json')
    teacher=[json.loads(s) for s in (T/'TEACHER_STATES.jsonl').read_text().splitlines()]
    checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('actual_supervised_exit0',load(B/'training_RUN_EXIT.json')['exit_code']==0)
    ck('all_actual_frozen_sources_unchanged',all(load(B/'INDEPENDENT_ACTUAL_END_SOURCE_CHECK.json').values()))
    ck('actual_registered_final_training_identity',sha(S/'RESEARCH_REGISTRATION.json')==result['registration_sha256']=='1f19554a2790d258b7fca4f081aeae38b7250acab817142b08825cf91ab18af1')
    ck('teacher_archive_matches_independent_acceptance',sha(T/'TEACHER_STATES.jsonl')==result['teacher_archive_sha256']==load(T/'ACCEPTED_TEACHER_STATES.json')['teacher_archive_sha256'])
    ck('exact_registered128_batch4_256',result['mode']=='research' and result['actual_states']==256 and result['fixed_steps_each']==128 and result['batch_size']==4)
    rng=np.random.default_rng(reg['seeds']['training_schedule'])
    fixed=np.concatenate([rng.permutation(256),rng.permutation(256)]).reshape(128,4).tolist()
    ck('independently_regenerated_schedule',fixed==result['fixed_schedule'])
    ck('all256_states_exactly_twice',Counter(x for batch in fixed for x in batch)==Counter({i:2 for i in range(256)}))
    ck('unique_all64x4_training_keys',len(teacher)==256 and {tuple(r['state_key']) for r in teacher}=={(i,j) for i in range(64) for j in range(4)})
    ck('actual_two_registered_arms',set(result['arms'])=={'reconstruction_control','paired_objective'})
    ck('reported_same_initial_identity',len({a['initial_identity'] for a in result['arms'].values()})==1)
    ck('distinct_fixed_final_tensor_digests',len({a['final_identity'] for a in result['arms'].values()})==2)
    for arm,coef in [('reconstruction_control',0.),('paired_objective',.2)]:
        a=result['arms'][arm];steps=B/(arm+'_STEPS.jsonl');rows=[json.loads(s) for s in steps.read_text().splitlines()]
        ck(arm+':complete_actual128_step_bytes',sha(steps)==a['steps_archive_sha256'] and [r['step'] for r in rows]==list(range(1,129)))
        ck(arm+':complete_same_schedule',[r['state_indices'] for r in rows]==fixed)
        previous=a['initial_identity']
        for i,(r,idx) in enumerate(zip(rows,fixed)):
            p=arm+':'+str(i+1)+':'
            ck(p+'teacher_keys',r['state_keys']==[teacher[j]['state_key'] for j in idx])
            ck(p+'teacher_state_sha',r['state_sha256']==[teacher[j]['state_sha256'] for j in idx])
            ck(p+'fixed_coefficient',r['coefficient']==coef)
            ck(p+'finite_losses',all(math.isfinite(r[k]) for k in ('total_loss','CE_loss','preference_loss')))
            ck(p+'independent_loss_decomposition',abs(r['total_loss']-r['CE_loss']-coef*r['preference_loss'])<2e-6)
            ck(p+'continuous_parameter_identity',r['actual_before_parameter_digest']==previous)
            ck(p+'actual_updated_parameter_digest',r['actual_before_parameter_digest']!=r['actual_after_parameter_digest'])
            ck(p+'actual_full_gradient_digest',r['actual_gradient_leaf_count']==234 and r['gradient_finite'] is True and len(r['actual_gradient_tensor_digest'])==64)
            previous=r['actual_after_parameter_digest']
        ck(arm+':actual_fixed_final_chain',previous==a['final_identity'])
        ck(arm+':fixed128_complete_model_update',a['fixed_steps']==128 and a['parameter_change_counts']=={'full':234,'encoder':160,'decoder':60})
    ck('snapshot_limits_retained',result['snapshot_only'] is True and result['dynamic_online_qualified'] is False and result['stable_gain_established'] is False)
    if final:
        raw=V/'downloaded';actual=load(raw/'ACTUAL_FIXED_TENSOR_VERIFICATION.json')
        whole=load(V/'INDEPENDENT_WHOLE_SAVED_WEIGHT_BYTES.json')
        ck('fresh_process_actual_reload_exit0',load(raw/'RUN_EXIT.json')['exit_code']==0)
        ck('actual_independent_verifier_source_bytes',all(load(raw/'INDEPENDENT_VERIFIER_ACTUAL_SOURCE_CHECK.json').values()))
        ck('tensor_verifier_bound_actual_training_bytes',actual['training_result_sha256']==sha(B/'TRAINING_RESULT.json'))
        ck('verifier_complete_frozen_source_end',all(actual['actual_source_end'].values()))
        initial=actual['initial_inventory'];ck('initial_full234_76M_elements',len(initial)==234 and sum(v['elements'] for v in initial.values())==76263424)
        initial_digest=digest({p:v['sha256'] for p,v in initial.items()})
        for arm,a in result['arms'].items():
            evidence=actual['arms'][arm];inventory=evidence['final_inventory'];out=raw/(arm+'_OUTPUTS.npz')
            p=arm+':independent_reload:'
            ck(p+'complete_remote_checkpoint_bytes',whole[arm]['whole_remote_bytes_read'] is True and whole[arm]['sha256']==a['checkpoint_sha256']==evidence['actual_sha256'] and whole[arm]['bytes']==evidence['actual_size_bytes'])
            ck(p+'initial_identity_recomputed_from_actual_arrays',initial_digest==evidence['actual_initial_digest']==a['initial_identity'])
            ck(p+'final_identity_recomputed_from_actual_arrays',digest({p:v['sha256'] for p,v in inventory.items()})==evidence['actual_final_digest']==a['final_identity'])
            ck(p+'full_same234_shapes',set(inventory)==set(initial) and all(inventory[p]['shape']==initial[p]['shape'] for p in initial))
            ck(p+'finite_actual76263424_elements',sum(v['elements'] for v in inventory.values())==76263424 and all(v['finite'] is True for v in inventory.values()))
            changes={group:sum(v['sha256']!=initial[p]['sha256'] for p,v in inventory.items() if group=='full' or p.startswith("('"+group+"',")) for group in ('full','encoder','decoder')}
            ck(p+'actual_parameter_update_recomputed',changes==evidence['actual_changes']==a['parameter_change_counts'])
            ck(p+'actual_saved_output_archive',sha(out)==evidence['output_archive_sha256'])
            z=np.load(out);masked=z['masked'];offdiag=~np.eye(51,dtype=bool)
            ck(p+'independent_output_reload_exact',np.array_equal(masked,z['reloaded']))
            ck(p+'finite_offdiagonal_masked_activation',np.isfinite(masked[:,offdiag]).all() and not np.array_equal(masked[:,offdiag],z['unmasked'][:,offdiag]))
    audit={'scope':'registered fixed training identity; deployment and gain are separate',
        'checks':checks,'checks_count':len(checks),'failed':sum(not r['pass'] for r in checks),
        'registration_sha256':sha(S/'RESEARCH_REGISTRATION.json'),'fixed_steps_each':128,'teacher_states':256,
        'actual_saved_tensor_reload_independently_qualified':final and all(r['pass'] for r in checks),
        'independent_per_step_gradient_reconstruction':False,
        'per_step_gradient_limits':'actual hashes and finite count recorded by trainer; this auditor independently checks schedule/identity chains and final tensors, not reruns128 gradients',
        'final_weights':{arm:a['checkpoint_sha256'] for arm,a in result['arms'].items()},
        'new_final_weights_budget_and_timeout_injection_qualified':False,
        'fresh_development_execution_qualified':False,'confirmation_enabled':False,'dynamic_online_gain':False}
    name='ACCEPTED_FIXED_TRAINING.json' if final else 'INDEPENDENT_FIXED_STEP_ARITHMETIC_AUDIT.json'
    with (B/name).open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2,allow_nan=False)
    print(json.dumps({k:audit[k] for k in ('checks_count','failed','fixed_steps_each','actual_saved_tensor_reload_independently_qualified','final_weights')}))
    return int(audit['failed']>0)
if __name__=='__main__':sys.exit(main('--final' in sys.argv))
