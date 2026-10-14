"""Independent schedule, objective arithmetic and remote byte identity review."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/review_3080_20261010/delivery/finite_training_preparation_v1'
B=ROOT/'research/review_3080_20261010/delivery/remote_finite_training_bundle_v1/downloaded'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    r=load(B/'TRAINING_RESULT.json');reg=load(SOURCE/'FINITE_ENGINEERING_REGISTRATION.json')
    teacher=[json.loads(x) for x in (SOURCE/'ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl').read_text().splitlines()]
    checks=[]
    def ck(name,passed):checks.append({'name':name,'pass':bool(passed)})
    ck('supervised_exit0',load(B/'RUN_EXIT.json')['exit_code']==0)
    ck('actual_300s_outer_bound',load(B/'RUN_EXIT.json')['wall_seconds']<300)
    ck('actual_source_end_bytes',all(load(B/'ACTUAL_END_SOURCE_CHECK.json').values()))
    ck('registered_exact_finite_source',sha(SOURCE/'FINITE_ENGINEERING_REGISTRATION.json')==r['registration_sha256']=='1387fbb41ee5fbce2496eca133dddb89c03ddf1bc67b3485ccc3e8f15ee77ade')
    ck('teacher_exact_archive',sha(SOURCE/'ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl')==r['teacher_archive_sha256'])
    ck('same_8_unique_states',len(teacher)==8 and len({tuple(x['state_key']) for x in teacher})==8)
    ck('batch4_fixed2_steps',r['actual_states']==8 and r['batch_size']==4 and r['fixed_steps_each']==2)
    fixed=np.random.default_rng(202610110073).permutation(8).reshape(2,4).tolist()
    ck('independent_numpy_registered_schedule',fixed==r['fixed_schedule'])
    ck('all_8_states_once',sorted(x for b in fixed for x in b)==list(range(8)))
    byte=load(B/'INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json')
    arms=r['arms'];ck('exact_two_registered_arms',set(arms)=={'reconstruction_control','paired_objective'})
    ck('same_reported_initial_parameters',len({x['initial_identity'] for x in arms.values()})==1)
    ck('distinct_reported_actual_final_parameters',len({x['final_identity'] for x in arms.values()})==2)
    for name,coefficient in [('reconstruction_control',0.),('paired_objective',.2)]:
        a=arms[name];p=B/(name+'_STEPS.jsonl');rows=[json.loads(s) for s in p.read_text().splitlines()]
        ck(name+':actual_step_archive_bytes',sha(p)==a['steps_archive_sha256'])
        ck(name+':unique_continuous_steps',[x['step'] for x in rows]==[1,2])
        ck(name+':same_schedule',[x['state_indices'] for x in rows]==fixed)
        for step,(row,indices) in enumerate(zip(rows,fixed),1):
            prefix=name+':step'+str(step)+':'
            ck(prefix+'actual_training_keys',row['state_keys']==[teacher[i]['state_key'] for i in indices])
            ck(prefix+'teacher_record_sha',row['state_sha256']==[teacher[i]['state_sha256'] for i in indices])
            ck(prefix+'fixed_objective',row['coefficient']==coefficient)
            ck(prefix+'finite_losses',all(np.isfinite(row[k]) for k in ('CE_loss','preference_loss','total_loss')))
            ck(prefix+'independent_loss_decomposition',abs(row['CE_loss']+coefficient*row['preference_loss']-row['total_loss'])<2e-6)
            ck(prefix+'reported_full_finite_gradient',row['gradient_finite'] is True and row['all234_parameter_leaves_in_gradient'] is True)
        actual=byte[name+'_FIXED_FINAL.pkl']
        ck(name+':independent_whole_remote_weight_bytes',actual['full_remote_bytes_read'] is True and actual['bytes']==305075564 and actual['sha256']==a['checkpoint_sha256'])
        ck(name+':reported_complete_update',a['parameter_change_counts']=={'full':234,'encoder':160,'decoder':60})
        ck(name+':fixed_final_save_reload_reported',a['fixed_steps']==2 and a['save_reload_params_equal'] is True)
    ck('actual_parameter_elements',r['actual_full_model_parameter_elements']==76263424)
    ck('not_research_or_online_gain',r['mode']=='engineering' and r['snapshot_only'] is True and all(r[k] is False for k in ('research_model_eligible','dynamic_online_qualified','stable_gain_established')))
    result={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),
        'independent_schedule_loss_and_byte_audit':True,'independent_local_saved_tensor_reload':False,
        'gradient_and_parameter_update_evidence':'actual runner reports; this auditor does not reconstruct unrecorded per-step tensors',
        'teacher_independent_report':str(SOURCE/'ENGINEERING_TEACHER_QA_v1/ROOT_ENGINEERING_TEACHER_AUDIT.json'),
        'research_model_qualified':False,'stable_core_online_gain':False}
    with (B/'INDEPENDENT_ROOT_FINITE_TRAINING_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':result['failed']}));return int(result['failed']>0)
if __name__=='__main__':sys.exit(main())
