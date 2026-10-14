"""Recompute engineering fixture, objective arithmetic, route and byte identities."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
B=ROOT/'research/review_3080_20261010/delivery/remote_features_bundle_v1/downloaded'
OLD=ROOT/'research/review_3080_20261010/delivery/remote_actual_bundle_v2/downloaded'

def main():
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    q=load(B/'FEATURES_OBJECTIVE_QA.json');f=load(B/'INPUT_AND_TEACHER_QUALIFICATION.json')
    byte=load(B/'INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json')
    checks=[]
    def check(name,ok):checks.append({'name':name,'pass':bool(ok)})
    check('exit0',load(B/'RUN_EXIT.json')['exit_code']==0)
    check('actual_remote_sources_unchanged',all(load(B/'ACTUAL_END_SOURCE_CHECK.json').values()))
    check('registration_pin',q['registration_sha256']=='c1110955d11e4f619d55fe11ab7f711d20d1f7bf30e354ef07122f62aa5e7a8a')
    check('original_checkpoint_pin',q['original_checkpoint_sha256']=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db')
    check('scope_not_scientific',q['research_model_eligible'] is False and q['dynamic_online_qualified'] is False and q['stable_gain_established'] is False)
    check('initial_front3_equals_prior_actual_input',np.array_equal(np.asarray(f['features'])[:,:,:3],np.asarray(load(OLD/'ACTUAL_TEACHER_FIXTURE.json')['raw_features'])))
    original=f['original_inventory'];mapped=f['mapped_inventory']
    check('234_same_parameter_keys',len(original)==len(mapped)==234 and original.keys()==mapped.keys())
    elements=0
    for name,record in mapped.items():
        elements+=int(np.prod(record['shape']))
        if name=="('init_proj', 'kernel')":
            check('only_expanded_projection_shape',original[name]['shape']==[3,512] and record['shape']==[10,512])
        else:check('unchanged_original_weight:'+name,record==original[name])
    check('actual_10D_parameter_elements',elements==f['mapped_parameter_elements']==76263424)
    check('3D_10D_forward_mapping',np.isfinite(f['mapping_max_logit_error']) and f['mapping_max_logit_error']<=1e-4)
    check('zero_initial_extra_fields',f['initial_attribute_removal_max_logit_error']<=1e-6)
    for name,passed in f['feature_rejections'].items():check('invalid_feature_rejected:'+name,passed is True)
    for sub in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(E/'scripts'/sub))
    from delivery_equal_schedule import generate
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_maskco import partial_adjacency
    contract=DeliveryContract();row=f['teacher']
    identity=load(E/'results/research_20261010_continuation_v2/delivery/train_split_v1/TRAIN_SPLIT_IDENTITY.json')
    data,_=generate(identity['train_seed'],row['instance'])
    # JSON serializes immutable contract tuples as lists. Compare canonical
    # payloads and the actual hash rather than Python container classes.
    check('actual_contract_identity',contract.identity()==f['contract_hash'] and json.loads(json.dumps(contract.payload(),allow_nan=False))==f['contract'])
    incumbent=evaluate(data,row['incumbent_routes_before'],contract)
    teacher=evaluate(data,row['candidate_routes'],contract)
    check('independent_incumbent_C0_cost',abs(incumbent['cost']-f['incumbent_full_C0_report']['cost'])<1e-8)
    check('independent_teacher_C0_cost',abs(teacher['cost']-f['teacher_full_C0_report']['cost'])<1e-8)
    check('complete_hard_pair',incumbent['hard'] is True and teacher['hard'] is True and incumbent['complete'] is True and teacher['complete'] is True)
    delta=incumbent['cost']-teacher['cost'];sign=0 if abs(delta)<=1e-8 else (1 if delta>0 else -1)
    check('independent_full_cost_preference_sign',sign==f['pair_sign'])
    x=np.asarray(f['features']);physics=contract.physics();classes=np.asarray(data['temp_class'],int)
    for column,key in enumerate(('tw_start','tw_end','service_time','reveal_time','ready_time'),3):
        expected=np.asarray(data[key],np.float32)/contract.horizon_h;expected[0]=0
        check('explicit_time_feature:'+key,np.max(np.abs(x[0,:,column]-expected))<1e-7)
    for column,expected in [(8,np.asarray(physics.thermal.target_temperature_c,np.float32)[classes]/20),
                            (9,np.asarray(physics.quality.initial_value,np.float32)[classes]/9)]:
        expected[0]=0;check('contract_attribute_feature:'+str(column),np.max(np.abs(x[0,:,column]-expected))<1e-7)
    context=partial_adjacency(row['incumbent_routes_before'],set(row['mask']),len(data['demands']))
    arms=q['arms'];a=arms['reconstruction_control'];b=arms['paired_objective']
    check('same_actual_initial_digest',a['same_initial_state_digest']==b['same_initial_state_digest'])
    check('CE_identical_before_step',a['loss_CE']==b['loss_CE'])
    check('preference_identical_before_step',a['loss_preference']==b['loss_preference'] and a['score_delta']==b['score_delta'])
    check('unique_registered_objective_difference',a['objective_coefficient']==0. and b['objective_coefficient']==.2)
    check('different_actual_final_parameters',a['parameter_inventory']!=b['parameter_inventory'])
    for name,arm in arms.items():
        preference=float(np.logaddexp(0.,-sign*arm['score_delta'])) if sign else 0.
        check(name+':preference_arithmetic',abs(preference-arm['loss_preference'])<1e-6)
        check(name+':objective_arithmetic',abs(arm['loss_total']-(arm['loss_CE']+arm['objective_coefficient']*preference))<1e-6)
        check(name+':fixed1step',arm['fixed_steps']==1)
        for group,stats in arm['gradient_stats'].items():check(name+':finite_nonzero_gradient:'+group,stats['finite']==stats['leaves'] and stats['nonzero']>0)
        for group in ('encoder','decoder'):check(name+':actual_parameter_update:'+group,arm['parameter_change_counts'][group]>0)
        check(name+':new_feature_update',arm['extra_feature_projection_gradient_norm']>0 and arm['attribute_removal_max_logit_effect_after_step']>0)
        r=evaluate(data,arm['routes'],contract)
        check(name+':independent_projection_hard_complete',r['hard'] is True and r['complete'] is True)
        check(name+':independent_projection_cost',abs(r['cost']-arm['projection']['cost'])<1e-8)
        check(name+':retained_edges',np.all(partial_adjacency(arm['routes'],set(),len(data['demands']))[context>0]>0))
        check(name+':actual_weight_full_byte_SHA',byte[name+'_ENGINEERING_ONLY.pkl']['sha256']==arm['save_sha256'] and byte[name+'_ENGINEERING_ONLY.pkl']['full_remote_bytes_read'] is True)
        check(name+':runner_save_reload_exact',arm['reload_logits_exact'] is True)
    for key,expected in q['source_end'].items():
        group,relative=key.split('/',1);p=(ROOT/'MASKCO_code' if group=='upstream' else E)/relative
        check('local_actual_source_end:'+key,hashlib.sha256(p.read_bytes()).hexdigest()==expected)
    report={'kind':'independent_actual_10D_paired_objective_engineering_audit','n_checks':len(checks),
        'n_failures':sum(not c['pass'] for c in checks),'engineering_pass':all(c['pass'] for c in checks),'checks':checks,
        'research_model_eligible':False,'stable_online_gain_established':False,
        'actual_operator_dtype':'explicit_float32_CPU_adapter_to_original_bfloat16_config',
        'limitations':['one already observed snapshot fixture, not dynamic/fresh effectiveness',
                       'C0 replay uses frozen evaluator, not independent physical calibration',
                       'full saved weight bytes independently hashed; saved tensors not independently reloaded locally',
                       'no all-in10s whole-day model deployment or fixed finite scientific training completed']}
    report['prior_audit_tool_failure']={'file':'INDEPENDENT_ROOT_ACTUAL_FEATURES_OBJECTIVE_AUDIT.json',
        'cause':'Python contract tuples versus JSON lists; actual hash and canonical payload already equal',
        'scientific_rerun':False}
    with (B/'INDEPENDENT_ROOT_ACTUAL_FEATURES_OBJECTIVE_AUDIT_v2.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('n_checks','n_failures','engineering_pass')}))
    return 0 if report['engineering_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
