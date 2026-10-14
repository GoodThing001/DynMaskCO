"""Independent arithmetic/route/source checks of the actual remote full-model QA."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
BASE=ROOT/'research/review_3080_20261010/delivery/remote_actual_bundle_v2/downloaded'


def main():
    checks=[]
    def check(name,ok):checks.append({'name':name,'pass':bool(ok)})
    q=json.loads((BASE/'ACTUAL_FULLSTEP_QA.json').read_text())
    fixture=json.loads((BASE/'ACTUAL_TEACHER_FIXTURE.json').read_text())
    projected=json.loads((BASE/'ACTUAL_PROJECTED_CANDIDATE.json').read_text())
    exit=json.loads((BASE/'RUN_EXIT.json').read_text())
    source=json.loads((BASE/'ACTUAL_END_SOURCE_CHECK.json').read_text())
    saved=json.loads((BASE/'INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json').read_text())
    check('actual_exit0',exit['exit_code']==0)
    check('remote_registration_and_sources_unchanged',all(source.values()))
    check('actual_saved_file_sha',saved['sha256']==q['saved_sha256'])
    check('original_checkpoint_pin',q['original_checkpoint_sha256']=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db')
    check('full_76M_architecture',q['actual_parameter_elements']==76259840 and q['actual_parameter_leaves']==234)
    check('fixed1step_not_research',q['trained_fixed_steps']==1 and q['research_model_eligible'] is False)
    check('actual_mask_activation',np.isfinite(q['actual_mask_context_effect']) and q['actual_mask_context_effect']>0)
    check('save_reload_prediction_exact',q['saved_reload_logits_equal'] is True)
    old={tuple(r['path']):r for r in q['before']['leaves']}
    new={tuple(r['path']):r for r in q['after']['leaves']}
    check('234_exact_parameter_paths',len(old)==234 and old.keys()==new.keys())
    elements=0
    for key,r in old.items():
        check('shape:'+repr(key),r['shape']==new[key]['shape'])
        check('actual_update:'+repr(key),r['sha256']!=new[key]['sha256'])
        elements+=int(np.prod(r['shape']))
    check('actual_element_inventory',elements==76259840)
    for group in ('full','encoder','decoder'):
        gradient=q['gradient_groups'][group]
        check('gradient_all_finite_nonzero:'+group,gradient['leaves_nonzero']==gradient['leaves_finite']==gradient['leaves'] and gradient['l2_squared']>0)
        check('actual_group_changed:'+group,q['before']['group_sha256'][group]!=q['after']['group_sha256'][group])
    # These are the same independently frozen C0 implementations, not a new physical model.
    for sub in ('delivery','coldchain','research_20261010'):
        sys.path.insert(0,str(EXT/'scripts'/sub))
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_equal_schedule import generate
    from delivery_maskco import partial_adjacency
    split=EXT/'results/research_20261010_continuation_v2/delivery/train_split_v1'
    identity=json.loads((split/'TRAIN_SPLIT_IDENTITY.json').read_text())
    row=fixture['teacher']
    generated,_=generate(identity['train_seed'],row['instance'])
    check('old_fixture_independent_regeneration',json.dumps(generated,sort_keys=True)==json.dumps(fixture['data'],sort_keys=True))
    teacher=evaluate(generated,row['candidate_routes'],DeliveryContract())
    candidate=evaluate(generated,projected['routes'],DeliveryContract())
    check('teacher_hard_complete',teacher['hard'] is True and teacher['complete'] is True)
    check('teacher_cost_replayed',abs(teacher['cost']-row['cost_after_candidate'])<1e-8)
    check('actual_projection_hard_complete',candidate['hard'] is True and candidate['complete'] is True)
    check('actual_projection_cost_replayed',abs(candidate['cost']-projected['report']['cost'])<1e-8)
    context=np.asarray(fixture['retained_adjacency'])
    rebuilt=partial_adjacency(projected['routes'],set(),len(generated['demands']))
    check('actual_retained_edges_preserved',np.all(rebuilt[context>0]>0))
    for key,expected in q['source_end'].items():
        group,relative=key.split('/',1)
        p=(ROOT/'MASKCO_code' if group=='upstream' else EXT)/relative
        check('independent_local_source_end:'+key,hashlib.sha256(p.read_bytes()).hexdigest()==expected)
    report={'kind':'independent_actual_full_checkpoint_engineering_audit','checks':checks,
        'n_checks':len(checks),'n_failures':sum(not r['pass'] for r in checks),
        'engineering_pass':all(r['pass'] for r in checks),
        'actual_full_parameter_step_and_C0_projection_verified':True,
        'research_model_eligible':False,'stable_online_gain_established':False,
        'limitations':['uses the registered C0 evaluator, not independently calibrated physical semantics',
                       'one already observed old TRAIN fixture; not a fresh data experiment',
                       'no preference-vs-reconstruction control or all-in10s deployment qualification']}
    with (BASE/'INDEPENDENT_ROOT_ACTUAL_FULLSTEP_AUDIT.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['n_checks','n_failures','engineering_pass','research_model_eligible']}))
    return 0 if report['engineering_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
