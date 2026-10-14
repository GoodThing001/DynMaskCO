"""Freeze actual sources and old engineering fixture before any new QA."""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'actual_checkpoint_engineering_v1'))
from actual_full_step_qa import sources, sha, dump
from features_objective import UNITS, FIELDS


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('root', 'upstream', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    olddir = HERE.parent / 'actual_checkpoint_engineering_v1'
    old = json.loads((olddir / 'ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    for folder in ('delivery', 'coldchain'):
        sys.path.insert(0, str(a.root / 'scripts' / folder))
    from delivery_contract import DeliveryContract
    reg = {'version': 'D-10D-PAIRED-ENGINEERING-v1', 'stage': 'old_fixture_engineering_only',
        'fixed_steps_each_arm': 1, 'optimizer': 'sgd_without_momentum', 'learning_rate': .00001,
        'preference_coefficient': .2, 'research_model_eligible': False, 'dynamic_online_qualified': False,
        'old_first_certified_train_only': True, 'new_scientific_data_generated': False,
        'teacher_split': old['teacher_split'], 'teacher_files': old['teacher_files'],
        'original_checkpoint_sha256': old['original_checkpoint_sha256'],
        'original_model_config': old['model_config'], 'sources': sources(a.root, a.upstream),
        'programs': {n: sha(HERE / n) for n in ('run_actual_features_qa.py', 'features_objective.py')},
        'old_dependency_programs': {n: sha(olddir / n) for n in ('actual_full_step_qa.py', 'inspect_checkpoint.py')},
        'contract_hash': DeliveryContract().identity(), 'contract_payload': DeliveryContract().payload(),
        'fields': FIELDS, 'units': UNITS, 'time_ceiling_seconds': 300, 'minimum_ram_gib': 8,
        'initialization': 'original_full_encoder_decoder; init_proj 3x512 to10x512 original3rows+7zero; all234leaves',
        'feature_qualification': 'all_released_ready_old_snapshot_only_no_dynamic_attention',
        'objective': 'upstream fullroute customerrow edge CE + lambda*softplus(-y*mean_route_logprob_difference)',
        'pair_sign': 'sign(cost_incumbent-cost_candidate); hard-certified full C0 traces; |delta|<=1e-8 means no preference',
        'coefficient_engineering_choice_not_literature_optimum': True,
        'controls': 'CEOnly lambda0 vs preference lambda0.2; exactsame initial/data/context/optimizer/fixedstep',
        'prohibitions': ['no_gain_from_surrogate_loss', 'no_physical_calibration_claim', 'no_reuse_engineering_weight_as_scientific_candidate',
                         'no_old_tree_modification', 'no_new_scientific_64TRAIN_before_separate_preregistration',
                         'no_final_20260927_28', 'no_dynamic_online_qualification', 'no_10second_budget_claim']}
    dump(a.out, reg)
    print(json.dumps({'registration_sha256': sha(a.out), 'dependency_sources': len(reg['sources'])}))
