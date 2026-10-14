"""Freeze existing inputs for actual-checkpoint engineering before its execution."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from actual_full_step_qa import sources, sha, dump


def register(root, upstream, inspection, out):
    inspected = json.loads(inspection.read_text(encoding='utf-8'))
    split = 'results/research_20261010_continuation_v2/delivery/train_split_v1'
    reg = {'version': 'D-ACTUAL-FULLSTEP-ENGINEERING-v1', 'stage': 'actual_checkpoint_engineering_only',
        'fixed_steps': 1, 'optimizer': 'sgd_without_momentum', 'learning_rate': .00001,
        'research_model_eligible': False, 'research_gain_claim': False,
        'teacher_first_certified_ordinal': 0, 'minimum_available_ram_gib': 8,
        'teacher_split': split, 'teacher_files': {n: sha(root / split / n)
             for n in ('samples.jsonl', 'TRAIN_SPLIT_IDENTITY.json', 'TRAIN_SPLIT_MANIFEST.json')},
        'original_checkpoint_sha256': inspected['checkpoint_sha256'], 'model_config': inspected['model_config'],
        'actual_parameters': {'leaves': inspected['parameter_leaves'], 'elements': inspected['parameter_elements']},
        'runtime_dtype': 'float32', 'feature_shape': [1, 51, 3],
        'loss': 'upstream_customer_row_positive_edge_logsoftmax_CE_full_encoder_decoder',
        'masked_context': 'old_first_certified_training_row_incumbent_immutable_unmasked_edges',
        'target': 'stored_complete_certified_candidate_routes_independent_C0_snapshot_replay_required',
        'projection': 'existing_regret2_insert_full_C0_snapshot_certification_retained_edges_immutable',
        'projection_budget': 'engineering_complete_path_no10s_qualification',
        'all_input_nodes_visible_ready_loaded': True, 'expanded10d_coldchain_feature_qualified': False,
        'random_control': 'actual_cfg_rng888_configuration_initialization_identity_not_fresh_random42_panel',
        'same_initial_reconstruction_control': 'current_actual_step_is_CE_only; no_preference_plus_CE_contrast_yet',
        'sources': sources(root, upstream),
        'engineering_scripts': {n: sha(Path(__file__).resolve().parent / n)
             for n in ('actual_full_step_qa.py', 'inspect_checkpoint.py')},
        'save': 'new_engineering_only_full_final_checkpoint_sha_and_reload_logits_equal',
        'prohibitions': ['no_old_result_or_original_weight_changes', 'no_D2_or_destroyEQ8_repeat',
                         'no_scientific_candidate_reuse_from_engineering_weight', 'no_final_20260927_28_use',
                         'no_physical_calibration_claim', 'no_GPU_claim_for_native_Windows_JAX'],
        'qualification': ['actual_model_and_input_identity', 'actual_all234_leaves_gradient',
                          'actual_encoder_decoder_parameter_change', 'actual_partial_adj_context_activation',
                          'complete_hard_feasible_retained_projection', 'saved_reloaded_identical_logits',
                          'source_and_original_checkpoint_end_hash_unchanged']}
    dump(out, reg)
    print(json.dumps({'registered_engineering_sha256': sha(out), 'source_files': len(reg['sources'])}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--inspection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    register(a.root, a.upstream, a.inspection, a.out)
