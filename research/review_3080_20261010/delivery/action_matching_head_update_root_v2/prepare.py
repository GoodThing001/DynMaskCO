"""Prospective registration of the delivery n12 head-only action-matching stage."""
import ast
import hashlib
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DELIVERY = HERE.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE / 'REGISTRATION.json').exists():
        raise SystemExit('REGISTRATION.json already exists')
    import numpy, jax, flax, optax
    sources = [HERE / n for n in ('head_action_matching.py', 'run.py', 'audit.py', 'prepare.py', 'supervise.py')]
    sources.append(ROOT / 'tools' / 'engineering_stage_closure_v1.py')
    for p in sources:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
    sealed_head = DELIVERY / 'head_intermediate_arithmetic_root_v1' / 'delivery_actual_action_head_v1.py'
    v1 = HERE.parent / 'action_matching_head_update_root_v1'
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p): sha(p) for p in sources},
        stage='delivery_n12_action_matching_head_update',
        goal=('Connect the action-matching scoring-head update / fixed order / save / reload: head-only four-leaf '
              'per-customer feasible-action cross-entropy over already sealed MaskCO decoder node latents.'),
        scope_boundary=('No backbone forward, no C0 physics, no solver, no teacher cost input, no model optimizer '
                        'update, no new scientific data, no formal trained control claim.'),
        cycle=5,
        preserved_prior_attempt=dict(
            directory='research/review_3080_20261010/delivery/action_matching_head_update_root_v1',
            outcome='failed_2_checks_exit_1_no_source_end',
            failures=['cpu_devices_only device-label string defect',
                      'actual_loaded_files_and_maps_start_end lazy encodings.cp437 codec load after the start closure'],
            closure_diagnosis='research/review_3080_20261010/delivery/action_matching_head_update_root_v1/CLOSURE_DIAGNOSIS.json',
            diagnosis_result='no changed or removed file; only encodings.cp437 module plus its pyc appeared',
            fix_scope='warm up the stdlib codec and the first sealed npz read before the start closure; CPU device check uses the device platform'),
        v2_corrections=['import encodings.cp437 and one sealed npz read inside the warm-up',
                        'cpu_devices_only checks jax.default_backend and device.platform',
                        'SOURCE_END.json and AUDIT_SOURCE_END.json are actually written',
                        'v1 FAIL preservation is asserted in both production and audit'],
        cycle_root='research/review_3080_20261010/dual_track_optimization_20261013_root_v1/cycle5_20261013_root_v1',
        sealed=dict(
            arm='pretrained_initial', records=30,
            npz_directory='research/review_3080_20261010/delivery/actual_action_head_execution_root_v1/downloaded',
            manifest='research/review_3080_20261010/delivery/actual_action_head_execution_root_v1/DOWNLOAD_MANIFEST.json',
            head_source=str(sealed_head), head_source_sha256=sha(sealed_head),
            head_seed=20261012, head_parameter_elements=196929,
            head_leaf_shapes={'k1': [3075, 64], 'b1': [64], 'k2': [64, 1], 'b2': [1]},
            action_options_sha256=sha(DELIVERY / 'feasible_action_options_root_v1' / 'ALL_ACTION_OPTIONS.json'),
            intermediates_sha256=sha(DELIVERY / 'head_intermediate_arithmetic_root_v1' / 'downloaded' / 'INTERMEDIATES.npz'),
        ),
        groups=dict(expected_supervised=29, min_eligible=2, max_eligible=6,
                    derivation='sealed eligible rows mapped into the saved head-input action grid; target must be in the subset'),
        update=dict(epochs=8, learning_rate=0.003, b1=0.9, b2=0.999, eps=1e-08,
                    fixed_order='ascending sealed record index over supervised groups'),
        tolerances=dict(
            forward_vs_sealed_scores_max_abs=2e-05,
            audit_concat_max_abs=1e-30,
            audit_reference_loss_max_abs=1e-05,
            audit_trajectory_loss_max_abs=1e-04,
            audit_final_param_max_abs=1e-04,
            audit_gradient_max_abs=1e-03,
            adam_step1_max_abs=1e-06,
            reload_bit_exact=0.0),
        runtime={'numpy': numpy.__version__, 'jax': jax.__version__, 'flax': flax.__version__, 'optax': optax.__version__},
        resource=dict(python='%d.%d.%d' % sys.version_info[:3], cpu_affinity=[7],
                      min_available_ram_gib_without_backbone=1.0, full_backbone_threshold_gib=8.0,
                      backbone_loaded=False),
        normal_runs=1, audit_runs=1, supervision_cap_seconds=600,
        papers=[
            dict(title='MaskCO: Masked Generation Drives Effective Representation Learning and Exploiting for '
                       'Combinatorial Optimization', venue='ICLR 2026', authors='Lvda Chen, Yang Li, Junchi Yan',
                 url='https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf',
                 read_scope='abstract, section 3 partial-solution/selection operators, section 4.1 variable BCE training',
                 use='per-customer feasible-action CE wiring inspiration only, not the original variable-BCE recipe'),
            dict(title='A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning',
                 venue='AISTATS 2011, PMLR 15:627-635', authors='Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell',
                 url='https://proceedings.mlr.press/v15/ross11a.html',
                 read_scope='sections 2.1 and 3 and Algorithm 3.1',
                 use='future learned-trajectory aggregation plan only; no DAgger executed in this round'),
            dict(title='Reinforcement and Imitation Learning via Interactive No-Regret Learning',
                 venue='arXiv:1406.5979v1 preprint', authors='Stephane Ross, J. Andrew Bagnell',
                 url='https://arxiv.org/abs/1406.5979',
                 read_scope='official abstract and version metadata only',
                 use='cost-to-go supervision direction only; not implemented, no AggreVaTe claim'),
        ],
        formal_trained_control_qualified=False, formal_controls_qualified=False,
        full_request_budget_qualified=False, full_scientific_loaded_closure_qualified=False,
        scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
        optimizer_updates_of_model=0, learned_output_qualified=False, stable_online_gain=False,
        shared_MD_uploaded=False, final_pickup_seeds_used=False,
    )
    with (HERE / 'REGISTRATION.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'))))


if __name__ == '__main__':
    main()
