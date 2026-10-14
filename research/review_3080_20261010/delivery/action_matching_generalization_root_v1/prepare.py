"""Prospective registration of delivery n13 (held-out generalisation + reproducible update)."""
import ast
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DELIVERY = HERE.parent
ROUND5 = DELIVERY / 'action_matching_head_update_root_v2'


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
    step0_grad = ROUND5 / 'HEAD_STEP0_GRAD.npz'
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p): sha(p) for p in sources},
        stage='delivery_n13_action_matching_generalization',
        goal=('Measure whether the action-matching scoring-head update transfers to sealed states it never saw, and '
              'whether its optimisation trajectory can be reproduced by an independent implementation.'),
        scope_boundary=('Head-only four-leaf CE over already sealed MaskCO decoder latents of the pretrained_initial '
                        'arm. No backbone forward, no C0, no solver, no teacher cost input, no model optimizer '
                        'update, no new data, no formal trained-control or gain claim.'),
        cycle=6,
        sealed=dict(
            arm='pretrained_initial', records=30,
            npz_directory='research/review_3080_20261010/delivery/actual_action_head_execution_root_v1/downloaded',
            manifest='research/review_3080_20261010/delivery/actual_action_head_execution_root_v1/DOWNLOAD_MANIFEST.json',
            head_source=str(sealed_head), head_source_sha256=sha(sealed_head),
            head_seed=20261012, head_parameter_elements=196929,
            action_options_sha256=sha(DELIVERY / 'feasible_action_options_root_v1' / 'ALL_ACTION_OPTIONS.json'),
            intermediates_sha256=sha(DELIVERY / 'head_intermediate_arithmetic_root_v1' / 'downloaded' / 'INTERMEDIATES.npz'),
        ),
        round5_artifacts=dict(
            step0_grad=str(step0_grad), step0_grad_sha256=sha(step0_grad),
            root='research/review_3080_20261010/delivery/action_matching_head_update_root_v2',
            role='source of the pre-declared matched step-scale derivation and of the recorded Adam control divergence'),
        split=dict(
            rule='ascending sealed record index over the 29 supervised groups; every third group from position 2 is held out',
            held_out_positions=[2, 5, 8, 11, 14, 17, 20, 23, 26],
            expected_train=20, expected_held_out=9,
            declared_before_execution=True, no_selection_by_outcome=True),
        arms=[
            dict(name='sgd_matched', kind='sgd', learning_rate=0.08607391167617187),
            dict(name='sgd_conservative', kind='sgd', learning_rate=0.01),
            dict(name='adam_control', kind='adam', learning_rate=0.003, b1=0.9, b2=0.999, eps=1e-08),
        ],
        primary_arm='sgd_matched',
        primary_arm_reason=('Plain SGD keeps the step proportional to the gradient, so it is the arm whose full '
                            'trajectory is asserted to be reproducible across arithmetics.'),
        arm_reporting='all three arms are reported in full; no arm is selected or discarded by outcome',
        update=dict(
            epochs=8, steps_per_arm=160,
            order='ascending train record index',
            matched_lr_derivation=dict(
                rule='mean |Adam t=1 step| / mean |g| over the sealed round-5 step-0 gradient artifact',
                mean_abs_adam_step_t1=0.0006951489918115264,
                mean_abs_gradient=0.00807618682913846,
                matched_lr=0.08607391167617187,
                derived_before_execution=True,
                source_artifact=str(step0_grad))),
        tolerances=dict(
            forward_vs_sealed_scores_max_abs=2e-05,
            audit_metric_max_abs=1e-05,
            audit_sgd_trajectory_max_abs=1e-04,
            audit_sgd_final_param_max_abs=1e-04,
            reload_bit_exact=0.0,
            adam_arm_trajectory_asserted=False),
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
            dict(title='Adam: A Method for Stochastic Optimization', venue='ICLR 2015, arXiv:1412.6980',
                 authors='Diederik P. Kingma, Jimmy Ba', url='https://arxiv.org/abs/1412.6980',
                 read_scope='algorithm 1 and the per-coordinate normalisation and epsilon role (optionally read this round)',
                 use='explains why a normalised update saturates to +/-lr and is not reproducible at the float32 floor'),
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
