"""Finite full-model snapshot research gates and exact predetermined schedule."""
import hashlib
import json
from pathlib import Path
import numpy as np

FORBIDDEN_FINAL_SEEDS = {20260927, 20260928}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def schedule(state_count, steps, batch_size, seed):
    if any(type(x) is not int or x <= 0 for x in (state_count, steps, batch_size, seed)):
        raise ValueError('Exact positive integer schedule parameters required')
    if (steps * batch_size) % state_count:
        raise ValueError('Fixed schedule must expose each state equally')
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(state_count) for _ in range(steps*batch_size//state_count)])
    return [list(map(int, row)) for row in order.reshape(steps, batch_size)]


def classify_pair(initial_report, candidate_report, candidate_routes, incumbent, status, late, epsilon=1e-8):
    if initial_report['hard'] is not True or initial_report['complete'] is not True:
        raise ValueError('Original initial hard failure must be retained and block dataset qualification')
    valid = (candidate_report['hard'] is True and candidate_report['complete'] is True
             and status == 'complete' and not late)
    delta = float(initial_report['cost']) - float(candidate_report['cost']) if valid else None
    sign = 0 if delta is None or abs(delta) <= epsilon else (1 if delta > 0 else -1)
    return {'teacher_pair_valid': valid, 'delta_cost': delta, 'preference_sign': sign,
            'category': 'invalid' if not valid else ('small' if sign == 0 else ('positive' if sign > 0 else 'negative')),
            'CE_target_routes': candidate_routes if valid and delta > epsilon else incumbent,
            'CE_target_choice': 'candidate' if valid and delta > epsilon else 'incumbent',
            'CE_target_rule': 'lower_full_C0_cost_of_certified_pair; tie_incumbent; invalid_incumbent',
            'all_states_retained': True, 'invalid_pair_reconstruction_incumbent_only': not valid}


def verify_research_gate(reg, gate_paths):
    """Fail closed: draft cannot generate64, train128 or launch confirmation."""
    expected = {'version': 'D-FULLMASK-SNAPSHOT-TRAIN-v1', 'status': 'REGISTERED_RESEARCH',
                'mode': 'pre_dispatch_snapshot', 'train_instances': 64, 'masks_per_instance': 4,
                'training_steps': 128, 'batch_size': 4, 'development_instances': 16,
                'confirmation_instances_each': 40, 'confirmation_cohorts': 3,
                'dynamic_online_claim_enabled': False, 'checkpoint_selection': 'fixed_final_step'}
    for name, value in expected.items():
        if type(reg.get(name)) is not type(value) or reg[name] != value:
            raise ValueError('Formal finite gate rejects draft or changed field: ' + name)
    seeds = reg.get('seeds', {})
    required = {'train', 'teacher_mask', 'training_schedule', 'development', 'arm_order',
                'bootstrap', 'confirmation1', 'confirmation2', 'confirmation3'}
    if set(seeds) != required or any(type(x) is not int or x <= 0 or x in FORBIDDEN_FINAL_SEEDS for x in seeds.values()):
        raise ValueError('Actual new science seeds must be registered before generation')
    cohort_seeds = [seeds[x] for x in ('train', 'development', 'confirmation1', 'confirmation2', 'confirmation3')]
    if len(set(cohort_seeds)) != 5:
        raise ValueError('Data cohorts not independent')
    if set(gate_paths) != {'teacher_engineering', 'actual_10D_objective', 'deployment_10s_budget'}:
        raise ValueError('Three exact independently accepted gates required')
    for name, path in gate_paths.items():
        if sha(path) != reg.get('accepted_gate_sha256', {}).get(name):
            raise ValueError('Accepted gate identity missing or changed: ' + name)
        gate = json.loads(Path(path).read_text(encoding='utf-8'))
        if gate.get('independent_acceptance') is not True or gate.get('gate_name') != name:
            raise ValueError('Runner self PASS is not independent acceptance: ' + name)
        if gate.get('contract_hash') != reg['contract_hash']:
            raise ValueError('Gate contract identity differs')
        if name == 'deployment_10s_budget' and gate.get('all_fixed_attempts_complete_within_total10s') is not True:
            raise ValueError('Actual end-to-end deployment budget not qualified')


def check_dependency_identity(reg, root, upstream):
    for name, expected in reg['dependency_sources'].items():
        family, relative = name.split('/', 1)
        path = (upstream if family == 'upstream' else root) / relative
        if sha(path) != expected:
            raise ValueError('Frozen dependency changed: ' + name)
    here = Path(__file__).resolve().parent
    for name, expected in reg['programs'].items():
        if sha(here / name) != expected:
            raise ValueError('Frozen finite program changed: ' + name)
    for name, expected in reg['sibling_programs'].items():
        if sha(here.parent / name) != expected:
            raise ValueError('Frozen actual model dependency changed: ' + name)
