"""Fail-closed package gate for prospective PROGRAM confirmation runs.

This checks bytes, metadata, raw-row coverage and arithmetic only. It NEVER
establishes actual model activation, causal semantics, hard feasibility, replay,
pre-data timing, or stable gain. Those need separately executed independent audits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PROGRAM_SHA = '8edbc3cd5cbe4563992112bf55acadef0be4242b12d2e8139684c3f146cdbbd8'
PROTECTED = {20260927, 20260928}
ROLES = {'source_closure', 'weights', 'dataset', 'contract', 'budget',
         'history', 'sampler', 'training_schedule', 'environment',
         'model_activation_inputs_outputs', 'training_parameter_updates',
         'independent_replay_recipe', 'strong_control_recipe'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def reject(message):
    raise ValueError(message)


def pairs(xs):
    d = {}
    for k, v in xs:
        if k in d:
            reject('duplicate JSON key: ' + k)
        d[k] = v
    return d


def load(p):
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda x: reject('nonfinite JSON ' + x))


def artifact(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        reject('artifact must have exact path/SHA fields')
    rel = Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts:
        reject('artifact traversal/absolute path')
    p = (root / rel).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        reject('artifact missing or outside project')
    if sha(p) != ref['sha256']:
        reject('actual artifact SHA mismatch: ' + str(rel))
    return p


def audit(root, registration_path, registry_path, bundle_path):
    reg, registry, bundle = load(registration_path), load(registry_path), load(bundle_path)
    if registry.get('schema') != 'external_new_run_registry_v1':
        reject('external registry schema')
    if registry.get('registration_sha256') != sha(registration_path):
        reject('registration not externally bound')
    if reg.get('schema') != 'prospective_formal_run_v1' or bundle.get('schema') != 'formal_run_raw_package_v1':
        reject('new run schemas')
    if reg.get('stage') != 'confirmation' or bundle.get('registration_sha256') != sha(registration_path):
        reject('engineering/development cannot qualify confirmation package')
    program = artifact(root, reg['program'])
    if sha(program) != PROGRAM_SHA or registry.get('program_sha256') != PROGRAM_SHA:
        reject('original PROGRAM binding')
    line, j = reg.get('line'), reg.get('family_index')
    if line not in ('pickup', 'delivery') or type(j) is not int or j < 1:
        reject('line/family index')
    cohorts = reg.get('cohorts')
    if not isinstance(cohorts, list) or len(cohorts) != 3:
        reject('exact three confirmation cohorts required')
    seeds = [c.get('seed') for c in cohorts]
    if any(type(s) is not int or s in PROTECTED for s in seeds) or len(set(seeds)) != 3:
        reject('distinct integer fresh seeds; old final forbidden')
    if any(c.get('day_ids') != list(range(40)) or any(type(d) is not int for d in c.get('day_ids', [])) for c in cohorts):
        reject('40 complete day IDs per cohort')
    arms = reg.get('arms')
    if not isinstance(arms, list) or len(arms) < 3 or any(type(a) is not str or not a for a in arms) or len(set(arms)) != len(arms):
        reject('unique arms')
    contrasts = reg.get('contrasts')
    if not isinstance(contrasts, list) or not contrasts:
        reject('registered contrasts missing')
    names = []
    for c in contrasts:
        if c.get('treatment') not in arms or c.get('control') not in arms or c['treatment'] == c['control']:
            reject('contrast arm identity')
        if not number(c.get('minimum_effect')):
            reject('finite fixed effect threshold')
        if line == 'delivery' and (c.get('metric') != 'relative_complete_cost_reduction' or c['minimum_effect'] < .01):
            reject('delivery registered complete relative cost minimum>=1%')
        names.append(c.get('id'))
    if any(type(x) is not str or not x for x in names) or len(set(names)) != len(names):
        reject('unique contrast IDs')
    roles = {c.get('control_role') for c in contrasts}
    if line == 'delivery' and not {'same_architecture_random', 'qualified_strong_nonlearned_rolling'} <= roles:
        reject('delivery random and strong nonlearned control registrations required')
    if line == 'pickup':
        by_role = {c.get('control_role'): c for c in contrasts}
        if not {'cond_hist', 'same_init_reconstruction_control'} <= set(by_role):
            reject('pickup main and learning attribution roles required')
        if by_role['cond_hist']['minimum_effect'] < 20 or by_role['same_init_reconstruction_control']['minimum_effect'] < 0:
            reject('pickup main minimum20 and positive learning attribution threshold')
        if any(c.get('metric') != 'complete_day_utility_difference' for c in contrasts):
            reject('pickup complete utility metric')
    alpha = .05 / (j * (j + 1)) / 2 / (len(contrasts) * 3)
    stat = reg.get('statistics', {})
    if not number(stat.get('two_sided_alpha')) or abs(stat['two_sided_alpha'] - alpha) > 1e-15:
        reject('PROGRAM family/line/contrast/cohort alpha allocation')
    reps = stat.get('bootstrap_replicates')
    if type(reps) is not int or reps < 1 or reps * alpha / 2 < 25:
        reject('bootstrap tail Monte Carlo expected count >=25')
    if stat.get('resampling_unit') != 'complete_day' or stat.get('paired') is not True:
        reject('complete-day paired resampling')
    if type(stat.get('bootstrap_seed')) is not int:
        reject('registered integer bootstrap seed')
    endpoints = stat.get('percentiles')
    expected = [alpha / 2 * 100, (1 - alpha / 2) * 100]
    if not isinstance(endpoints, list) or len(endpoints) != 2 or any(not number(x) or abs(x-y) > 1e-12 for x, y in zip(endpoints, expected)):
        reject('family adjusted percentile endpoint')
    identity = bundle.get('identities', {})
    if set(identity) != ROLES or registry.get('identities') != identity:
        reject('exact externally registered identity roles')
    identities = {k: artifact(root, v) for k, v in identity.items()}
    raw = bundle.get('days')
    if not isinstance(raw, list):
        reject('actual day raw table missing')
    expected_keys = {(s, d, a) for s in seeds for d in range(40) for a in arms}
    seen, raw_hashes, checks = set(), set(), []
    for row in raw:
        key = (row.get('seed'), row.get('day_id'), row.get('arm'))
        if type(key[0]) is not int or type(key[1]) is not int or key not in expected_keys or key in seen:
            reject('raw key duplicate/mismatch/type')
        seen.add(key)
        p = artifact(root, row['raw'])
        if row['raw']['sha256'] in raw_hashes:
            reject('same raw bytes reused across independent units/arms')
        raw_hashes.add(row['raw']['sha256'])
        value = load(p)
        if value.get('schema') != 'complete_online_day_raw_v1' or (value.get('seed'), value.get('day_id'), value.get('arm')) != key:
            reject('raw embedded identity')
        if type(value.get('day_id')) is not int or type(value.get('seed')) is not int:
            reject('raw integer IDs')
        if value.get('complete') is not True or value.get('hard') is not True or not number(value.get('utility_or_cost')):
            reject('finite complete literal-hard day')
        if value.get('identity_sha256') != {k: v['sha256'] for k, v in identity.items()}:
            reject('raw identity role binding')
        artifact(root, value['event_journal'])
        artifact(root, value['model_call_records'])
        artifact(root, value['runtime_process_records'])
        for e in value.get('admissions', []):
            start, submitted, cap = e.get('entry_monotonic'), e.get('submit_monotonic'), e.get('cap_s')
            if not all(number(x) for x in [start, submitted, cap]) or start > submitted or cap != 10:
                reject('all-in admission clock/cap invalid')
            if submitted - start > cap and e.get('own_incumbent_fallback') is not True:
                reject('late submission lacks retained fallback')
            if e.get('excluded_internal_setup_s') != 0:
                reject('unregistered inside-admission setup exclusion')
        if not isinstance(value.get('admissions'), list) or not value['admissions']:
            reject('admission raw records required')
        checks.append({'seed': key[0], 'day_id': key[1], 'arm': key[2], 'raw_sha256': sha(p)})
    if seen != expected_keys:
        reject('all three fresh40day/arm raw coverage required')
    # Rehash after reading to catch changed raw or input files in this audit.
    for k, p in identities.items():
        if sha(p) != identity[k]['sha256']:
            reject('identity changed during audit')
    return {'kind': 'registered_confirmation_raw_package_subset_gate_v1',
            'checked_design_subset_consistent': True, 'registered_raw_byte_coverage_pass': True,
            'declared_budget_record_arithmetic_pass': True, 'raw_days': checks,
            'alpha_per_contrast_cohort': alpha, 'expected_bootstrap_tail_count': reps * alpha / 2,
            'independent_actual_model_activation_established': False,
            'independent_causal_information_validation_established': False,
            'independent_process_kill_budget_proven': False,
            'independent_full_trajectory_replay_established': False,
            'pre_data_registration_timing_established': False,
            'formal_online_qualification_established': False, 'stable_maskco_core_online_gain': False,
            'limitations': ['External registry binds bytes, not timing or freshness.',
                            'Stored model and process records are inputs for independent rerun, not proof of activation.',
                            'Hard and complete flags require independent semantic replay; no such proof in this adapter.',
                            'This adapter does not compute effect CIs or certify strong controls.']}


def main():
    ap = argparse.ArgumentParser()
    for name in ['project', 'registration', 'registry', 'bundle', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    try:
        report = audit(a.project, a.registration, a.registry, a.bundle)
        code = 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {'checked_design_subset_consistent': False, 'error': str(exc),
                  'formal_online_qualification_established': False, 'stable_maskco_core_online_gain': False}
        code = 2
    if a.output.resolve().is_relative_to(a.project.resolve()):
        reject('write only outside frozen project')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'exit_code': code, 'stable_maskco_core_online_gain': False}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
