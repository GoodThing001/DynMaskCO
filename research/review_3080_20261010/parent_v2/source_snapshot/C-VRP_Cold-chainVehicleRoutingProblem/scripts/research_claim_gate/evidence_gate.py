"""Fail-closed claim qualification from a preregistered, file-backed evidence bundle.

This is a reporting gate, not an optimizer or a cryptographic witness that supplied
files are truthful. External independent trajectory and identity audits remain required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re


class EvidenceError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + ': finite number required')
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + ': integer required')
    return value


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'invalid SHA256')
    return value


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise EvidenceError('non-standard JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid_constant)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(root, entry):
    rel = Path(entry['path'])
    require(not rel.is_absolute() and '..' not in rel.parts, 'artifact path must stay within bundle')
    path = (root / rel).resolve()
    require(path.is_relative_to(root.resolve()), 'resolved artifact leaves bundle')
    require(path.is_file(), 'missing artifact: ' + str(rel))
    require(file_hash(path) == sha(entry['sha256']), 'artifact hash mismatch: ' + str(rel))
    return path


def percentile(values, quantile):
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def paired_bootstrap(differences, seed, repetitions=2000):
    require(len(differences) > 1, 'at least two independent units needed')
    rng = random.Random(seed)
    n = len(differences)
    values = [sum(differences[rng.randrange(n)] for _ in range(n)) / n
              for _ in range(repetitions)]
    return {'mean': sum(differences) / n,
            'ci95': [percentile(values, .025), percentile(values, .975)],
            'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
            'algorithm': 'python_random_randrange_percentile_linear_v1'}


def validate_program_binding(root, bundle, protocol, program_path):
    """Check proposed binding against a caller-supplied trusted program anchor.

    A bundle cannot select its own authoritative PROGRAM. This only validates the
    binding's form and conservative Monte Carlo prerequisites; it does not execute
    the adjusted inference or prove adaptive confirmation isolation.
    """
    if 'program_binding' not in bundle:
        return False
    require(program_path is not None, 'program binding requires an external trusted PROGRAM anchor')
    binding = load_json(artifact(root, bundle['program_binding']))
    require(binding.get('schema') == 'maskco_program_binding_v1', 'unsupported program binding schema')
    trusted_program = Path(program_path)
    require(trusted_program.is_file(), 'trusted PROGRAM missing')
    require(file_hash(trusted_program) == sha(binding['program_sha256']), 'trusted PROGRAM hash mismatch')
    require(binding['protocol_sha256'] == bundle['protocol']['sha256'], 'binding protocol mismatch')
    require(protocol['units_per_run'] == 40 and protocol['resampling_unit'] == 'day' and
            protocol['scope'] == 'dynamic_online', 'stable program requires complete 40-day online runs')
    family = integer(binding['family_index'], 'family_index', 1)
    comparisons = len(protocol['comparisons'])
    require(protocol['confirmation_runs'] == 3, 'bound program requires exactly three confirmation batches')
    expected_alpha = .05 / (family * (family + 1)) / 2 / (comparisons * 3)
    alpha = finite(binding['allocation_alpha'], 'allocation_alpha')
    require(math.isclose(alpha, expected_alpha, rel_tol=1e-12, abs_tol=1e-15), 'alpha spending mismatch')
    endpoints = binding['bootstrap_endpoints']
    require(len(endpoints) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in endpoints),
            'bootstrap endpoints invalid')
    require(math.isclose(endpoints[0], alpha / 2, rel_tol=1e-12, abs_tol=1e-15) and
            math.isclose(endpoints[1], 1 - alpha / 2, rel_tol=1e-12, abs_tol=1e-15),
            'adjusted bootstrap endpoints mismatch')
    reps = integer(binding['bootstrap_repetitions'], 'bound bootstrap repetitions', 1)
    require(reps == protocol['bootstrap_repetitions'], 'bound repetitions differ from registered protocol')
    require(reps * endpoints[0] >= 25, 'insufficient bootstrap tail Monte Carlo count')
    # v1's fixed 95%/2000 evaluator is deliberately not promoted to an adjusted gate.
    return True


def audit_bundle(bundle_path, program_path=None):
    """Return qualification; schema/identity violations raise EvidenceError.

    Each run is one independently generated day/instance batch, with no filtering.
    Every arm shares unit-level data hashes; timeout fallback is a retained row.
    CI across zero is reported as inconclusive, never converted to equivalence.
    """
    bundle_path = Path(bundle_path)
    root = bundle_path.parent.resolve()
    bundle = load_json(bundle_path)
    require(bundle.get('schema') == 'maskco_claim_bundle_v1', 'unsupported bundle schema')
    protocol_path = artifact(root, bundle['protocol'])
    protocol = load_json(protocol_path)
    require(protocol.get('schema') == 'maskco_claim_protocol_v1', 'unsupported protocol schema')
    require(protocol['registered_before_outcomes'] is True, 'protocol not registered before outcomes')
    require(protocol['frozen_selection_before_confirmation'] is True, 'confirmation model was not frozen')
    arms = protocol['arms']
    require(isinstance(arms, list) and len(set(arms)) == len(arms) and len(arms) >= 2,
            'distinct arms required')
    target = protocol['target_arm']
    require(target in arms, 'target arm missing')
    units = integer(protocol['units_per_run'], 'units_per_run', 2)
    require(protocol['resampling_unit'] in ('day', 'instance'), 'unsupported resampling unit')
    require(protocol['metric_direction'] in ('higher', 'lower'), 'metric direction required')
    require(protocol['scope'] in ('snapshot', 'fixed_state', 'dynamic_online'), 'scientific scope required')
    repetitions = integer(protocol['bootstrap_repetitions'], 'bootstrap_repetitions', 1)
    require(repetitions == 2000, 'this version requires exactly 2000 paired bootstrap samples')
    binding_form_valid = validate_program_binding(root, bundle, protocol, program_path)
    sign = 1 if protocol['metric_direction'] == 'higher' else -1
    comparisons = protocol['comparisons']
    require(isinstance(comparisons, list) and len(comparisons) >= 1, 'comparisons missing')
    require(len({c['arm'] for c in comparisons}) == len(comparisons), 'duplicate comparisons')
    primary = [c for c in comparisons if c['role'] == 'strong_primary']
    require(len(primary) >= 1, 'strong primary comparator required')
    for comp in comparisons:
        require(comp['arm'] in arms and comp['arm'] != target, 'invalid comparator')
        require(comp['role'] in ('strong_primary', 'random_same_schedule', 'nonlearning_same_schedule'),
                'unsupported comparator role')
        require(comp['scale'] in ('absolute', 'relative'), 'unsupported effect scale')
        require(finite(comp['minimum_mean'], 'minimum_mean') >= 0, 'negative gain threshold')
    protected = {integer(x, 'protected seed') for x in protocol['protected_seeds']}
    sha(protocol['selection_sha256'])
    integer(protocol['bootstrap_seed_offset'], 'bootstrap_seed_offset')
    required_confirmation = integer(protocol['confirmation_runs'], 'confirmation_runs', 1)
    require(required_confirmation >= 3, 'stable gain requires at least three fixed confirmation runs')
    identities = protocol['identity_roles']
    needed = {'source', 'weights', 'data_generator', 'contract', 'budget', 'pool', 'training'}
    require(isinstance(identities, list) and needed.issubset(identities), 'incomplete identity roles')
    # These file-backed prerequisites are not replaced by any summary "PASS" string.
    verified_mechanisms = True
    mechanism_keys = ('native_masked_reconstruction', 'event_mask', 'frozen_prefix',
                      'equal_total_budget', 'equal_iteration_schedule_controls')
    runner_entry = bundle['runner_source']
    artifact(root, runner_entry)
    for key in ('independent_identity_audit', 'independent_trajectory_audit'):
        proof = load_json(artifact(root, bundle[key]))
        require(proof.get('schema') == 'maskco_independent_audit_v1', 'independent audit schema missing')
        require(proof.get('passed') is True, key + ': literal true required')
        require(proof.get('protocol_sha256') == bundle['protocol']['sha256'], 'audit protocol mismatch')
        auditor_entry = bundle['auditor_sources'][key]
        artifact(root, auditor_entry)
        require(proof.get('auditor_source_sha256') == auditor_entry['sha256'], 'actual auditor source mismatch')
        require(proof.get('auditor_source_sha256') != runner_entry['sha256'],
                'runner cannot serve as its own independent auditor')
        sha(proof['auditor_source_sha256'])
        verified_mechanisms = verified_mechanisms and all(
            proof.get('mechanism_checks', {}).get(k) is True for k in mechanism_keys)
    data_objects = bundle['data_objects']
    require(isinstance(data_objects, dict) and len(data_objects) > 0, 'data inventory missing')
    for expected, entry in data_objects.items():
        require(sha(expected) == entry['sha256'], 'data inventory hash/key mismatch')
        artifact(root, entry)
    seen_seeds, seen_keys, seen_data = set(), set(), set()
    run_results = []
    stage_counts = {'development': 0, 'confirmation': 0}
    expected_keys = protocol['expected_units_by_seed']
    require(len(bundle['runs']) == len(expected_keys), 'registered run omitted or extra run supplied')
    run_inventory = []
    for run in bundle['runs']:
        seed = integer(run['seed'], 'seed')
        require(seed not in protected and seed not in seen_seeds, 'protected or duplicate seed')
        seen_seeds.add(seed)
        stage = run['stage']
        require(stage in stage_counts, 'invalid stage')
        require(protocol['stage_by_seed'].get(str(seed)) == stage, 'registered split changed')
        stage_counts[stage] += 1
        require(run['exit_code'] == 0 and type(run['exit_code']) is int, 'nonzero/invalid exit')
        require(run['selection_sha256'] == protocol['selection_sha256'], 'selected model/config changed')
        registered = expected_keys.get(str(seed))
        require(isinstance(registered, list) and len(registered) == units and len(set(registered)) == units,
                'registered unique unit set invalid')
        for role in identities:
            entry = run['identity'][role]
            artifact(root, entry)
            require(entry['sha256'] == protocol['identity_hashes'][role], 'frozen identity changed: ' + role)
            if role == 'source':
                closure = load_json(artifact(root, entry))
                require(isinstance(closure.get('files'), list) and len(closure['files']) > 0,
                        'source closure file list missing')
                require(len({e['path'] for e in closure['files']}) == len(closure['files']),
                        'source closure duplicates')
                for source_file in closure['files']:
                    artifact(root, source_file)
        rows_path = artifact(root, run['rows'])
        run_inventory.append(run['rows']['sha256'])
        rows = load_json(rows_path)
        require(isinstance(rows, list) and len(rows) == units * len(arms), 'row count mismatch')
        matrix = {}
        for row in rows:
            key = row['unit_id']
            require(isinstance(key, str) and key in registered, 'unknown/missing unit id')
            arm = row['arm']
            require(arm in arms, 'unexpected arm')
            pair = (key, arm)
            require(pair not in matrix, 'duplicate unit-arm row')
            matrix[pair] = row
            finite(row['value'], 'row value')
            require(row['hard'] is True and row['complete'] is True, 'hard/complete must be literal true')
            require(type(row['timeout']) is bool, 'timeout must be literal boolean')
            finite(row['wall_seconds'], 'wall_seconds')
            require(row['wall_seconds'] >= 0, 'negative wall time')
            integer(row['completed_candidates'], 'completed_candidates')
            integer(row['model_calls'], 'model_calls')
            sha(row['data_sha256'])
            require(row['data_sha256'] in data_objects, 'paired data artifact not independently hashed')
        for key in registered:
            require(key not in seen_keys, 'same independent unit reused across splits/runs')
            seen_keys.add(key)
            require(len({matrix[(key, arm)]['data_sha256'] for arm in arms}) == 1,
                    'paired arms do not share data')
        this_data = {row['data_sha256'] for row in rows}
        require(not seen_data.intersection(this_data), 'data reused across development/confirmation batches')
        seen_data.update(this_data)
        results = {}
        for comp in comparisons:
            differences = []
            for key in registered:
                a = matrix[(key, target)]['value']
                b = matrix[(key, comp['arm'])]['value']
                delta = sign * (a - b)
                if comp['scale'] == 'relative':
                    require(b > 0, 'relative comparator must be strictly positive')
                    delta /= b
                differences.append(delta)
            result = paired_bootstrap(differences, seed + protocol['bootstrap_seed_offset'], repetitions)
            result['passes'] = result['mean'] >= comp['minimum_mean'] and result['ci95'][0] > 0
            result['interpretation'] = ('positive_gain' if result['passes'] else
                                        'negative_effect' if result['ci95'][1] < 0 else
                                        'no_qualified_positive_gain')
            result['equivalence_established'] = False
            results[comp['arm']] = result
        # Activation is observed execution count, not model-card presence.
        model_active = all(matrix[(key, target)]['model_calls'] > 0 and
                           matrix[(key, target)]['completed_candidates'] > 0 for key in registered)
        schedule_arms = [target] + [c['arm'] for c in comparisons
                                   if c['role'] in ('random_same_schedule', 'nonlearning_same_schedule')]
        same_completed_schedule = all(len({matrix[(key, arm)]['completed_candidates']
                                             for arm in schedule_arms}) == 1 for key in registered)
        run_results.append({'seed': seed, 'stage': stage, 'comparisons': results,
                            'all_comparisons_pass': all(x['passes'] for x in results.values()),
                            'model_active_every_unit': model_active,
                            'equal_completed_candidates_controls': same_completed_schedule,
                            'timeouts_by_arm': {arm: sum(matrix[(key, arm)]['timeout'] for key in registered)
                                                for arm in arms}})
    # Independent auditors must cover the exact rows actually recomputed here.
    for key in ('independent_identity_audit', 'independent_trajectory_audit'):
        proof = load_json(artifact(root, bundle[key]))
        require(sorted(proof['rows_sha256']) == sorted(run_inventory), 'audit row coverage incomplete')
        require(proof['unit_arm_rows'] == units * len(arms) * len(run_results), 'audit row count mismatch')
    require(stage_counts['development'] >= 1, 'development evidence missing')
    confirmations = [x for x in run_results if x['stage'] == 'confirmation']
    require(stage_counts['confirmation'] == required_confirmation, 'confirmation count mismatch')
    development = [x for x in run_results if x['stage'] == 'development']
    stable = (all(x['all_comparisons_pass'] for x in development + confirmations) and
              all(x['model_active_every_unit'] for x in development + confirmations))
    roles = {c['role'] for c in comparisons}
    semantics = protocol['method_semantics']
    core = (stable and verified_mechanisms and
            all(x['equal_completed_candidates_controls'] for x in run_results) and
            protocol['scope'] == 'dynamic_online' and
            {'strong_primary', 'random_same_schedule', 'nonlearning_same_schedule'}.issubset(roles) and
            all(semantics.get(k) is True for k in ('native_masked_reconstruction', 'event_mask',
                                                   'frozen_prefix', 'equal_total_budget',
                                                   'equal_iteration_schedule_controls')))
    return {'schema': 'maskco_claim_adjudication_v1', 'qualified_evidence': True,
            'candidate_gain_under_supplied_v1_protocol': stable,
            'candidate_core_online_gain_under_supplied_v1_protocol': core,
            'program_binding_form_valid': binding_form_valid,
            'stable_gain_in_registered_setting': False, 'stable_maskco_core_online_gain': False,
            'equivalence_established': False, 'full_from_zero_reproduction': False,
            'scope': protocol['scope'], 'runs': run_results,
            'limitations': ['File hashes prove identity, not truth of execution.',
                            'Audit payloads require external independent review and raw trajectory replay.',
                            'This finite gate does not prove universal gain or repeated-search error control.',
                            'v1 fixed 95%/2000 bootstrap is not the PROGRAM alpha-spending confirmation gate.',
                            'Stable gain is never asserted by this component; adjusted independent confirmation is not implemented.',
                            'Original missing training identities are not repaired by new source-complete runs.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle')
    parser.add_argument('--output', required=True)
    parser.add_argument('--program', help='External trusted PROGRAM anchor; never selected by the evidence bundle')
    args = parser.parse_args()
    try:
        result = audit_bundle(args.bundle, args.program)
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as error:
        result = {'schema': 'maskco_claim_adjudication_v1', 'qualified_evidence': False,
                  'stable_maskco_core_online_gain': False, 'reason': str(error)}
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    raise SystemExit(0 if result['qualified_evidence'] else 2)
