"""Adjudicate the pickup full-scenario stage: preserved FAILs and what was verified."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PK = ROOT / 'research/review_3080_20261010/pickup'
V1 = PK / 'lazy_base_distance_full_scenario_root_v1'
V2 = PK / 'lazy_base_distance_full_scenario_root_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


R = json.loads((V2 / 'RESULT.json').read_text(encoding='utf-8'))
A = json.loads((V2 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
RUN = json.loads((V2 / 'RUN_EXIT.json').read_text(encoding='utf-8'))
AX = json.loads((V2 / 'AUDIT_RUN_EXIT.json').read_text(encoding='utf-8'))
V1_RUN = json.loads((V1 / 'RUN_SUPERVISION_RECEIPT.json').read_text(encoding='utf-8'))

adj = {
    'stage': 'pickup_full_scenario_paired_lazy_base_reuse',
    'registration_sha256': sha(V2 / 'REGISTRATION.json'),
    'preserved_prior_attempt': dict(
        directory='research/review_3080_20261010/pickup/lazy_base_distance_full_scenario_root_v1',
        outcome='producer raised AttributeError and exited 1 before writing any RESULT.json',
        cause='a leftover list initialiser was used as a dict for the deletion relation',
        exit_code=V1_RUN['returncode'], artifacts=['SOURCE_START.json', 'run.stdout.txt', 'run.stderr.txt',
                                                   'RUN_SUPERVISION_RECEIPT.json'],
        original_bytes_unchanged=True),
    'producer': dict(checks=R['checks_count'], failed=R['failed'], exit_code=RUN['exit_code'],
                     failed_checks=[c['name'] for c in R['checks'] if not c['pass_']]),
    'audit': dict(checks=A['checks_count'], failed=A['failed'], exit_code=AX['exit_code'],
                  failure_names=A['failure_names']),
    'failure_classification': {
        'identical_scored_candidates': {
            'class': 'harness design flaw, not a scientific result',
            'detail': ('the unmodified original EnergyCompleteSAA._insert_u computes trial-minus-base but does not '
                       'record its scored candidates, so the registered "identical scored candidates" gate has no '
                       'reference-side log to compare and records FAIL.'),
            'consequence': ('the candidate-side score arithmetic is therefore not compared against the original; the '
                            'insertion choice itself is still compared, because both arms must produce identical '
                            'events and those gates passed.')},
        'producer_reported_no_failure / producer_exit_zero': {
            'class': 'correct propagation',
            'detail': 'the audit refuses to certify a producer that exited 1; this is intended behaviour.'},
        'at_least_one_trial_kept_per_removed_base': {
            'class': 'audit-check defect',
            'detail': ('the check identified base entries by Python object identity across two separately loaded JSON '
                       'logs, so every candidate entry was mis-classified as a trial. The producer-side deletion '
                       'check does not use identity across objects and passed.'),
            'consequence': 'not a finding about the component; recorded as a defect in this audit check.'},
    },
    'verified_substantive_result': {
        'sequence_length': R['inserted'],
        'feasibly_inserted': R['accepted'],
        'scored_feasible_candidates': len(R['score_log']),
        'reference_distance_calls': R['reference_distance_calls'],
        'candidate_distance_calls': R['candidate_distance_calls'],
        'removed_repeated_base_calls': R['proved_removed_base_calls'],
        'removed_fraction': R['proved_removed_base_calls'] / R['reference_distance_calls'],
        'gates_that_passed': [
            'identical insertion choices between the unmodified original and the candidate',
            'identical feasibility calls and values',
            'identical C0 physical transitions (80 segments)',
            'identical terminal energy and distance hex',
            'candidate distance log is the reference log with in-order deletions',
            'every deleted entry is a same-loop base call',
            'all 1042 reference and 733 candidate distance values independently recomputed by ordered double sums',
            'cross-repeat determinism of both arms',
            'immutable distance matrix and public parameter digest unchanged'],
        'timing': {
            'reference_seconds': R['reference_seconds'],
            'candidate_seconds': R['candidate_seconds'],
            'reference_mean': sum(R['reference_seconds']) / len(R['reference_seconds']),
            'candidate_mean': sum(R['candidate_seconds']) / len(R['candidate_seconds']),
            'reading': ('the candidate removes 29.7 percent of the route-distance evaluations but is not faster: its '
                        'mean timed section is higher than the reference in both interleaved pairs. Reducing call '
                        'counts is not a speedup, and this bounded section is small relative to process noise, so no '
                        'acceleration claim is made in either direction.'),
        },
        'conclusion': ('extending the lazy reuse to the whole fixed archived scenario raises the removed-work fraction '
                       'from 13/109 to 309/1042 while leaving every exactness gate that could run intact, and it still '
                       'does not produce a measured speedup. The pickup runtime path therefore stays unqualified and '
                       'is not the route to an online gain.'),
    },
    'no_rerun_to_chase_pass': True,
    'v3_created': False,
    'flags': dict(formal_strong_dynamic_qualified=False, full_request_budget_qualified=False,
                  scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                  stable_online_gain=False),
    'saved_sha256': {n: sha(V2 / n) for n in ('RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json',
                                              'AUDIT_RUN_EXIT.json')},
}
out = V2 / 'STAGE_ADJUDICATION.json'
assert not out.exists()
out.write_text(json.dumps(adj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'written': str(out.relative_to(ROOT)), 'producer_failed': R['failed'], 'audit_failed': A['failed'],
                  'removed_fraction': round(adj['verified_substantive_result']['removed_fraction'], 4)}))
