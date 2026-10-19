"""P2-C-ANCHOR-01 registration: commitment-anchored masked reconstruction (CAMR) scope
feasibility + synthetic persistence qualification. Read-only; registered before running.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT = Path(__file__).resolve().parent
SAA = ROOT / 'research/p2_c_opp_01_20261019'
CTRL = ROOT / 'research/p2_c_ctrl_01_20261019'
DOM = ROOT / 'research/p2_c_domain_01_20261019'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


INPUTS = [SAA / 'phases.json', SAA / 'result.json',
          CTRL / 'phases.json', CTRL / 'result.json',
          DOM / 'domain_summary.json', DOM / 'event_type_census.json']

registration = {
    'run_id': 'p2_c_anchor_01_20261019',
    'baseline': '59c1251cab86774dda5d0aaf083a72b73f7b2a77',
    'parent': 'P2-C-DOMAIN-01 (which measured that the lock is the single in-transit vehicle)',
    'corrected_premise': ('DOMAIN-01 attributed the not-yet-executed suffix of an in-transit vehicle '
                          'to the commitment contract. That attribution is NOT established: '
                          'P2-POS-01 declares it as a conservative action domain ("not a general '
                          'optimum domain"), and build_vehicle_plans() already represents a committed '
                          'vehicle as anchor=(committed_next, committed_finish, load after pickup) '
                          'with the remaining route as its suffix. This package tests whether the '
                          'suffix is legally reconstructable while the anchor stays frozen.'),
    'inputs': {str(p.relative_to(WORKSPACE)): sha(p) for p in INPUTS},
    'H1': ('Freezing the in-transit anchor (committed_next / committed_arrive / committed_finish and '
           'the current physical state), at least one of the R14 reveal phases admits >=2 DISTINCT '
           'candidate plans for the not-yet-executed suffix that pass the same route-level reference '
           'check, AND those candidates are distinguishable by the first action AFTER the anchor.'),
    'primary_control': ('the current conservative action domain: identical enumeration and identical '
                        'reference check, but the in-transit vehicle left out of allowed_vehicle_ids'),
    'primary_metrics': {
        'n_anchor_candidates': 'distinct plan_hash values over the full fleet, anchor frozen',
        'n_next_action_after_anchor': ('number of distinct first elements of the in-transit vehicle '
                                       'suffix after the anchor; 0 = depot return'),
        'phase_verdict': ['ANCHOR_LEGAL_MULTI', 'ANCHOR_LEGAL_SAME_NEXT', 'NO_LEGAL_CANDIDATE',
                          'INSUFFICIENT_QUEUE'],
    },
    'anchor_freeze_assertions': [
        'candidate plans[committed_vid].anchor_node == original',
        'candidate plans[committed_vid].anchor_time == original',
        'candidate plans[committed_vid].anchor_load == original',
        'the committed_next order is NOT in any candidate suffix',
        'all accepted-unserved orders still appear exactly once across the fleet',
    ],
    'mask_scope_under_test': 'size-2 masks over the queue (P2-POS-01 keeps the fixed pair rule); size-1 enumerated for context only',
    'reference_check': ('route_check (customer TW + capacity + depot return) via the shared '
                        'certify/enumerate path; energy and quality are NOT checked because the C0 '
                        'budget B is AWAITING_USER_BUDGET_RULING'),
    'data': 'the 289 saved public phases of OPP-01/CTRL-01; no new root day',
    'qualification': 'REFERENCE_ONLY',
    'synthetic_test': ('a hand-built minimal instance (NOT a source day) driven by the real unmodified '
                       'StrictOnlineEnv, to check that a suffix installed behind a frozen anchor is '
                       'actually consumed in order and is not silently dropped at a reveal or a '
                       'service completion; plan-storage/persistence consistency is a STOP CONDITION'),
    'prohibited': ['modifying the execution engine', 'modifying controllers or thresholds',
                   're-running a SOURCE day', 'label collection', 'training', 'GPU', 'final seeds',
                   'claiming online utility', 'claiming novelty of CAMR'],
    'resource': {'CPU_workers': 1, 'BLAS_threads': 1, 'GPU': False, 'wall_s_cap': 1800},
    'outputs': ['anchor_scope.csv', 'anchor_summary.json', 'synthetic_tx.json'],
}

with (OUT / 'registration.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(registration, f, ensure_ascii=False, indent=1, sort_keys=True)
    f.write('\n')
print('registered', registration['run_id'], 'inputs', len(registration['inputs']))
