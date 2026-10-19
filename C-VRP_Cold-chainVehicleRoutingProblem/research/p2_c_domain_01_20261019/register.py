"""P2-C-DOMAIN-01 registration: legal mask scope and decision-timing audit (read-only).

Registered BEFORE the audit is run. The audit re-reads the frozen CTRL/OPP phase
records and the 289 saved public phases; it never executes a controller, a repair,
a rollout, an evaluator or a source day.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT = Path(__file__).resolve().parent
SAA = ROOT / 'research/p2_c_opp_01_20261019'
CTRL = ROOT / 'research/p2_c_ctrl_01_20261019'


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


INPUTS = [SAA / 'phases.json', SAA / 'result.json', SAA / 'verification.json',
          CTRL / 'phases.json', CTRL / 'result.json', CTRL / 'verification.json']

registration = {
    'run_id': 'p2_c_domain_01_20261019',
    'baseline': '2e461046ad690ce8d727948074fcbf616191ce4a',
    'purpose': ('Attribute where accepted-but-unserved orders are locked out of the legal mask '
                'scope, and census whether alternative legal mask granularities would create '
                'distinct candidate plan identities on the SAME saved public states.'),
    'analyzed_unit': 'the 289 saved natural phases: (arm, day, ordinal) with arm in {SAA, MYOPIC}',
    'inputs': {str(p.relative_to(WORKSPACE)): sha(p) for p in INPUTS},
    'analysis_scope': {
        'Q1_lock_attribution': ('per phase, classify every accepted-and-unserved order into exactly '
                                'one bucket: MUTABLE / PROTECTED / EXCLUDED_FROM_REPLAN / '
                                'STATUS_NOT_IDLE_READY / CLOSED_VEHICLE / COMMITTED_ANCHOR / '
                                'UNREPRESENTED. Buckets are derived from the saved public_view '
                                'only (protected, replan_ids, vehicle status, committed_next, '
                                'mutable_suffix).'),
        'Q2_mask_granularity_census': ('per phase, count DISTINCT candidate plan identities '
                                       '(action_contract.plan_hash over the full fleet plan) '
                                       'reachable by legality-only enumeration (route_check), '
                                       'under four pre-declared policies: CURRENT(size-2, 4 '
                                       'heuristics, seed 0) / S1_ONLY / MIXED_1_2 / MIXED_1_2_3.'),
        'Q3_timing': ('all 289 phases are the single frozen type POST_ADMISSION_PRE_REPAIR; any '
                      'alternative capture point is therefore UNASSESSABLE from these artifacts. '
                      'Only two artifact-based quantities are reported: (a) orders that were in the '
                      'mutable pool at phase k-1 and are frozen at phase k; (b) a scope-only '
                      'structural upper bound that is explicitly NOT a legal option.'),
    },
    'declared_policies': ['CURRENT', 'S1_ONLY', 'MIXED_1_2', 'MIXED_1_2_3'],
    'candidate_identity': 'action_contract.plan_hash(plans) of the reconstructed fleet plan; no evaluation',
    'qualification': 'DOMAIN_ONLY',
    'prohibited': ['MaskCO repair execution', 'counterfactual continuation', 'utility / distance / '
                   'coldchain_cost computation', 'training', 'GPU', 'final seeds',
                   're-running either SOURCE day', 'changing controller or thresholds',
                   'backfilling OPP/CTRL receipts'],
    'resource': {'CPU_workers': 1, 'BLAS_threads': 1, 'GPU': False, 'wall_s_cap': 1800},
    'outputs': ['lock_attribution.csv', 'mask_scope_census.csv', 'domain_summary.json'],
}

with (OUT / 'registration.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(registration, f, ensure_ascii=False, indent=1, sort_keys=True)
    f.write('\n')
print('registered', registration['run_id'], 'inputs=', len(registration['inputs']))
