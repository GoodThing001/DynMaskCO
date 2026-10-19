"""P2-C-CAMR-STATIC-01  T1: pre-declaration and resource freeze.

Writes `registration.json` BEFORE any implementation file exists, as required by the
authorisation ("T1 必须先于实现和实验"). It therefore freezes:

  * the single hypothesis and the L1-A / L1-B criteria
  * the 19 phase-mask units, each with a full identity key (arm, day_id, ordinal,
    public_identity, mask) and a sha256 over the ordered list
  * the shared legal-action interface, the capacity convention, the anchor assertions
  * the unified tie-break rule
  * Pass A / Pass B budgets and the resource caps
  * the exception taxonomy and the denominator policy
  * the source FILE LIST (names only, NO hashes: the implementation does not exist yet and
    the ruling keeps the pre-declaration freeze and the source-identity freeze at two
    distinct time points; hashes belong to T2)
  * the model identity that T2 must verify after loading
  * prohibitions, stop conditions, execution order

Read-only with respect to the hypothesis inputs. Refuses to overwrite.

Usage: python register.py [--out registration.json]
"""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../C-VRP_Cold-chainVehicleRoutingProblem
WORKSPACE = ROOT.parent
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / 'scripts'))
from project_paths import configure_imports                       # noqa: E402
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation', 'training'))

import p2_pos01_reference as ref                                   # noqa: E402
from p2_pos01c_gap import view_from_json                           # noqa: E402

PHASES = {'SAA': ROOT / 'research/p2_c_opp_01_20261019/phases.json',
          'MYOPIC': ROOT / 'research/p2_c_ctrl_01_20261019/phases.json'}
SCOPE = ROOT / 'research/p2_c_anchor_01_20261019/anchor_scope.csv'
BASELINE = 'f4b7e6594acd4a3c1c2805367aba37bf6771c7c4'

CHECKPOINT = 'MASKCO_code/ckpts/cvrp100.ckpt'
CHECKPOINT_SHA = '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
CHECKPOINT_BYTES = 305061150

# Source FILE LIST frozen at T1 (names only; hashes are frozen at T2).
SOURCE_FILES = [
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/camr_shared.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/synthetic_regression.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/freeze_identity.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/degeneration_gate.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/run_pass.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/verify_camr_static.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_camr_static_01_20261019/register.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/p2_pos01_mask_value.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre_trained.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/dynmaskco_cc_graph.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/dynmaskco_cc.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/action_contract.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/mpre_replanner.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/mpre_policy.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/experimental/p2_pos01_reference.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/experimental/p2_pos01c_gap.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_sgbs_fixed_state.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/strict_online_env.py',
]

INPUT_FILES = {
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_opp_01_20261019/phases.json': None,
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_ctrl_01_20261019/phases.json': None,
    'C-VRP_Cold-chainVehicleRoutingProblem/research/p2_c_anchor_01_20261019/anchor_scope.csv': None,
}

RESOURCES = {
    'unit_wall_s': 300,
    'package_wall_s': 3600,
    'memory_rss_gib': 6,
    'scoring_calls_per_unit_per_pass': 500,
    'warmup_calls_per_side': 1,
    'warmup_excluded_from_unit_wall': True,
    'cpu_workers': 1,
    'gpu': False,
    'jax_platform': 'cpu',
    'blas_threads': 1,
}

EXCEPTIONS = {
    'side_level': ['NO_LEGAL_ACTION', 'NONFINITE_SCORE', 'TIMEOUT',
                   'ANCHOR_MISMATCH', 'REFERENCE_REJECT'],
    'pair_level': ['PAIR_INVALID_NO_OUTPUT', 'PAIR_INVALID_TIMEOUT',
                   'PAIR_INVALID_NONFINITE', 'PAIR_INVALID_ILLEGAL'],
    'rule_two_sided_none_is_not_identical':
        'if either side has no valid plan the pair is PAIR_INVALID_*, which is NOT counted as '
        '"identical"; only two completed, anchor-preserving, reference-passing plans may be '
        'compared by plan_hash',
    'denominator_policy':
        'every invalid pair keeps its slot in the 19; the denominator never shrinks; the report '
        'must print n_evaluable, n_invalid and the list of invalid unit_ids',
}


def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_units():
    """Rebuild the ANCHOR-01 mask rule verbatim and attach the full identity key."""
    import csv
    with SCOPE.open('r', encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['verdict'] == 'ANCHOR_LEGAL_MULTI']
    units = []
    for r in rows:
        arm, day, ordinal = r['arm'], int(r['day']), int(r['ordinal'])
        ph = next(p for p in json.loads(PHASES[arm].read_text(encoding='utf-8'))
                  if int(p['day']) == day and int(p['ordinal']) == ordinal)
        view = view_from_json({'public_view': ph['public_view'],
                               'public_identity': ph['public_identity']})
        plans = ref.plans_from_view(view)
        committed = [v for v in view.vehicles
                     if v.status == 'committed' and v.committed_next not in (None, 0)]
        if len(committed) != 1:
            raise SystemExit('unit rebuild: expected exactly one committed vehicle')
        cvid = int(committed[0].vehicle_id)
        base = plans[cvid]
        pending = {int(x) for x in view.accepted} - {int(x) for x in view.served}
        queue = [c for c in base.suffix if c in pending]
        masks = [tuple(sorted(p)) for p in combinations(queue, 2)]
        if len(masks) != int(r['n_masks']):
            raise SystemExit('unit rebuild: mask count %d != anchor_scope %s'
                             % (len(masks), r['n_masks']))
        for m in masks:
            units.append({'unit_id': {'arm': arm, 'day_id': day, 'ordinal': ordinal,
                                      'public_identity': ph['public_identity'],
                                      'mask': list(m)},
                          'committed_vid': cvid,
                          'committed_next': int(base.anchor_node),
                          'queue': [int(c) for c in queue],
                          'anchor': [int(base.anchor_node), round(float(base.anchor_time), 6),
                                     round(float(base.anchor_load), 6)]})
    units.sort(key=lambda u: (u['unit_id']['arm'], u['unit_id']['day_id'],
                              u['unit_id']['ordinal'], u['unit_id']['mask']))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'registration.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    units = build_units()
    unit_sha = hashlib.sha256(json.dumps(units, sort_keys=True).encode()).hexdigest()

    inputs = {}
    for rel in INPUT_FILES:
        inputs[rel] = sha_file(WORKSPACE / rel)

    payload = {
        'schema': 'p2_c_camr_static_01_registration_v1',
        'run_id': 'p2_c_camr_static_01_20261019',
        'package': 'P2-C-CAMR-STATIC-01',
        'plan_version': 'R2.1',
        'baseline_commit': BASELINE,
        'authorisation': ('GO - limited execution, granted for R2.1 only: T1 freeze, isolated '
                          'implementation, synthetic regression, T2 source-identity freeze, '
                          'degeneration gate, Pass A, Pass B, two deterministic runs, records, '
                          'failure ledger, independent verification and the adjudication report'),
        'qualification': 'REFERENCE_ONLY / L1',
        'hypothesis': ('H-CAMR-STATIC-01: under identical strict-online public state, frozen '
                       'anchor, FIXED two-order mask, SHARED legal-action space and controlled '
                       'budget, does the frozen MaskCO reconstruction score produce a legal '
                       'complete plan different from distance-greedy?'),
        'not_answered_here': ['learned mask selection', 'quality advantage',
                              'installability / L3', 'online utility / L2'],
        'criteria': {
            'L1-A': ('behavioural difference of the learned score: compare the two GREEDY '
                     'outputs only. Go requires a unit where the plan_hash differs and BOTH '
                     'plans are anchor-preserving and pass the shared reference check, produced '
                     'under the same allowed set, the same tie-break, the same per-side budget '
                     'and the same shared interface, and the whole package reproduces bit for '
                     'bit on a second run'),
            'L1-A_nogo_reasons': ['no difference at all',
                                  'not reproducible',
                                  'illegal (anchor changed or reference check failed)',
                                  'fully explained by budget / action-set / tie-break asymmetry'],
            'L1-B': ('coverage by the non-learned search: is the MaskCO-greedy output inside the '
                     'distance-SGBS completion set? Full coverage forbids an "irreplaceable '
                     'search value" claim but does NOT retract the observed L1-A fact; uncovered '
                     'legal plans only justify further investigation and never a quality claim'),
            'L2_L3': 'blocked; a static L1 result grants no exemption',
        },
        'units': {'n': len(units), 'expected_n': 19, 'sha256': unit_sha, 'list': units},
        'inputs': inputs,
        'input_discipline': 'read-only; no SOURCE day is re-run; no new instance is drawn',
        'source_file_list': {'note': ('T1 freezes the FILE LIST only. Hashes are frozen at T2, '
                                      'after implementation and synthetic regression, per the '
                                      'authorisation ruling that the pre-declaration freeze and '
                                      'the source-identity freeze are two distinct time points.'),
                             'files': SOURCE_FILES},
        'shared_interface': {
            'definition': ('legal_actions_shared(view, plans, customer, allowed) keeps (c, action) '
                           'iff cand.feasible (certify_route certificate) AND every plan of the '
                           'trial plan set passes route_check AND the action targets a vehicle in '
                           'the explicit allowed set'),
            'allowed': 'mutable_ids(view) UNION {committed_vid} for BOTH sides (same object)',
            'forbidden': ('calling mpre_policy.enumerate_legal_actions(..., mutable_ids) directly '
                          'on either side'),
            'capacity_convention': 'route_check capacity_ok: load <= capacity - 1e-4 '
                                   '(the stricter of the two conventions present in the repo)',
            'anchor_assertions': ['plans[committed_vid].anchor_node == recorded anchor node',
                                  'plans[committed_vid].anchor_time == recorded anchor time',
                                  'plans[committed_vid].anchor_load == recorded anchor load',
                                  'committed_next never appears in any suffix',
                                  'every accepted-unserved order appears exactly once fleet-wide'],
        },
        'tie_break': {
            'rule': ('both sides order actions by (primary ascending, action_id() ascending) with '
                     'action_id() = FleetAction.action_id(); primary = incremental_distance for '
                     'the distance side and -score for the model side'),
            'why': ('_model_propose in the historical module used bare np.argsort(-scores) with no '
                    'action_id tie-break while _distance_propose already used '
                    '(incremental_distance, action_id()); that asymmetry could manufacture a fake '
                    'model difference'),
            'must_be_applied_before': 'the degeneration gate',
        },
        'budget': {
            'pass_a': {'object': 'single greedy completion per side',
                       'sides': ['MaskCO-greedy', 'distance-greedy'],
                       'completions_per_side': 1},
            'pass_b': {'object': 'completion set per side',
                       'sides': ['MaskCO-SGBS', 'distance-SGBS'],
                       'beam': 2, 'max_per_parent': 2, 'max_completions': 8},
            'single_output_rule': ('Pass A is the sole primary comparison. The Pass B set is a '
                                   'secondary descriptive object and may never rewrite the L1-A '
                                   'verdict. min(J_vis) selection is banned outright.'),
            'runs': 2,
        },
        'resources': RESOURCES,
        'exceptions': EXCEPTIONS,
        'outputs': ['registration.json', 'camr_shared.py', 'synthetic_regression.json',
                    'run_identity.json', 'preflight.json', 'pass_a_run1.json', 'pass_b_run1.json',
                    'pass_a_run2.json', 'pass_b_run2.json', 'verification.json', 'REPORT.md'],
        'model_identity_to_freeze_at_T2': {
            'checkpoint': CHECKPOINT,
            'checkpoint_sha256': CHECKPOINT_SHA,
            'checkpoint_bytes': CHECKPOINT_BYTES,
            'loader': 'p2_pos01_mask_value.FrozenBackbone',
            'scoring_entry': 'FrozenBackbone.insertion_scores(view, base, partial, actions)',
            'must_record': ['checkpoint_sha256', 'parameter_sha256', 'encoder_dim',
                            'decoder_dim', 'input_dim', 'layers', 'step', 'platform',
                            'public_node_bucket'],
            'no_trained_insertion_head': ('the module header states there is no decoder '
                                          'fine-tuning and no insertion head; the mask-value head '
                                          'is zero-initialised (KEEP) and MUST NOT be used to '
                                          'produce or pick a mask'),
        },
        'execution_order': ['T1 registration freeze (this file)',
                            'isolated implementation',
                            'synthetic regression (no SOURCE day)',
                            'T2 run-identity freeze (source hashes + synthetic record + model '
                            'parameter identity)',
                            'degeneration gate -> preflight.json bound to run_identity_sha256',
                            'Pass A and Pass B, two complete deterministic runs',
                            'independent verification',
                            'adjudication report'],
        'prohibitions': [
            'modifying the execution engine or upstream MASKCO_code/',
            're-running a SOURCE day', 'using the final test seeds',
            'counterfactual continuation', 'utility label collection', 'value-head training',
            'formal L2/L3', 'production qualification',
            'GPU', 'more than one CPU worker',
            'changing mask rule, capacity convention, shared interface, tie-break, budget or '
            'timeout after the freeze', 'dropping an invalid unit from the 19-unit denominator',
            'using the Pass B set to rewrite the L1-A verdict',
            'using min(J_vis) selection as a substitute for the greedy output',
            'reusing a mixed-version result after a source repair',
        ],
        'stop_conditions': [
            'T1/T2 hash mismatch, or checkpoint / model parameter identity mismatch',
            'the shared legal-action set, anchor constraint, capacity convention or tie-break '
            'cannot be unified',
            'the degeneration gate fails',
            'source repaired without re-freezing T2 and re-running every prerequisite check',
            'declared resource cap exceeded, or a non-reproducible result',
        ],
    }
    a.out.open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

    print('units', len(units), 'sha256', unit_sha[:16])
    for u in units[:3]:
        print(' ', u['unit_id']['arm'], u['unit_id']['day_id'], u['unit_id']['ordinal'],
              u['unit_id']['public_identity'][:10], u['unit_id']['mask'])
    print('inputs', json.dumps(inputs, indent=1))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
