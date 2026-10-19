"""P2-C-CAMR-STATIC-01  degeneration gate (preflight).

Hard precondition for Pass A. For every one of the 19 registered units it runs the REAL model
plumbing with the model's score discarded and replaced by -incremental_distance, and requires the
result to be bit-identical to distance-greedy. If it is not, then the two sides differ because of
the plumbing (batching, index mapping, score sign, tie-break, skeleton) rather than because of the
learned score, and Pass A must not run.

The receipt is written to `preflight.json` and binds to `run_identity.json` by sha256, because T2
is frozen before this gate executes and must not be edited afterwards.

Runs AFTER T2. Does not write an analysis result: `run_pass.py` refuses to start unless this file
exists with all_ok true and the binding hash matches.

Usage: python degeneration_gate.py [--out preflight.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import camr_shared as cs                                            # noqa: E402

class CountingBackbone:
    """Proxy that proves the model path really executed, and counts its invocations."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def insertion_scores(self, *a, **k):
        self.calls += 1
        return self.inner.insertion_scores(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'preflight.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    reg = cs.load_json(OUT / 'registration.json')
    ident = cs.load_json(OUT / 'run_identity.json')
    ident_sha = cs.sha_file(OUT / 'run_identity.json')
    if ident['T1']['registration_sha256'] != cs.sha_file(OUT / 'registration.json'):
        raise SystemExit('STOP: run_identity does not bind the current registration')
    if ident['gate_result_included'] is not False:
        raise SystemExit('STOP: run_identity must not contain a gate result at T2')

    from p2_pos01_mask_value import FrozenBackbone
    bb = CountingBackbone(FrozenBackbone(str(cs.WORKSPACE / ident['model']['checkpoint'])))
    if dict((k, list(v) if isinstance(v, tuple) else v) for k, v in bb.inner.manifest.items()) \
            != ident['model']['manifest']:
        raise SystemExit('STOP: model manifest differs from the T2 record')

    beam, per_parent, max_comp = 2, 2, 8
    rows = []
    for unit in reg['units']['list']:
        uid = unit['unit_id']
        u = cs.unit_from_row({'arm': uid['arm'], 'day': uid['day_id'],
                              'ordinal': uid['ordinal']})
        mask = tuple(int(x) for x in uid['mask'])
        if (u['public_identity'] != uid['public_identity']
                or u['committed_next'] != unit['committed_next']
                or [int(x) for x in u['queue']] != [int(x) for x in unit['queue']]):
            raise SystemExit('STOP: unit reconstruction mismatch %s' % (uid,))

        view, plans, env, allowed = u['view'], u['plans'], u['env'], u['allowed']
        base = plans
        before = bb.calls
        t0 = time.time()
        g_gate = cs.greedy(cs.proposer(view, env, allowed, score='model_path_distance',
                                       backbone=bb, base=base), plans, mask, allowed)
        g_dist = cs.greedy(cs.proposer(view, env, allowed, score='distance'), plans, mask, allowed)
        wall = time.time() - t0
        h_gate = None if g_gate['plan'] is None else cs.signature(g_gate['plan'])
        h_dist = None if g_dist['plan'] is None else cs.signature(g_dist['plan'])
        rows.append({'unit_id': uid, 'gate_plan_hash': h_gate, 'distance_plan_hash': h_dist,
                     'equal': (h_gate is not None and h_gate == h_dist),
                     'gate_reason': g_gate['reason'], 'distance_reason': g_dist['reason'],
                     'model_calls_in_gate': bb.calls - before,
                     'wall_s': round(wall, 3)})

    checks = {
        'all_units_equal': all(r['equal'] for r in rows),
        'all_units_produced_a_plan': all(r['gate_plan_hash'] is not None for r in rows),
        'model_path_executed': all(r['model_calls_in_gate'] > 0 for r in rows),
        'unit_count_is_19': len(rows) == reg['units']['n'] == 19,
        'wall_within_package_cap': sum(r['wall_s'] for r in rows) <= reg['resources']['package_wall_s'],
    }
    payload = {
        'schema': 'p2_c_camr_static_01_preflight_v1',
        'run_id': reg['run_id'],
        'run_identity_sha256': ident_sha,
        'registration_sha256': ident['T1']['registration_sha256'],
        'gate': ('model path executed (insertion_scores really called) with the model score '
                 'discarded and replaced by -incremental_distance, then compared bit-for-bit '
                 'with distance-greedy'),
        'params': {'beam': beam, 'max_per_parent': per_parent, 'max_completions': max_comp},
        'checks': checks,
        'all_ok': all(checks.values()),
        'n_units': len(rows),
        'n_equal': sum(1 for r in rows if r['equal']),
        'rows': rows,
        'note': ('preflight is a precondition receipt, not an analysis result; it is bound to '
                 'run_identity_sha256 so that T2 could be frozen first'),
    }
    cs.write_json(a.out, payload)
    for r in rows:
        print('%-7s d%s ord%-3s %-10s equal=%s' % (r['unit_id']['arm'], r['unit_id']['day_id'],
                                                   r['unit_id']['ordinal'],
                                                   str(r['unit_id']['mask']), r['equal']))
    print(json.dumps(checks, indent=1))
    print('GATE', 'PASS' if payload['all_ok'] else 'FAIL')
    sys.exit(0 if payload['all_ok'] else 5)


if __name__ == '__main__':
    main()
