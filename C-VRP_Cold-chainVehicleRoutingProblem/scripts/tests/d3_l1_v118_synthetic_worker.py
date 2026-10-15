#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SYNTHETIC scorer worker for the V1.18 end-to-end acceptance.

This worker drives the REAL production path -- `d3_l1_v118_targets.verify_slot_record`,
`publish_new(..., verify=True)`, the V1.14 phase journal, the V1.16 contained supervisor and the
round-revocation adjudicator -- but replaces the F model with a deterministic SYNTHETIC scorer.

It exists because the real F forward is the single item awaiting the user's signature.  Everything
it publishes is labelled `SYNTHETIC_SCORER_NOT_F`, so a synthetic PASS can never be read as a real
F result (hard rule: "不得把合成 64 行 PASS 冒充真实 F 64 状态 PASS").

Behaviour is selected by `--mode`:

  pass            publish all 64 verified rows, the trailing identity record, then 'completed'
  rows_63         publish 63 rows only (the row count is short by one)
  block_slot      declare a slot and then sleep far past the slot cap, so the real deadline kills it
  mutate_watched  publish all 64 rows, then modify a WATCHED file so the trailing identity fails
  corrupt_phase   publish all 64 rows and then leave a corrupt phase record behind
  no_tail         publish all 64 rows and then exit WITHOUT the trailing identity record
  bad_record      publish row 0 with a deliberately wrong score length (a real verification FAIL)
  slow_init       sleep `--init-delay` BEFORE publishing READY, then publish all 64 rows; used to
                  prove that a real initialization overrun is killed at the initialization deadline
  one_slot_then_hold
                  publish READY and one verified row, then sleep, so the batch outlives the
                  initialization cap without ever exceeding the batch cap
  fail_finite     publish row 0 with NaN scores (the finite check must fail)
  fail_shape      publish row 0 with a 2-D score vector (the shape check must fail)
  fail_repeat     publish row 0 whose two forwards differ (the bitwise-repeat check must fail)
  fail_len        publish row 0 with one score too few (the shape/ordered-legal check must fail)
  stop_after      for the `fail_*` modes the worker STOPS after the failing row, exactly like the
                  production worker, so the harness can prove no later state was scored
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

LABEL = 'SYNTHETIC_SCORER_NOT_F / ENGINEERING_ACCEPTANCE_ONLY'


def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


journal = _import('d3_l1_v112_phase_journal_synth',
                  EXT / 'scripts/tests/d3_l1_v112_phase_journal.py')
import d3_l1_v118_persistence as persistence                                # noqa: E402
import d3_l1_v118_targets as targets                                        # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def synthetic_scores(n, *, day, pos, seed=20261015):
    """Deterministic bounded float32 vector of the right length -- explicitly NOT F output."""
    out = []
    x = (seed + 1009 * day + 9176 * pos) % 2147483647
    for _ in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out.append(((x % 20000) - 10000) / 10000.0)
    import numpy as np
    return np.asarray(out, dtype=np.float32)


# Defect injection: each entry makes the FIRST slot violate exactly one frozen check, so the
# production worker's own check set (not a harness assertion) is what must catch it.
DEFECT_MODES = {
    'fail_finite': 'finite',
    'fail_shape': 'shape',
    'fail_repeat': 'repeat',
    'fail_len': 'len',
}


def _inject(defect, a, b, n_expected):
    """Return `(checks_expected_to_fail, a, b)` for a defect mode; `(None, a, b)` when not a defect."""
    import numpy as np
    if defect is None:
        return None, a, b
    if defect == 'finite':
        a = a.copy()
        a[0] = np.float32('nan')
        b = a.copy()
        return ['all_finite'], a, b
    if defect == 'shape':
        a = np.zeros((1, max(1, len(a))), dtype=np.float32)
        b = a.copy()
        return ['shape_matches_ordered_legal'], a, b
    if defect == 'repeat':
        b = b.copy()
        b[0] = np.float32(b[0] + 1.0)
        return ['repeat_exact_type_dtype_shape_bytes'], a, b
    if defect == 'len':
        a = a[:-1].copy() if len(a) > 1 else a.copy()
        b = a.copy()
        return ['shape_matches_ordered_legal'], a, b
    raise ValueError('unknown defect %r' % defect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--init-delay', type=float, default=0.0,
                    help='seconds to sleep BEFORE publishing the READY record')
    args = ap.parse_args()
    out = Path(args.out_dir)
    persisted = json.loads((out / 'FREEZE_REPORT.json').read_text(encoding='utf-8'))
    # The frozen records are published to unique names; the reader is the only supported accessor.
    ident = persistence.VerifiedReader(out).latest('RUN_IDENTITY')
    fwd = persistence.VerifiedReader(out).latest('FORWARD_TARGETS')
    idt = persistence.VerifiedReader(out).latest('IDENTITY_TARGETS')
    phases = out / 'phases'
    phases.mkdir(parents=True, exist_ok=True)
    jr = journal.PhaseJournal(phases)
    run_id = ident['run_id']
    persistence.publish_new(out, 'MODEL_LOAD',
                            {'run_id': run_id, 'label': LABEL, 'scorer': 'synthetic',
                             'restored_exact': None, 'synthetic': True,
                             'not_evidence': 'this is not an F model load'})
    persistence.publish_new(out, 'WARMUP',
                            {'run_id': run_id, 'label': LABEL, 'calls': 1, 'synthetic': True})
    persistence.publish_new(phases, 'identity_start', {'stage': 'identity_start',
                                                       'run_id': run_id,
                                                       'paths': idt['paths']})
    if args.init_delay > 0:
        # A REAL initialization overrun: nothing is READY yet, so the supervisor's initialization
        # deadline is the only thing that can end this run.
        time.sleep(args.init_delay)
    persistence.publish_new(out, 'READY',
                            {'run_id': run_id, 'kind': 'initialization_and_compilation_finished',
                             'mono': time.monotonic(), 'n_warmup_calls': 1,
                             'warmup_scores_sha256': 'synthetic', 'worker_pid': os.getpid(),
                             'synthetic': True, 'not_evidence': 'not an F warm-up'})
    n_rows = 63 if args.mode == 'rows_63' else len(fwd['rows'])
    defect = DEFECT_MODES.get(args.mode)
    for index, row in enumerate(fwd['rows'][:n_rows]):
        day, pos = int(row['day_id']), int(row['position'])
        if args.mode == 'one_slot_then_hold' and index == 1:
            # READY was already published, so the initialization cap must no longer apply.  This
            # sleep happens BETWEEN two completed slots: slot 0 is already 'slot_finished' and slot
            # 1 has NOT been declared open, so no slot deadline is involved.  The only deadline that
            # could still fire is the batch cap, which is far away.  A supervisor keyed to a fixed
            # READY.json (which this worker never writes) would instead kill this at the
            # initialization deadline.
            time.sleep(float(os.environ.get('V119_HOLD_S', '20')))
        jr.publish('slot', day_id=day, position=pos)
        if args.mode == 'block_slot' and (day, pos) == (0, 2):
            time.sleep(120)                       # the supervisor's real deadline must kill this
            return 0
        n = int(row['n_legal_expected'])
        if args.mode == 'bad_record' and (day, pos) == (0, 1):
            n = max(1, n - 1)                     # a real, detectable verification failure
        a = synthetic_scores(n, day=day, pos=pos)
        b = a.copy()
        expected_failures = None
        if defect and index == 0:
            expected_failures, a, b = _inject(defect, a, b, n)
        raw = persistence.publish_new(out / 'scores', 'SCORES',
                                      pickle.dumps({'synthetic': True, 'scores_first': a,
                                                    'scores_repeat': b,
                                                    'input_sha256': row['input_sha256'],
                                                    'legal_sha256': row['legal_sha256'],
                                                    'label': LABEL}, protocol=4))
        rec = {'run_id': run_id, 'day_id': day, 'position': pos,
               'status': 'FORWARD_CHECKS_PASS',
               'failure_reason': None,
               'checks': {'synthetic': True},
               'raw_relpath': str(Path(raw['path']).relative_to(out).as_posix()),
               'raw_sha256': raw['sha256'],
               'input_sha256': row['input_sha256'], 'legal_sha256': row['legal_sha256'],
               'state_hash': row['state_hash'],
               'scores_dtype': a.dtype.str, 'scores_shape': list(a.shape),
               'scores_first_hex': a.tobytes().hex(), 'scores_repeat_hex': b.tobytes().hex(),
               'forward_first_wall_s': 0.0, 'forward_repeat_wall_s': 0.0,
               'span_wall_s': 0.0, 'F_calls': 2, 'label': LABEL, 'synthetic': True}
        persistence.publish_new(out / 'scores', 'SLOT_day%02d_pos%d' % (day, pos), rec)
        jr.publish('slot_finished', day_id=day, position=pos, status='FORWARD_CHECKS_PASS')
        if expected_failures is not None:
            # Terminal, exactly like the production worker: record the failure, publish the tail so
            # the round can be adjudicated, stop, and exit non-zero.  Any later state would be a
            # defect, and the harness asserts that no later row record exists at all.
            persistence.publish_new(out, 'FIRST_FAILURE',
                                    {'run_id': run_id, 'day_id': day, 'position': pos,
                                     'expected_failing_checks': expected_failures,
                                     'row_index': index, 'synthetic': True})
            persistence.publish_new(phases, 'identity_end',
                                    {'stage': 'identity_end', 'run_id': run_id,
                                     'paths': {str(p): sha256_file(p) for p in idt['paths']},
                                     'ended_because': 'slot_failure'})
            persistence.publish_new(out, 'WORKER_RESULT',
                                    {'run_id': run_id, 'ok': False, 'label': LABEL,
                                     'terminated_on_first_failure': True,
                                     'first_failure_row_index': index})
            return 1
    if args.mode == 'mutate_watched':
        watched = os.environ.get('V118_WATCHED')
        if watched:
            Path(watched).write_text('mutated by the synthetic worker\n', encoding='utf-8')
    if args.mode == 'corrupt_phase':
        (phases / 'phase_999999998_corrupt.json').write_bytes(b'{ this is not json')
    if args.mode != 'no_tail':
        persistence.publish_new(phases, 'identity_end',
                                {'stage': 'identity_end', 'run_id': run_id,
                                 'paths': {str(p): sha256_file(p) for p in idt['paths']}})
    jr.publish('completed', run_id=run_id)
    persistence.publish_new(out, 'WORKER_RESULT', {'run_id': run_id, 'ok': True, 'label': LABEL})
    return 0


if __name__ == '__main__':
    sys.exit(main())
