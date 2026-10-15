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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--mode', required=True)
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
    n_rows = 63 if args.mode == 'rows_63' else len(fwd['rows'])
    for row in fwd['rows'][:n_rows]:
        day, pos = int(row['day_id']), int(row['position'])
        jr.publish('slot', day_id=day, position=pos)
        if args.mode == 'block_slot' and (day, pos) == (0, 2):
            time.sleep(120)                       # the supervisor's real deadline must kill this
            return 0
        n = int(row['n_legal_expected'])
        if args.mode == 'bad_record' and (day, pos) == (0, 1):
            n = max(1, n - 1)                     # a real, detectable verification failure
        a = synthetic_scores(n, day=day, pos=pos)
        b = a.copy()
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
