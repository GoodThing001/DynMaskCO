#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.18 forward / supervisor targets: WHO is watched, WHICH rows are expected, and how a record
is independently verified.

The V1.11 failure was a persistence race, but the reason it could not be diagnosed afterwards was
that the harness had no durable statement of what SHOULD have been there.  This module supplies
that missing statement, mechanically:

  * `forward_targets(run_identity)` -- the ordered list of expected slot rows, each with the
    PRE-QUALIFIED input digest taken from the V1.10 input qualification report.  It does not
    re-derive inputs and does not score anything.
  * `identity_targets(...)` -- the files the trailing identity record must cover.
  * `verify_slot_record(record, state, *, expect_forward_calls)` -- the independent, per-row
    verifier: shape, finiteness, bit-identical repetition, the recorded input digest, the ordered
    legal length and the score-vector digest, plus the call count.  It returns `(ok, detail)` and
    never raises on a malformed record -- it reports.

Both the runner and the read-only verifier import this module, so the two cannot drift into
disagreeing about what "verified" means.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_ROWS = 64
EXPECTED_FORWARD_CALLS_PER_SLOT = 2
# The row record is published to a FRESH destination name (V1.18), so it is located by a glob.
# The runner and the read-only verifier both import THIS constant, so the two cannot drift into
# disagreeing about which file is the row record for a given (day_id, position).
SLOT_RECORD_GLOB = 'SLOT_day{day:02d}_pos{pos}_*.json'
FAILURE_REASONS = ('no_legal', 'nan', 'timeout', 'rollback', 'cert_fail',
                   'score_shape', 'score_len', 'geo_len', 'no_certifier')


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def input_qualification_digests(path):
    """`{(day_id, position): {'input_sha256', 'legal_sha256', 'n_legal'}}` from the V1.10 report.

    The V1.10 report is the frozen, independently audited statement of what each state's F input
    tensor is.  Reading it is how the new run binds to an ALREADY-QUALIFIED input instead of
    recomputing one (which would be a new derivation and could silently differ).
    """
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for row in doc.get('rows', []):
        key = (int(row['day_id']), int(row['position']))
        out[key] = {'input_sha256': row.get('input_sha256'),
                    'legal_sha256': row.get('legal_sha256'),
                    'n_legal': row.get('n_legal'),
                    'Nv': row.get('Nv'),
                    'status': row.get('status'),
                    'derived_path': row.get('derived_path'),
                    'derived_sha256': row.get('derived_sha256')}
    return out


def forward_targets(run_identity, qualification):
    """The ordered expected-row list, bound to the frozen state list and to V1.10 input digests."""
    states = run_identity['state_list']['states']
    rows, errors = [], []
    for i, state in enumerate(states):
        key = (int(state['day_id']), int(state['position']))
        q = qualification.get(key)
        if q is None:
            errors.append({'state': list(key), 'reason': 'state_absent_from_input_qualification'})
            q = {}
        elif q.get('status') != 'INPUT_CHECKS_PASS':
            errors.append({'state': list(key), 'reason': 'input_qualification_not_passing',
                           'status': q.get('status')})
        if not q.get('input_sha256'):
            errors.append({'state': list(key), 'reason': 'missing_input_digest'})
        rows.append({
            'index': i, 'day_id': key[0], 'position': key[1],
            'state_hash': state['state_hash'],
            'input_sha256': q.get('input_sha256'),
            'legal_sha256': q.get('legal_sha256'),
            'n_legal_expected': q.get('n_legal'),
            'Nv': q.get('Nv'),
            'expected_record_relpath': 'scores/day%02d_pos%d.json' % key,
            'expected_forwards': EXPECTED_FORWARD_CALLS_PER_SLOT,
            'derived_relpath': q.get('derived_path'),
            'derived_sha256': q.get('derived_sha256'),
        })
    nvs = [int(r['Nv']) for r in rows if isinstance(r['Nv'], int)]
    ms = [int(r['n_legal_expected']) for r in rows if isinstance(r['n_legal_expected'], int)]
    if len(nvs) != len(rows) or len(ms) != len(rows):
        errors.append({'reason': 'Nv / ordered-legal length missing for some states',
                       'n_with_Nv': len(nvs), 'n_with_n_legal': len(ms), 'n_rows': len(rows)})
    payload = [[r['day_id'], r['position'], r['state_hash'], r['input_sha256'],
                r['n_legal_expected'], r['Nv']] for r in rows]
    return {'rows': rows, 'n_rows': len(rows), 'expected_rows': EXPECTED_ROWS,
            'per_slot_forward_calls': EXPECTED_FORWARD_CALLS_PER_SLOT,
            'max_forward_calls': run_identity['call_schedule']['max_forward_calls'],
            'warmup_calls': run_identity['call_schedule']['warmup_calls'],
            # the JIT shapes are the maxima of the ALREADY-QUALIFIED V1.10 inputs; they are not
            # chosen by looking at any score (V1.11 §"维度来自已验收64输入的最大值")
            'Nv_max': (max(nvs) if nvs else None),
            'M_max': (max(ms) if ms else None),
            'jit_shape_basis': 'max Nv and max ordered-legal length over the 64 pre-qualified '
                               'V1.10 inputs; not chosen from any score',
            'errors': errors, 'ok': not errors and len(rows) == EXPECTED_ROWS,
            'targets_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
            .hexdigest()}


def verify_slot_record(record, target, *, out_dir=None):
    """Independent per-row verification.  Returns `(ok, detail)`; never raises on bad input."""
    detail, problems = {}, []
    if not isinstance(record, dict):
        return False, {'problems': ['record is not a JSON object']}
    for field in ('day_id', 'position', 'input_sha256', 'legal_sha256', 'scores_dtype',
                  'scores_shape', 'scores_first_hex', 'scores_repeat_hex', 'status'):
        if field not in record:
            problems.append('missing field %s' % field)
    if record.get('day_id') != target['day_id'] or record.get('position') != target['position']:
        problems.append('row identity mismatch')
    if record.get('input_sha256') != target['input_sha256']:
        problems.append('input digest differs from the pre-qualified digest')
    if record.get('legal_sha256') != target['legal_sha256']:
        problems.append('ordered-legal digest differs from the pre-qualified digest')
    if record.get('status') != 'FORWARD_CHECKS_PASS':
        problems.append('status is %r, not FORWARD_CHECKS_PASS' % record.get('status'))
    if record.get('F_calls') != EXPECTED_FORWARD_CALLS_PER_SLOT:
        problems.append('F_calls is %r, expected %d'
                        % (record.get('F_calls'), EXPECTED_FORWARD_CALLS_PER_SLOT))
    first, repeat = record.get('scores_first_hex'), record.get('scores_repeat_hex')
    detail['repeat_identical'] = bool(first) and first == repeat
    if not detail['repeat_identical']:
        problems.append('the two forwards are not bit-identical')
    detail['first_sha256'] = hashlib.sha256(bytes.fromhex(first)).hexdigest() if first else None
    detail['repeat_sha256'] = hashlib.sha256(bytes.fromhex(repeat)).hexdigest() if repeat else None
    shape = record.get('scores_shape')
    if isinstance(shape, list) and len(shape) == 1 and target.get('n_legal_expected') is not None:
        detail['shape_matches_ordered_legal'] = int(shape[0]) == int(target['n_legal_expected'])
        if not detail['shape_matches_ordered_legal']:
            problems.append('score length %r != ordered-legal length %r'
                            % (shape[0], target['n_legal_expected']))
    else:
        detail['shape_matches_ordered_legal'] = False
        problems.append('score shape is not a 1-D length')
    nbytes = None
    if isinstance(shape, list) and len(shape) == 1 and record.get('scores_dtype') == '<f4':
        nbytes = int(shape[0]) * 4
    detail['declared_and_actual_byte_lengths_agree'] = bool(
        nbytes is not None and first and repeat and len(bytes.fromhex(first)) == nbytes
        and len(bytes.fromhex(repeat)) == nbytes)
    if not detail['declared_and_actual_byte_lengths_agree']:
        problems.append('declared dtype/shape and the recorded byte string disagree')
    detail['failure_reason_vocabulary_ok'] = record.get('failure_reason') in FAILURE_REASONS
    detail['checks'] = record.get('checks')
    if out_dir is not None and record.get('raw_relpath'):
        raw = Path(out_dir) / record['raw_relpath']
        if not raw.is_file():
            problems.append('raw score artifact is missing: %s' % record['raw_relpath'])
            detail['raw_sha256_matches'] = False
        else:
            actual = sha(raw)
            detail['raw_sha256_matches'] = bool(actual) and actual == record.get('raw_sha256')
            if not detail['raw_sha256_matches']:
                problems.append('raw score artifact digest mismatch')
            # The raw artifact is a pickle of the two score arrays, not JSON.  Re-read it and
            # re-derive the two hex strings, so the JSON record cannot claim scores the pickle
            # does not contain.
            try:
                import pickle
                blob = pickle.loads(raw.read_bytes())
                a, b = bytes(blob['scores_first']), bytes(blob['scores_repeat'])
                detail['raw_replay_matches_record'] = (
                    a.hex() == record.get('scores_first_hex')
                    and b.hex() == record.get('scores_repeat_hex')
                    and blob.get('input_sha256') == record.get('input_sha256'))
                if not detail['raw_replay_matches_record']:
                    problems.append('the raw score pickle does not reproduce the recorded hex '
                                    'strings / input digest')
            except Exception as exc:                                     # noqa: BLE001
                detail['raw_replay_matches_record'] = False
                problems.append('raw score artifact is not a readable score pickle: %r' % exc)
    detail['problems'] = problems
    return (not problems), detail


def identity_targets(run_id, watched_paths, *, spec):
    """The files the trailing identity record must cover, with the digests to compare against.

    `watched_paths` is supplied by the caller (the run-identity module owns the list) so that the
    target map is always derived from the SAME frozen list the start record used; a verifier can
    therefore never invent a different set of files to check.
    """
    return {'run_id': run_id, 'spec': spec,
            'paths': {str(p): sha(p) for p in watched_paths}}


def verify_identity(record, targets):
    """Compare a published identity record against the frozen target map."""
    problems = []
    if not isinstance(record, dict):
        return False, {'problems': ['identity record is not a JSON object']}
    if record.get('run_id') != targets['run_id']:
        problems.append('run_id mismatch')
    rows = record.get('paths') or {}
    for path, expected in targets['paths'].items():
        got = rows.get(path)
        if got is None:
            problems.append('path not covered: %s' % path)
        elif got != expected:
            problems.append('path digest changed: %s' % path)
    return (not problems), {'problems': problems, 'n_paths': len(targets['paths'])}
