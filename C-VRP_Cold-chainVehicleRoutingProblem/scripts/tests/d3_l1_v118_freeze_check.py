#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 V1.18 freeze check: turn the archived 64-state asset + the F identity into a signable
execution pre-registration, WITHOUT running any new F forward.

Read-only.  It answers, mechanically, the questions the user requires frozen before the single
authorised real-F batch can be signed:
  * are the 64 archived states complete, in frozen order, each with a state_hash?
  * do the data / profile / contract identities still match their registrations?
  * do the F checkpoint / backbone files match their registered SHAs, and is the historical-backbone
    label preserved (HISTORICAL_BACKBONE_UNVERIFIED)?
  * are the V1.14 phase journal and the V1.16 round-revocation supervisor importable and wired?
  * what exactly is still missing before READY_FOR_USER_SIGNATURE?
It never scores a state and never writes outside --outdir.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))
EV = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
F_CKPT = EXT / 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'
BACKBONE = EXT.parent / 'MASKCO_code/ckpts/cvrp100.ckpt'
PROFILE = EXT / 'results/o0cc/scale_v2/objective_profile.json'
DEV_CHECK = EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'
REG = {'f_ckpt': 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5',
       'backbone': '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db',
       'dev_check': '66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89',
       'profile_hash': 'df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6',
       'effective_contract': 'edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833'}
CASES = []
REC = {}


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:250]))


def load_states():
    """Locate the archived 64-state asset in the REAL V1.9 collection layout and re-verify it.

    The ledger is an append-only event log; a state record is its ``event == 'PUBLISHED'`` line.
    Authoritative order = publication order, which must already be the canonical (day_id, position)
    order: nothing is re-sorted here, because the pre-registration freezes "ledger order" and a
    re-sorted read would silently freeze a different object than the collector published.
    The same read also harvests the OBSERVED per-slot / per-day wall-clock budgets, so the new run's
    budgets can be derived from real measurements instead of invented numbers.
    """
    root = EV / 'v19_evidence/formal_collection'
    info = {'root': root.relative_to(EV).as_posix(), 'root_exists': root.is_dir(),
            'ledger': None, 'n_ledger_rows': 0, 'n_published': 0, 'n_rows': 0,
            'with_state_hash': 0, 'verified_against_originals': 0,
            'expected_original_hashes': 0, 'order_keys_unique': False,
            'frozen_order_matches_ledger': False, 'first_rows': [], 'errors': [],
            'integrity_failures': [], 'events': {}, 'run_ids': [], 'budget_observed': {}}
    ledger = root / 'STATE_LEDGER.jsonl'
    if ledger.is_file():
        info['ledger'] = ledger.relative_to(EV).as_posix()
    else:
        info['errors'].append('STATE_LEDGER.jsonl missing under %s' % info['root'])
    rows = []
    if ledger.is_file():
        for i, line in enumerate(ledger.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                info['errors'].append('ledger line %d unparsable: %s: %s'
                                      % (i, type(exc).__name__, exc))
    for r in rows:
        if isinstance(r, dict):
            info['events'][str(r.get('event'))] = info['events'].get(str(r.get('event')), 0) + 1
    info['run_ids'] = sorted({str(r.get('run_id')) for r in rows if isinstance(r, dict)})
    published = [r for r in rows if isinstance(r, dict) and r.get('event') == 'PUBLISHED']
    attempts = [r for r in rows if isinstance(r, dict) and r.get('event') == 'ATTEMPT_STARTED']
    info['n_ledger_rows'] = len(rows)
    info['n_published'] = len(published)
    info['n_rows'] = len(published)
    info['with_state_hash'] = sum(1 for r in published if r.get('state_hash'))
    keys = [(r.get('day_id'), r.get('position')) for r in published]
    info['order_keys_unique'] = len(set(keys)) == len(keys)
    info['frozen_order_matches_ledger'] = bool(published) and info['order_keys_unique'] \
        and keys == sorted(keys)
    # per-row integrity bits the collector itself asserted at publication time
    for r in published:
        for flag in ('bitwise_parity_ok', 'projection_causal', 'suffix_boundary_sha_equal'):
            if r.get(flag) is not True:
                info['integrity_failures'].append('PUBLISHED %s/%s %s=%r'
                                                  % (r.get('day_id'), r.get('position'), flag,
                                                     r.get(flag)))
    for r in attempts:
        for flag in ('ledger_integrity_ok', 'registered_before_capture'):
            if r.get(flag) is not True:
                info['integrity_failures'].append('ATTEMPT_STARTED %s/%s %s=%r'
                                                  % (r.get('day_id'), r.get('position'), flag,
                                                     r.get(flag)))
    verified, missing_pairs = 0, []
    for r in published:
        for sha_field, path_field in (('snapshot_sha256', 'snapshot_path'),
                                      ('validation_sha256', 'validation_path')):
            rel = r.get(path_field)
            if not rel:
                missing_pairs.append('%s@%s/%s absent' % (path_field, r.get('day_id'),
                                                          r.get('position')))
                continue
            q = Path(rel)
            if not q.is_absolute():
                q = root / rel
            if not q.is_file():
                q = EV / rel
            if q.is_file() and sha(q) == r.get(sha_field):
                verified += 1
            else:
                missing_pairs.append('%s@%s/%s mismatch' % (path_field, r.get('day_id'),
                                                            r.get('position')))
    info['verified_against_originals'] = verified
    info['expected_original_hashes'] = 2 * len(published)
    info['first_rows'] = [{'day_id': r.get('day_id'), 'position': r.get('position'),
                           'state_hash': (r.get('state_hash') or '')[:16]}
                          for r in published[:3]]
    def _nums(rs, field):
        vals = sorted(float(r[field]) for r in rs
                      if isinstance(r, dict) and isinstance(r.get(field), (int, float)))
        if not vals:
            return None
        return {'n': len(vals), 'min': vals[0], 'median': vals[len(vals) // 2], 'max': vals[-1]}
    info['budget_observed'] = {
        'slot_span_wall_s': _nums(published, 'slot_span_wall_s'),
        'day_rollout_wall_s': _nums([r for r in rows if isinstance(r, dict)
                                     and r.get('event') == 'DAY_ROLLOUT_FINISHED'],
                                    'day_rollout_wall_s'),
        'source': 'measured by the V1.9 collector; provenance only -- the new run needs its own freeze'}
    info['errors'] += missing_pairs[:8]
    info['n_mismatches'] = len(missing_pairs)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    REC['identities'] = {
        'f_ckpt': {'path': str(F_CKPT.relative_to(EXT)), 'sha256': sha(F_CKPT),
                   'registered': REG['f_ckpt'], 'matches': sha(F_CKPT) == REG['f_ckpt']},
        'backbone': {'path': 'MASKCO_code/ckpts/cvrp100.ckpt', 'sha256': sha(BACKBONE),
                     'registered': REG['backbone'], 'matches': sha(BACKBONE) == REG['backbone'],
                     'label': 'HISTORICAL_BACKBONE_UNVERIFIED'},
        'dev_check_npz': {'sha256': sha(DEV_CHECK), 'registered': REG['dev_check'],
                          'matches': sha(DEV_CHECK) == REG['dev_check']}}
    prof = json.loads(PROFILE.read_text(encoding='utf-8'))
    REC['identities']['profile'] = {'sha256': sha(PROFILE),
                                    'profile_hash': prof.get('profile_hash'),
                                    'matches': prof.get('profile_hash') == REG['profile_hash']}
    ck('V118-1 the F checkpoint, the backbone, the dev_check data and the registered profile still '
       'match their recorded identities, and the historical-backbone label is preserved',
       all([REC['identities']['f_ckpt']['matches'], REC['identities']['backbone']['matches'],
            REC['identities']['dev_check_npz']['matches'], REC['identities']['profile']['matches']]),
       {k: v.get('matches') for k, v in REC['identities'].items()})

    # effective contract recomputation
    try:
        import dataclasses
        from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                        default_pilot_contract)
        flds = {f.name for f in dataclasses.fields(ObjectiveProfile)}
        eff = apply_objective_profile(default_pilot_contract(),
                                      ObjectiveProfile(**{k: v for k, v in prof.items()
                                                          if k in flds}))
        eh = getattr(eff, 'contract_hash', None)
        REC['identities']['effective_contract'] = {'recomputed': eh, 'registered':
                                                   REG['effective_contract'],
                                                   'matches': eh == REG['effective_contract']}
        ck('V118-2 the effective contract hash recomputes to the frozen value',
           eh == REG['effective_contract'], REC['identities']['effective_contract'])
    except BaseException as exc:
        REC['identities']['effective_contract'] = {'error': '%s: %s' % (type(exc).__name__, exc)}
        ck('V118-2 the effective contract hash recomputes to the frozen value', False,
           REC['identities']['effective_contract'])

    # 64-state manifest (real V1.9 collection archive, ledger order preserved)
    REC['state_asset'] = load_states()
    ck('V118-3 the archived 64 states are located in FROZEN ORDER (STATE_LEDGER.jsonl append order, '
       'which is already the canonical (day_id, position) order -- never re-sorted), every row '
       'carries a state_hash, and all 128 snapshot+validation originals still match their recorded SHA',
       REC['state_asset']['n_rows'] == 64
       and REC['state_asset']['with_state_hash'] == 64
       and REC['state_asset']['expected_original_hashes'] == 128
       and REC['state_asset']['verified_against_originals'] == 128
       and REC['state_asset']['frozen_order_matches_ledger']
       and not REC['state_asset']['integrity_failures']
       and len(REC['state_asset']['run_ids']) == 1, REC['state_asset'])

    # supervisor + phase journal wiring
    wire = {}
    for name in ('d3_l1_v116_forward_control', 'd3_l1_v112_phase_journal'):
        hits = [p for p in (EXT / 'scripts').rglob(name + '.py') if '__pycache__' not in str(p)]
        wire[name] = {'found': bool(hits),
                      'sha256': sha(hits[0]) if hits else None}
    try:
        import d3_l1_v116_forward_control as FC
        wire['round_verdict_present'] = hasattr(FC, 'round_verdict')
        wire['run_contained_present'] = hasattr(FC, 'run_contained')
    except BaseException as exc:
        wire['import_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
    REC['supervisor_wiring'] = wire
    ck('V118-4 the round-revocation supervisor and the phase journal are present and importable',
       bool(wire.get('round_verdict_present')) and bool(wire.get('run_contained_present'))
       and all(v.get('found') for k, v in wire.items() if isinstance(v, dict)), {
           k: (v if not isinstance(v, dict) else v.get('found')) for k, v in wire.items()})

    missing = []
    if REC['state_asset']['n_rows'] != 64:
        missing.append('the 64 published state records could not be located/parsed from the archive')
    if not REC['state_asset']['frozen_order_matches_ledger']:
        missing.append('the archived 64 states are not readable in frozen ledger order')
    missing += ['a frozen new run_id isolated from V1.11 (not yet created); the V1.9 observed '
                'slot/day budgets above are provenance, not a freeze for the new run',
                'per-state forward call counts and initialization/slot/batch wall-clock budgets '
                'frozen for the NEW run',
                'the Windows new-destination persistence acceptance executed on the REAL F worker',
                'the independent read-only verifier frozen against the new run identity']
    REC['readiness'] = {'verdict': 'READY_FOR_USER_SIGNATURE' if not missing else 'NOT_READY',
                        'missing_before_signature': missing,
                        'not_run_here': ['no F forward', 'no F-vs-D-Geo comparison',
                                         'no training', 'no L2/L3', 'no final test']}
    fails = [c for c in CASES if not c[1]]
    REC['summary'] = {'rows': len(CASES), 'pass': len(CASES) - len(fails), 'fail': len(fails)}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, **REC['summary'],
                               'tool_sha256': sha(Path(__file__))},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== V1.18 freeze check: %d rows, %d PASS, %d FAIL | %s ==='
          % (len(CASES), len(CASES) - len(fails), len(fails), REC['readiness']['verdict']))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
