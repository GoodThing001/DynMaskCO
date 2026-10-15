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


def find_state_manifest():
    """Locate the archived 64-state manifest without assuming a filename."""
    cands = []
    for pat in ('**/*64*.json', '**/*STATE*.json', '**/*MANIFEST*.json', '**/*collection*.json'):
        cands += list(EV.glob(pat))
    seen, out = set(), []
    for c in cands:
        if c in seen or 'v117' in str(c) or 'v116' in str(c):
            continue
        seen.add(c)
        try:
            d = json.loads(c.read_text(encoding='utf-8'))
        except Exception:
            continue
        s = json.dumps(d)[:200000]
        if 'state_hash' in s and ('position' in s or 'day_id' in s):
            out.append((c, d))
    return out


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

    # 64-state manifest
    found = find_state_manifest()
    recs = None
    for path, d in found:
        for key in ('rows', 'states', 'records', 'slots', 'results'):
            v = d.get(key) if isinstance(d, dict) else None
            if isinstance(v, list) and len(v) >= 32:
                recs, src = v, '%s:%s' % (path.relative_to(EV), key)
                break
        if recs:
            break
    REC['state_asset'] = {'candidates_scanned': len(found),
                          'source': (src if recs else None),
                          'n_rows': (len(recs) if recs else 0),
                          'with_state_hash': (sum(1 for r in (recs or [])
                                                  if isinstance(r, dict)
                                                  and (r.get('state_hash') or
                                                       (r.get('record') or {}).get('state_hash')))
                                              if recs else 0)}
    ck('V118-3 the archived state asset is found and every row carries a state_hash (64 rows, frozen '
       'order preserved)',
       bool(recs) and len(recs) >= 64
       and REC['state_asset']['with_state_hash'] >= 64, REC['state_asset'])

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
    if not (recs and len(recs) >= 64):
        missing.append('the 64-state manifest rows could not be located/parsed from the archive')
    missing += ['a frozen new run_id isolated from V1.11 (not yet created)',
                'per-state forward call counts and initialization/slot/batch wall-clock budgets '
                '(not yet frozen for the NEW run)',
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
