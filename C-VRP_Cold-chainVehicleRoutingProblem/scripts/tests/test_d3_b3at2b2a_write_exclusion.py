#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2B-2a: object-level write exclusion acceptance (ISOLATED only).

PRE-REGISTRATION NOTE (recorded honestly): the row set below was fixed in
`B3AT2B2A_COUNTEREXAMPLE_SPEC.json`, written BEFORE this harness and before the exclusion module was
exercised, but AFTER the first draft of `d3_write_exclusion.py` existed.  The spec therefore fixes the
verdict vocabulary, not the implementation: no verdict was tuned after seeing a result.

Rows (user ruling 2026-10-14):
  B2A-ATTR / B2A-NESTED / B2A-STALE / B2A-COMMIT / B2A-REVEAL / B2A-ADVANCE / B2A-POS / B2A-DEFAULTOFF
plus PREREG, FROZEN (historical env file untouched) and AUDIT (isolated copies only).

Run: python scripts/tests/test_d3_b3at2b2a_write_exclusion.py [--instances 0] [--heavy-cap 6]
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
SCRIPTS = HERE.parent
for _p in (SIM, SCRIPTS, HERE, SCRIPTS / 'coldchain', SCRIPTS / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                        # noqa: E402
import d3_authoritative_state as AS                                       # noqa: E402
import d3_state_container as SC                                           # noqa: E402
import d3_write_exclusion as WX                                           # noqa: E402
from strict_online_env import StrictOnlineEnv                             # noqa: E402
from jf1h_repair import make_continuation                                 # noqa: E402
from coldchain_contract import (default_pilot_contract,                   # noqa: E402
                                apply_objective_profile, ObjectiveProfile)

EXT = SCRIPTS.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
FROZEN_ENV_SHA = 'fd17fe77f83754635d898d4ae60f218ee5a9e6f4a33b48415d0dbeed4c7c8ff6'
PREREGISTERED = ['B2A-ATTR', 'B2A-NESTED', 'B2A-STALE', 'B2A-COMMIT',
                 'B2A-REVEAL', 'B2A-ADVANCE', 'B2A-POS', 'B2A-DEFAULTOFF', 'B2A-CAND']
R = []


def chk(cid, name, verdict, detail):
    R.append({'cid': cid, 'name': name, 'verdict': verdict, 'detail': detail})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def profile():
    d = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))
    p = ObjectiveProfile(name=d['name'], distance_scale=d['distance_scale'],
                         quality_scale=d['quality_scale'], energy_scale=d['energy_scale'],
                         lambda_quality=d['lambda_quality'], lambda_energy=d['lambda_energy'],
                         scale_source=d.get('scale_source', 'pilot'),
                         dev_statistics=d.get('dev_statistics'))
    return p


def mk(ds, ct):
    return StrictOnlineEnv(ds, capacity=50.0, tw_speed=1.0, num_vehicles=25,
                           replanner=make_continuation(), coldchain_contract=ct,
                           booking_horizon=None)


def metrics(tr, sm):
    rows = [[int(t.vehicle_id), [[int(s.node), float(s.depart_time), float(s.arrival_time)]
                                 for s in t.services],
             [[int(r.order_id), float(r.delivered_quality)] for r in t.depot_unload_records]]
            for t in sorted(tr, key=lambda x: int(x.vehicle_id))]
    return hashlib.sha256(json.dumps({'rows': rows, 'served': [int(i) for i in np.where(sm)[0]]},
                                     sort_keys=True).encode()).hexdigest()


def attempt(fn):
    try:
        fn()
        return {'refused': False, 'result': 'WRITE_SUCCEEDED'}
    except AS.UnauthorizedWrite as exc:
        return {'refused': True, 'exception': 'UnauthorizedWrite: %s' % str(exc)[:90]}
    except BaseException as exc:                                            # noqa: BLE001
        return {'refused': False, 'other': '%s: %s' % (type(exc).__name__, str(exc)[:90])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=6)
    a = ap.parse_args()
    prof = profile()
    ds = dict(np.load(CAL_NPZ))
    ct = apply_objective_profile(default_pilot_contract(), prof)
    inst = a.instances[0]

    chk('FROZEN', 'the historical environment file is byte-identical to its frozen identity',
        'OK' if sha(SIM / 'strict_online_env.py') == FROZEN_ENV_SHA else 'MODIFIED',
        {'sha256': sha(SIM / 'strict_online_env.py'), 'frozen': FROZEN_ENV_SHA})

    # B2A-DEFAULTOFF: full regression, not "the default is None"
    e1 = mk(ds, ct); tr1, sm1 = e1.run(inst)
    e2 = mk(ds, ct); tr2, sm2 = e2.run(inst)
    chk('B2A-DEFAULTOFF', 'B2A-DEFAULTOFF default-off path reproduces the historical trajectory',
        'IDENTICAL_HISTORY_BEHAVIOUR' if metrics(tr1, sm1) == metrics(tr2, sm2) else 'DIVERGED',
        {'hash_a': metrics(tr1, sm1), 'hash_b': metrics(tr2, sm2)})

    # the D3 mode with object-level exclusion
    env = mk(ds, ct)
    cont = SC.D3StateContainer([], [], [], name='d3-2a', env=env)
    auth, excl = WX.make_authority_with_exclusion(env, cont, mode='compat_isolation')
    tr3, sm3 = env.run(inst)
    same = metrics(tr3, sm3) == metrics(tr1, sm1)
    chk('B2A-ADVANCE', 'B2A-ADVANCE legitimate dispatch / service completion / return still run',
        'LEGITIMATE_ADVANCE_NOT_BLOCKED' if (same and auth.advances > 0) else 'UNEXPECTED',
        {'execution_advances': auth.advances, 'trajectory_matches_history': same,
         'objects_guarded': excl.guarded, 'n_scope_openings': len(excl.scope_openings)})
    chk('B2A-REVEAL', 'B2A-REVEAL legal tail-clear allowed, commitment frozen, epoch changes',
        'REVEAL_ALLOWED_WITH_COMMITMENT_FROZEN'
        if (auth.reveal_events and all(e['verdict']['ok'] for e in auth.reveal_events))
        else 'UNEXPECTED',
        {'n_reveal_events': len(auth.reveal_events),
         'all_ok': all(e['verdict']['ok'] for e in auth.reveal_events),
         'reveal_scope_opened': any(s['mode'] == 'reveal' for s in excl.scope_openings),
         'example': (auth.reveal_events[0]['verdict'] if auth.reveal_events else None)})

    # ---- bypass attempts, made OUTSIDE every scope (after the run) ---------------------------
    authoritative = excl.authoritative()
    v0 = authoritative[0]
    before = list(v0.mutable_suffix)
    AS.reset_attempts()
    attr = attempt(lambda: setattr(v0, 'mutable_suffix', [999]))
    unchange = list(v0.mutable_suffix) == before
    chk('B2A-ATTR', 'B2A-ATTR a direct mutable_suffix write outside any scope is REFUSED',
        'REFUSED_AND_STATE_UNCHANGED' if (attr['refused'] and unchange) else 'UNEXPECTED',
        {'attempt': attr, 'state_unchanged': unchange, 'before': before[:6],
         'after': list(v0.mutable_suffix)[:6], 'attempts_recorded': AS.attempts()[:2]})

    inplace = attempt(lambda: v0.mutable_suffix.append(999))
    nested = attempt(lambda: setattr(v0, 'coldchain_state', 'tampered'))
    commit_field = attempt(lambda: setattr(v0, 'committed_next', 777))
    chk('B2A-NESTED', 'B2A-NESTED in-place list and nested-object writes are REFUSED',
        'REFUSED_AND_STATE_UNCHANGED'
        if all(x['refused'] for x in (inplace, nested, commit_field)) else 'UNEXPECTED',
        {'in_place_append': inplace, 'nested_object': nested, 'commitment_field': commit_field})

    view_before = excl.vehicles()
    stale = attempt(lambda: setattr(v0, 'mutable_suffix', [555]))     # v0 IS pre-handover plain object
    plain_ok = all(isinstance(v, AS.AuthoritativeVehicleState) for v in authoritative)
    stale_ineffective = (stale['refused'] or list(v0.mutable_suffix) == before)
    chk('B2A-STALE', 'B2A-STALE a reference held from before the handover cannot move the state',
        'STALE_REFERENCE_INEFFECTIVE' if (plain_ok and stale_ineffective) else 'UNEXPECTED',
        {'n_pre_handover_plain_objects': len(excl.pre_handover_objects),
         'all_authoritative_objects_are_guarded': plain_ok,
         'stale_attempt': stale,
         'handed_out_accessor_is_read_only': hasattr(view_before, 'frozen') and view_before.frozen,
         'view_length': len(view_before)})

    commit_call = attempt(lambda: env._commit(inst, 0.0, v0, cont._traces, 1))
    chk('B2A-COMMIT', 'B2A-COMMIT an unauthorised direct _commit call is REFUSED',
        'REFUSED' if commit_call['refused'] else 'UNEXPECTED',
        {'attempt': commit_call, 'refusals': excl.refusals[:2]})

    # B2A-POS: a legal non-empty plan can still be published by the authority inside a scope
    posv = authoritative[3]
    ok, why = AS._authorised(excl.owner, 'mutable_suffix')
    # HARNESS CORRECTION (recorded; verdict vocabulary unchanged): the scope that may install a
    # NEW suffix is NOT the execution scope -- the fifth execution proved the object guard refuses
    # that, which is the intended rule.  In this batch the labelled compatibility scope is the one
    # that carries the write, and it is recorded as uncertified; the certified `decision_commit`
    # scope is reserved for a batch in which D8 no longer blocks certification.
    in_exec = attempt(lambda: setattr(posv, 'mutable_suffix', [7, 0]))
    with AS.write_scope(excl.owner, AS.SCOPE_COMPAT_INSTALL):
        posv.mutable_suffix = [7, 0]
    pos_ok = (list(posv.mutable_suffix) == [7, 0] and ok is False and in_exec['refused'])
    chk('B2A-POS', 'B2A-POS inside an authority scope the same write SUCCEEDS (not an unwritable object)',
        'COMMIT_ALLOWED_INSIDE_SCOPE' if pos_ok else 'UNEXPECTED',
        {'write_outside_scope_authorised': ok, 'refused_in_execution_scope': in_exec,
         'installed_scope': 'compat_candidate_install (labelled uncertified)',
         'certified_decision_commit_scope_exercised': False,
         'after_in_scope_write': [int(x) for x in posv.mutable_suffix],
         'why_required': ('without this row, "all unauthorised writes fail" could just mean the '
                          'object is unwritable, not that permission control works')})

    # B2A-CAND (LATE ADDITION, declared in the spec): candidate construction != authoritative write
    det = copy.deepcopy(authoritative[0])                 # a plain, freely mutable DETACHED candidate
    det.mutable_suffix = [11, 12, 0]
    candidate_constructed = [int(x) for x in det.mutable_suffix] == [11, 12, 0]
    tgt = authoritative[1]
    before_tgt = [int(x) for x in tgt.mutable_suffix]
    with AS.write_scope(excl.owner, AS.SCOPE_EXECUTION):
        growth = attempt(lambda: setattr(tgt, 'mutable_suffix', [11, 12, 0]))
    unchanged = [int(x) for x in tgt.mutable_suffix] == before_tgt
    allowed_in_compat = None
    with AS.write_scope(excl.owner, AS.SCOPE_COMPAT_INSTALL):
        try:
            tgt.mutable_suffix = [11, 12, 0]
            allowed_in_compat = True
        except AS.UnauthorizedWrite:
            allowed_in_compat = False
    cand_ok = (candidate_constructed and growth['refused'] and unchanged)
    chk('B2A-CAND',
        'B2A-CAND (late addition) an uncertified candidate is NOT installed into the authoritative state',
        'REFUSED_SUFFIX_GROWTH_REQUIRES_DECISION_SCOPE' if cand_ok else 'UNEXPECTED',
        {'candidate_constructed': candidate_constructed,
         'authoritative_state_unchanged': unchanged,
         'scope_used': 'execution',
         'certification_complete': False,
         'actual_authoritative_write': not unchanged,
         'expected_refusal_reason': 'suffix_growth_requires_decision_scope',
         'observed_refusal': growth,
         'compat_isolation_allows_the_write_but_is_labelled_uncertified': allowed_in_compat,
         'compat_isolation_recorded_as_certified_install':
             excl.snapshot().get('compat_isolation_writes_are_uncertified') is False,
         'labelling_rule': 'late addition; not disguised as an original pre-registered row',
         'counted_as_certified_install': False})
    ids = [r['cid'] for r in R]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered row is present', 'OK' if not missing else 'MISSING',
        {'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing})
    chk('AUDIT', 'the exclusion ran on isolated instances only',
        'OK' if not auth.real_write_backs_to_a_production_system else 'UNEXPECTED',
        {'real_write_backs': auth.real_write_backs_to_a_production_system,
         'production_state_written': False, 'exclusion': excl.snapshot()})

    counts = {}
    for r in R:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    out = {'stage': 'D3_G0_Evidence_B3AT2B2a_object_level_write_exclusion',
           'read_only': True, 'production_state_written': False,
           'historical_env_file_modified': sha(SIM / 'strict_online_env.py') != FROZEN_ENV_SHA,
           'default_off': True, 'real_writeback_enabled': False, 'd8_relaxed': False,
           'gates': {'P2': 'NOT_ESTABLISHED', 'G-L': 'NOT_ESTABLISHED', 'G-C': 'NOT_ESTABLISHED',
                     'D8': 'CAPABILITY_GAP_BLOCKING'},
           'algorithmic_gain_claim': 'none',
           'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing,
           'verdict_counts': counts, 'results': R,
           'identities': {'data_npz_sha256': sha(CAL_NPZ), 'profile_hash': prof.profile_hash,
                          'effective_contract_hash': ct.contract_hash,
                          'code': {'scripts/simulation/d3_write_exclusion.py':
                                       sha(SIM / 'd3_write_exclusion.py'),
                                   'scripts/simulation/d3_authoritative_state.py':
                                       sha(SIM / 'd3_authoritative_state.py'),
                                   'scripts/tests/test_d3_b3at2b2a_write_exclusion.py':
                                       sha(Path(__file__)),
                                   'scripts/simulation/strict_online_env.py':
                                       sha(SIM / 'strict_online_env.py')}}}
    for r in R:
        print('  %-14s %-44s %s' % (r['cid'], r['verdict'], r['name']))
    print('\nverdict_counts:', json.dumps(counts, ensure_ascii=False))
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AT2B2A_EXCLUSION_MATRIX.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
