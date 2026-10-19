"""P2-C-CAMR-STATIC-01  independent verification.

Recomputes the aggregates from the raw per-unit rows instead of trusting the summaries, verifies
the whole identity chain (registration -> run_identity -> preflight -> pass_a/pass_b), re-derives
the 19 unit identities from the saved data, enforces the frozen denominator and failure
accounting, and checks the two complete runs agree bit for bit.

Usage: python verify_camr_static.py [--out verification.json]     (refuses to overwrite)
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import camr_shared as cs                                            # noqa: E402

ENGINE = cs.ROOT / 'scripts' / 'simulation' / 'strict_online_env.py'
# action_contract.plan_hash returns sha256(...)[:16], not a full 64-char digest
PLAN_HASH = re.compile(r'^[0-9a-f]{16}$')


def git_blob(path):
    rel = str(path.relative_to(cs.WORKSPACE)).replace('\\', '/')
    r = subprocess.run(['git', 'cat-file', 'blob', ':' + rel], cwd=str(cs.WORKSPACE),
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit('git cat-file failed for %s' % rel)
    return hashlib.sha256(r.stdout).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'verification.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    checks = []

    def rec(name, ok, detail=None):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'ok': bool(ok),
                       'detail': detail})

    reg = cs.load_json(OUT / 'registration.json')
    ident = cs.load_json(OUT / 'run_identity.json')
    pre = cs.load_json(OUT / 'preflight.json')
    syn = cs.load_json(OUT / 'synthetic_regression.json')
    pa1 = cs.load_json(OUT / 'pass_a_run1.json')
    pb1 = cs.load_json(OUT / 'pass_b_run1.json')
    pa2 = cs.load_json(OUT / 'pass_a_run2.json')
    pb2 = cs.load_json(OUT / 'pass_b_run2.json')

    # ---- A. identity chain ------------------------------------------------
    reg_sha = cs.sha_file(OUT / 'registration.json')
    ident_sha = cs.sha_file(OUT / 'run_identity.json')
    pre_sha = cs.sha_file(OUT / 'preflight.json')
    chain = {
        'registration_is_registered_schema': reg.get('schema') == 'p2_c_camr_static_01_registration_v1',
        'run_identity_binds_registration': ident['T1']['registration_sha256'] == reg_sha,
        'run_identity_has_no_gate_result': ident.get('gate_result_included') is False,
        'preflight_binds_run_identity': pre['run_identity_sha256'] == ident_sha,
        'preflight_binds_registration': pre['registration_sha256'] == reg_sha,
        'preflight_passed': pre['all_ok'] is True,
        'synthetic_regression_passed': syn['all_ok'] is True,
        'synthetic_used_no_model': syn['model_loaded'] is False,
        'pass_a_run1_binds_identity': pa1['run_identity_sha256'] == ident_sha,
        'pass_a_run2_binds_identity': pa2['run_identity_sha256'] == ident_sha,
        'pass_b_run1_binds_identity': pb1['run_identity_sha256'] == ident_sha,
        'pass_b_run2_binds_identity': pb2['run_identity_sha256'] == ident_sha,
        'pass_a_run1_binds_preflight': pa1['preflight_sha256'] == pre_sha,
        'pass_a_run2_binds_preflight': pa2['preflight_sha256'] == pre_sha,
    }
    rec('A_identity_chain', all(chain.values()), chain)

    # ---- B. source identity ----------------------------------------------
    drift = {rel: cs.sha_file(cs.WORKSPACE / rel)
             for rel, want in ident['source_files'].items()
             if cs.sha_file(cs.WORKSPACE / rel) != want}
    rec('B1_no_source_drift_since_T2', not drift, drift or 'all sources match run_identity')
    rec('B2_engine_worktree_equals_head_blob', git_blob(ENGINE) == cs.sha_file(ENGINE),
        {'engine_sha256': cs.sha_file(ENGINE)})

    # ---- C. unit identities recomputed from the saved data ----------------
    units = []
    for u in reg['units']['list']:
        uid = u['unit_id']
        row = next(r for r in cs.read_scope()
                   if r['arm'] == uid['arm'] and int(r['day']) == uid['day_id']
                   and int(r['ordinal']) == uid['ordinal'])
        rebuilt = cs.unit_from_row(row)
        ok = (rebuilt['public_identity'] == uid['public_identity']
              and rebuilt['committed_next'] == u['committed_next']
              and [int(x) for x in rebuilt['queue']] == [int(x) for x in u['queue']]
              and list(uid['mask']) in [list(m) for m in rebuilt['masks']])
        units.append(ok)
    rec('C1_all_19_units_rebuild_identically', all(units) and len(units) == 19,
        {'n_units': len(units), 'all_ok': all(units)})
    rec('C2_registration_unit_sha256_is_self_consistent',
        hashlib.sha256(json.dumps(reg['units']['list'], sort_keys=True).encode()).hexdigest()
        == reg['units']['sha256'], {'units_sha256': reg['units']['sha256']})

    # ---- D. denominators and failure accounting ---------------------------
    want_ids = json.dumps([u['unit_id'] for u in reg['units']['list']], sort_keys=True)
    denom = {}
    for tag, pa in (('run1', pa1), ('run2', pa2)):
        got = json.dumps([r['unit_id'] for r in pa['rows']], sort_keys=True)
        denom['%s_rows_19' % tag] = len(pa['rows']) == 19
        denom['%s_ids_match_registration' % tag] = got == want_ids
        denom['%s_denominator_intact' % tag] = (
            pa['n_evaluable'] + pa['n_invalid'] == 19 and pa['n_units_run'] == 19)
        denom['%s_invalid_list_matches' % tag] = (
            pa['n_invalid'] == len(pa['invalid_units'])
            and all(r['pair_status'] != 'OK' for r in pa['invalid_units']))
        denom['%s_invalid_reasons_registered' % tag] = all(
            r['pair_status'].startswith(('PAIR_INVALID_',))
            for r in pa['invalid_units'])
    rec('D_denominator_and_failure_ledger', all(denom.values()), denom)

    # ---- E. two runs agree bit for bit -----------------------------------
    def side_hashes(rows):
        return {(r['unit_id']['arm'], r['unit_id']['day_id'], r['unit_id']['ordinal'],
                 tuple(r['unit_id']['mask'])): {k: (v.get('plan_hash') if v else None)
                                                for k, v in (r.get('sides') or {}).items()}
                for r in rows}

    h1, h2 = side_hashes(pa1['rows']), side_hashes(pa2['rows'])
    mism = {str(k): [h1[k], h2.get(k)] for k in h1 if h1[k] != h2.get(k)}
    rec('E1_pass_a_two_runs_identical', not mism, mism or 'all 19 units × 2 sides identical')

    def set_hashes(rows):
        return {(r['unit_id']['arm'], r['unit_id']['day_id'], r['unit_id']['ordinal'],
                 tuple(r['unit_id']['mask'])): {k: (sorted(v.get('valid_plan_hashes') or [])
                                                    if v else None)
                                                for k, v in (r.get('sides') or {}).items()}
                for r in rows}

    s1, s2 = set_hashes(pb1['rows']), set_hashes(pb2['rows'])
    mism_b = {str(k): [s1[k], s2.get(k)] for k in s1 if s1[k] != s2.get(k)}
    rec('E2_pass_b_two_runs_identical', not mism_b, mism_b or 'all completion sets identical')

    # ---- F. verdicts recomputed from the raw rows ------------------------
    valid = [r for r in pa1['rows'] if r['pair_status'] == 'OK']
    diff = [r for r in valid if r['same_plan'] is False]
    same = [r for r in valid if r['same_plan'] is True]
    rec('F1_row_level_consistency',
        all(HD['plan_hash'] is None or PLAN_HASH.match(HD['plan_hash']) for r in valid
            for HD in r['sides'].values()),
        {'n_valid': len(valid), 'plan_hash_format': 'sha256(...)[:16]'})
    recomputed = {
        'n_units': len(pa1['rows']),
        'n_evaluable': len(valid),
        'n_invalid': len(pa1['rows']) - len(valid),
        'n_different_plan': len(diff),
        'n_same_plan': len(same),
        'n_different_but_same_first_after_anchor':
            sum(1 for r in diff if r['same_first_after_anchor']),
        'l1a_go': len(diff) > 0,
        'l1b_model_greedy_covered': pb1['n_model_greedy_covered'],
        'l1b_model_greedy_uncovered': pb1['n_model_greedy_uncovered'],
        'l1b_unknown': pb1['n_model_greedy_unknown'],
        'l1b_fully_covered': (pb1['n_model_greedy_uncovered'] == 0
                              and pb1['n_model_greedy_covered'] > 0),
        'difference_units': [r['unit_id'] for r in diff],
        'same_units': [r['unit_id'] for r in same],
    }
    rec('F2_summary_matches_rows',
        pa1['n_evaluable'] == recomputed['n_evaluable']
        and pa1['n_invalid'] == recomputed['n_invalid']
        and pa1['n_different_plan'] == recomputed['n_different_plan'],
        {'summary': [pa1['n_evaluable'], pa1['n_invalid'], pa1['n_different_plan']],
         'rows': [recomputed['n_evaluable'], recomputed['n_invalid'],
                  recomputed['n_different_plan']]})
    rec('F3_first_after_anchor_matches_suffix',
        all(r['sides']['model']['suffix_first_after_anchor']
            == (r['sides']['model']['cvid_suffix'][0] if r['sides']['model']['cvid_suffix'] else 0)
            for r in valid),
        {'rule': 'suffix_first_after_anchor == first element of the transit vehicle suffix, '
                 '0 meaning depot return'})

    # ---- G. the gate really was a gate -----------------------------------
    rec('G_gate_covered_all_units',
        pre['n_units'] == 19 and pre['n_equal'] == 19 and pre['all_ok'] is True,
        {'n_units': pre['n_units'], 'n_equal': pre['n_equal']})

    artefacts = {}
    for name in ('registration.json', 'run_identity.json', 'preflight.json',
                 'synthetic_regression.json', 'pass_a_run1.json', 'pass_b_run1.json',
                 'pass_a_run2.json', 'pass_b_run2.json', 'camr_shared.py',
                 'synthetic_regression.py', 'freeze_identity.py', 'degeneration_gate.py',
                 'run_pass.py', 'verify_camr_static.py', 'register.py'):
        artefacts[name] = cs.sha_file(OUT / name) if (OUT / name).is_file() else 'MISSING'
    artefacts['scripts/simulation/strict_online_env.py'] = cs.sha_file(ENGINE)

    payload = {
        'schema': 'p2_c_camr_static_01_verification_v1',
        'run_id': reg['run_id'],
        'registration_sha256': reg_sha,
        'run_identity_sha256': ident_sha,
        'preflight_sha256': pre_sha,
        'recomputed': recomputed,
        'n_checks': len(checks),
        'n_ok': sum(1 for c in checks if c['ok']),
        'all_ok': all(c['ok'] for c in checks),
        'checks': checks,
        'artefact_sha256': artefacts,
    }
    cs.write_json(a.out, payload)
    for c in checks:
        print('%-5s %s' % (c['status'], c['check']))
    print('%d/%d ok' % (payload['n_ok'], payload['n_checks']))
    print(json.dumps(recomputed, ensure_ascii=False, indent=1))
    sys.exit(0 if payload['all_ok'] else 6)


if __name__ == '__main__':
    main()
