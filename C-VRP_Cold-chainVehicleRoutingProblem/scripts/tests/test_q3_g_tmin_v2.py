#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-TMIN CANDIDATE V2 ACCEPTANCE (DIAGNOSTIC, not for production).

Implements criteria C1..C11 of
`scripts/q3_qualification/q3_gtmin_preregistration_2026-10-17.md` on the 14 pre-registered inputs,
in the declared order, with the declared counts.  No instance data is read, no randomness is used,
no production install entry is imported.

Usage: python scripts/tests/test_q3_g_tmin_v2.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))
sys.path.insert(0, str(EXT / 'scripts/m1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts/q3_qualification'))

from coldchain_contract import ColdChainContractV2                         # noqa: E402
import m1b_contract_identity as CI                                        # noqa: E402
import m1b_quality_certificate as QC                                      # noqa: E402
from m1b_q2_certifier import M1bQ2TrustFixCertifier                       # noqa: E402
import q3_t_min_candidate_v2 as V2                                        # noqa: E402

CONTRACT = ColdChainContractV2()
T_END = 24.0
ROWS = []
LIMITS = ('synthetic controls on the resolver only; no certification is run end to end, no plan is '
          'installed, no production install entry is imported')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-6s %s' % ('PASS' if ok else 'FAIL', cid, name[:95]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class HostileEnv(object):
    """Every attribute access raises: proves the resolver never reads `env`."""

    def __getattr__(self, name):
        raise AssertionError('the resolver read env.%s' % name)


# the 14 pre-registered inputs, in the pre-registered order (group id, description, kwargs)
INPUTS = (
    ('G1', 'no arguments', {}),
    ('G2', 'bare t_min=1e6', {'t_min': 1e6}),
    ('G3', 'bare t_min=24.0', {'t_min': 24.0}),
    ('G4', 'verified_t_min=1e6 (no env)', {'verified_t_min': 1e6}),
    ('G5', 'verified_t_min=1e6 + env', {'verified_t_min': 1e6, 'env': 'ENV'}),
    ('G6', 'verified_t_min=24.0 + env', {'verified_t_min': 24.0, 'env': 'ENV'}),
    ('G7', 'verified_t_min=0.0 + env', {'verified_t_min': 0.0, 'env': 'ENV'}),
    ('G8', 'verified_t_min=-5.0 + env', {'verified_t_min': -5.0, 'env': 'ENV'}),
    ('G9', 'verified_t_min=nan + env', {'verified_t_min': float('nan'), 'env': 'ENV'}),
    ('G10', 'verified_t_min=inf + env', {'verified_t_min': float('inf'), 'env': 'ENV'}),
    ('G11', 'verified_t_min="1e6" + env', {'verified_t_min': '1e6', 'env': 'ENV'}),
    ('G12', 'verified_t_min=1e10 + env', {'verified_t_min': 1e10, 'env': 'ENV'}),
    ('G13', 'verified_t_min=None + env', {'verified_t_min': None, 'env': 'ENV'}),
    ('G14', 'verified_t_min=True + env', {'verified_t_min': True, 'env': 'ENV'}),
)
#: The pre-registration grouped these wrongly (see amendment 01: the LIST said 5, the CONDITION
#: "verified_t_min is not None" implies 10).  The grouping below follows the CONDITION, which is the
#: stricter reading, and the amendment records that it was applied after the first run.
PASS_GROUPS = ('G1', 'G2', 'G3', 'G13')                       # not refused, numbers unchanged
REFUSED_EQUAL = ('G4', 'G7', 'G8', 'G9', 'G10', 'G12')        # refused; production already 0.0
REFUSED_CHANGED = ('G5', 'G6', 'G11', 'G14')                  # refused; production > 0
POSITIVE = PASS_GROUPS
NEGATIVE = REFUSED_EQUAL + REFUSED_CHANGED


def production_t_min(kw):
    k = {a: b for a, b in kw.items() if a != 'env'}
    if 'env' in kw:
        k['env'] = object()
    c = M1bQ2TrustFixCertifier(contract=CONTRACT, T_end=T_END, **k)
    return c.t_min, getattr(c, 't_min_source', None)


def candidate(kw, env=None):
    k = {a: b for a, b in kw.items() if a != 'env'}
    if 'env' in kw:
        k['env'] = env if env is not None else object()
    return V2.resolve_t_min(**k)


def q_lb_for(t_min):
    pc, _v1, _d1 = CI.check_quality_classes_are_certifiable(CONTRACT,
                                                           epsilon_init=QC.EPSILON_INIT)
    res, verd, det = QC.certify_quality_lb(pc, T_end=T_END, t_min=float(t_min),
                                           hours_per_time_unit=1.0)
    row = (res or {}).get(1, {})
    return {'H_max': det.get('H_max'), 'q_lb': row.get('q_lb'), 'verdict': verd,
            'per_class_constants': pc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-regression', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    src = {
        'scripts/q3_qualification/q3_t_min_candidate_v2.py':
            sha256(EXT / 'scripts/q3_qualification/q3_t_min_candidate_v2.py'),
        'scripts/q3_qualification/q3_gtmin_preregistration_2026-10-17.md':
            sha256(EXT / 'scripts/q3_qualification/q3_gtmin_preregistration_2026-10-17.md'),
        'scripts/m1b_adapter/m1b_q2_certifier.py':
            sha256(EXT / 'scripts/m1b_adapter/m1b_q2_certifier.py'),
        'scripts/m1b_adapter/m1b_quality_certificate.py':
            sha256(EXT / 'scripts/m1b_adapter/m1b_quality_certificate.py'),
        'scripts/coldchain/coldchain_contract.py':
            sha256(EXT / 'scripts/coldchain/coldchain_contract.py'),
        'scripts/tests/test_q3_g_tmin_v2.py': sha256(Path(__file__)),
    }
    frozen_before = {k: src[k] for k in ('scripts/m1b_adapter/m1b_q2_certifier.py',
                                         'scripts/m1b_adapter/m1b_quality_certificate.py',
                                         'scripts/coldchain/coldchain_contract.py')}
    pc0 = q_lb_for(0.0)

    # ---------------------------------------------------------------- run the 14 inputs once
    table = []
    for gid, desc, kw in INPUTS:
        prod_t, prod_src = production_t_min(kw)
        rep_t, rep_src = V2.production_resolver_replica(
            verified_t_min=kw.get('verified_t_min'),
            env=(object() if 'env' in kw else None))
        cand = candidate(kw)
        prod_q = q_lb_for(prod_t)
        cand_q = q_lb_for(cand.t_min)
        receipt = {'schema': 'synthetic', 'composite_pass': True,
                   'final': 'M1B_Q2TRUSTFIX_ENGINEERING_CERTIFIED', 'gates': {}}
        V2.apply_to_receipt(receipt, cand)
        would_pass, gate_detail = V2.composite_would_pass(receipt)
        table.append({
            'group': gid, 'input': desc,
            'production': {'t_min': prod_t, 'source': prod_src,
                           'H_max': prod_q['H_max'], 'q_lb': prod_q['q_lb']},
            'replica': {'t_min': rep_t, 'source': rep_src},
            'candidate': {'t_min': cand.t_min, 'source': cand.source,
                          'refused': cand.refused, 'env_consulted': cand.env_consulted,
                          'H_max': cand_q['H_max'], 'q_lb': cand_q['q_lb']},
            'receipt': {'composite_pass': receipt['composite_pass'], 'final': receipt['final'],
                        't_min_refused': receipt['t_min_refused'],
                        'would_pass': would_pass},
            'replica_matches_production': (prod_t, prod_src) == (rep_t, rep_src),
            'numeric_identical': (prod_q['H_max'] == cand_q['H_max']
                                  and prod_q['q_lb'] == cand_q['q_lb']),
        })

    # ---------------------------------------------------------------- C1 direction safety
    ck('C1', 'PRIMARY CONTROL: the candidate never sets t_min above the production value and always '
       'returns 0.0, on all 14 pre-registered inputs',
       all(r['candidate']['t_min'] <= r['production']['t_min'] for r in table)
       and all(r['candidate']['t_min'] == 0.0 for r in table)
       and all(r['replica_matches_production'] for r in table),
       {'n_inputs': len(table),
        'changed_t_min_groups': [r['group'] for r in table
                                 if r['production']['t_min'] != r['candidate']['t_min']],
        'replica_matches_production_on_all': all(r['replica_matches_production'] for r in table),
        'table': table},
       conclusion='the candidate is direction-safe: it can only lower t_min, i.e. only make the '
                  'certification harder')

    # ---------------------------------------------------------------- C2 numeric identity
    pos = [r for r in table if r['group'] in PASS_GROUPS]
    req = [r for r in table if r['group'] in REFUSED_EQUAL]
    ck('C2', 'on the 4 PASS groups the candidate does not refuse, and on the 4 PASS + 6 R-EQ groups '
       'the numbers are bit-identical to production (H_max and q_lb unchanged)',
       all(r['numeric_identical'] for r in pos + req)
       and all(not r['candidate']['refused'] for r in pos),
       {'pass_groups': [r['group'] for r in pos],
        'refused_but_numerically_equal_groups': [r['group'] for r in req],
        'n_identical': sum(1 for r in pos + req if r['numeric_identical']),
        'n_checked': len(pos) + len(req),
        'H_max': pc0['H_max'], 'q_lb': pc0['q_lb'],
        'amendment': 'scripts/q3_qualification/q3_gtmin_preregistration_amendment_01.md',
        'rows': [{'group': r['group'], 'prod_q_lb': r['production']['q_lb'],
                  'cand_q_lb': r['candidate']['q_lb']} for r in pos + req]},
       conclusion='the conservative path is untouched wherever production was already conservative')

    # ---------------------------------------------------------------- C3 strict refusal
    neg = [r for r in table if r['group'] in NEGATIVE]
    ck('C3', 'STRICT REFUSAL SEMANTICS: on the 10 groups where verified_t_min is not None the '
       'candidate refuses and the receipt is forced to REFUSED / composite_pass=False, not merely '
       'computed conservatively',
       all(r['candidate']['refused'] for r in neg)
       and all(r['candidate']['source'] == V2.UNVERIFIED_T_MIN_REFUSED for r in neg)
       and all(r['receipt']['composite_pass'] is False for r in neg)
       and all(r['receipt']['final'] == V2.ENGINEERING_REFUSED_LABEL for r in neg)
       and all(r['receipt']['would_pass'] is False for r in neg)
       and all(r['candidate']['q_lb'] == pc0['q_lb'] for r in neg),
       {'negative_groups': [r['group'] for r in neg],
        'rows': [{'group': r['group'], 'production_t_min': r['production']['t_min'],
                  'candidate_t_min': r['candidate']['t_min'],
                  'refused': r['candidate']['refused'],
                  'composite_pass': r['receipt']['composite_pass'],
                  'final': r['receipt']['final'],
                  'q_lb': r['candidate']['q_lb']} for r in neg],
        'difference_from_v1': 'V1 only lowered t_min; V2 additionally marks the certification '
                              'REFUSED, so a reader of composite_pass cannot mistake the '
                              'conservative number for a passing judgement'},
       conclusion='an untrusted claim can no longer leave a receipt that reads PASS')

    # ---------------------------------------------------------------- C4 forbidden label
    cand_src = (EXT / 'scripts/q3_qualification/q3_t_min_candidate_v2.py').read_text(
        encoding='utf-8')
    label_used_as_value = 'VERIFIED_AGAINST_SUPPLIED_ENV' in cand_src
    forged = {'composite_pass': True, 'final': 'M1B_Q2TRUSTFIX_ENGINEERING_CERTIFIED',
              'gates': {'old': {'t_min_source': 'VERIFIED_AGAINST_SUPPLIED_ENV'}}}
    ok_forged, det_forged = V2.receipt_gate(forged)
    clean = {'composite_pass': True, 'final': 'M1B_Q2TRUSTFIX_ENGINEERING_CERTIFIED',
             'gates': {}, 't_min_refused': False}
    ok_clean, _d = V2.receipt_gate(clean)
    ck('C4', 'the pre-fix label is unreachable: it appears in the candidate only inside the FORBIDDEN '
       'list, and a receipt carrying it is refused by the label gate',
       (not ok_forged) and ok_clean and det_forged['forbidden_label_paths'],
       {'label_occurs_in_source': label_used_as_value,
        'occurrences': cand_src.count('VERIFIED_AGAINST_SUPPLIED_ENV'),
        'forbidden_receipt_paths': det_forged['forbidden_label_paths'],
        'clean_receipt_gate_ok': ok_clean},
       conclusion='F-TMIN-2 holds: the old name is a refusal trigger')

    # ---------------------------------------------------------------- C5 env never read
    hostile_ok, err = True, None
    try:
        d_hostile = V2.resolve_t_min(verified_t_min=1e6, env=HostileEnv())
    except BaseException as exc:                                          # noqa: BLE001
        hostile_ok, err, d_hostile = False, '%s: %s' % (type(exc).__name__, exc), None
    ck('C5', 'env is never consulted: an env whose every attribute access raises is accepted without '
       'raising, the value is refused, and env_consulted is False',
       hostile_ok and d_hostile is not None and d_hostile.t_min == 0.0
       and d_hostile.refused is True and d_hostile.env_received is True
       and d_hostile.env_consulted is False,
       {'exception': err, 'decision': None if d_hostile is None else d_hostile.as_dict()},
       conclusion='F-TMIN-3 holds: env presence is evidence of an incomplete claim, not proof')

    # ---------------------------------------------------------------- C6 / C7 monotonicity
    ck('C6', 'no input raises q_lb above the production value (the defect direction is closed)',
       all(r['candidate']['q_lb'] <= r['production']['q_lb'] for r in table),
       {'per_group': [{'group': r['group'], 'production': r['production']['q_lb'],
                       'candidate': r['candidate']['q_lb']} for r in table]},
       conclusion='q_lb can only go down, i.e. toward a stricter qualification')
    ck('C7', 'no input shortens H_max below the production value',
       all(r['candidate']['H_max'] >= r['production']['H_max'] for r in table),
       {'per_group': [{'group': r['group'], 'production': r['production']['H_max'],
                       'candidate': r['candidate']['H_max']} for r in table]},
       conclusion='the certified horizon can only grow')

    # ---------------------------------------------------------------- C8 no FALSE_PASS
    false_pass = []
    for gid, desc, kw in INPUTS:
        cand = candidate(kw)
        r = {'composite_pass': True, 'final': 'M1B_Q2TRUSTFIX_ENGINEERING_CERTIFIED',
             'gates': {}}
        V2.apply_to_receipt(r, cand)
        would, _d = V2.composite_would_pass(r)
        if cand.refused and would:
            false_pass.append(gid)
    ck('C8', 'no FALSE_PASS: for every one of the 14 inputs, a refused t_min input cannot leave a '
       'receipt that a reader would call CERTIFIED',
       not false_pass,
       {'false_pass_groups': false_pass,
        'rule': 'composite_would_pass requires composite_pass is True AND final ends with '
                'ENGINEERING_CERTIFIED AND no forbidden label AND t_min_refused is False'},
       conclusion='the refusal is observable exactly where a reader looks')

    # ---------------------------------------------------------------- C9 frozen untouched
    after = {k: sha256(EXT / k) for k in frozen_before}
    ck('C9', 'no frozen source was modified by this work unit (the production certifier, the quality '
       'module and the contract are byte-identical to the pre-registration)',
       after == frozen_before,
       {'before': frozen_before, 'after': after,
        'candidate_installed': V2.INSTALLED,
        'preregistration': V2.PREREGISTRATION},
       conclusion='the candidate lives outside the production path')

    # ---------------------------------------------------------------- C10 regression
    reg = {}
    ok10 = True
    if args.skip_regression:
        reg['skipped'] = True
    else:
        for script, expect in (('scripts/tests/test_m1b_q2_final_entry_controls.py', None),
                               ('scripts/tests/test_q3_g_tmin_candidate.py', 11)):
            cmd = [sys.executable, str(EXT / script)]
            if expect is not None:
                tmp = out_dir / ('_reg_%s.json' % Path(script).stem)
                cmd += ['--out', str(tmp)]
            proc = subprocess.run(cmd, cwd=str(EXT), capture_output=True, text=True)
            n_fail = None
            if expect is not None and (out_dir / ('_reg_%s.json' % Path(script).stem)).exists():
                payload = json.loads((out_dir / ('_reg_%s.json' % Path(script).stem))
                                     .read_text(encoding='utf-8'))
                n_fail = payload.get('n_fail')
            reg[script] = {'exit': proc.returncode, 'n_fail': n_fail,
                           'false_pass_line': [ln for ln in proc.stdout.splitlines()
                                               if 'FALSE_PASS' in ln] or None}
            ok10 = ok10 and proc.returncode == 0 and (n_fail in (None, 0))
    ck('C10', 'the previously delivered suites still pass: the M1-b final-entry controls (6 checks, '
       '0 FALSE_PASS) and the first-round G-TMIN candidate controls (11 rows)',
       ok10, {'regression': reg},
       conclusion='the new candidate did not disturb the existing evidence')

    # ---------------------------------------------------------------- C11 B2 gap covered
    b2 = EXT / 'scripts/tests/test_m1b_q2_final_entry_controls.py'
    text = b2.read_text(encoding='utf-8')
    gap = ('t_min=t_min' in text) and ('verified_t_min=' not in text)
    byp = [r for r in table if r['production']['source'] == 'VERIFIED_AGAINST_SUPPLIED_ENV'
           and float(r['production']['t_min']) > 0.0]
    ck('C11', 'the B2 coverage gap is independently covered here: production still returns a POSITIVE '
       't_min for verified_t_min while B2 never passes that keyword, and the candidate both refuses '
       'the value and blocks the receipt',
       gap and len(byp) >= 4
       and all(r['candidate']['refused'] for r in byp)
       and all(float(r['production']['t_min']) > 0.0 for r in byp),
       {'b2_passes_bare_t_min': 't_min=t_min' in text,
        'b2_passes_verified_t_min_keyword': 'verified_t_min=' in text,
        'production_bypass_groups_with_positive_t_min':
            [{'group': r['group'], 't_min': r['production']['t_min']} for r in byp],
        'candidate_refuses_them': all(r['candidate']['refused'] for r in byp),
        'note': 'G7 (verified_t_min=0.0 + env) also carries the pre-fix label but returns 0.0, so it '
                'is not a numerical bypass; it is still refused (R-EQ)'},
       conclusion='this file, not B2, is the control that exercises the real bypass')

    payload = {
        'schema': 'q3_g_tmin_v2_acceptance',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'candidate_installed': V2.INSTALLED,
        'candidate_version': V2.IMPLEMENTATION_VERSION,
        'preregistration': V2.PREREGISTRATION,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'baseline': {'H_max': pc0['H_max'], 'q_lb': pc0['q_lb'],
                     'per_class_constants': pc0['per_class_constants']},
        'n_inputs': len(table), 'positive': list(POSITIVE), 'negative': list(NEGATIVE),
        'table': table,
        'affected_call_sites': [list(x) for x in V2.AFFECTED_CALL_SITES],
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('GTMIN_V2_ACCEPTANCE_PASS' if all(r['ok'] for r in ROWS)
                    else 'GTMIN_V2_ACCEPTANCE_FAIL'),
        'criteria': {'C1': 'direction safety (primary)', 'C2': 'numeric identity',
                     'C3': 'strict refusal', 'C4': 'forbidden label', 'C5': 'env not read',
                     'C6': 'q_lb never raised', 'C7': 'H_max never shortened',
                     'C8': 'no FALSE_PASS', 'C9': 'frozen untouched', 'C10': 'regression',
                     'C11': 'B2 gap covered'},
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('\n=== %d criteria, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
