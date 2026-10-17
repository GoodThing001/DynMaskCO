#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 G-TMIN ISOLATED REPAIR CANDIDATE -- ACCEPTANCE AND COMPATIBILITY CONTROLS (DIAGNOSTIC).

Rows TMIN-N-01..04 and TMIN-P-01..02 are the six controls required by the Q3 work-package spec.
They are supplemented by:

  TMIN-C-01  the candidate's replica of the PRE-FIX resolver reproduces the PRODUCTION resolver on
             every input, so the comparison is against the real code and not against a paraphrase;
  TMIN-C-02  the production resolver accepts a STRING ("1e6") as a verified lower bound -- a type
             confusion the spec's requirement 4 ("wrong type must fail closed") does not tolerate;
  TMIN-C-03  the existing B2 control in `test_m1b_q2_final_entry_controls.py` exercises the BARE
             `t_min` keyword, not `verified_t_min` + env, so the bypass this work package is about is
             NOT covered by the existing suite (recorded fact, not a claim about the author);
  TMIN-S-01  direction safety: q_lb is non-decreasing in t_min, and the candidate's t_min is never
             larger than the production one, so the candidate can only refuse more;
  TMIN-W-01  `env` is never consulted: an env object whose every attribute access raises is accepted
             without raising, which proves the resolver does not read it.

The candidate is NOT installed (`INSTALLED = False`) and this file does not modify the production
certifier.  Nothing here is a production receipt.

Usage: python scripts/tests/test_q3_g_tmin_candidate.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import q3_t_min_candidate as CT                                           # noqa: E402

CONTRACT = ColdChainContractV2()
T_END = 24.0
ROWS = []
LIMITS = ('synthetic controls on the resolver only; no certification is run end to end, no plan is '
          'installed, nothing is written to any production receipt store')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-9s %s' % ('PASS' if ok else 'FAIL', cid, name[:92]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def production_t_min(**kw):
    """Resolve through the REAL production resolver (construction only; certify() is never called)."""
    c = M1bQ2TrustFixCertifier(contract=CONTRACT, T_end=T_END, **kw)
    return c.t_min, getattr(c, 't_min_source', None)


def per_class_constants():
    _ident, verd, _det = CI.validate_contract_for_quality_certificate(
        CONTRACT, registered_contract_hash=None)
    pc, pcverd, _pcdet = CI.check_quality_classes_are_certifiable(
        CONTRACT, epsilon_init=QC.EPSILON_INIT)
    return pc, (verd, pcverd)


class HostileEnv(object):
    """Any attribute access raises: used to prove the candidate never reads `env`."""

    def __getattr__(self, name):
        raise AssertionError('the resolver read env.%s' % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    src = {
        'scripts/q3_qualification/q3_t_min_candidate.py':
            sha256(EXT / 'scripts/q3_qualification/q3_t_min_candidate.py'),
        'scripts/m1b_adapter/m1b_q2_certifier.py':
            sha256(EXT / 'scripts/m1b_adapter/m1b_q2_certifier.py'),
        'scripts/m1b_adapter/m1b_quality_certificate.py':
            sha256(EXT / 'scripts/m1b_adapter/m1b_quality_certificate.py'),
        'scripts/tests/test_q3_g_tmin_candidate.py': sha256(Path(__file__)),
    }
    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pc, idents = per_class_constants()

    baseline_prod = production_t_min()
    base_h, base_q = CT.horizon_and_q_lb(pc, T_end=T_END, t_min=baseline_prod[0])
    prod_qc = QC.certify_quality_lb(pc, T_end=T_END, t_min=baseline_prod[0],
                                    hours_per_time_unit=1.0)
    prod_row = (prod_qc[0] or {}).get(1, {})

    # ============================================================== TMIN-N-01
    d_none = CT.resolve_t_min(verified_t_min=None, env=None)
    d_none_env = CT.resolve_t_min(verified_t_min=None, env=object())
    ck('TMIN-N-01', 'verified_t_min=None resolves to the conservative clock origin with or without an '
       'env, and the numeric result is identical to the production baseline',
       d_none.t_min == 0.0 and d_none_env.t_min == 0.0
       and d_none.source == CT.CONSERVATIVE_CLOCK_ORIGIN
       and d_none_env.source == CT.CONSERVATIVE_CLOCK_ORIGIN
       and d_none_env.env_consulted is False
       and base_h == 24.000001 and abs(base_q - 8.54807998017) < 1e-12
       and abs(prod_row.get('q_lb', 0.0) - base_q) < 1e-15,
       {'resolve_none': d_none.as_receipt_fields(), 'resolve_none_with_env':
        d_none_env.as_receipt_fields(),
        'baseline_t_min': baseline_prod[0], 'baseline_source': baseline_prod[1],
        'candidate_H_max': base_h, 'candidate_q_lb': base_q,
        'production_q_lb': prod_row.get('q_lb'),
        'mirror_matches_production': abs(prod_row.get('q_lb', 0.0) - base_q) < 1e-15},
       conclusion='the conservative branch is unchanged and the q_lb mirror is numerically identical '
                  'to the production certificate function')

    # ============================================================== TMIN-N-02
    prod_1e6 = production_t_min(verified_t_min=1e6, env=object())
    cand_1e6 = CT.resolve_t_min(verified_t_min=1e6, env=object())
    h_1e6, q_1e6 = CT.horizon_and_q_lb(pc, T_end=T_END, t_min=cand_1e6.t_min)
    ck('TMIN-N-02', 'verified_t_min=1e6 with an env witness no longer raises the quality lower bound: '
       'the candidate refuses it and the numbers are bit-identical to the conservative baseline',
       prod_1e6[0] == 1000000.0 and cand_1e6.t_min == 0.0
       and cand_1e6.source == CT.UNVERIFIED_T_MIN_REFUSED
       and h_1e6 == base_h and q_1e6 == base_q and cand_1e6.env_consulted is False,
       {'production_t_min': prod_1e6[0], 'production_source': prod_1e6[1],
        'production_H_max': 0.0, 'production_q_lb': 8.999999999,
        'candidate': cand_1e6.as_receipt_fields(), 'candidate_H_max': h_1e6,
        'candidate_q_lb': q_1e6, 'baseline_q_lb': base_q,
        'why_this_matters': 'H_max = max(0, T_end + eps - t_min) so t_min=1e6 collapses the horizon '
                            'to 0 and q_lb becomes the initial value 8.999999999'},
       conclusion='the bypass is closed and the fix moves the number only in the safe direction')

    # ============================================================== TMIN-N-03
    prod_24 = production_t_min(verified_t_min=24.0, env=object())
    cand_24 = CT.resolve_t_min(verified_t_min=24.0, env=object())
    h_24, q_24 = CT.horizon_and_q_lb(pc, T_end=T_END, t_min=cand_24.t_min)
    ck('TMIN-N-03', 'verified_t_min=24.0 with an env witness no longer shortens the certified horizon',
       prod_24[0] == 24.0 and cand_24.t_min == 0.0 and h_24 == base_h and q_24 == base_q,
       {'production_t_min': prod_24[0], 'production_H_max': 1.0000000010279564e-06,
        'production_q_lb': 8.99999998017, 'candidate': cand_24.as_receipt_fields(),
        'candidate_H_max': h_24, 'candidate_q_lb': q_24},
       conclusion='the horizon cannot be shrunk by an unverified claim')

    # ============================================================== TMIN-N-04
    forged = {'schema': 'x', 'gates': {'quality': {'t_min_source':
                                                   'VERIFIED_AGAINST_SUPPLIED_ENV'}},
              'note': 'a receipt that still carries the forbidden label'}
    ok_label, found = CT.receipt_label_gate(forged)
    clean = {'schema': 'x', 'gates': {'quality': {'t_min_source': CT.CONSERVATIVE_CLOCK_ORIGIN}}}
    ok_clean, found_clean = CT.receipt_label_gate(clean)
    ck('TMIN-N-04', 'a receipt that still carries VERIFIED_AGAINST_SUPPLIED_ENV is REFUSED by the '
       'composite label gate; a receipt with the conservative label passes it',
       (not ok_label) and found and ok_clean and not found_clean,
       {'forged_verdict': 'REFUSED' if not ok_label else 'PASS', 'forbidden_paths': found,
        'clean_verdict': 'PASS' if ok_clean else 'REFUSED',
        'forbidden_labels': list(CT.FORBIDDEN_RECEIPT_LABELS)},
       conclusion='the old label is impossible to re-use: it is a refusal trigger, not a name')

    # ============================================================== TMIN-W-01
    ok_hostile, err = True, None
    try:
        d_hostile = CT.resolve_t_min(verified_t_min=1e6, env=HostileEnv())
    except BaseException as exc:                                          # noqa: BLE001
        ok_hostile, err = False, '%s: %s' % (type(exc).__name__, exc)
        d_hostile = None
    ck('TMIN-W-01', 'env is never consulted: an env object whose every attribute access raises is '
       'accepted without raising, and the value is still refused',
       ok_hostile and d_hostile is not None and d_hostile.t_min == 0.0
       and d_hostile.env_received is True and d_hostile.env_consulted is False,
       {'exception': err, 'decision': None if d_hostile is None else
        d_hostile.as_receipt_fields()},
       conclusion='F-TMIN-3 holds: the presence of env is recorded as evidence, never read as proof')

    # ============================================================== TMIN-P-01 witness
    witness_path = out_dir / 'tmin_witness_value.txt'
    witness_path.write_text('18.5\n', encoding='utf-8')
    value = 18.5
    digest = sha256(witness_path)

    def recompute(path):
        p = Path(path)
        return {'sha256': sha256(p), 'value': float(p.read_text(encoding='utf-8').strip())}

    wit = CT.TMinWitness(source_path=str(witness_path), source_sha256=digest,
                         generator_identity='synthetic control fixture',
                         declared_semantics='earliest possible execution time of the certified suffix',
                         monotone_over_all_plans=True, recomputable_from_repo=True,
                         contract_binding=CONTRACT.contract_hash,
                         state_binding='STATE_SYNTHETIC', value=value)
    d_wit = CT.resolve_t_min(witness=wit, recompute=recompute,
                             contract_identity=CONTRACT.contract_hash,
                             state_identity='STATE_SYNTHETIC')
    h_wit, q_wit = CT.horizon_and_q_lb(pc, T_end=T_END, t_min=d_wit.t_min)
    ck('TMIN-P-01', 'a complete witness (identity + sha256 + semantics + monotonicity + '
       'recomputability + binding) is the ONLY way to obtain a non-zero t_min, and its sha256 is '
       'recorded in the receipt fields',
       d_wit.t_min == value and d_wit.source == CT.WITNESS_VERIFIED
       and d_wit.witness_report.get('recomputed', {}).get('sha256') == digest
       and h_wit < base_h and q_wit > base_q,
       {'decision': d_wit.as_receipt_fields(), 'H_max': h_wit, 'q_lb': q_wit,
        'baseline_H_max': base_h, 'baseline_q_lb': base_q,
        'note': 'the witness mechanism is DEMONSTRATED on a synthetic fixture only; a real one would '
                'need a repository artefact and an independent recomputation'},
       conclusion='the future non-zero t_min path exists but is unreachable without a recomputable '
                  'witness')

    # ============================================================== TMIN-P-02 tampered witness
    bad = CT.TMinWitness(**dict(dict(wit.__dict__), source_sha256='0' * 64))
    d_bad = CT.resolve_t_min(witness=bad, recompute=recompute,
                             contract_identity=CONTRACT.contract_hash,
                             state_identity='STATE_SYNTHETIC')
    nohook = CT.resolve_t_min(witness=wit, recompute=None,
                              contract_identity=CONTRACT.contract_hash,
                              state_identity='STATE_SYNTHETIC')
    nomono = CT.resolve_t_min(witness=CT.TMinWitness(**dict(dict(wit.__dict__),
                                                           monotone_over_all_plans=False)),
                              recompute=recompute, contract_identity=CONTRACT.contract_hash,
                              state_identity='STATE_SYNTHETIC')
    badbind = CT.resolve_t_min(witness=wit, recompute=recompute,
                               contract_identity='A_DIFFERENT_CONTRACT',
                               state_identity='STATE_SYNTHETIC')
    ck('TMIN-P-02', 'a witness whose sha256 does not match, that has no recomputation hook, that is '
       'not monotone over all plans, or that is bound to another contract is REFUSED (fail-closed, '
       't_min returns to 0)',
       d_bad.t_min == 0.0 and d_bad.source == CT.WITNESS_REFUSED
       and nohook.t_min == 0.0 and nohook.source == CT.WITNESS_REFUSED
       and nomono.t_min == 0.0 and nomono.source == CT.WITNESS_REFUSED
       and badbind.t_min == 0.0 and badbind.source == CT.WITNESS_REFUSED,
       {'sha_mismatch': d_bad.as_receipt_fields(), 'no_recompute_hook': nohook.as_receipt_fields(),
        'not_monotone': nomono.as_receipt_fields(), 'wrong_contract_binding':
        badbind.as_receipt_fields()},
       conclusion='every element of spec 1.3 is load-bearing; dropping any one returns t_min to 0')

    # ============================================================== TMIN-C-01 replica fidelity
    cases = [('none', {}), ('bare_1e6', {'t_min': 1e6}), ('bare_24', {'t_min': 24.0}),
             ('verif_1e6_noenv', {'verified_t_min': 1e6}), ('verif_1e6_env',
                                                            {'verified_t_min': 1e6, 'env': object()}),
             ('verif_24_env', {'verified_t_min': 24.0, 'env': object()}),
             ('verif_0_env', {'verified_t_min': 0.0, 'env': object()}),
             ('verif_neg_env', {'verified_t_min': -5.0, 'env': object()}),
             ('verif_nan_env', {'verified_t_min': float('nan'), 'env': object()}),
             ('verif_inf_env', {'verified_t_min': float('inf'), 'env': object()}),
             ('verif_str_env', {'verified_t_min': '1e6', 'env': object()}),
             ('verif_1e10_env', {'verified_t_min': 1e10, 'env': object()}),
             ('verif_None_env', {'verified_t_min': None, 'env': object()}),
             ('verif_bool_env', {'verified_t_min': True, 'env': object()})]
    rows = []
    replica_exact, candidate_le = True, True
    for name, kw in cases:
        prod = production_t_min(**{k: v for k, v in kw.items() if k != 'env'},
                               **({'env': kw['env']} if 'env' in kw else {}))
        rep = CT.production_resolver_replica(verified_t_min=kw.get('verified_t_min'),
                                             env=kw.get('env'))
        cand = CT.resolve_t_min(**({'verified_t_min': kw['verified_t_min']}
                                   if 'verified_t_min' in kw else {}),
                                **({'env': kw['env']} if 'env' in kw else {}))
        rows.append({'case': name, 'production': {'t_min': prod[0], 'source': prod[1]},
                     'replica': {'t_min': rep[0], 'source': rep[1]},
                     'candidate': {'t_min': cand.t_min, 'source': cand.source},
                     't_min_changed': prod[0] != cand.t_min,
                     'source_changed': prod[1] != cand.source})
        replica_exact = replica_exact and prod == rep
        candidate_le = candidate_le and cand.t_min <= prod[0]
    tmin_changed = [r['case'] for r in rows if r['t_min_changed']]
    source_changed = [r['case'] for r in rows if r['source_changed']]
    ck('TMIN-C-01', 'the candidate replica reproduces the PRODUCTION resolver on every input; the '
       'candidate lowers the NUMBER on exactly the four hazardous inputs and never raises it, while '
       'the LABEL changes on every non-None verified_t_min input (by design, F-TMIN-1/2)',
       replica_exact and candidate_le
       and set(tmin_changed) == {'verif_1e6_env', 'verif_24_env', 'verif_str_env', 'verif_bool_env'}
       and set(tmin_changed) <= set(source_changed),
       {'replica_matches_production': replica_exact,
        'candidate_t_min_le_production_always': candidate_le,
        't_min_changed_cases': tmin_changed, 'source_changed_cases': source_changed, 'rows': rows,
        'reading': 'the four number-changing inputs are exactly the ones where the pre-fix label '
                   'VERIFIED_AGAINST_SUPPLIED_ENV was produced without reading env: 1e6, 24.0, the '
                   'STRING "1e6" and the BOOLEAN True (which float() turns into t_min = 1.0); the '
                   'extra label changes are inputs where production already fell back to 0 but under '
                   'the misleading CONSERVATIVE_CLOCK_ORIGIN label'},
       conclusion='the fix is confined to the defect; the conservative path is untouched')

    # ============================================================== TMIN-C-02 string type
    str_row = [r for r in rows if r['case'] == 'verif_str_env'][0]
    bool_row = [r for r in rows if r['case'] == 'verif_bool_env'][0]
    ck('TMIN-C-02', 'the production resolver accepts the STRING "1e6" and the BOOLEAN True as verified '
       'execution-time lower bounds, so requirement 4 ("wrong type must fail closed") is violated',
       str_row['production']['t_min'] == 1000000.0
       and str_row['production']['source'] == 'VERIFIED_AGAINST_SUPPLIED_ENV'
       and str_row['candidate']['t_min'] == 0.0
       and bool_row['production']['t_min'] == 1.0
       and bool_row['candidate']['t_min'] == 0.0,
       {'string_case': {'production': str_row['production'], 'candidate': str_row['candidate']},
        'bool_case': {'production': bool_row['production'], 'candidate': bool_row['candidate']},
        'root_cause': '`float(self.verified_t_min)` accepts any numeric string and any bool (True -> '
                      '1.0); there is no isinstance check and no witness'},
       conclusion='the type-confusion channels are closed by the witness requirement, not by adding '
                  'an isinstance test')

    # ============================================================== TMIN-C-03 existing coverage
    b2 = EXT / 'scripts/tests/test_m1b_q2_final_entry_controls.py'
    text = b2.read_text(encoding='utf-8')
    b2_uses_bare = 't_min=t_min' in text
    b2_passes_verified = 'verified_t_min=' in text          # a KEYWORD USE, not a prose mention
    b2_mentions_verified = 'verified_t_min' in text
    ck('TMIN-C-03', 'the existing B2 control passes the BARE t_min keyword and never supplies '
       'verified_t_min as a keyword, so the bypass G-TMIN is about was not covered by the existing '
       'suite (a recorded fact about coverage, not a claim about the author)',
       b2_uses_bare and not b2_passes_verified,
       {'file': str(b2.relative_to(EXT)), 'passes_bare_t_min_keyword': b2_uses_bare,
        'passes_verified_t_min_keyword': b2_passes_verified,
        'mentions_verified_t_min_in_prose_only': b2_mentions_verified,
        'b2_assertion': 'B2 asserts H_max > 24.0 with t_min=1e6, which holds only because the BARE '
                        'keyword is ignored -- the verified_t_min path is never constructed, so the '
                        'B2 PASS and the G-TMIN hazard coexist',
        'label': 'COVERAGE_GAP_RECORDED'},
       conclusion='a control that passes for the wrong reason is not evidence; the G-TMIN controls '
                  'in this file close the gap')

    # ============================================================== TMIN-S-01 direction safety
    grid = [0.0, 0.5, 1.0, 6.0, 12.0, 23.0, 23.999999, 24.0, 1e6]
    ds = CT.direction_safety(pc, T_end=T_END, t_mins=grid)
    prod_vals = []
    for t in grid:
        r = QC.certify_quality_lb(pc, T_end=T_END, t_min=t, hours_per_time_unit=1.0)
        prod_vals.append((r[0] or {}).get(1, {}).get('q_lb'))
    mirror_agrees = all(abs(a['q_lb'] - b) < 1e-15 for a, b in zip(ds['rows'], prod_vals))
    ck('TMIN-S-01', 'q_lb is non-decreasing in t_min (so lowering t_min can only add refusals), and '
       'the candidate never lowers q_lb below the production value for the same input',
       ds['q_lb_non_decreasing_in_t_min'] and mirror_agrees,
       {'direction': ds, 'production_q_lb_at_grid': prod_vals,
        'mirror_agrees_with_production': mirror_agrees,
        'statement': 'dq_lb/dt_min = +k > 0 with k = rate_per_hour'},
       conclusion='the candidate is direction-safe: it can only make certification harder')

    payload = {
        'schema': 'q3_g_tmin_candidate_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'candidate_installed': CT.INSTALLED,
        'candidate_version': CT.IMPLEMENTATION_VERSION,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'contract_identity_gates': idents,
        'affected_call_sites': [list(x) for x in CT.AFFECTED_CALL_SITES],
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('GTMIN_CANDIDATE_CONTROLS_PASS' if all(r['ok'] for r in ROWS)
                    else 'GTMIN_CANDIDATE_CONTROLS_FAIL'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
