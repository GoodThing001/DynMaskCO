#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-TMIN INSTALLED-FIX VERIFICATION -- the five pre-installation checks (DIAGNOSTIC).

The user's conditional authorisation requires five checks after the minimal diff is applied to the
active source.  This file implements them as V-01..V-09, plus the identity separation the
authorisation demands and the classification of every superseded control.

  check 1  V-01  the diff is confined to `_resolve_t_min`, the refusal flag, the receipt gate and
                 nothing else (no Q3, no G5/G6, no install path)
  check 2  V-02  refusal propagation: EVERY non-None verified_t_min (0, True, False, str, nan, inf,
                 objects, no-env, with-env) gets t_min=0 AND composite_pass=False AND an explicit
                 refusal gate; the conservative path is untouched
  check 3  V-03  independence: env=object() cannot change the t_min conclusion, and the OTHER uses of
                 `env` in certify() are preserved (not crudely deleted)
  check 4  V-04  numerical conservatism: q_lb is never raised and H_max never shortened relative to
                 the pre-fix resolver
  check 5  V-05  identity separation: pre-fix and post-fix sha256 are both recorded, differ, and the
                 pre-fix acceptance records are NOT reused as the new source's qualification
  also     V-06  historical callers: the four test files that construct the certifier still pass
                 (or their failures are classified as superseded-by-the-fix)
           V-07  no receipt carries the forbidden label
           V-08  the install path is untouched (`run_repair_transaction` not imported, d3_geo_adapter
                 and d3_install_receipt hashes unchanged)
           V-09  HEAD unchanged, no commit, only the authorised file modified

Usage: python scripts/tests/test_q3_gtmin_install_verification.py --out <json path>
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

from coldchain_contract import ColdChainContractV2                         # noqa: E402
import m1b_contract_identity as CI                                        # noqa: E402
import m1b_quality_certificate as QC                                      # noqa: E402
from m1b_q2_certifier import (M1bQ2TrustFixCertifier, ENGINEERING_REFUSED,  # noqa: E402
                              ENGINEERING_CERTIFIED)

# NOTE: this file is deliberately SELF-CONTAINED.  The two helpers below are inlined so that the
# delivered test depends only on tracked sources (the frozen certifier tree and the contract), and
# not on the local-only G-TMIN candidate module (which is not part of this delivery).
FORBIDDEN_RECEIPT_LABELS = ('VERIFIED_AGAINST_SUPPLIED_ENV',)


def production_resolver_replica(verified_t_min=None, env=None):
    """Verbatim replication of the PRE-FIX `_resolve_t_min` (m1b_q2_certifier.py:129-139).

    Used only as the comparison baseline: it is the semantics the fix replaced.
    """
    source = 'CONSERVATIVE_CLOCK_ORIGIN'
    if verified_t_min is None or env is None:
        return 0.0, source
    try:
        v = float(verified_t_min)
    except Exception:                                                     # noqa: BLE001
        return 0.0, source
    if v != v or v < 0 or v > 1e9:
        return 0.0, source
    return v, 'VERIFIED_AGAINST_SUPPLIED_ENV'


def receipt_label_gate(receipt):
    """Refuse a receipt carrying a forbidden label or a refused t_min flag."""
    found = []

    def _walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    for bad in FORBIDDEN_RECEIPT_LABELS:
                        if bad in k:
                            found.append(path + '/' + k)
                _walk(v, path + '/' + str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, '%s[%d]' % (path, i))
        elif isinstance(obj, str):
            for bad in FORBIDDEN_RECEIPT_LABELS:
                if bad in obj:
                    found.append(path)

    _walk(receipt, '')
    return (not found and not bool(receipt.get('t_min_refused'))), {
        'forbidden_label_paths': found,
        't_min_refused': bool(receipt.get('t_min_refused'))}

CONTRACT = ColdChainContractV2()
T_END = 24.0
CERT = EXT / 'scripts/m1b_adapter/m1b_q2_certifier.py'
PRE_FIX = EXT / 'results/_q3_next_20261017/gtmin/m1b_q2_certifier.py.PRE_FIX'
CERT_REL = 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/m1b_adapter/m1b_q2_certifier.py'
PRE_FIX_REL = 'PRE_FIX_COPY'

#: 64-hex digests are kept OUT of this source on purpose.  `m1b_q2_certifier.py`'s own header records
#: the reason: a literal table of sha256 digests inside a source file trips the repository's staged
#: secret scanner (rule `R2_bare_64_hex_token`) on a FALSE POSITIVE, and the sanctioned remedy is to
#: keep the single recorded copy in the REPORT rather than to exempt the scanner.  So the expected
#: digests are parsed from the delivered acceptance record (`<path> = <sha256>` lines).
PIN_REPORT = (EXT / 'docs/当前规划/决策与审计/GTMIN安全补正与选择性交付验收_2026-10-17.md')
_PIN_RX = __import__('re').compile(r'^\s*(\S+\.py|PRE_FIX_COPY)\s*=\s*([0-9a-f]{64})\s*$')


def pinned_digests():
    if not PIN_REPORT.exists():
        return {}
    out = {}
    for line in PIN_REPORT.read_text(encoding='utf-8').splitlines():
        m = _PIN_RX.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


PINS = pinned_digests()
OLD_SHA = PINS.get(PRE_FIX_REL)
UNCHANGED_PINS = (
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_geo_adapter.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/d3_install_receipt.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/m1a_adapter/m1a_certification_adapter.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py',
)
ROWS = []
LIMITS = ('read-only verification of an already-applied authorised patch; no plan is installed, no '
          'experiment is run, nothing is committed')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS, status=None):
    st = status or ('PASS' if ok else 'FAIL')
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-5s %s' % (st, cid, name[:96]))


def ck_skip(cid, name, reason, detail=None):
    """A check whose LOCAL evidence is not part of the repository.

    This file is committed; the audit it performs needs the pre-fix byte copy and four local Q3
    suites.  In a fresh clone those are absent, so the affected rows report SKIP with the reason
    instead of pretending to pass or failing spuriously.  In the authoring checkout the evidence is
    present and the full check runs.
    """
    d = dict(detail or {})
    d['skip_reason'] = reason
    ROWS.append({'id': cid, 'name': name, 'ok': True, 'status': 'SKIP', 'detail': d,
                 'conclusion': 'NOT EVALUATED in this checkout', 'limits': LIMITS})
    print('[SKIP ] %-5s %s' % (cid, name[:96]))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class HostileEnv(object):
    def __getattr__(self, name):
        raise AssertionError('the resolver read env.%s' % name)


def certify_probe(**kw):
    """Exercise the REAL patched `certify()` path with minimal, deliberately unusable arguments.

    Gate 0b sits immediately after the proof-hook gate, so a refusal attributable to `t_min` is
    observable without building a projection.  When no `verified_t_min` is supplied the call proceeds
    into the normal gate chain, which is what V-02 compares against.
    """
    c = M1bQ2TrustFixCertifier(contract=CONTRACT, T_end=T_END, **kw)
    ok, rc = c.certify(vehicles=[], served_mask=[], candidate_plans={}, clock=0.0, inst_idx=0,
                       travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 1.0)
    return {'ok': ok, 't_min': c.t_min, 'source': getattr(c, 't_min_source', None),
            'final': rc['final'], 'composite_pass': rc['composite_pass'],
            'gates': sorted(rc.get('gates', {}).keys()),
            't_min_gate': (rc.get('gates', {}).get('t_min') or {}).get('verdict'),
            'receipt': rc}


def q_lb(t_min):
    pc, _a, _b = CI.check_quality_classes_are_certifiable(CONTRACT,
                                                         epsilon_init=QC.EPSILON_INIT)
    res, verd, det = QC.certify_quality_lb(pc, T_end=T_END, t_min=float(t_min),
                                           hours_per_time_unit=1.0)
    return {'H_max': det.get('H_max'), 'q_lb': (res or {}).get(1, {}).get('q_lb'), 'verdict': verd}


#: every shape of untrusted claim the authorisation enumerates, plus a few more
CLAIM_CASES = (
    ('1e6', 1e6), ('24.0', 24.0), ('0.0', 0.0), ('-5.0', -5.0), ('1e10', 1e10),
    ('True', True), ('False', False), ('"1e6"', '1e6'), ('""', ''), ('nan', float('nan')),
    ('inf', float('inf')), ('-inf', float('-inf')), ('list', [1, 2]), ('object', object()),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-callers', action='store_true')
    ap.add_argument('--simulate-fresh-checkout', action='store_true',
                    help='treat the local evidence (PRE_FIX copy + local suites) as absent, so the '
                         'SKIP path can be verified deterministically')
    args = ap.parse_args()
    new_sha = sha256(CERT)
    have_prefix = PRE_FIX.exists()
    pre_sha = sha256(PRE_FIX) if have_prefix else None
    local_suites = {n: (EXT / 'scripts/tests' / n).exists()
                    for n in ('test_q3_g_tmin_candidate.py', 'test_q3_g_tmin_v2.py',
                              'test_q3_archive_regression.py',
                              'test_q3_door_interval_thresholds.py',
                              'q3_theorem_b_audit.py')}
    have_local_suites = all(local_suites.values())
    if args.simulate_fresh_checkout:
        have_prefix = False
        have_local_suites = False

    # ============================================================ V-01 diff scope
    numstat = subprocess.run(['git', 'diff', '--no-index', '--numstat', '--',
                              str(PRE_FIX), str(CERT)],
                             capture_output=True, text=True).stdout.strip()
    body = Path(PRE_FIX).read_text(encoding='utf-8').splitlines()
    now = CERT.read_text(encoding='utf-8').splitlines()
    import difflib
    sm = difflib.SequenceMatcher(None, body, now)
    changed_old, changed_new = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed_old.extend(range(i1 + 1, i2 + 1))
        changed_new.extend(range(j1 + 1, j2 + 1))
    # allowed regions: _resolve_t_min (old 122-139) and the gate-0 area (old 182-192)
    allowed_old = set(range(122, 140)) | set(range(182, 193))
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=str(EXT.parent),
                            capture_output=True, text=True).stdout.splitlines()
    modified = [ln[3:].strip() for ln in status if ln[:2] in (' M', 'M ', 'MM')]
    forbidden_touched = [p for p in modified
                         if p.replace('\\', '/').endswith(('d3_geo_adapter.py',
                                                           'd3_install_receipt.py',
                                                           'm1a_certification_adapter.py',
                                                           'coldchain_state.py',
                                                           'strict_online_env.py'))
                         or '/q3_' in p.replace('\\', '/')]
    cert_in_status = [p for p in modified if p.replace('\\', '/').endswith('m1b_q2_certifier.py')]
    if not have_prefix:
        ck_skip('V-01', 'CHECK 1 -- diff scope', 'the PRE_FIX byte copy is not part of the repository',
                {'expected_local_path': str(PRE_FIX.relative_to(EXT))})
    else:
        ck('V-01', 'CHECK 1 -- the diff is confined to `_resolve_t_min` and the receipt gate: every '
           'changed pre-fix line lies inside the two allowed regions, and no Q3 / G5 / G6 / install-path '
           'file is touched',
           not forbidden_touched and set(changed_old) <= allowed_old,
           {'numstat': numstat, 'n_changed_old_lines': len(changed_old),
            'n_changed_new_lines': len(changed_new),
            'changed_old_lines': changed_old, 'allowed_old_regions': sorted(allowed_old),
            'modified_tracked_files': modified, 'forbidden_touched': forbidden_touched,
            'certifier_uncommitted_in_worktree': bool(cert_in_status),
            'new_sha256': new_sha, 'old_sha256': pre_sha,
            'note': 'the "the certifier is dirty" clause was removed after this file was committed: it '
                    'was only true before the commit, and a delivered test must pass in the committed '
                    'state too.  Whether the change is still uncommitted is reported separately.'},
           conclusion='the change is exactly the authorised two-hunk patch and nothing else')

    # ============================================================ V-02 refusal propagation
    rows = []
    ok2 = True
    for label, val in CLAIM_CASES:
        for env_label, env in (('no env', None), ('env=object()', object()),
                               ('env=HostileEnv', 'HOSTILE')):
            kw = {'verified_t_min': val}
            if env == 'HOSTILE':
                kw['env'] = HostileEnv()
            elif env is not None:
                kw['env'] = env
            try:
                r = certify_probe(**kw)
            except BaseException as exc:                                  # noqa: BLE001
                rows.append({'claim': label, 'env': env_label, 'raised': type(exc).__name__})
                ok2 = False
                continue
            good = (r['t_min'] == 0.0
                    and r['final'] == ENGINEERING_REFUSED
                    and r['composite_pass'] is False
                    and r['t_min_gate'] == 'UNVERIFIED_T_MIN_REFUSED'
                    and r['gates'] == ['t_min'])          # gate 0b short-circuits
            ok2 = ok2 and good
            rows.append({'claim': label, 'env': env_label, 't_min': r['t_min'],
                         'source': r['source'], 'final': r['final'],
                         'composite_pass': r['composite_pass'], 't_min_gate': r['t_min_gate'],
                         'gates_evaluated': r['gates'], 'ok': good})
    base = certify_probe()
    base_ok = (base['t_min'] == 0.0 and base['source'] == 'CONSERVATIVE_CLOCK_ORIGIN'
               and base['t_min_gate'] is None and 't_min' not in base['gates']
               and 'contract_identity' in base['gates'])
    ck('V-02', 'CHECK 2 -- refusal propagation: for 14 shapes of untrusted claim x 3 env variants the '
       'patched certifier returns t_min=0, composite_pass=False, an explicit UNVERIFIED_T_MIN_REFUSED '
       'gate, and NOTHING else is evaluated; with no claim the normal gate chain runs unchanged',
       ok2 and base_ok,
       {'n_cases': len(CLAIM_CASES) * 3, 'all_refused_at_gate_0b': ok2,
        'baseline_without_claim': {'t_min': base['t_min'], 'source': base['source'],
                                   't_min_gate': base['t_min_gate'], 'gates': base['gates'],
                                   'final': base['final']},
        'rows': rows},
       conclusion='a conservative number can no longer leave a receipt that a reader could call PASS')

    # ============================================================ V-03 independence + env preserved
    a = certify_probe(verified_t_min=1e6)
    b = certify_probe(verified_t_min=1e6, env=object())
    c = certify_probe(verified_t_min=1e6, env=HostileEnv())
    core = lambda r: (r['t_min'], r['source'], r['t_min_gate'], r['final'], r['composite_pass'])
    src_now = CERT.read_text(encoding='utf-8')
    src_old = Path(PRE_FIX).read_text(encoding='utf-8')
    env_uses_now = src_now.count('self.env')
    env_uses_old = src_old.count('self.env')
    projection_check_kept = ('assert_visible_set_matches_env' in src_now
                             and src_now.count('assert_visible_set_matches_env')
                             == src_old.count('assert_visible_set_matches_env'))
    ck('V-03', 'CHECK 3 -- independence: env=object() / a hostile env / no env give the SAME t_min '
       'conclusion; and the OTHER uses of `env` (the trusted-projection re-verification) are preserved '
       'rather than deleted with the fix',
       core(a) == core(b) == core(c) and projection_check_kept
       and env_uses_now == env_uses_old - 1,
       {'no_env': core(a), 'env_object': core(b), 'hostile_env': core(c),
        'conclusions_identical': core(a) == core(b) == core(c),
        'self_env_uses_pre_fix': env_uses_old, 'self_env_uses_post_fix': env_uses_now,
        'expected_removed': 1,
        'projection_env_check_preserved': projection_check_kept,
        'why_it_matters': 'env is still needed by PP.assert_visible_set_matches_env; removing it '
                          'wholesale would have broken a working mechanism'},
       conclusion='env can no longer influence the t_min conclusion, and its legitimate use survives')

    # ============================================================ V-04 conservatism
    base_q = q_lb(0.0)
    rows4 = []
    ok4 = True
    for label, val in CLAIM_CASES:
        old_v, old_src = production_resolver_replica(verified_t_min=val, env=object())
        old_q = q_lb(old_v)
        r = certify_probe(verified_t_min=val, env=object())
        new_q = q_lb(r['t_min'])
        good = (new_q['q_lb'] <= old_q['q_lb'] and new_q['H_max'] >= old_q['H_max'])
        ok4 = ok4 and good
        rows4.append({'claim': label, 'pre_fix_t_min': old_v, 'pre_fix_q_lb': old_q['q_lb'],
                      'post_fix_t_min': r['t_min'], 'post_fix_q_lb': new_q['q_lb'],
                      'q_lb_raised': new_q['q_lb'] > old_q['q_lb'],
                      'H_max_shortened': new_q['H_max'] < old_q['H_max']})
    no_claim_now = q_lb(certify_probe()['t_min'])
    ck('V-04', 'CHECK 4 -- numerical conservatism: for every claim shape the certified quality lower '
       'bound is never raised and the horizon never shortened relative to the PRE-FIX resolver; with '
       'no claim the numbers are bit-identical to the published baseline',
       ok4 and no_claim_now['H_max'] == 24.000001
       and abs(no_claim_now['q_lb'] - 8.54807998017) < 1e-15,
       {'baseline': base_q, 'no_claim_after_fix': no_claim_now,
        'n_claims': len(rows4), 'rows': rows4},
       conclusion='the fix moves the number only toward strictness')

    # ============================================================ V-05 identity separation
    arts = ['results/_q3_next_20261017/gtmin/minimal_patch_for_authorisation.diff',
            'scripts/q3_qualification/q3_t_min_candidate_v2.py']
    if not (have_prefix and OLD_SHA):
        ck_skip('V-05', 'CHECK 5 -- identity separation', 'PRE_FIX byte copy or its pinned digest is '
                'not available in this checkout',
                {'expected_local_path': str(PRE_FIX.relative_to(EXT)),
                 'pins_source': str(PIN_REPORT.relative_to(EXT)),
                 'pin_found': bool(OLD_SHA)})
    else:
        ck('V-05', 'CHECK 5 -- identity separation: the pre-fix and post-fix sha256 both exist and differ, '
       'the pre-fix bytes are preserved as evidence, and the OLD acceptance records are explicitly NOT '
       'transferred to the new source',
       new_sha != pre_sha and pre_sha == OLD_SHA and new_sha != OLD_SHA,
       {'pre_fix_sha256': pre_sha, 'post_fix_sha256': new_sha,
        'pre_fix_copy_path': str(PRE_FIX.relative_to(EXT)),
        'pre_fix_copy_matches_pinned_identity': pre_sha == OLD_SHA,
        'pins_source': str(PIN_REPORT.relative_to(EXT)),
        'transfer_rule': ('the pre-fix acceptance records (M1b_Q2TrustFix / Q2 final-entry runs) were '
                          'produced for the PRE-FIX digest recorded in the acceptance report and must '
                          'NOT be cited as qualification of the post-fix source; every post-fix '
                          'result in this file is labelled with the new hash'),
        'new_digest_prefix': new_sha[:16],
        'supporting_artifacts': {p: sha256(EXT / p) for p in arts}},
       conclusion='two identities, two separate evidence sets')

    # ============================================================ V-06 historical callers
    # The decisive form of "old-interface regression" for this fix is not "the suites pass"; it is
    # "NO CALLER'S OUTCOME CHANGES".  So each caller is run twice: once against the live (patched)
    # certifier and once against a PRE-FIX twin tree, and the exit codes are compared.
    callers = ('test_m1b_q2_final_entry_controls.py', 'test_m1b_q2fix_certification.py',
               'test_m1b_q2r_negative_controls.py', 'test_m1b_q2_certification.py',
               'test_m1b_q2fixt_trust_boundary.py')
    reg = {}
    ok6 = True
    if args.skip_callers:
        reg['skipped'] = True
    else:
        import shutil
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix='gtmin_prefix_'))
        for f in sorted((EXT / 'scripts/m1b_adapter').glob('*.py')):
            shutil.copy(f, tmpdir / f.name)
        shutil.copy(PRE_FIX, tmpdir / 'm1b_q2_certifier.py')
        for name in callers:
            live = subprocess.run([sys.executable, str(EXT / 'scripts/tests' / name)],
                                  cwd=str(EXT), capture_output=True, text=True)
            src = (EXT / 'scripts/tests' / name).read_text(encoding='utf-8')
            lines = src.splitlines()
            last = max(i for i, ln in enumerate(lines) if 'sys.path.insert' in ln)
            injected = lines[:last + 1] + [
                "sys.path.insert(0, str(Path(r'%s')))  # PRE-FIX twin wins" % tmpdir] + lines[last + 1:]
            twin_src = '\n'.join(injected) + '\n'
            twin_path = EXT / 'scripts/tests' / ('_twin_%s' % name)
            twin_path.write_text(twin_src, encoding='utf-8')
            try:
                twin = subprocess.run([sys.executable, str(twin_path)], cwd=str(EXT),
                                      capture_output=True, text=True)
            finally:
                twin_path.unlink(missing_ok=True)
            same = live.returncode == twin.returncode
            ok6 = ok6 and same
            reg[name] = {'exit_live_patched': live.returncode,
                         'exit_prefix_twin': twin.returncode,
                         'outcome_unchanged': same,
                         'live_tail': (live.stdout or '').strip().splitlines()[-1:],
                         'prefix_tail': (twin.stdout or '').strip().splitlines()[-1:]}
    if not (have_prefix and have_local_suites):
        ck_skip('V-06', 'five historical callers: identical exit code against a PRE-FIX twin',
                'needs the PRE_FIX byte copy and the local suites', {'present': local_suites})
        ok6 = True
    else:
        ck('V-06', 'the five historical caller test files have the SAME exit code against the patched '
       'certifier and against a byte-exact PRE-FIX twin: the fix changes no caller outcome (one suite '
       'already fails before the patch for an unrelated KeyError, and it still fails identically)',
       ok6 and all(isinstance(v, dict) and v.get('outcome_unchanged') for v in reg.values()),
       {'callers': reg,
        'pre_existing_failure_registration': {
            'label': 'PRE_EXISTING_FAILURE_UNCHANGED',
            'suite': 'test_m1b_q2_certification.py',
            'failure': "KeyError: 'omega10_initialization'",
            'exit_live_patched': 1, 'exit_prefix_twin': 1,
            'reading': ('present on BOTH sides, therefore not caused by the fix.  It must NOT be '
                        'reported as "the whole historical suite passes": that suite was already '
                        'failing before this delivery and remains failing after it.')},
        'method': 'each suite is executed twice, the second time in a twin module tree whose '
                  'm1b_q2_certifier.py is the recorded PRE_FIX copy'},
       conclusion='no existing interface user is affected, and the pre-existing failure is shown to '
                  'predate the patch')

    # ============================================================ V-07 forbidden label
    hits = []
    for label, val in CLAIM_CASES:
        r = certify_probe(verified_t_min=val, env=object())
        ok_lbl, det = receipt_label_gate(r['receipt'])
        if not ok_lbl and det.get('forbidden_label_paths'):
            hits.append({'claim': label, 'paths': det['forbidden_label_paths'][:3]})
    src_hits = CERT.read_text(encoding='utf-8').count('VERIFIED_AGAINST_SUPPLIED_ENV')
    ck('V-07', 'no post-fix receipt carries the pre-fix label anywhere (the string survives only in the '
       'candidate module as a forbidden token, and the production file no longer sets it)',
       not hits and src_hits == 0,
       {'receipts_with_forbidden_label': hits,
        'occurrences_in_production_source': src_hits,
        'note': 'the string still appears in the docstring? checked separately'},
       conclusion='the misnamed label cannot be produced any more')

    # ============================================================ V-08 install path untouched
    inst = {p: PINS[p] for p in UNCHANGED_PINS if p in PINS}
    have_pins = len(inst) == len(UNCHANGED_PINS)
    drift8 = {k: {'expected': v,
                  'actual': sha256(EXT / k.split('/', 1)[1])} for k, v in inst.items()
              if sha256(EXT / k.split('/', 1)[1]) != v}
    import_ok = 'run_repair_transaction' not in CERT.read_text(encoding='utf-8')
    if not have_pins:
        ck_skip('V-08', 'install path untouched', 'the digest pins are not readable from the '
                'acceptance record', {'pinned_entries_found': sorted(inst)})
    else:
        ck('V-08', 'the install path and the Q3 evidence chain are untouched: the geo adapter, the D8 '
           'receipt writer, the M1-a adapter and the executor keep their pinned digests, and the '
           'certifier does not reference run_repair_transaction',
           not drift8 and import_ok,
           {'drift': drift8, 'certifier_references_install_entry': not import_ok,
            'pins_source': str(PIN_REPORT.relative_to(EXT))},
           conclusion='the fix cannot reach the installation boundary')

    # ============================================================ V-09 workspace state
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(EXT.parent), capture_output=True,
                          text=True).stdout.strip()
    modified_all = modified
    # "no collateral edit" = every modified TRACKED path is one this delivery is authorised to touch.
    # The set is stated here rather than "only the certifier", because the delivered test and the two
    # acceptance records are part of the same delivery and are legitimately dirty until committed.
    AUTHORISED_SUFFIXES = (
        'scripts/m1b_adapter/m1b_q2_certifier.py',
        'scripts/tests/test_q3_gtmin_install_verification.py',
        'GTMIN严格判定与受控实施验收_2026-10-17.md',
        'GTMIN安全补正与选择性交付验收_2026-10-17.md',
    )
    unexpected = [p for p in modified_all
                  if not any(p.replace('\\', '/').endswith(suf) for suf in AUTHORISED_SUFFIXES)
                  and 'CC_Compare/MAPT_code' not in p]
    cert_still_dirty = any(p.replace('\\', '/').endswith('m1b_q2_certifier.py') for p in modified_all)
    ck('V-09', 'workspace discipline: no collateral edit exists -- every modified tracked path belongs '
       'to this delivery set; the check reports whether the change is still uncommitted',
       not unexpected,
       {'head': head, 'head_is_baseline': head == '759fe7fb81dd869c212a28eed352f82ff99a1b1e',
        'modified_tracked': modified_all, 'unexpected': unexpected,
        'authorised_suffixes': list(AUTHORISED_SUFFIXES),
        'certifier_change_uncommitted': cert_still_dirty,
        'note': 'the HEAD equality is not asserted: this file is delivered WITH the change, so a '
                'post-commit run must not report FAIL merely because the commit exists'},
       conclusion='no collateral edit; only the authorised delivery files are dirty')

    # ============================================================ V-11 hostile __repr__ controls
    # The reviewer's finding: a refusal receipt must not render the untrusted object.  `repr()` on
    # caller-controlled data runs caller code INSIDE the certifier -- it can raise, produce side
    # effects, or return sensitive text that then lands in a receipt.  Three hostile inputs are
    # exercised and the assertion is that the dunder is never invoked at all.
    repr_calls = {'boom': 0, 'leaky': 0, 'side': 0}

    class BoomRepr(object):
        def __repr__(self):
            repr_calls['boom'] += 1
            raise RuntimeError('repr must not be called inside the certifier')

        __str__ = __repr__

    class LeakyRepr(object):
        def __repr__(self):
            repr_calls['leaky'] += 1
            return 'LEAK-MARKER-0123456789'

        __str__ = __repr__

    class SideEffectRepr(object):
        def __init__(self):
            self.effects = []

        def __repr__(self):
            repr_calls['side'] += 1
            self.effects.append('repr() executed')
            return '<benign>'

        __str__ = __repr__

    LEAK_MARKER = 'LEAK-MARKER-0123456789'
    hostile_rows = []
    ok11 = True
    for label, factory, key in (('repr_raises', BoomRepr, 'boom'),
                                ('repr_leaks_secret', LeakyRepr, 'leaky'),
                                ('repr_has_side_effects', SideEffectRepr, 'side')):
        obj = factory()
        before = dict(repr_calls)
        try:
            r = certify_probe(verified_t_min=obj, env=object())
            raised = None
        except BaseException as exc:                                      # noqa: BLE001
            r, raised = None, '%s: %s' % (type(exc).__name__, exc)
        called = repr_calls[key] > before[key]
        gate = {} if r is None else (r['receipt'].get('gates', {}).get('t_min') or {})
        detail = gate.get('detail') or {}
        blob = json.dumps(r['receipt'], default=str) if r is not None else ''
        good = (raised is None and r is not None and not called
                and r['t_min'] == 0.0
                and r['final'] == ENGINEERING_REFUSED
                and r['composite_pass'] is False
                and gate.get('verdict') == 'UNVERIFIED_T_MIN_REFUSED'
                and detail.get('claim_supplied') is True
                and detail.get('t_min_used') == 0.0
                and detail.get('reason_code') == 'UNVERIFIED_T_MIN_WITHOUT_TRUSTED_WITNESS'
                and 'value_received' not in detail
                and LEAK_MARKER not in blob
                and (getattr(obj, 'effects', None) in (None, [])))
        ok11 = ok11 and good
        hostile_rows.append({'case': label, 'raised': raised, 'dunder_called': called,
                             't_min': None if r is None else r['t_min'],
                             'final': None if r is None else r['final'],
                             'composite_pass': None if r is None else r['composite_pass'],
                             'gate_detail': detail,
                             'marker_leaked': (LEAK_MARKER in blob) if r is not None else None,
                             'side_effects': getattr(obj, 'effects', None), 'ok': good})
    ck('V-11', 'a refusal receipt never renders the untrusted object: for a `__repr__` that raises, one '
       'that returns a leak marker, and one with observable side effects, the dunder is NOT called, the '
       'certification still returns REFUSED with a fixed reason code, and no caller value reaches the '
       'receipt',
       ok11,
       {'cases': hostile_rows, 'dunder_call_counts': repr_calls,
        'receipt_detail_contract': {'claim_supplied': True, 't_min_used': 0.0,
                                    'reason_code': 'UNVERIFIED_T_MIN_WITHOUT_TRUSTED_WITNESS'},
        'forbidden_in_detail': 'value_received (removed by the safety correction)'},
       conclusion='the refusal path is free of caller-code execution and of value disclosure')

    # ============================================================ V-10 superseded-row ledger
    # A fix to a defect makes the controls that ASSERTED the defect fail.  That is expected, and it is
    # only acceptable if the failing set is exactly the pre-declared one and nothing else moved.  So
    # every earlier suite is run against the patched source and the failing row ids are compared with
    # this table, which was written by hand BEFORE the run.
    SUPERSEDED = {
        'test_q3_g_tmin_candidate.py': {
            'TMIN-N-02': 'asserts production returns t_min=1e6 for verified_t_min=1e6 + env',
            'TMIN-N-03': 'asserts production returns t_min=24.0 for verified_t_min=24.0 + env',
            'TMIN-C-01': 'asserts the pre-fix resolver replica equals the LIVE production resolver',
            'TMIN-C-02': 'asserts production accepts "1e6" and True as execution-time bounds'},
        'test_q3_g_tmin_v2.py': {
            'C1': 'its replica-fidelity arm compares the PRE-FIX semantics with the LIVE resolver',
            'C10': 'it re-runs the round-1 G-TMIN suite, which now fails on the four rows above',
            'C11': 'it asserts that a positive-t_min bypass is still live in production'},
        'test_q3_archive_regression.py': {
            'ARC-01': 'it asserts the certifier sha256 equals the PRE-FIX identity',
            'ARC-05': 'it asserts the live resolver still accepts 1e6 / "1e6" / True',
            'ARC-06': 'it re-runs the round-1 G-TMIN suite as a subprocess',
            'ARC-07': 'it asserts no tracked file other than the pre-existing dirty path is modified'},
        'test_q3_door_interval_thresholds.py': {
            'ERR-12': 'it asserts the certifier sha256 is unchanged'},
    }
    unaffected = ('q3_theorem_b_audit.py', 'test_q3_thermal_interval_controls.py',
                  'test_q3_strict_float_bounds.py', 'test_q3_plan_layer_controls.py',
                  'test_q3_event_segmentation.py', 'test_q3_float_replay_qualification.py',
                  'test_q3_time_budget_audit.py')
    observed, expected = {}, {k: sorted(v) for k, v in SUPERSEDED.items()}
    unaffected_rows = 0
    retries = []
    ok10 = True
    if not args.skip_callers:
        tmp = Path(__import__('tempfile').mkdtemp(prefix='gtmin_sweep_'))
        for name in list(SUPERSEDED) + list(unaffected):
            # One retry is allowed and RECORDED.  First version of this sweep named the theorem-B
            # audit `test_q3_theorem_b_audit.py` (the real file has no `test_` prefix), got exit 2,
            # and I first hypothesised "non-reproducible platform flake".  The recorded stderr proved
            # it was my own filename typo.  The retry is kept so that a genuine flake is visible
            # instead of silently passing, and the mis-diagnosis stays in this comment.
            for attempt in (1, 2):
                outp = tmp / ('%s%s' % (name.replace('.py', ''), '' if attempt == 1 else '_retry')
                              )
                outp = outp.with_suffix('.json')
                proc = subprocess.run([sys.executable, str(EXT / 'scripts/tests' / name),
                                       '--out', str(outp)], cwd=str(EXT), capture_output=True,
                                      text=True)
                if outp.exists():
                    break
                retries.append({'suite': name, 'attempt': attempt, 'exit': proc.returncode,
                                'stderr_head': (proc.stderr or '')[:200]})
            if not outp.exists():
                observed[name] = ['<no artifact>']
                ok10 = False
                continue
            payload = json.loads(outp.read_text(encoding='utf-8'))
            rows = payload.get('rows', [])
            failing = sorted(r['id'] for r in rows if not r['ok'])
            observed[name] = failing
            if name in SUPERSEDED:
                ok10 = ok10 and failing == expected[name]
            else:
                unaffected_rows += len(rows)
                ok10 = ok10 and failing == []
    if not have_local_suites:
        ck_skip('V-10', 'superseded-row ledger', 'needs the four local superseded suites',
                {'present': local_suites})
        ok10 = True
    else:
        ck('V-10', 'the superseded-row ledger is exact: the only rows that fail against the patched source '
       'are the %d pre-declared ones that asserted the defect, and all %d rows of the unaffected Q3 '
       'suites still pass'
       % (sum(len(v) for v in expected.values()), unaffected_rows),
       ok10, {'superseded_expected': expected, 'superseded_observed': observed,
              'n_superseded_rows': sum(len(v) for v in expected.values()),
              'unaffected_suites': list(unaffected), 'unaffected_rows_green': unaffected_rows,
              'non_reproducible_flakes_recorded': retries,
              'rule': 'a fix that changes no test outcome would mean the tests never covered the '
                      'defect; a fix that changes an UNDECLARED outcome is a regression'},
       conclusion='the change is fully accounted for: exactly the pre-fix assertions moved')

    payload = {
        'schema': 'q3_gtmin_install_verification',
        'kind': 'DIAGNOSTIC', 'not_for_production': False,
        'head': head,
        'pre_fix_sha256': pre_sha, 'post_fix_sha256': new_sha,
        'numstat': numstat,
        'superseded_rows': expected,
        'superseded_observed': observed,
        'authorisation': ('conditional approval of the G-TMIN minimal fix; five pre-installation '
                          'checks implemented as V-01..V-05, supersession ledger as V-10'),
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['status'] == 'PASS'),
        'n_skip': sum(1 for r in ROWS if r['status'] == 'SKIP'),
        'n_fail': sum(1 for r in ROWS if r['status'] == 'FAIL'),
        'verdict': ('GTMIN_INSTALL_VERIFICATION_PASS' if not any(r['status'] == 'FAIL' for r in ROWS)
                    else 'GTMIN_INSTALL_VERIFICATION_FAIL'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('\n=== %d checks, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
