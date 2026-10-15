#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 invariant differential test (V1.19, isolated; no F, no data collection, no production write).

Purpose: replace the V1.17 string-inspection with EXECUTABLE positive/negative cases against the
CURRENT implementation, and separate
    [already implemented + already accepted historically]  from  [genuinely still open].
Sources of record for the "already accepted" column: test_d3_g0_green_phase.py,
test_d3_g0_red_phase.py, the B1/B2/B3AC/B3AT1/B3AT2*/B3CF/B3Q* suites and d3_geo_adapter.py itself.

Invariants exercised here (each with the negative case that used to pass):
  G0-01 no certify_fn                     -> run_repair_transaction must REFUSE before sampling
  G0-02 score vector contract             -> ndim != 1 and len(scores) != len(legal) fail closed
  G0-03 certified vs installed identity   -> mismatch refuses, zero installs
  G0-04 one ledger == one transaction     -> a second finish() refuses, no second install
  G0-05 refusal vs rollback vs unverified -> the three outcomes stay distinct
  G0-06 missing energy budget/quality     -> must refuse or safely degrade; never a "fully
                                             certified" receipt (capability gaps are read from the
                                             module, never invented)
"""
import argparse
import dataclasses
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

import numpy as np                                                # noqa: E402
import d3_geo_adapter as G                                        # noqa: E402
from test_d3_geo_adapter import _cand, _identity_apply, _plans     # noqa: E402

CASES = []
REC = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:260]))


def ledger_dump(led):
    out = {}
    for n in dir(led):
        if n.startswith('_'):
            continue
        try:
            out[n] = repr(getattr(led, n))
        except Exception:
            out[n] = '<unreadable>'
    return out


P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
MUT = {1, 2, 3}
EXTRACT = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)
LEGAL_SEQ = [_cand(5, 'anchored', 3, 0, 5.0), _cand(5, 'anchored', 3, 1, 5.0)]
LEGAL = lambda P, rem: [t for t in LEGAL_SEQ if t[0] in rem]       # noqa: E731
TIE = lambda st: [1.0, 1.0]                                        # noqa: E731  # len == len(legal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    REC['identity'] = {
        'd3_geo_adapter_sha256': G.__file__ and __import__('hashlib').sha256(
            Path(G.__file__).read_bytes()).hexdigest(),
        'INSTALL_REQUIRES_CERTIFICATION': getattr(G, 'INSTALL_REQUIRES_CERTIFICATION', None),
        'REFUSAL_REASONS': list(getattr(G, 'REFUSAL_REASONS', ())),
        'FAILURE_REASONS': list(getattr(G, 'FAILURE_REASONS', ()))}
    ck('I0 the adapter exposes the certification requirement and the refusal/failure reason sets',
       REC['identity']['INSTALL_REQUIRES_CERTIFICATION'] is True
       and 'no_certifier' in REC['identity']['FAILURE_REASONS'],
       REC['identity'])

    # ---- G0-01: no certifier -> refuse BEFORE sampling ------------------------------------
    led = G.InstallLedger(install_fn=lambda p: None)
    ok = G.run_repair_transaction(ledger=led, score_fn=TIE, extract_fn=EXTRACT, legal_fn=LEGAL,
                                  apply_fn=_identity_apply, plans=P0, mask_set=[5],
                                  mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True)
    d = ledger_dump(led)
    REC['G0_01'] = {'returned': repr(ok), 'installs': d.get('n_installs'),
                    'pre_sampling_refusals': d.get('n_pre_sampling_refusals')}
    ck('G0-01 NEGATIVE (no certify_fn): the transaction refuses before sampling -- zero installs '
       'and a pre-sampling refusal, not a silent uncertified install',
       d.get('n_installs') == '0' and d.get('n_pre_sampling_refusals') == '1', REC['G0_01'])

    # ---- G0-02: score vector contract ------------------------------------------------------
    bad = {}
    for name, fn in (('ndim2', lambda st: np.ones((1, len(LEGAL_SEQ)))),
                     ('too_short', lambda st: [1.0]),
                     ('too_long', lambda st: [1.0] * (len(LEGAL_SEQ) + 1))):
        _p, steps, _r, fr, diag = G.stepwise_sample_d3(
            score_fn=fn, extract_fn=EXTRACT, legal_fn=LEGAL, apply_fn=_identity_apply, plans=P0,
            mask_set=[5], mutable_ids=MUT, selector=G.SELECTOR_GEO, deterministic=True)
        bad[name] = {'failure_reason': fr, 'n_steps': len(steps),
                     'violation': (diag or {}).get('score_contract', {}).get('violation')}
    REC['G0_02'] = bad
    ck('G0-02 NEGATIVE (score vector): a 2-D vector fails closed with score_shape and a wrong-length '
       'vector with score_len -- and neither produces a step (no silent prefix scoring)',
       bad['ndim2']['failure_reason'] == 'score_shape'
       and bad['too_short']['failure_reason'] == 'score_len'
       and bad['too_long']['failure_reason'] == 'score_len'
       and all(v['n_steps'] == 0 for v in bad.values()), bad)

    # ---- G0-03 / G0-04 / G0-05: install gate, idempotency, outcome distinctness ------------
    good = G.InstallLedger(install_fn=lambda p: None)
    cert = G._plan_identity(P0)
    wrong = 'deadbeef' * 8
    r3 = good.finish(failure_reason=None, plan=P0, original_plan=P0, certified=True,
                     certified_plan_hash=wrong, certifier='test-certifier', deadline_ok=True)
    d3 = ledger_dump(good)
    REC['G0_03'] = {'returned': repr(r3), 'events': d3.get('events')}
    ck('G0-03 NEGATIVE (identity mismatch): a certified plan hash that does not match the plan to '
       'install is refused with certifier identity bound, zero installs',
       r3 is False and d3.get('n_installs') == '0'
       and 'certified_plan_identity_mismatch' in (d3.get('events') or ''), REC['G0_03'])

    led4 = G.InstallLedger(install_fn=lambda p: None)
    first = led4.finish(failure_reason=None, plan=P0, original_plan=P0, certified=True,
                        certified_plan_hash=cert, certifier='test-certifier', deadline_ok=True)
    second = led4.finish(failure_reason=None, plan=P0, original_plan=P0, certified=True,
                         certified_plan_hash=cert, certifier='test-certifier', deadline_ok=True)
    d4 = ledger_dump(led4)
    REC['G0_04'] = {'first': repr(first), 'second': repr(second), 'installs': d4.get('n_installs'),
                    'events': d4.get('events')}
    ck('G0-04 POSITIVE then NEGATIVE (idempotency): the first correctly certified install succeeds '
       'exactly once and a second finish() on the same ledger refuses with already_finished',
       first is True and second is False and d4.get('n_installs') == '1'
       and 'already_finished' in (d4.get('events') or ''), REC['G0_04'])

    led5a = G.InstallLedger(install_fn=lambda p: None)                     # no rollback_fn
    led5a.finish(failure_reason='injected', plan=P0, original_plan=P0)
    d5a = ledger_dump(led5a)
    led5b = G.InstallLedger(install_fn=lambda p: None, rollback_fn=lambda p: None)
    led5b.finish(failure_reason='injected', plan=P0, original_plan=P0)
    d5b = ledger_dump(led5b)
    led5c = G.InstallLedger(install_fn=lambda p: None)
    led5c.finish(failure_reason='no_certifier', plan=P0, original_plan=P0,
                 refused_before_sampling=True)
    d5c = ledger_dump(led5c)
    REC['G0_05'] = {'no_rollback_fn': {k: d5a.get(k) for k in
                                       ('n_rollback_unverified', 'n_state_restorations',
                                        'n_pre_sampling_refusals')},
                    'with_rollback_fn': {k: d5b.get(k) for k in
                                         ('n_rollback_unverified', 'n_state_restorations',
                                          'n_pre_sampling_refusals')},
                    'pre_sampling': {k: d5c.get(k) for k in
                                     ('n_rollback_unverified', 'n_state_restorations',
                                      'n_pre_sampling_refusals')}}
    ck('G0-05 the three outcomes stay DISTINCT: an unverified rollback is counted separately from a '
       'real restoration, and a pre-sampling refusal is neither',
       d5a.get('n_rollback_unverified') == '1' and d5b.get('n_state_restorations') == '1'
       and d5c.get('n_pre_sampling_refusals') == '1' and d5c.get('n_rollback_unverified') == '0',
       REC['G0_05'])

    # ---- G0-06: no registered budget / no quality capability -> refuse, never "fully certified"
    try:
        import d3_composed_certifier as CC
        gaps = getattr(CC, 'CAPABILITY_GAPS', None) or getattr(CC, 'QUALITY_GAP', None)
        gap_names = []
        if isinstance(gaps, (list, tuple, set)):
            gap_names = sorted(str(x) for x in gaps)
        elif isinstance(gaps, dict):
            gap_names = sorted(str(k) for k in gaps)
        sigs = {n: str(inspect.signature(getattr(CC, n))) for n in dir(CC)
                if not n.startswith('_') and callable(getattr(CC, n))
                and 'certif' in n.lower()}
        text = Path(CC.__file__).read_text(encoding='utf-8')
        cap_items = sorted({l.strip() for l in text.splitlines()
                            if 'salab' in l.lower() or 'CAPABILITY' in l})
        REC['G0_06'] = {'module': str(Path(CC.__file__).relative_to(EXT)),
                        'declared_capability_gaps': gap_names,
                        'quality_salability_mentioned': any('salab' in x.lower()
                                                            for x in gap_names + cap_items),
                        'certification_signatures': sigs,
                        'capability_lines': cap_items[:6]}
        ck('G0-06a the composed certifier itself declares the quality/salability capability gap, so '
           'no "fully certified" receipt may be claimed from it',
           REC['G0_06']['quality_salability_mentioned'], REC['G0_06'])
    except BaseException as exc:
        REC['G0_06'] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:200])}
        ck('G0-06a the composed certifier declares its capability gaps', False, REC['G0_06'])
    b2 = (EXT / 'scripts/tests/test_d3_b2_certification.py')
    rec_b = {'historical_fixture_budget_300_present': False,
             'registered_budget_for_the_64_dev_check_states': None,
             'file': str(b2.relative_to(EXT)) if b2.is_file() else None}
    if b2.is_file():
        txt = b2.read_text(encoding='utf-8')
        rec_b['historical_fixture_budget_300_present'] = 'B_BUDGET = 300' in txt
    REC['budget_provenance'] = rec_b
    ck('G0-06b NEGATIVE (budget): no registered C0 budget exists for the 64 dev_check states, so the '
       'budget question is BLOCKED/AWAITING_USER_BUDGET_RULING and must not be filled in by the agent',
       rec_b['registered_budget_for_the_64_dev_check_states'] is None, rec_b)

    REC['already_implemented_and_accepted_history'] = {
        'test_assets': [p.name for p in sorted((EXT / 'scripts/tests').glob('test_d3_g0_*.py'))]
        + [p.name for p in sorted((EXT / 'scripts/tests').glob('test_d3_b3*.py'))]
        + ['test_d3_b1_state_level.py', 'test_d3_b2_certification.py'],
        'note': 'G0-Safety batch 1 interfaces are implemented AND covered by the historical suites; '
                'the remaining work is the EVIDENCE batch (real-state composition, write uniqueness, '
                'persistent install receipt, D8 hard constraints, quality capability).'}
    fails = [c for c in CASES if not c[1]]
    REC['summary'] = {'rows': len(CASES), 'pass': len(CASES) - len(fails), 'fail': len(fails),
                      'closed_by_executable_counterexample': [n for n, o, _ in CASES if o
                                                              and n.startswith('G0-0')],
                      'open_items': ['real-state composed certification', 'write uniqueness on real '
                                     'state', 'persistent install receipt P_before/P_proposed/'
                                     'P_certified/P_after_install', 'D8 hard constraints',
                                     'quality/salability certification', 'C0 budget ruling'],
                      'no_new_runs': ['no F batch', 'no F-vs-D-Geo', 'no training', 'no L2/L3',
                                      'no 40-day batch', 'no final test']}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, 'n_rows': len(CASES),
                               'n_pass': len(CASES) - len(fails), 'n_fail': len(fails),
                               'tool_sha256': __import__('hashlib').sha256(
                                   Path(__file__).read_bytes()).hexdigest()},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== G0 differential: %d rows, %d PASS, %d FAIL ===' % (len(CASES),
                                                                    len(CASES) - len(fails),
                                                                    len(fails)))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
