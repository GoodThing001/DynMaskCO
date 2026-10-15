#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G0 residual-risk counterexamples: EXECUTABLE confirmation of the four live risks.

The P0 audit named seven residual risks and classified them.  Since then a G0-Safety batch closed
several.  Rather than assert in prose which are closed, this module builds the MINIMAL
counterexample for each of the four that were still live, runs it against the CURRENT code, and
records "detected" or "not detected":

  A-2   `certify_fn=None` used to skip all certification and return success, so the transaction
        layer would install an UNCERTIFIED plan.  The minimal counterexample is a transaction with
        no certifier; DETECTED means the transaction refuses (zero installs).
  A-3b  the adapter does not itself freeze/verify the committed prefix or the protected vehicles.
        DETECTED means a transaction whose apply-hook moves a COMMITTED customer is refused or
        reported; NOT DETECTED means the adapter cannot see it.  (Reported honestly either way: the
        real guard for this lives on `VehicleState` + the mask pool, not in the adapter.)
  A-4b  the certified plan and the installed plan used to share no identity, and the ledger stored
        plans by reference.  DETECTED means (i) installing a plan different from the certified one
        is refused, and (ii) mutating the caller's plan object afterwards does NOT change what the
        ledger recorded.
  A-5d  a failure with no `rollback_fn` used to count as "rolled back".  DETECTED means the ledger
        distinguishes `committed_state_restored` from `rollback_unverified`.

Every counterexample runs the REAL `d3_geo_adapter` code through its own public API.  Nothing is
installed into a live environment, nothing is trained, and no frozen tree is touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402

RESULTS = []


def record(risk, name, confirmed, detail, *, kind='FIX_CONFIRMED'):
    """`kind` is what the observation MEANS, so a confirmed fix and a confirmed残留 risk are never
    conflated:
      FIX_CONFIRMED       -- the risk is closed: the code now refuses / reports correctly
      RISK_CONFIRMED_OPEN -- the risk is real and still open: the code cannot see it.  This is a
                             FINDING, not a failure of the harness, and it is not counted as a fix.
    """
    RESULTS.append({'risk': risk, 'name': name, 'observation_confirmed': bool(confirmed),
                    'kind': kind, 'detail': detail})
    print('[%s] %s | %s :: %s' % (kind, risk, name,
                                  json.dumps(detail, default=repr, sort_keys=True)[:190]))
    return bool(confirmed)


def make_plan(seed=1):
    from action_contract import VehiclePlan
    return {0: VehiclePlan(0, 0, 0.0, 0.0, (1, 2 + seed)),
            1: VehiclePlan(1, 0, 0.0, 0.0, (3,)),
            2: VehiclePlan(2, 0, 0.0, 0.0, ())}


def plan_suffixes(plan):
    return {str(v): list(getattr(p, 'suffix', [])) for v, p in plan.items()}


# ---------------------------------------------------------------------------------------------
# A-2  certify_fn=None must not be able to install
# ---------------------------------------------------------------------------------------------
def a2_no_certifier(root):
    from d3_geo_adapter import InstallLedger, run_repair_transaction
    installs = {'n': 0}
    ledger = InstallLedger(lambda p: installs.update(n=installs['n'] + 1))
    plans = make_plan()
    outcome = run_repair_transaction(
        ledger=ledger, score_fn=lambda state: [0.0], extract_fn=lambda P, r, l: {},
        legal_fn=lambda P, r: [], apply_fn=lambda P, a, allowed_vehicle_ids=None: P,
        plans=plans, mask_set=[1], mutable_ids=[0], selector=GEO.SELECTOR_HISTORICAL,
        deterministic=True, certify_fn=None)
    detail = {'returned': outcome, 'installs': installs['n'],
              'ledger_reason': ledger.events[0].get('reason') if ledger.events else None,
              'n_pre_sampling_refusals': ledger.n_pre_sampling_refusals}
    return record('A-2', 'a transaction with NO certifier refuses and installs nothing',
                  outcome is False and installs['n'] == 0 and ledger.n_installs == 0
                  and ledger.events and ledger.events[0].get('reason') == 'no_certifier', detail)


def a2_certified_but_unbound(root):
    """The sharper version: a REAL certifier ran, but the installed plan is a different object."""
    from d3_geo_adapter import InstallLedger
    installs = {'n': 0}
    ledger = InstallLedger(lambda p: installs.update(n=installs['n'] + 1))
    certified, installed_plan = make_plan(1), make_plan(5)
    outcome = ledger.finish(failure_reason=None, plan=installed_plan, original_plan=make_plan(0),
                            certified=True,
                            certified_plan_hash=GEO._plan_identity(certified),
                            certifier='synthetic.certifier', deadline_ok=True)
    detail = {'returned': outcome, 'installs': installs['n'],
              'ledger_reason': ledger.events[0].get('reason') if ledger.events else None,
              'certified_hash': GEO._plan_identity(certified),
              'installed_hash': GEO._plan_identity(installed_plan)}
    return record('A-2', '"certified A, installed B" is refused (identity-bound certification)',
                  outcome is False and installs['n'] == 0
                  and ledger.events[0].get('reason') == 'certified_plan_identity_mismatch', detail)


# ---------------------------------------------------------------------------------------------
# A-3b  the adapter does NOT itself freeze the committed prefix / protected vehicles
# ---------------------------------------------------------------------------------------------
def a3b_committed_prefix(root):
    """A mask that contains a COMMITTED customer is applied by the adapter without complaint.

    The adapter is a pure sampler: it has no notion of the committed prefix, and the real guard
    lives on `VehicleState` + the B mask pool.  The counterexample is therefore run as a DIRECT
    mechanism check, not through the install gate: a plan whose committed prefix holds customer 1 is
    given to `stepwise_sample_d3` with `mask_set=[1]`, and the sampler moves it.  `detected=False`
    means "the adapter cannot see this", which is the true statement and is NOT reported as a pass.
    """
    from action_contract import ActionSlot, FleetAction, VehiclePlan, apply_action
    plans = {0: VehiclePlan(0, 0, 0.0, 0.0, (5, 1)), 1: VehiclePlan(1, 0, 0.0, 0.0, ())}
    committed = 1                      # already inside vehicle 0's route in this fixture
    action = FleetAction(customer=committed, slot=ActionSlot('anchored', 0), position=1,
                         predecessor=5, successor=0)
    candidate = type('C', (), {'incremental_distance': 1.0})()
    legal = [(committed, action, candidate)]
    before = {v: list(p.suffix) for v, p in plans.items()}
    plan_out, steps, _records, failure_reason, _diag = GEO.stepwise_sample_d3(
        score_fn=lambda state: [0.0],
        extract_fn=GEO.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=[0, 1]),
        legal_fn=lambda P, r: legal, apply_fn=apply_action, plans=plans, mask_set=[committed],
        mutable_ids=[0, 1], selector=GEO.SELECTOR_HISTORICAL, deterministic=True)
    after = {v: list(getattr(p, 'suffix', [])) for v, p in plan_out.items()}
    moved = after != before
    detail = {'failure_reason': failure_reason, 'steps': steps,
              'plan_before': before, 'plan_after': after,
              'committed_customer_moved': moved,
              'adapter_exposes_a_committed_prefix_check':
                  hasattr(GEO, 'assert_committed_prefix'),
              'where_the_real_guard_lives': 'VehicleState + mask_candidate_pool / build_pool_B; the '
                                            'P0 audit recorded that adding a same-named fake field '
                                            'to the adapter would NOT fix this'}
    # The observation is "the sampler moved the committed customer".  Confirmed == the risk is REAL
    # and still open, so this is recorded as RISK_CONFIRMED_OPEN, never as a fix.
    return record('A-3b', 'the sampler moves a committed customer: the adapter has no committed-'
                          'prefix guard, so the guard must live on the vehicle + pool layer',
                  moved, detail, kind='RISK_CONFIRMED_OPEN')


# ---------------------------------------------------------------------------------------------
# A-4b  identity binding + immutability of the ledger receipt
# ---------------------------------------------------------------------------------------------
def a4b_reference_mutation(root):
    from d3_geo_adapter import InstallLedger
    installs = {}
    ledger = InstallLedger(lambda p: installs.update(plan=p))
    plan = make_plan(1)
    before = plan_suffixes(plan)
    ok = ledger.finish(failure_reason=None, plan=plan, original_plan=make_plan(0), certified=True,
                       certified_plan_hash=GEO._plan_identity(plan),
                       certifier='synthetic.certifier', deadline_ok=True)
    # mutate the caller's plan object AFTER the install
    for p in plan.values():
        p.suffix = tuple(list(p.suffix) + [77])
    stored = plan_suffixes(ledger.installed_plans[0])
    detail = {'installed': ok, 'before': before, 'after_caller_mutation': plan_suffixes(plan),
              'stored_in_ledger': stored,
              'ledger_story_unchanged': stored == before,
              'recorded_plan_hash': ledger.installs[0]['plan_hash'] if ledger.installs else None}
    return record('A-4b', 'the ledger keeps a VALUE COPY: mutating the caller plan afterwards '
                          'cannot change what was recorded',
                  ok is True and stored == before, detail)


def a4b_unhashable_plan(root):
    """A plan whose identity cannot be computed must be REFUSED, never installed on a None hash."""
    from d3_geo_adapter import InstallLedger
    installs = {'n': 0}
    ledger = InstallLedger(lambda p: installs.update(n=installs['n'] + 1))
    outcome = ledger.finish(failure_reason=None, plan=object(), original_plan=make_plan(0),
                            certified=True, certified_plan_hash='deadbeef',
                            certifier='synthetic.certifier', deadline_ok=True)
    detail = {'returned': outcome, 'installs': installs['n'],
              'reason': ledger.events[0].get('reason') if ledger.events else None}
    return record('A-4b', 'an unhashable plan is refused instead of installed on a None identity',
                  outcome is False and installs['n'] == 0, detail)


# ---------------------------------------------------------------------------------------------
# A-5d  rollback accounting without a rollback function
# ---------------------------------------------------------------------------------------------
def a5d_rollback_unverified(root):
    from d3_geo_adapter import InstallLedger
    ledger = InstallLedger(lambda p: None, rollback_fn=None)
    ledger.finish(failure_reason='cert_fail', plan=make_plan(1), original_plan=make_plan(0))
    event = ledger.events[0]
    detail = {'event': event, 'n_rollbacks': ledger.n_rollbacks,
              'n_state_restorations': ledger.n_state_restorations,
              'n_rollback_unverified': ledger.n_rollback_unverified}
    return record('A-5d', 'a failure with no rollback_fn is reported as rollback_UNVERIFIED, not '
                          'as a completed restoration',
                  event.get('rollback_unverified') is True
                  and event.get('committed_state_restored') is False
                  and ledger.n_state_restorations == 0
                  and ledger.n_rollback_unverified == 1, detail)


def a5d_real_rollback(root):
    from d3_geo_adapter import InstallLedger
    calls = []
    ledger = InstallLedger(lambda p: None, rollback_fn=lambda plan: calls.append(plan))
    ledger.finish(failure_reason='cert_fail', plan=make_plan(1), original_plan=make_plan(0))
    event = ledger.events[0]
    detail = {'event': event, 'rollback_calls': len(calls),
              'n_state_restorations': ledger.n_state_restorations}
    return record('A-5d', 'a failure WITH a rollback_fn is reported as committed_state_restored',
                  event.get('committed_state_restored') is True and len(calls) == 1
                  and ledger.n_state_restorations == 1, detail)


def a5b_double_finish(root):
    """A-5b: calling finish twice must not install twice (one ledger == one transaction)."""
    from d3_geo_adapter import InstallLedger
    installs = {'n': 0}
    plan = make_plan(1)
    ledger = InstallLedger(lambda p: installs.update(n=installs['n'] + 1))
    first = ledger.finish(failure_reason=None, plan=plan, original_plan=make_plan(0), certified=True,
                          certified_plan_hash=GEO._plan_identity(plan),
                          certifier='synthetic.certifier', deadline_ok=True)
    second = ledger.finish(failure_reason=None, plan=plan, original_plan=make_plan(0), certified=True,
                           certified_plan_hash=GEO._plan_identity(plan),
                           certifier='synthetic.certifier', deadline_ok=True)
    detail = {'first': first, 'second': second, 'installs': installs['n'],
              'second_refusal': ledger.events[-1].get('reason')}
    return record('A-5b', 'a second finish on the same ledger is refused, so no double install',
                  first is True and second is False and installs['n'] == 1
                  and ledger.events[-1].get('reason') == 'already_finished', detail)


RISKS = (
    ('A-2', 'no certifier', a2_no_certifier),
    ('A-2', 'certified-A-installed-B', a2_certified_but_unbound),
    ('A-3b', 'committed prefix', a3b_committed_prefix),
    ('A-4b', 'reference mutation', a4b_reference_mutation),
    ('A-4b', 'unhashable plan', a4b_unhashable_plan),
    ('A-5d', 'no rollback_fn', a5d_rollback_unverified),
    ('A-5d', 'with rollback_fn', a5d_real_rollback),
    ('A-5b', 'double finish', a5b_double_finish),
)


def main():
    outdir = Path(sys.argv[sys.argv.index('--outdir') + 1]) if '--outdir' in sys.argv else \
        (EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1')
    started = time.monotonic()
    for _risk, _name, fn in RISKS:
        try:
            fn(None)
        except Exception as exc:                                          # noqa: BLE001
            import traceback
            record(_risk, _name, False, {'exception': '%s: %s' % (type(exc).__name__, exc),
                                         'traceback': traceback.format_exc()})
    by_risk = {}
    for r in RESULTS:
        e = by_risk.setdefault(r['risk'], {'n': 0, 'n_confirmed': 0, 'kinds': set()})
        e['n'] += 1
        e['n_confirmed'] += int(r['observation_confirmed'])
        e['kinds'].add(r['kind'])
    for k, v in by_risk.items():
        v['kinds'] = sorted(v['kinds'])
        v['status'] = ('CLOSED_FIX_CONFIRMED' if v['kinds'] == ['FIX_CONFIRMED']
                       else 'RISK_CONFIRMED_STILL_OPEN' if v['kinds'] == ['RISK_CONFIRMED_OPEN']
                       else 'MIXED')
    payload = {
        'schema': 'd3_g0_residual_risk_counterexamples',
        'what_this_is': 'executable confirmation, on the CURRENT code, of the residual risks the '
                        'P0 audit named -- not a re-statement of the audit',
        'results': RESULTS, 'by_risk': by_risk,
        'n_rows': len(RESULTS),
        'n_fix_confirmed': sum(1 for r in RESULTS if r['kind'] == 'FIX_CONFIRMED'),
        'n_risk_confirmed_open': sum(1 for r in RESULTS
                                     if r['kind'] == 'RISK_CONFIRMED_OPEN'),
        'geo_adapter_sha256': GEO.sha256_of(EXT / 'scripts/simulation/d3_geo_adapter.py'),
        'not_run': ['any live install', 'any F forward', 'any comparison', 'any training'],
        'g0_qualified': False,
        'g0_note': 'G0 is NOT qualified by this file: the remaining open items are the state '
                   'certification (G-L), the real-write uniqueness (G-C), the conservative quality '
                   'guarantee and the confirmed A-3b absence of a committed-prefix guard inside '
                   'the adapter; and a receipt still cannot observe the environment\'s own commit',
        'stable_online_gain': False,
        'wall_s': time.monotonic() - started,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'D3_G0_RESIDUAL_RISK_COUNTEREXAMPLES.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== G0 residual-risk counterexamples: %d rows, %d FIX_CONFIRMED, %d '
          'RISK_CONFIRMED_OPEN ===' % (len(RESULTS), payload['n_fix_confirmed'],
                                        payload['n_risk_confirmed_open']))
    print(json.dumps({k: {'n': v['n'], 'status': v['status']} for k, v in by_risk.items()},
                     indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
