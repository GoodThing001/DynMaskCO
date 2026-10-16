#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 ARM-ORDER / SHARED-ALLOWANCE ACCEPTANCE: isolated synthetic counterexamples.

The two arms run SEQUENTIALLY in the frozen order and are charged from ONE monotonic origin, because
L1 pre-registers `per_state_wall_clock_s = 10.0` for the STATE rather than for each arm.  The second
arm's remaining time is therefore `10.0 s - (what the first arm spent)`, so its time budget depends on
the FIRST ARM'S COST.  This file fixes that down as executable statements rather than prose.

Rows
----
  O1  the arm order is FROZEN and matches the protocol's `ARMS`, and the shared origin is what the L1
      run actually constructs (not merely what a keyword allows)
  O2  the same shared state scored with a FAST first arm and a SLOW second arm -> BOTH complete
  O3  the same state with a SLOW first arm and a FAST second arm -> the SECOND arm times out
      (`O2` and `O3` are the counterexample pair: the ONLY thing that changed is which arm ran first)
  O4  V1 is ORDER-INVARIANT: the first-step verdict, the chosen first action and the first-step legal
      fingerprint are identical under both orders on the same state
  O5  `classify_order_confound` separates the three attributions deterministically and puts the O3
      timeout on the SECOND arm only
  O6  a timeout on the FIRST arm is NOT order-attributable (nothing had run before it)
  O7  an order-attributable difference and a genuine mechanism difference are DISTINGUISHED: the
      classifier flags `ORDER_ATTRIBUTABLE`, and the adjudication records it without removing it
  O8  the adjudication keeps every affected state in the 64-state denominator
  O9  the frozen caps are untouched: 10 / 60 / 180 s and 128 / 2 / 2
  O10 no extra run is created: the acceptance needs ONE formal opportunity, and this file runs zero
      formal comparisons

Every worker/scorer here is SYNTHETIC.  Nothing in this file is evidence about MaskCO or D-Geo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

RUNNER = EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'
PROTOCOL = EXT / 'scripts/simulation/d3_l1_two_arm_protocol.py'
ROWS = []


def adjudicate_source(entry_src_or=None):
    """The `adjudicate` body of the L1 entry, so a row can inspect it without importing it twice."""
    import ast as _ast
    src = entry_src_or or RUNNER.read_text(encoding='utf-8')
    for node in _ast.parse(src).body:
        if isinstance(node, _ast.FunctionDef) and node.name == 'adjudicate':
            return _ast.get_source_segment(src, node) or ''
    return ''


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:200]))


def runner_module():
    spec = importlib.util.spec_from_file_location('l1_entry_order', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['l1_entry_order'] = module
    spec.loader.exec_module(module)
    return module


def _ordered_run(protocol, geo, fixture, order, *, first_sleep, second_sleep, budget_s):
    """Run BOTH arms on ONE shared state, in `order`, from ONE monotonic wall-clock origin.

    This is the production mechanism, reproduced: two `BudgetGate`s built with the SAME
    `started_mono`, run sequentially in `order`.  The sharing is what makes the second arm's
    remaining allowance depend on the first arm's cost.

    The scorers are synthetic but WIRED LIKE THE REAL ONES:
      * the F slot uses a length-correct synthetic logits vector -- a wrong-length vector is
        `score_len`, which would confound the timing counterexample with a scoring defect;
      * the D-Geo slot uses the REAL `d3_geo_adapter.geo_score` over the shared extractor wrapped by
        the production `make_geo_extract_fn`.  D-Geo is not a synthetic stand-in for anything: it
        reads only the environment's own distance vector, exactly as in the L1 run.
    The only synthetic element is the per-call COST, which is what the counterexample manipulates.
    """
    import numpy as np
    fx = fixture
    gates = {a: protocol.BudgetGate(max_scorer_calls=128, max_wall_seconds=budget_s,
                                    started_mono=fx['origin']) for a in protocol.ARMS}
    sleeps = dict(zip(order, (first_sleep, second_sleep)))
    f_scorer = fx['f_scorer_factory'](sleeps[protocol.ARM_F])

    def d_geo_scorer(state):
        time.sleep(sleeps[protocol.ARM_D_GEO])
        return geo.geo_score(state)

    results = {}
    for arm in order:
        scorer = f_scorer if arm == protocol.ARM_F else d_geo_scorer
        results[arm] = protocol.run_arm(arm, fx['shared'], scorer=scorer,
                                        extract_fn=fx['extract_fn'], legal_fn=fx['legal_fn'],
                                        apply_fn=fx['apply_fn'], gate=gates[arm])
    return results, gates


def build_synthetic_shared(protocol, acceptance):
    """A REAL shared input from the accepted fixture, plus the production extractor wiring."""
    import numpy as np
    import d3_geo_adapter as geo
    fx = acceptance.build_fixture()
    arms, legal_fn, apply_fn = acceptance.make_arms(fx)
    shared = acceptance.shared_input(fx, legal_fn=legal_fn)
    mat = fx['materialized']
    # `legal` reaches the scorer through the extractor, so the F slot's vector length must come from
    # the extractor's own record of this step's ordered legal list
    carry = {'legal': []}
    bind = {'allowed': mat['mutable_ids']}

    def f_scorer_factory(sleep_s):
        def scorer(state):
            time.sleep(sleep_s)
            return np.zeros(len(carry['legal']), dtype=np.float32)
        return scorer

    geo_extract = geo.make_geo_extract_fn(arms['extract_fn'],
                                          allowed_vehicle_ids=bind['allowed'])

    def counting_extract(P, remaining, legal):
        carry['legal'] = list(legal)
        return geo_extract(P, remaining, legal)

    return {'shared': shared, 'extract_fn': counting_extract, 'legal_fn': legal_fn,
            'apply_fn': apply_fn, 'origin': None, 'f_scorer_factory': f_scorer_factory,
            'geo': geo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--budget-s', type=float, default=1.0,
                    help='the SHARED per-state allowance used by the counterexamples.  The production '
                         'value is the frozen 10.0 s; a smaller value is used ONLY so the rejection '
                         'costs a second instead of ten.  O9 asserts the production value is intact.')
    args = ap.parse_args()
    module = runner_module()
    import importlib.util as ilu
    spec = ilu.spec_from_file_location('two_arm_acc_order',
                                       str(HERE / 'd3_l1_two_arm_acceptance.py'))
    acceptance = ilu.module_from_spec(spec)
    sys.modules['two_arm_acc_order'] = acceptance
    spec.loader.exec_module(acceptance)
    import d3_l1_two_arm_protocol as protocol

    budget_s = float(args.budget_s)
    results = {}

    # O1 -------------------------------------------------------------------------------------
    o1_ok = (tuple(module.ARM_ORDER) == tuple(protocol.ARMS) == ('F', 'D-Geo')
             and 'started_mono=wall_origin_mono' in RUNNER.read_text(encoding='utf-8')
             and 'shared_origin = started_mono is not None' in PROTOCOL.read_text(encoding='utf-8'))
    ck('O1 the sequential arm order is FROZEN and the L1 run really constructs BOTH gates from ONE '
       'monotonic origin', o1_ok,
       {'entry_ARM_ORDER': list(module.ARM_ORDER), 'protocol_ARMS': list(protocol.ARMS),
        'l1_run_passes_one_origin_to_both_gates':
            'started_mono=wall_origin_mono' in RUNNER.read_text(encoding='utf-8'),
        'gate_records_shared_origin':
            'shared_origin = started_mono is not None' in PROTOCOL.read_text(encoding='utf-8'),
        'wall_clock_scope': module.WALL_CLOCK_SCOPE})

    # O2 / O3 / O4: the counterexample PAIR on ONE state ------------------------------------
    # The state, the mask, the ordered legal lists, the call ceilings and the two scorers' OWN costs
    # are held FIXED; the only thing that differs between the pair is which arm runs FIRST.
    fx = build_synthetic_shared(protocol, acceptance)
    geo = fx['geo']
    per_call = 0.25
    fx_fast_first = dict(fx)
    fx_fast_first['origin'] = time.monotonic()
    # first arm cheap -> the second arm still has room
    fast_first, _g2 = _ordered_run(protocol, geo, fx_fast_first, ('F', 'D-Geo'),
                                  first_sleep=per_call, second_sleep=per_call, budget_s=budget_s)
    if not (fast_first['F']['failure_reason'] is None
            and fast_first['D-Geo']['failure_reason'] is None):
        # widen the allowance until this arm is out of the way; the PAIR is what matters, not the
        # absolute number, and O9 asserts the production value is untouched either way
        for candidate in (2.0, 4.0, 8.0):
            fx_fast_first['origin'] = time.monotonic()
            fast_first, _g2 = _ordered_run(protocol, geo, fx_fast_first, ('F', 'D-Geo'),
                                          first_sleep=per_call, second_sleep=per_call,
                                          budget_s=candidate)
            if (fast_first['F']['failure_reason'] is None
                    and fast_first['D-Geo']['failure_reason'] is None):
                budget_s = candidate
                break

    fx_slow_first = dict(fx)
    fx_slow_first['origin'] = time.monotonic()
    # The FIRST arm is made costly enough to EAT THE ALLOWANCE but still COMPLETE within it, so the
    # pair isolates the order rather than the first arm's own failure: the second arm's own cost is
    # unchanged at `per_call` and it starts with the remainder.
    for frac in (0.8, 0.7, 0.9, 0.6):
        fx_slow_first['origin'] = time.monotonic()
        slow_first, _g3 = _ordered_run(protocol, geo, fx_slow_first, ('F', 'D-Geo'),
                                       first_sleep=budget_s * frac, second_sleep=per_call,
                                       budget_s=budget_s)
        if (slow_first['F']['failure_reason'] is None
                and slow_first['D-Geo']['failure_reason'] == 'timeout'):
            break

    o2_ok = (fast_first['F']['failure_reason'] is None
             and fast_first['D-Geo']['failure_reason'] is None
             and fast_first['F']['n_steps'] > 0 and fast_first['D-Geo']['n_steps'] > 0)
    ck('O2 a state where the SECOND arm is cheap: BOTH arms complete (baseline for the pair)', o2_ok,
       {'order': ['F', 'D-Geo'], 'first_arm': 'F', 'second_arm': 'D-Geo',
        'F_failure': fast_first['F']['failure_reason'],
        'D_Geo_failure': fast_first['D-Geo']['failure_reason'],
        'F_steps': fast_first['F']['n_steps'], 'D_Geo_steps': fast_first['D-Geo']['n_steps'],
        'shared_allowance_s': budget_s})

    o3_ok = (slow_first['F']['failure_reason'] is None
             and slow_first['D-Geo']['failure_reason'] == 'timeout'
             and slow_first['D-Geo']['returned_original_plan'] is True)
    ck('O3 the SAME state with a SLOW FIRST arm and a FAST SECOND arm: the SECOND arm times out -- '
       'the second arm paid for time the first arm spent', o3_ok,
       {'order': ['F', 'D-Geo'], 'first_arm': 'F', 'second_arm': 'D-Geo',
        'F_failure': slow_first['F']['failure_reason'],
        'D_Geo_failure': slow_first['D-Geo']['failure_reason'],
        'D_Geo_returned_original_plan': slow_first['D-Geo']['returned_original_plan'],
        'what_changed_between_O2_and_O3':
            'ONLY which arm ran first.  The state, the mask, the ordered legal lists, the scorers\' '
            'own cost and both call ceilings are identical, so O3 is an instrument effect.',
        'shared_allowance_s': budget_s})

    # O4: V1 under both orders
    def first_step_of(res):
        return {a: (res[a]['first_step_action'] or None) for a in res}

    def v1_of(res):
        fa, da = res['F']['first_step_action'], res['D-Geo']['first_step_action']
        if fa is None or da is None:
            return 'NOT_COMPARABLE'
        keys = ('customer', 'vehicle_id', 'slot_kind', 'slot_anchor', 'position', 'predecessor',
                'successor')
        return 'IDENTICAL' if all(fa[k] == da[k] for k in keys) else 'DIVERGENT'

    # O4: V1 is order-invariant -- TWO statements, because the first cut of this row conflated them
    #
    # (a) WHEN BOTH ARMS PRODUCE A FIRST STEP, the first step is the same under any order.  A scorer
    #     is a pure function of its step's state, so the first call either happens or it does not; it
    #     is never "different" because another arm ran earlier.
    # (b) ORDER CAN PREVENT THE FIRST STEP FROM HAPPENING AT ALL.  Under the shared allowance an arm
    #     whose predecessor consumed it is refused at its FIRST call, so it produces no first step and
    #     V1 becomes `NOT_COMPARABLE`.  That is order-driven LOSS OF COMPARABILITY, not a mechanism
    #     difference -- and it is exactly why the classifier must exist.
    #
    # The clean order-invariance test therefore needs the two arms' costs SWAPPED, so each order has
    # both arms cheap and each order produces both first steps.
    fx_swap_a = dict(fx)
    fx_swap_a['origin'] = time.monotonic()
    swap_a, _ga = _ordered_run(protocol, geo, fx_swap_a, ('F', 'D-Geo'),
                               first_sleep=per_call, second_sleep=per_call, budget_s=budget_s)
    fx_swap_b = dict(fx)
    fx_swap_b['origin'] = time.monotonic()
    swap_b, _gb = _ordered_run(protocol, geo, fx_swap_b, ('D-Geo', 'F'),
                               first_sleep=per_call, second_sleep=per_call, budget_s=budget_s)
    both_delivered = (swap_a['F']['first_step_action'] is not None
                      and swap_a['D-Geo']['first_step_action'] is not None
                      and swap_b['F']['first_step_action'] is not None
                      and swap_b['D-Geo']['first_step_action'] is not None)
    v1_invariance = both_delivered and all(
        swap_a[a]['first_step_action'] == swap_b[a]['first_step_action'] for a in protocol.ARMS)
    legal_invariance = all(
        swap_a[a]['legal_first_step_sha256'] == swap_b[a]['legal_first_step_sha256']
        for a in protocol.ARMS)
    # (b) the loss-of-comparability case, recorded rather than hidden
    lost_comparability = (v1_of(slow_first) == 'NOT_COMPARABLE'
                          and slow_first['D-Geo']['first_step_action'] is None
                          and slow_first['D-Geo']['failure_reason'] == 'timeout')
    o4_ok = bool(v1_invariance and legal_invariance and lost_comparability)
    ck('O4 V1 is ORDER-INVARIANT when both arms deliver a first step (same chosen action and same '
       'first-step legal fingerprint under both orders), and order destroys COMPARABILITY -- not a '
       'scorer difference -- when the shared allowance kills the second arm before its first step',
       o4_ok,
       {'both_arms_delivered_a_first_step_in_both_orders': both_delivered,
        'first_step_action_identical_across_orders': v1_invariance,
        'first_step_legal_fingerprint_identical_across_orders': legal_invariance,
        'order_F_then_D_Geo': {a: swap_a[a]['first_step_action'] is not None for a in protocol.ARMS},
        'order_D_Geo_then_F': {a: swap_b[a]['first_step_action'] is not None for a in protocol.ARMS},
        'loss_of_comparability_case': {
            'verdict': v1_of(slow_first),
            'second_arm': 'D-Geo', 'second_arm_first_step': slow_first['D-Geo']['first_step_action'],
            'second_arm_failure': slow_first['D-Geo']['failure_reason'],
            'reading': 'NOT_COMPARABLE means exactly "no usable first step existed on both arms"; it '
                       'is NOT a difference between the scorers'},
        'why': 'a scorer is a pure function of its step state, so the first call either happens or it '
               'does not; it is never different because another arm ran earlier'});

    # O5 / O6: the classifier ---------------------------------------------------------------
    def rec_from(res, arm_order):
        return {'V1': {'verdict': v1_of(res)},
                'V2': {'arm_order': list(arm_order),
                       'wall_budget_source': 'shared_single_allowance'},
                'per_arm_failure': {a: res[a]['failure_reason'] for a in res}}

    r3 = rec_from(slow_first, ('F', 'D-Geo'))
    r2 = rec_from(fast_first, ('F', 'D-Geo'))
    c3 = module.classify_order_confound(r3)
    c2 = module.classify_order_confound(r2)
    # the mirror case: the SECOND arm is the slow one and the FIRST times out -- impossible under a
    # shared origin (the first arm has the whole allowance and nothing precedes it), so build it by
    # making the FIRST arm the one that reports timeout
    r_first_timeout = {'V1': {'verdict': 'DIVERGENT'},
                       'V2': {'arm_order': ['F', 'D-Geo'],
                              'wall_budget_source': 'shared_single_allowance'},
                       'per_arm_failure': {'F': 'timeout', 'D-Geo': None}}
    c_first = module.classify_order_confound(r_first_timeout)
    r_both = {'V1': {'verdict': 'NOT_COMPARABLE'},
              'V2': {'arm_order': ['F', 'D-Geo'],
                     'wall_budget_source': 'shared_single_allowance'},
              'per_arm_failure': {'F': 'timeout', 'D-Geo': 'timeout'}}
    c_both = module.classify_order_confound(r_both)
    o5_ok = (c3 == 'ORDER_ATTRIBUTABLE' and c2 == 'NOT_ORDER_ATTRIBUTABLE'
             and c_both == 'ORDER_INDETERMINATE')
    ck('O5 `classify_order_confound` is DETERMINISTIC and separates the three attributions, putting '
       'the O3 timeout on the SECOND arm only',
       o5_ok, {'O3_state': c3, 'O2_state': c2, 'both_arms_timeout': c_both,
               'rule': module.ORDER_CONFOUND_RULE[:180] + '...'})
    o6_ok = c_first == 'NOT_ORDER_ATTRIBUTABLE'
    ck('O6 a `timeout` on the FIRST arm is NOT order-attributable -- nothing had run before it, so '
       'the shared allowance cannot have been consumed by anyone else', o6_ok,
       {'first_arm_timeout': c_first, 'first_arm': 'F', 'second_arm': 'D-Geo'})

    # O7 / O8: adjudication keeps the states and does not launder the difference ------------
    def full_record(day, pos, attribution_rec):
        r = module.blank_state_record(day, pos, state_hash='h%02d%d' % (day, pos),
                                      status='COMPLETED', synthetic=True)
        r['V1'] = {'verdict': attribution_rec['V1']['verdict']}
        r['V2'] = dict(attribution_rec['V2'], both_completed=False,
                       F={'full_repair_completed': attribution_rec['per_arm_failure'].get('F') is None},
                       **{'D-Geo': {'full_repair_completed':
                                    attribution_rec['per_arm_failure'].get('D-Geo') is None}})
        r['per_arm_failure'] = dict(attribution_rec['per_arm_failure'])
        r['audit'] = {'single_factor_ok': True, 'state_identity_agrees_both_arms': True}
        r['ledger_ok'] = True
        r['ledger_problems'] = []
        r['v2_order_attribution'] = module.classify_order_confound(r)
        return r

    states = module.build_l1_identity(created_utc='2026-01-01T00:00:00Z')['state_list']['states']
    records = [full_record(int(s['day_id']), int(s['position']), r2) for s in states]
    clean = module.adjudicate(records, synthetic=True, expected_states=64)
    k = 7
    records[k] = full_record(int(states[k]['day_id']), int(states[k]['position']), r3)
    with_order = module.adjudicate(records, synthetic=True, expected_states=64)
    ao = with_order['arm_order']
    o7_ok = (ao['counts']['ORDER_ATTRIBUTABLE'] == 1
             and ao['counts']['NOT_ORDER_ATTRIBUTABLE'] == 63
             and len(ao['order_attributable_states']) == 1
             and ao['order_attributable_states'][0]['day_id'] == int(states[k]['day_id'])
             and 'NOT evidence' in ao['must_not_be_read_as'])
    ck('O7 an order-attributable V2 difference and a genuine mechanism difference are DISTINGUISHED: '
       'the state is flagged, its V1 verdict is untouched, and the record says the difference may not '
       'be read as a scorer advantage', o7_ok,
       {'counts': ao['counts'], 'flagged_state': ao['order_attributable_states'],
        'V1_verdict_of_the_flagged_state': with_order['V1_counts'],
        'status': with_order['status'], 'must_not_be_read_as': ao['must_not_be_read_as'],
        'clean_run_counts': clean['arm_order']['counts']})

    o8_ok = (with_order['n_states_in_denominator'] == 64
             and with_order['denominator_is_the_preregistered_list'] is True
             and len(with_order['timeout_states']) >= 1
             and with_order['status'] == 'ADJUDICATED'
             and with_order['arm_order']['no_state_is_removed'].startswith('every state affected'))
    ck('O8 the order-affected state KEEPS its record and stays in the 64-state denominator; it is '
       'reported, not removed, and the round is still adjudicable', o8_ok,
       {'denominator': with_order['n_states_in_denominator'],
        'denominator_is_the_preregistered_list':
            with_order['denominator_is_the_preregistered_list'],
        'n_timeout_states': len(with_order['timeout_states']), 'status': with_order['status']})

    # O9 -------------------------------------------------------------------------------------
    caps = module.BUDGET
    o9_ok = (caps['per_state_wall_clock_s'] == 10.0 and caps['initialize_compile_s'] == 60.0
             and caps['batch_wall_clock_s'] == 180.0
             and caps['per_arm_total_scorer_calls'] == 128
             and caps['per_arm_per_state_scorer_calls'] == 2)
    ck('O9 the frozen 10 / 60 / 180 s and 128 / 2 / 2 are UNCHANGED; only the counterexamples use a '
       'SHORTER allowance so a rejection costs a second instead of ten', o9_ok,
       {'production_caps': caps, 'counterexample_allowance_s': budget_s,
        'counterexample_only': 'the shortened allowance is a local argument to this harness and is '
                               'never written into any frozen constant or record'})

    # O10 ------------------------------------------------------------------------------------
    # The classification must add NO execution: it is a pure function of an already-recorded
    # outcome.  Checked structurally rather than asserted in prose: the entry's adjudication reads
    # `v2_order_attribution` off records it was already given, and no counter is added to any cap.
    entry_src = RUNNER.read_text(encoding='utf-8')
    cls = module.classify_order_confound
    import inspect
    cls_src = inspect.getsource(cls)
    o10_ok = (caps['per_arm_total_scorer_calls'] == 128
              and 'records' in adjudicate_source(entry_src_or=entry_src)
              and 'v2_order_attribution' in adjudicate_source(entry_src_or=entry_src)
              and 'time.' not in cls_src and 'run_arm' not in cls_src and 'score' not in cls_src)
    ck('O10 no extra formal run is created: the classification is a PURE READING RULE over the single '
       'recorded run -- it calls no scorer, no arm and no clock', o10_ok,
       {'classifier_is_pure': 'no time./run_arm/score reference in classify_order_confound',
        'adjudication_reads_the_recorded_attribution':
            'v2_order_attribution' in adjudicate_source(entry_src_or=entry_src),
        'call_ceiling_unchanged': caps['per_arm_total_scorer_calls'],
        'mechanism': 'the rule maps an already-recorded per-state outcome to an attribution; it adds '
                     'no state, no arm, no repetition and no second opportunity',
        'no_real_comparison_in_this_file': True})

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_l1_arm_order_acceptance',
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'arm_order': list(protocol.ARMS),
        'wall_clock_scope': module.WALL_CLOCK_SCOPE,
        'order_confound_rule': module.ORDER_CONFOUND_RULE,
        'v1_order_invariance_statement': (
            'V1 compares the first step, and no scorer call precedes it, so the second arm has spent '
            'nothing and no deadline can have fired.  V1 is order-invariant by construction, and the '
            'O4 counterexample confirms it by running the same state under two different first arms.'),
        'v2_order_sensitivity_statement': (
            'V2 compares per-arm completion, which is exactly what the shared deadline bounds, so a '
            'second-arm `timeout` may be caused by the order rather than by the scorer.  O2 and O3 '
            'are the counterexample pair: only the identity of the first arm changed.'),
        'violations_of_the_claim_rule': {
            'forbidden': 'reading a V2 difference on an ORDER_ATTRIBUTABLE state as a scorer '
                         'mechanism advantage',
            'enforced_by': 'classify_order_confound + adjudicate.arm_order (counts, the affected '
                           'state list and `must_not_be_read_as`)'},
        'scorer': 'SYNTHETIC ONLY: this file runs no real F scorer and no real D-Geo arm and says '
                  'nothing about MaskCO or D-Geo',
        'not_run': ['the real F-vs-D-Geo comparison', 'training', 'L2', 'L3', 'the 40-day batch',
                    'the final test'],
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'tool_sha256': sha(Path(__file__)), 'runner_sha256': sha(RUNNER),
        'protocol_sha256': sha(PROTOCOL),
        'verdict': ('L1_ARM_ORDER_CONFOUND_BOUNDED' if not fails else 'L1_ARM_ORDER_CONFOUND_OPEN'),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== L1 arm-order acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails)))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
