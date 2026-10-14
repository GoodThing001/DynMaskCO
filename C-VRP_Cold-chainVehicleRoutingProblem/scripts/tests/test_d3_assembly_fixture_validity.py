#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 A2/A3 FIXTURE-VALIDITY REVIEW (user-authorised round, 2026-10-14).

Scope (deliberately narrow, read-only):
  * attribute, AT RUNTIME, the exact cause of the three failing assertions in the FROZEN
    historical suite `test_d3_assembly_invariants.py`
    (2x A2 end-to-end / only-added-field, 1x A3 "success path installs exactly once");
  * redo the A2 comparison on a LEGAL fixture (one customer, two legal actions,
    `len(scores) == len(legal)`, exact tie) and compare the assembly's first choice with the
    HISTORICAL F projection, using the SAME evidence the frozen suite uses (the step record);
  * A3 TWO-WAY control: (a) no certifier -> must REFUSE, (b) a clearly TEST-ONLY
    deterministic certifier -> installs EXACTLY ONCE with identity match;
  * prove the test-only certifier cannot masquerade as the real `certify_plan` / D8 path;
  * record, as raw evidence, whether the action the assembly APPLIES is coherent with the
    action it RECORDS as chosen (a disagreement here would be a real implementation defect).

NOT in scope: any F-vs-D-Geo effect comparison, any metric, any threshold, any freeze change.
The frozen historical suite is NOT edited; it is re-executed from the new freeze tree instead.

`ck()` rows are the evidence: a FAIL here is a finding about the CODE, not about a fixture.
If a LEGAL fixture still fails, that is recorded as a real implementation defect -- the fixture
is never bent again and the certification is never relaxed.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE.parent))

import d3_geo_adapter as G                                        # noqa: E402
import mpre_policy as MP                                          # noqa: E402
import action_contract                                            # noqa: E402
from test_d3_geo_adapter import _cand, _identity_apply, _plans    # noqa: E402

CASES = []
RECORDS = {}

STEP_FIELDS = ('customer', 'slot_kind', 'slot_anchor', 'position', 'predecessor', 'successor',
               'incremental_distance')


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def rec(key, value):
    RECORDS[key] = value


def _public(obj):
    """Defensive public-attribute dump: never raises, never assumes an API name."""
    out = {}
    for name in dir(obj):
        if name.startswith('_'):
            continue
        try:
            out[name] = repr(getattr(obj, name))
        except Exception as exc:                                   # pragma: no cover
            out[name] = '<unreadable: %r>' % (exc,)
    return out


def _reason(dump):
    for key in ('failure_reason', 'reason', 'last_failure', 'refusal', 'status'):
        if key in dump:
            return '%s=%s' % (key, dump[key])
    return None


def _find_call(value, needle):
    return needle in json.dumps(value, default=repr)


def _cand_tuple(cand):
    """The candidate's identity as it appears in a chosen step record."""
    customer, act, candobj = cand
    return (customer, act.slot.kind, act.slot.anchor, act.position, act.predecessor,
            act.successor, float(candobj.incremental_distance))


def _step_tuple(step):
    return tuple(step.get(f) for f in STEP_FIELDS)


def _index_by_step(step, order):
    if step is None:
        return None
    target = _step_tuple(step)
    for idx, cand in enumerate(order):
        try:
            if _cand_tuple(cand) == target:
                return idx
        except Exception:
            pass
    return None


def _index_by_action(reprs, order):
    """Which candidate's FleetAction (by repr) appears among the recorded apply_fn arguments."""
    for idx, cand in enumerate(order):
        if repr(cand[1]) in reprs:
            return idx
    return None


# --------------------------------------------------------------------------- fixtures
P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
MUT = {1, 2, 3}
EXTRACT = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)

# the FROZEN A2/A3 caller fixture (verbatim from test_d3_assembly_invariants.py:121-135)
SEQ_F = [_cand(4, 'anchored', 3, 0, 3.0), _cand(5, 'new_route', -1, 0, 3.0)]
PLANS_F = _plans({1: (3, [4]), 2: (0, [])})
EXT_F = G.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids={1, 2})
ILLEGAL_SCORES = lambda st: [1.0, 1.0]                            # noqa: E731  (2 != 1 legal)
LEGAL_SCORES = lambda st: [1.0, 1.0]                              # noqa: E731  (2 == 2 legal)

# LEGAL A2 fixture: ONE customer (5), TWO legal actions, exact tie on every numeric field.
# position 0 keeps predecessor==successor==anchor; position 1 moves the predecessor to 0, so the
# two entries are distinct actions of the same customer with identical incremental_distance.
A_POS1 = _cand(5, 'anchored', 3, 1, 5.0)
A_POS0 = _cand(5, 'anchored', 3, 0, 5.0)
LEGAL_ORDER_POS1_HEAD = [A_POS1, A_POS0]      # legal head carries the LARGER canonical key
LEGAL_ORDER_POS0_HEAD = [A_POS0, A_POS1]      # legal head carries the SMALLER canonical key

# COHERENCE probe: two DIFFERENT customers, canonical order reversed (as in the frozen suite),
# so "which action was chosen" is unambiguous from the customer field alone.
TIE_SEQ = [_cand(5, 'anchored', 3, 0, 5.0), _cand(4, 'anchored', 3, 1, 5.0)]

APPLY_CALLS = []


def _recording_apply(*args, **kwargs):
    """Records what the assembly actually applied, then leaves the plan untouched."""
    APPLY_CALLS.append({'args': [repr(a) for a in args],
                        'kwargs': {k: repr(v) for k, v in kwargs.items()}})
    return args[0] if args else None


CERT_CALLS = []


def _test_only_deterministic_certifier(plan):
    """TEST-ONLY stand-in. It is NOT `certify_plan`, it certifies nothing physical, and it is
    named so that its recorded callable identity cannot be mistaken for the real certifier."""
    CERT_CALLS.append(plan)
    return True, 'TEST_ONLY_DETERMINISTIC_FIXTURE_CERTIFIER (not a C0 certification)'


def _legal_fn(seq):
    return lambda P, rem: [t for t in seq if t[0] in rem]


def _count_legal(seq, rem):
    return len(_legal_fn(seq)(None, rem))


# --------------------------------------------------------------------------- V: attribution
def verify_attribution():
    print('--- V: runtime attribution of the frozen A2/A3 failures ---')
    rem = {5}
    n_legal = _count_legal(SEQ_F, rem)
    rec('V_frozen_A2_fixture', {'legal_actions_for_mask_5': n_legal,
                               'legal_customers': [t[0] for t in _legal_fn(SEQ_F)(None, rem)],
                               'scores_len': len(ILLEGAL_SCORES(None)),
                               'score_fn': 'lambda st: [1.0, 1.0]'})
    ck('V-A2a the FROZEN A2/A3 caller fixture is ILLEGAL: len(scores)=2 but len(legal)=1 '
       'for mask_set=[5]',
       n_legal == 1, {'len_legal': n_legal, 'len_scores': 2})

    plan_b, steps_b, recs_b, fr_b, diag_b = G.stepwise_sample_d3(
        score_fn=ILLEGAL_SCORES, extract_fn=EXT_F, legal_fn=_legal_fn(SEQ_F),
        apply_fn=action_contract.apply_action, plans=PLANS_F, mask_set=[5],
        mutable_ids={1, 2}, selector=G.SELECTOR_GEO, deterministic=True)
    rec('V_new_assembly_on_illegal_fixture',
        {'failure_reason': fr_b, 'n_steps': len(steps_b), 'n_records': len(recs_b),
         'plan_unchanged': repr(plan_b) == repr(PLANS_F), 'diag': repr(diag_b)})
    ck('V-A2b the NEW assembly FAILS CLOSED on the illegal fixture (score_len, zero steps) -- '
       'this is the runtime cause of the 2 frozen A2 assertion failures',
       fr_b == 'score_len' and len(steps_b) == 0, {'failure_reason': fr_b,
                                                   'n_steps': len(steps_b)})

    real_enum = MP.enumerate_legal_actions
    MP.enumerate_legal_actions = lambda env, inst, P, rem_, mut: _legal_fn(SEQ_F)(P, rem_)
    try:
        _pa, s_f, _ra, fr_f = MP.stepwise_sample(
            score_fn=ILLEGAL_SCORES, extract_fn=EXT_F, env=None, inst_idx=0, plans=PLANS_F,
            mask_set=[5], mutable_ids={1, 2}, deterministic=True)
    finally:
        MP.enumerate_legal_actions = real_enum
    rec('V_legacy_sampler_on_illegal_fixture',
        {'n_steps': len(s_f), 'failure_reason': fr_f,
         'first_step': repr(s_f[0]) if s_f else None})
    ck('V-A2c the LEGACY sampler still returns a step on the same illegal fixture (no '
       'score-length guard) -- the frozen A2 comparison therefore compared 1 step against 0',
       len(s_f) == 1, {'legacy_steps': len(s_f), 'new_steps': len(steps_b)})

    # A3: the frozen "success path installs exactly once" control, verbatim (no certify_fn)
    installed = []
    ledger3 = G.InstallLedger(install_fn=lambda p: installed.append(p))
    raised = None
    try:
        ok3 = G.run_repair_transaction(
            ledger=ledger3, score_fn=G.geo_score, extract_fn=EXT_F, legal_fn=_legal_fn(SEQ_F),
            apply_fn=_identity_apply, plans=PLANS_F, mask_set=[5], mutable_ids={1, 2},
            selector=G.SELECTOR_GEO, deterministic=True)
    except BaseException as exc:
        ok3, raised = None, '%s: %s' % (type(exc).__name__, exc)
    dump3 = _public(ledger3)
    rec('V_frozen_A3_call_no_certify_fn',
        {'returned': repr(ok3), 'raised': raised, 'install_fn_calls': len(installed),
         'ledger': dump3, 'reason': _reason(dump3)})
    ck('V-A3 the FROZEN A3 "installs exactly once" call passes NO certify_fn, so with '
       'INSTALL_REQUIRES_CERTIFICATION=%s the transaction refuses (n_installs=0) -- the runtime '
       'cause of the 3rd frozen failure' % (G.INSTALL_REQUIRES_CERTIFICATION,),
       _find_call(dump3, 'no_certifier') and dump3.get('n_installs') == '0'
       and len(installed) == 0 and raised is None,
       {'returned': repr(ok3), 'n_installs': dump3.get('n_installs'),
        'install_fn_calls': len(installed), 'reason': _reason(dump3)})


# --------------------------------------------------------------------------- C: legal A2 control
def verify_legal_control(order, tag):
    APPLY_CALLS.clear()
    proj_idx = G.select_historical_argmax([-t[2].incremental_distance for t in order])
    plan_g, steps_g, recs_g, fr_g, diag_g = G.stepwise_sample_d3(
        score_fn=LEGAL_SCORES, extract_fn=EXTRACT, legal_fn=_legal_fn(order),
        apply_fn=_recording_apply, plans=P0, mask_set=[5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    step0 = steps_g[0] if steps_g else None
    chosen = _index_by_step(step0, order)
    applied = _index_by_action(APPLY_CALLS[0]['args'] + list(APPLY_CALLS[0]['kwargs'].values()),
                               order) if APPLY_CALLS else None
    keys = recs_g[0].get('canonical_keys') if recs_g else None
    canon_head = None
    if keys is not None and len(keys) == len(order):
        try:
            canon_head = sorted(range(len(order)), key=lambda i: repr(keys[i]))[0]
        except Exception:
            canon_head = None
    detail = {'legal_order_positions': [t[1].position for t in order],
              'n_legal': _count_legal(order, {5}), 'n_scores': len(LEGAL_SCORES(None)),
              'customers': [t[0] for t in order], 'failure_reason': fr_g,
              'n_steps': len(steps_g), 'chosen_index_from_step_record': chosen,
              'applied_index_from_apply_fn': applied,
              'historical_projection_index': proj_idx,
              'chosen_step': repr(step0), 'canonical_keys': repr(keys),
              'canonical_key_head_index': canon_head,
              'canonical_order_differs_from_legal_order':
                  (None if canon_head is None else canon_head != 0),
              'apply_calls': APPLY_CALLS}
    rec('C_%s' % tag, detail)
    ck('C-%s precondition: SAME customer, 2 legal actions, len(scores)==len(legal)==2' % tag,
       _count_legal(order, {5}) == 2 and len(LEGAL_SCORES(None)) == 2
       and len({t[0] for t in order}) == 1, {'len_legal': _count_legal(order, {5})})
    ck('C-%s the legal fixture samples cleanly (no fail-closed reason, exactly one step)' % tag,
       fr_g is None and len(steps_g) == 1, {'failure_reason': fr_g, 'n_steps': len(steps_g)})
    ck('C-%s on an exact tie the assembly RECORDS the FIRST action in legal order as chosen, '
       'matching the historical F projection' % tag,
       chosen == 0 and proj_idx == 0,
       {'chosen_index_from_step_record': chosen, 'historical_projection_index': proj_idx,
        'legal_head_position': order[0][1].position, 'chosen_step': repr(step0)})
    ck('C-%s canonical keys are still recorded for audit (len == len(legal))' % tag,
       keys is not None and len(keys) == len(order), {'canonical_keys': repr(keys)})
    return {'chosen': chosen, 'proj': proj_idx, 'canon': canon_head, 'applied': applied}


def verify_coherence_probe():
    """Two DIFFERENT customers with the canonical order reversed: the chosen customer is
    unambiguous, so the recorded choice can be compared with the applied action."""
    print('--- C4/C5: choice-vs-apply coherence probe (distinct customers) ---')
    APPLY_CALLS.clear()
    proj_idx = G.select_historical_argmax([-t[2].incremental_distance for t in TIE_SEQ])
    _p, steps_g, recs_g, fr_g, _d = G.stepwise_sample_d3(
        score_fn=LEGAL_SCORES, extract_fn=EXTRACT, legal_fn=_legal_fn(TIE_SEQ),
        apply_fn=_recording_apply, plans=P0, mask_set=[4, 5], mutable_ids=MUT,
        selector=G.SELECTOR_GEO, deterministic=True)
    step0 = steps_g[0] if steps_g else None
    recs = APPLY_CALLS[0]['args'] + list(APPLY_CALLS[0]['kwargs'].values()) if APPLY_CALLS else []
    applied = _index_by_action(recs, TIE_SEQ)
    rec('C_coherence_probe',
        {'failure_reason': fr_g, 'n_steps': len(steps_g), 'chosen_step': repr(step0),
         'historical_projection_index': proj_idx,
         'chosen_customer_from_step': step0.get('customer') if step0 else None,
         'projected_customer': TIE_SEQ[proj_idx][0],
         'applied_customer_from_apply_fn': (TIE_SEQ[applied][0] if applied is not None else None),
         'apply_calls': APPLY_CALLS,
         'interpretation': 'a disagreement here is an implementation defect, not a fixture '
                           'question'})
    ck('C4 with two distinct customers the assembly records the FIRST legal-order action as '
       'chosen (customer 5), matching the historical projection',
       step0 is not None and step0.get('customer') == TIE_SEQ[proj_idx][0] == 5,
       {'chosen_customer': step0.get('customer') if step0 else None,
        'projected_customer': TIE_SEQ[proj_idx][0]})
    ck('C5 the action the assembly APPLIES is coherent with the action it records as chosen',
       applied is not None and step0 is not None
       and TIE_SEQ[applied][0] == step0.get('customer'),
       {'applied_index_from_apply_fn': applied,
        'applied_customer': TIE_SEQ[applied][0] if applied is not None else None,
        'recorded_customer': step0.get('customer') if step0 else None,
        'apply_calls': APPLY_CALLS})


# --------------------------------------------------------------------------- D: A3 two-way
def verify_a3_two_way():
    print('--- D: A3 two-way certifier control ---')

    def _refusal_run(tag):
        installed = []
        ledger = G.InstallLedger(install_fn=lambda p: installed.append(p))
        raised = None
        try:
            ok = G.run_repair_transaction(
                ledger=ledger, score_fn=G.geo_score, extract_fn=EXT_F,
                legal_fn=_legal_fn(SEQ_F), apply_fn=_identity_apply, plans=PLANS_F,
                mask_set=[5], mutable_ids={1, 2}, selector=G.SELECTOR_GEO, deterministic=True)
        except BaseException as exc:
            ok, raised = None, '%s: %s' % (type(exc).__name__, exc)
        dump = _public(ledger)
        rec('D_%s' % tag, {'returned': repr(ok), 'raised': raised,
                           'install_fn_calls': len(installed), 'ledger': dump,
                           'reason': _reason(dump)})
        return ok, dump, installed, raised

    ok_a, dump_a, inst_a, raised_a = _refusal_run('no_certifier_A')
    ck('D1a NO certifier => the transaction must REFUSE: no_certifier, zero installs, no '
       'exception escaping',
       _find_call(dump_a, 'no_certifier') and dump_a.get('n_installs') == '0'
       and len(inst_a) == 0 and raised_a is None,
       {'returned': repr(ok_a), 'n_installs': dump_a.get('n_installs'),
        'install_fn_calls': len(inst_a), 'raised': raised_a, 'reason': _reason(dump_a)})
    ok_b, dump_b, inst_b, raised_b = _refusal_run('no_certifier_B')
    ck('D1b the refusal is deterministic across a second, fresh ledger',
       dump_a == dump_b and len(inst_a) == len(inst_b) == 0,
       {'identical_dump': dump_a == dump_b, 'raised_b': raised_b})

    CERT_CALLS.clear()
    installed2 = []
    ledger_p = G.InstallLedger(install_fn=lambda p: installed2.append(p))
    raised = None
    try:
        ok_p = G.run_repair_transaction(
            ledger=ledger_p, score_fn=G.geo_score, extract_fn=EXT_F, legal_fn=_legal_fn(SEQ_F),
            apply_fn=_identity_apply, plans=PLANS_F, mask_set=[5], mutable_ids={1, 2},
            selector=G.SELECTOR_GEO, deterministic=True,
            certify_fn=_test_only_deterministic_certifier)
    except BaseException as exc:
        ok_p, raised = None, '%s: %s\n%s' % (type(exc).__name__, exc, traceback.format_exc())
    dump_p = _public(ledger_p)
    cert_id = G._callable_identity(_test_only_deterministic_certifier) \
        if hasattr(G, '_callable_identity') else repr(_test_only_deterministic_certifier)
    installed_ids = [G._plan_identity(p) for p in installed2] if hasattr(G, '_plan_identity') \
        else [repr(p) for p in installed2]
    certified_ids = [G._plan_identity(p) for p in CERT_CALLS] if hasattr(G, '_plan_identity') \
        else [repr(p) for p in CERT_CALLS]
    rec('D_test_only_certifier',
        {'returned': repr(ok_p), 'raised': raised, 'install_fn_calls': len(installed2),
         'certifier_calls': len(CERT_CALLS), 'certifier_identity': cert_id,
         'certifier_module': _test_only_deterministic_certifier.__module__,
         'installed_plan_ids': installed_ids, 'certified_plan_ids': certified_ids,
         'ledger': dump_p, 'reason': _reason(dump_p)})
    ck('D2 with the TEST-ONLY deterministic certifier the transaction installs EXACTLY ONCE',
       ok_p is True and dump_p.get('n_installs') == '1' and len(installed2) == 1
       and len(CERT_CALLS) == 1,
       {'returned': repr(ok_p), 'n_installs': dump_p.get('n_installs'),
        'install_fn_calls': len(installed2), 'certifier_calls': len(CERT_CALLS),
        'raised': raised})
    ck('D2b the certified plan identity equals the installed plan identity (binding, not a '
       'bare boolean)',
       len(certified_ids) == 1 and installed_ids == certified_ids,
       {'certified': certified_ids, 'installed': installed_ids})

    real_ids = {}
    for modname in ('rolling_optimizer', 'd3_composed_certifier', 'd3_install_gate',
                    'd3_commit_transaction'):
        try:
            mod = __import__(modname)
        except Exception as exc:
            real_ids[modname] = '<unimportable: %r>' % (exc,)
            continue
        real_ids[modname] = {n: repr(getattr(mod, n)) for n in dir(mod)
                             if 'certif' in n.lower() and not n.startswith('_')}
    rec('D_real_certifier_surfaces', real_ids)
    flat_real = json.dumps(real_ids)
    ck('D3 anti-masquerade: the ledger records the TEST-ONLY callable identity itself (a '
       'function of this test process, not the real certifier surface), no real certifier '
       'appears anywhere in the ledger, and no D8 install receipt is produced',
       _test_only_deterministic_certifier.__module__ == '__main__'
       and '_test_only_deterministic_certifier' in cert_id
       and 'compose_certification' not in cert_id
       and 'certify_plan' not in json.dumps(dump_p, default=repr)
       and 'receipt' not in json.dumps(dump_p, default=repr).lower(),
       {'certifier_identity': cert_id,
        'certifier_module': _test_only_deterministic_certifier.__module__,
        'real_surfaces_seen': flat_real[:400],
        'ledger_mentions_certify_plan': 'certify_plan' in json.dumps(dump_p, default=repr),
        'ledger_mentions_receipt': 'receipt' in json.dumps(dump_p, default=repr).lower()})

    installed3 = []
    ledger_x = G.InstallLedger(install_fn=lambda p: installed3.append(p))
    raised = None
    try:
        ok_x = G.run_repair_transaction(
            ledger=ledger_x, score_fn=ILLEGAL_SCORES, extract_fn=EXT_F,
            legal_fn=_legal_fn(SEQ_F), apply_fn=_identity_apply, plans=PLANS_F, mask_set=[5],
            mutable_ids={1, 2}, selector=G.SELECTOR_GEO, deterministic=True,
            certify_fn=_test_only_deterministic_certifier)
    except BaseException as exc:
        ok_x, raised = None, '%s: %s' % (type(exc).__name__, exc)
    dump_x = _public(ledger_x)
    rec('D_certifier_with_illegal_fixture',
        {'returned': repr(ok_x), 'raised': raised, 'install_fn_calls': len(installed3),
         'ledger': dump_x, 'reason': _reason(dump_x)})
    ck('D4 a present certifier does NOT bypass the score-length guard: the illegal fixture '
       'still fails closed with score_len and zero installs',
       _find_call(dump_x, 'score_len') and dump_x.get('n_installs') == '0'
       and len(installed3) == 0,
       {'returned': repr(ok_x), 'n_installs': dump_x.get('n_installs'),
        'reason': _reason(dump_x)})


# --------------------------------------------------------------------------- identity proof
def record_self_identity():
    import hashlib
    mods = {}
    for name in ('d3_geo_adapter', 'mpre_policy', 'action_contract'):
        mod = sys.modules.get(name) or __import__(name)
        path = Path(getattr(mod, '__file__', '<none>'))
        sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        mods[name] = {'file': str(path), 'sha256': sha}
    rec('imported_modules', mods)
    rec('primary_constants', {
        'EPS_D_PRIMARY_REL': getattr(G, 'EPS_D_PRIMARY_REL', '<missing>'),
        'EPS_D_PRIMARY_ABS': getattr(G, 'EPS_D_PRIMARY_ABS', '<missing>'),
        'TIE_BREAK_LEGAL_ORDER': getattr(G, 'TIE_BREAK_LEGAL_ORDER', '<missing>'),
        'TIE_BREAK_CANONICAL': getattr(G, 'TIE_BREAK_CANONICAL', '<missing>'),
        'INSTALL_REQUIRES_CERTIFICATION': getattr(G, 'INSTALL_REQUIRES_CERTIFICATION',
                                                  '<missing>'),
        'REFUSAL_REASONS': repr(getattr(G, 'REFUSAL_REASONS', '<missing>'))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    record_self_identity()
    verify_attribution()
    print('--- C: legal-fixture A2 control (same customer, two legal actions, exact tie) ---')
    c1 = verify_legal_control(LEGAL_ORDER_POS1_HEAD, 'legal_head_is_larger_canonical_key')
    c2 = verify_legal_control(LEGAL_ORDER_POS0_HEAD, 'legal_head_is_smaller_canonical_key')
    rec('C_order_reversal_discriminates_legal_order_from_canonical_key',
        {'run1_chosen_index': c1['chosen'], 'run2_chosen_index': c2['chosen'],
         'canonical_heads': [c1['canon'], c2['canon']],
         'verdict': 'legal-order rule' if c1['chosen'] != c2['chosen'] else
                    'NOT distinguishable from the recorded step (both runs recorded index 0)'})
    ck('C3 the canonical-key head differs from the legal-order head in run1 (so this fixture '
       'really does discriminate the two tie rules) while the recorded choice stays legal-first',
       c1['canon'] == 1 and c1['chosen'] == 0,
       {'run1_canonical_head': c1['canon'], 'run1_chosen': c1['chosen'],
        'run2_canonical_head': c2['canon'], 'run2_chosen': c2['chosen']})
    verify_coherence_probe()
    verify_a3_two_way()

    fails = [c for c in CASES if not c[1]]
    print('\n=== %d rows, %d PASS, %d FAIL ===' % (len(CASES), len(CASES) - len(fails),
                                                   len(fails)))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
             'records': RECORDS,
             'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails), 'n_fail': len(fails)},
            indent=2, default=repr, sort_keys=True), encoding='utf-8')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
