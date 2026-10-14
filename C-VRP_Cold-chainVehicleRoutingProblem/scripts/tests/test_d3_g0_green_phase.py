#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 G0-Safety Green Phase (batch 1): DIFFERENTIAL acceptance test.

This is a NEW artefact.  It does NOT replace, rewrite or re-interpret the Red Phase harness
(`test_d3_g0_red_phase.py`), which stays byte-intact as the pre-fix counterexample evidence at
commit 87c6a5c.  Instead it runs the SAME registered invariants against TWO modules:

    PREFIX  = the frozen pre-fix `d3_geo_adapter.py` (byte-verified against 87c6a5c)
    LIVE    = the current `scripts/simulation/d3_geo_adapter.py`

so the evidence chain is explicit and falsifiable:

    block P : pre-fix  -> invariant VIOLATED   (the defect is real and this test detects it)
    block G : post-fix -> invariant SATISFIED  (cleanly: no exception standing in for fail-closed)
    block E : legal input -> live and prefix AGREE (F's selection rule is unchanged)
    block R : the residuals this batch does NOT own are still declared OPEN

A failing block G is a real defect.  A failing block P means the "counterexample" was not one.
Block R exists so this suite can never be read as claiming A-3b / A-5c are fixed.

Scope (user ruling 2026-10-14): G0-Safety batch 1 only -- mandatory certification, score
shape/length validation, GEO index alignment, install idempotency, certification<->install
identity binding, three-state rollback.  Persistence (A-5c) and the real `VehicleState`
commitment freeze (A-3b) belong to G0-Evidence batch 2 and are deliberately NOT implemented.

Synthetic fixtures and injected hooks only: no MaskCO checkpoint, no scientific data, no training.
Run: python scripts/tests/test_d3_g0_green_phase.py        (exit 0 == chain complete)
"""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np                                              # noqa: E402
import d3_geo_adapter as LIVE                                   # noqa: E402
import action_contract as AC                                    # noqa: E402
from test_d3_geo_adapter import _cand, _identity_apply, _plans, MUT   # noqa: E402

RESULTS = []
P0 = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
LF = lambda seq: (lambda P, rem: [t for t in seq if t[0] in rem])     # noqa: E731
# three physically-consistent candidates; the distance-true best is customer 6 (delta 1.0)
SEQ3 = [_cand(4, 'anchored', 3, 0, 5.0), _cand(5, 'anchored', 3, 1, 6.0),
        _cand(6, 'new_route', -1, 0, 1.0)]
SEQ1 = [_cand(4, 'anchored', 3, 0, 3.0)]
SEQ2T = [_cand(5, 'anchored', 3, 0, 5.0), _cand(4, 'anchored', 3, 1, 5.0)]   # exact distance tie


def chk(block, name, ok, detail=None):
    RESULTS.append({'block': block, 'name': name, 'ok': bool(ok), 'detail': detail})


def attempt(fn):
    """Return `(value, None)` or `(None, 'Type: message')` so a crash is never mistaken for a pass."""
    try:
        return fn(), None
    except BaseException as exc:                                 # noqa: BLE001 (classified on purpose)
        return None, '%s: %s' % (type(exc).__name__, str(exc)[:140])


def repo_root():
    p = HERE
    while p is not None and not (p / 'MASKCO_code').is_dir():
        p = p.parent
    if p is None:
        raise SystemExit('repo root not found (no MASKCO_code/ above %s)' % HERE)
    return p


def load_prefix():
    """Load the frozen pre-fix module under its own name (never shadowing the live module)."""
    root = repo_root()
    path = (root / 'research' / 'review_3080_20261010' / 'd3_g0_green_phase_root_v1' /
            'frozen_prefix' / 'd3_geo_adapter__d3_geo_adapter.py')
    if not path.is_file():
        raise SystemExit('frozen pre-fix module missing: %s  (run freeze_prefix.py first)' % path)
    spec = importlib.util.spec_from_file_location('d3_geo_adapter_prefix', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['d3_geo_adapter_prefix'] = mod
    spec.loader.exec_module(mod)
    return mod, path


def accepts(mod, kwargs_name):
    """Does this module's InstallLedger.finish accept the certification-evidence keyword?"""
    return kwargs_name in inspect.signature(mod.InstallLedger.finish).parameters


def evidence(mod, **kw):
    """Pass certification evidence only where the gate has a concept for it.

    The pre-fix gate has no such concept at all -- that absence IS audit item A-4b's first half.
    """
    return kw if accepts(mod, 'certified_plan_hash') else {}


def extract_of(mod):
    return mod.make_geo_extract_fn(lambda P, r, l: {}, allowed_vehicle_ids=MUT)


def sample(mod, *, score_fn, extract_fn=None, seq=None, mask=None, selector=None, **kw):
    return mod.stepwise_sample_d3(
        score_fn=score_fn, extract_fn=extract_fn or extract_of(mod), legal_fn=LF(seq or SEQ3),
        apply_fn=_identity_apply, plans=P0, mask_set=mask or [4, 5, 6], mutable_ids=MUT,
        selector=selector or mod.SELECTOR_GEO, deterministic=True, **kw)


# ===========================================================================
# cases -- each runs against ONE module and returns raw observations
# ===========================================================================
def c_A2_no_certifier(mod):
    installed = []
    led = mod.InstallLedger(install_fn=lambda p: installed.append(p))
    returned = mod.run_repair_transaction(
        ledger=led, score_fn=mod.geo_score, extract_fn=extract_of(mod), legal_fn=LF(SEQ1),
        apply_fn=_identity_apply, plans=P0, mask_set=[4], mutable_ids=MUT,
        selector=mod.SELECTOR_GEO, deterministic=True)
    return {'returned': returned, 'n_installs': led.n_installs, 'n_install_fn_calls': len(installed),
            'n_pre_sampling_refusals': getattr(led, 'n_pre_sampling_refusals', 'ABSENT'),
            'n_rollbacks': led.n_rollbacks,
            'last_event': led.events[-1] if led.events else None}


def c_A2_identity_mismatch(mod):
    """Certify plan A, then try to install plan B: counting certifications cannot catch this."""
    led = mod.InstallLedger(install_fn=lambda p: None)
    plan_a = _plans({1: (3, [4]), 2: (0, []), 3: (0, [5])})
    plan_b = _plans({1: (3, [4, 9]), 2: (0, []), 3: (0, [5])})
    h_a, h_b = AC.plan_hash(plan_a), AC.plan_hash(plan_b)
    returned = led.finish(failure_reason=None, plan=plan_b, original_plan=plan_a,
                          **evidence(mod, certified=True, certified_plan_hash=h_a,
                                     certifier='test.certifier', deadline_ok=True))
    return {'returned': returned, 'n_installs': led.n_installs, 'hash_certified': h_a,
            'hash_installed': h_b, 'hashes_differ': h_a != h_b,
            'last_event': led.events[-1] if led.events else None}


def c_A2_deadline(mod):
    led = mod.InstallLedger(install_fn=lambda p: None)
    h = AC.plan_hash(P0)
    returned = led.finish(failure_reason=None, plan=P0, original_plan=P0,
                          **evidence(mod, certified=True, certified_plan_hash=h,
                                     certifier='test.certifier', deadline_ok=False))
    return {'returned': returned, 'n_installs': led.n_installs,
            'last_event': led.events[-1] if led.events else None}


def c_A1_short(mod):
    """A-1a is about the SCORE vector vs `legal`, so it uses the HISTORICAL selector -- exactly
    the path the Red Phase harness measured.  (On the GEO path a short score vector does not
    silently pick a prefix: it crashes in the evidence recorder at `scores[idx]`, a different
    manifestation of the same unchecked contract.)"""
    _p, steps, _r, fr, diag = sample(mod, score_fn=lambda st: [0.0],
                                     selector=mod.SELECTOR_HISTORICAL)
    return {'fr': fr, 'chosen': [s['customer'] for s in steps],
            'n_legal': 3, 'score_len': 1, 'diag_score_contract': diag.get('score_contract')}


def c_A1_long(mod):
    """An over-long score vector is NOT reliably fail-loud: whether it is silent depends on
    where the flattened argmax lands relative to the shrinking `legal` list, step by step.
    Both branches are measured separately so the outcome cannot be reported as one thing."""
    def run(scores):
        return attempt(lambda: sample(mod, score_fn=lambda st: scores,
                                      selector=mod.SELECTOR_HISTORICAL))

    inside, e_in = run([5.0, 4.0, 3.0, 2.0])      # argmax stays index 0 -> inside every step
    beyond, e_by = run([1.0, 2.0, 9.0, 3.0])      # argmax index 2 -> outside once legal shrinks
    return {'inside_exception': e_in,
            'inside_fr': inside[3] if inside else None,
            'inside_chosen': [s['customer'] for s in inside[1]] if inside else None,
            'beyond_exception': e_by,
            'beyond_fr': beyond[3] if beyond else None,
            'n_legal': 3, 'score_len': 4}


def c_A1_2d(mod):
    scores_2d = np.array([[0.0, 0.0], [9.0, 0.0], [1.0, 0.0]])
    flat_idx = mod.select_historical_argmax(scores_2d)      # F's primitive on a 2-D input
    intended_idx = int(np.argmax(scores_2d[:, 0]))
    out, exc = attempt(lambda: sample(mod, score_fn=lambda st: scores_2d))
    return {'primitive_flat_index': flat_idx, 'intended_row_index': intended_idx,
            'primitive_mis_indexes': flat_idx != intended_idx,
            'adapter_exception': exc,
            'adapter_fr': out[3] if out else None,
            'adapter_diag': (out[4].get('score_contract') if out else None)}


def c_A1d_geo(mod):
    """GEO index alignment: aligned scorer, but the injected state's GEO vector is truncated."""
    base = extract_of(mod)

    def truncating(P, rem, legal):
        st = base(P, rem, legal)
        st[mod.GEO_SCORE_KEY] = st[mod.GEO_SCORE_KEY][:2]
        return st

    out, exc = attempt(lambda: sample(mod, score_fn=lambda st: [0.0, 0.0, 0.0],
                                      extract_fn=truncating))
    return {'exception': exc, 'fr': out[3] if out else None,
            'chosen': [s['customer'] for s in out[1]] if out else None,
            'true_best_customer': SEQ3[2][0], 'delta_len_injected': 2, 'n_legal': 3,
            'diag_geo_contract': (out[4].get('geo_contract') if out else None)}


def c_A1d_vid(mod):
    """The vehicle_target_vid index feeds the tie KEY, so a short one mis-keys the tie."""
    base = extract_of(mod)

    def truncating(P, rem, legal):
        st = base(P, rem, legal)
        st['vehicle_target_vid'] = st['vehicle_target_vid'][:2]
        return st

    out, exc = attempt(lambda: sample(mod, score_fn=lambda st: [0.0, 0.0, 0.0],
                                      extract_fn=truncating))
    return {'exception': exc, 'fr': out[3] if out else None,
            'diag_geo_contract': (out[4].get('geo_contract') if out else None)}


def c_A4b_identity(mod):
    """Install once with full evidence; then mutate the source plan and re-read the ledger."""
    retained = []
    led = mod.InstallLedger(install_fn=lambda p: retained.append(p))
    plan, _s, _r, fr, diag = mod.stepwise_sample_d3(
        score_fn=mod.geo_score, extract_fn=extract_of(mod), legal_fn=LF(SEQ1),
        apply_fn=AC.apply_action, plans=P0, mask_set=[4], mutable_ids=MUT,
        selector=mod.SELECTOR_GEO, deterministic=True, certify_fn=lambda p: (True, {}))
    h_cert = diag.get('certified_plan_hash')
    expected_hash = AC.plan_hash(plan)              # BEFORE any mutation (see below)
    returned = led.finish(failure_reason=fr, plan=plan, original_plan=P0,
                          **evidence(mod, certified=diag.get('certified'),
                                     certified_plan_hash=h_cert,
                                     certifier=diag.get('certifier'), deadline_ok=True))
    entry_before = ({v: list(p.suffix) for v, p in led.installed_plans[0].items()}
                    if led.installed_plans else None)
    receipt_before = dict(led.installs[0]) if getattr(led, 'installs', None) else None
    # external mutation of the object that was handed to the ledger (the mutation is REAL: the
    # source plan's own hash changes, which is what `hash_after_mutation != expected_hash` shows)
    if retained:
        vid = sorted(retained[0].keys())[0]
        retained[0][vid] = AC.VehiclePlan(vid, 0, 0.0, 0.0, [999])
    hash_after_mutation = AC.plan_hash(plan)
    entry_after = ({v: list(p.suffix) for v, p in led.installed_plans[0].items()}
                   if led.installed_plans else None)
    receipt_after = dict(led.installs[0]) if getattr(led, 'installs', None) else None
    return {'returned': returned, 'fr': fr, 'n_installs': led.n_installs,
            'certifier': diag.get('certifier'), 'certified_plan_hash': h_cert,
            'expected_hash': expected_hash, 'hash_after_mutation': hash_after_mutation,
            'mutation_was_real': hash_after_mutation != expected_hash,
            'receipt_plan_hash': (receipt_before or {}).get('plan_hash'),
            'entry_before': entry_before, 'entry_after': entry_after,
            'ledger_entry_stable_under_external_mutation': entry_before == entry_after,
            'receipt_stable_under_external_mutation': receipt_before == receipt_after,
            'install_event': led.events[-1] if led.events else None}


def c_A5b_idempotent(mod):
    led = mod.InstallLedger(install_fn=lambda p: None)
    h = AC.plan_hash(P0)
    first = led.finish(failure_reason=None, plan=P0, original_plan=P0,
                       **evidence(mod, certified=True, certified_plan_hash=h,
                                  certifier='test.certifier', deadline_ok=True))
    second = led.finish(failure_reason=None, plan=P0, original_plan=P0,
                        **evidence(mod, certified=True, certified_plan_hash=h,
                                   certifier='test.certifier', deadline_ok=True))
    return {'first': first, 'second': second, 'n_installs': led.n_installs,
            'n_refusals': getattr(led, 'n_refusals', 'ABSENT'),
            'last_event': led.events[-1] if led.events else None}


def c_A5d_rollback(mod):
    unbacked = mod.InstallLedger(install_fn=lambda p: None, rollback_fn=None)
    unbacked.finish(failure_reason='no_legal', plan=dict(P0), original_plan=dict(P0))
    ev_unbacked = unbacked.events[-1]
    restored_calls = []
    backed = mod.InstallLedger(install_fn=lambda p: None,
                               rollback_fn=lambda p: restored_calls.append(p))
    backed.finish(failure_reason='no_legal', plan=dict(P0), original_plan=dict(P0))
    ev_backed = backed.events[-1]
    return {'unbacked_event': ev_unbacked,
            'unbacked_flagged': any('unverified' in str(k).lower() for k in ev_unbacked),
            'unbacked_restored_flag': ev_unbacked.get('committed_state_restored'),
            'backed_event': ev_backed, 'backed_rollback_fn_calls': len(restored_calls),
            'backed_n_state_restorations': getattr(backed, 'n_state_restorations', 'ABSENT')}


def c_failure_preserves_plan(mod):
    """Acceptance #4: NO failure path may install, return a partial plan, or mutate the source.

    Measured at both layers, because they answer different questions:
      * `stepwise_sample_d3` -> the RETURNED plan (must be the original, not a partial repair);
      * `run_repair_transaction` -> the ledger (must install ZERO times and report a failure).
    `run_repair_transaction` returns a boolean, so the returned plan is observed on the sampler.
    """
    def mutating_apply(P, action, allowed_vehicle_ids=None):
        out = {v: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load, list(p.suffix))
               for v, p in P.items()}
        out[1] = type(out[1])(1, out[1].anchor_node, 0.0, 0.0,
                              list(out[1].suffix) + [int(action.customer)])
        return out

    original = {v: [int(x) for x in p.suffix] for v, p in P0.items()}
    src_before = {v: [int(x) for x in p.suffix] for v, p in P0.items()}
    specs = {
        # name: (kwargs, sampled-after-a-success then no_legal / nan / timeout / score_len)
        'no_legal': dict(legal_fn=LF(SEQ1), mask_set=[4, 5], score_fn=mod.geo_score),
        'nan': dict(legal_fn=LF(SEQ3), mask_set=[4, 5, 6],
                    score_fn=lambda st: [float('nan')] * 3),
        'timeout': dict(legal_fn=LF(SEQ3), mask_set=[4, 5, 6], score_fn=mod.geo_score,
                        wall_clock_limit_s=-1.0),
        'score_len': dict(legal_fn=LF(SEQ3), mask_set=[4, 5, 6], score_fn=lambda st: [0.0]),
    }
    rows = {}
    for name, kw in specs.items():
        plan_out, exc_s = attempt(lambda kw=kw: mod.stepwise_sample_d3(
            extract_fn=extract_of(mod), apply_fn=mutating_apply, plans=P0, mutable_ids=MUT,
            selector=mod.SELECTOR_GEO, deterministic=True, **kw))
        installed = []
        led = mod.InstallLedger(install_fn=lambda p: installed.append(p))
        returned, exc_t = attempt(lambda kw=kw, led=led: mod.run_repair_transaction(
            ledger=led, extract_fn=extract_of(mod), apply_fn=mutating_apply, plans=P0,
            mutable_ids=MUT, selector=mod.SELECTOR_GEO, deterministic=True, **kw))
        rows[name] = {
            'sampler_exception': exc_s,
            'sampler_fr': plan_out[3] if plan_out else None,
            'sampler_returned_plan': ({v: [int(x) for x in p.suffix] for v, p in plan_out[0].items()}
                                      if plan_out else None),
            'tx_exception': exc_t, 'tx_returned': returned,
            'tx_n_installs': led.n_installs, 'tx_n_install_fn_calls': len(installed),
        }
    src_after = {v: [int(x) for x in p.suffix] for v, p in P0.items()}
    return {'rows': rows, 'original': original, 'source_plan_unchanged': src_before == src_after,
            'all_failed_closed': all(r['sampler_fr'] is not None for r in rows.values()),
            'all_sampler_returned_original':
                all(r['sampler_returned_plan'] == original for r in rows.values()),
            'all_tx_zero_installs': all(r['tx_n_installs'] == 0 for r in rows.values()),
            'no_exception': all(r['sampler_exception'] is None and r['tx_exception'] is None
                                for r in rows.values())}


# ---------------------------------------------------------------- block R residuals
def r_A5c_no_persistence(mod):
    led = mod.InstallLedger(install_fn=lambda p: None)
    surface = [k for k in dir(led) if any(t in k.lower() for t in
                                         ('receipt', 'persist', 'write', 'path', 'save', 'dump'))]
    return {'persistence_api_surface': surface, 'events_in_memory_only': True}


def r_A3b_no_commitment_guard(mod):
    sig = inspect.signature(mod.stepwise_sample_d3)
    body = Path(mod.__file__).read_text(encoding='utf-8')
    return {'adapter_accepts_committed_next': 'committed_next' in sig.parameters,
            'adapter_mentions_committed_next': 'committed_next' in body,
            'note': ('the commitment guard lives on VehicleState + the mask pool, OUTSIDE this '
                     'plan-level adapter; batch 1 must not fake it with a same-named field')}


# ---------------------------------------------------------------- block E equivalence
def eq_battery(mod):
    """Legal-input behaviour: the same decisions the pre-fix module made.

    The scorer is LENGTH-AWARE on purpose.  A fixed-length score list only stays aligned with
    `legal` at the first step (the mask shrinks every step), so a fixed list would be an ILLEGAL
    input from step 2 on -- and illegal input is exactly what the fix now refuses.  Using one
    would compare two different things and measure the new guard, not the selection rule.
    """
    score_fn = lambda st: [0.0] * len(st[mod.GEO_SCORE_KEY])      # noqa: E731
    out = {'geo': {}, 'historical': {}, 'primitive': {}}
    fixtures = {
        'exact_tie_2': (SEQ2T, [4, 5]),
        'distinct_3': (SEQ3, [4, 5, 6]),
        'one_legal': (SEQ1, [4]),
    }
    for name, (seq, mask) in fixtures.items():
        _p, steps, _r, fr, _d = sample(mod, score_fn=score_fn, seq=seq, mask=mask)
        out['geo'][name] = {'steps': steps, 'fr': fr}
        _p, steps_h, _r, fr_h, _d = mod.stepwise_sample_d3(
            score_fn=score_fn, extract_fn=extract_of(mod), legal_fn=LF(seq),
            apply_fn=_identity_apply, plans=P0, mask_set=mask, mutable_ids=MUT,
            selector=mod.SELECTOR_HISTORICAL, deterministic=True)
        out['historical'][name] = {
            'steps': [{k: v for k, v in s.items() if k != 'incremental_distance'}
                      for s in steps_h], 'fr': fr_h}
    battery = [[2.0, 2.0, 1.0], [0.0, 1e-13, 0.0], [1.0, 1.0], [3.0], [-1.0, -2.0, -2.0]]
    out['primitive'] = [mod.select_historical_argmax(b) for b in battery]
    return out


# ===========================================================================
def main():
    prefix, prefix_path = load_prefix()
    freeze = json.loads((repo_root() / 'research' / 'review_3080_20261010' /
                         'd3_g0_green_phase_root_v1' / 'PREFIX_FREEZE.json')
                        .read_text(encoding='utf-8'))

    # ---------------- block F: the differential is anchored to a FROZEN artefact -------------
    import hashlib
    want = freeze['files']['d3_geo_adapter']['sha256_at_base_commit']
    got = hashlib.sha256(prefix_path.read_bytes()).hexdigest()
    chk('F', 'the pre-fix module under test is byte-identical to commit %s'
        % freeze['base_commit'], want == got, {'want': want, 'got': got})

    # ---------------- block P: pre-fix must VIOLATE each registered invariant ----------------
    p_rows = {}
    for name, fn, violated in (
        ('P1 A-2   install without certification', c_A2_no_certifier,
         lambda r: r['n_installs'] == 1),
        ('P2 A-2   no identity concept to bind', c_A2_identity_mismatch,
         # pre-fix: plan B is installed although only plan A's hash existed -> 1 install, A != B
         lambda r: r['n_installs'] == 1 and r['hashes_differ']),
        ('P3 A-1a  short score silently used', c_A1_short,
         # pre-fix: a 1-element score vector drives every step (argmax 0) and the run reports
         # SUCCESS, i.e. 2 of the 3 candidates were never scored at all
         lambda r: r['fr'] is None and r['chosen'] == [4, 5, 6]),
        ('P4 A-1b  over-long score silently used', c_A1_long,
         # pre-fix: silent success when the argmax stays inside `legal` (the in-range branch);
         # the out-of-range branch raises IndexError instead, so the behaviour is position
         # dependent -- which is exactly why "fail-loud, acceptable" was the wrong verdict
         lambda r: r['inside_exception'] is None and r['inside_fr'] is None
         and r['inside_chosen'] == [4, 5, 6]),
        ('P5 A-1c  2-D score flattened by the primitive', c_A1_2d,
         lambda r: r['primitive_mis_indexes']),
        ('P6 A-1d  truncated GEO vector silently used', c_A1d_geo,
         lambda r: r['fr'] is None and r['chosen'][0] != r['true_best_customer']),
        ('P7 A-4b  no identity recorded', c_A4b_identity,
         lambda r: r['receipt_plan_hash'] is None and r['certified_plan_hash'] is None),
        ('P8 A-5b  finish twice installs twice', c_A5b_idempotent,
         lambda r: r['n_installs'] == 2),
        ('P9 A-5d  unbacked rollback unflagged', c_A5d_rollback,
         lambda r: not r['unbacked_flagged']),
    ):
        res, exc = attempt(lambda fn=fn: fn(prefix))
        ok = (exc is None and res is not None and violated(res))
        # a pre-fix API that cannot even EXPRESS the invariant is a violation too (P2)
        if not ok and exc is not None and 'unexpected keyword' in exc:
            ok, res = True, {'api_cannot_express_the_invariant': exc}
        p_rows[name] = {'violated_as_expected': ok, 'exception': exc, 'observed': res}
        chk('P', name, ok, {'exception': exc, 'observed': res})

    # ---------------- block G: post-fix must SATISFY the same invariants ---------------------
    g = {}
    for key, fn in (('A2_no_certifier', c_A2_no_certifier),
                    ('A2_identity_mismatch', c_A2_identity_mismatch),
                    ('A2_deadline', c_A2_deadline),
                    ('A1_short', c_A1_short), ('A1_long', c_A1_long), ('A1_2d', c_A1_2d),
                    ('A1d_geo', c_A1d_geo), ('A1d_vid', c_A1d_vid),
                    ('A4b_identity', c_A4b_identity), ('A5b_idempotent', c_A5b_idempotent),
                    ('A5d_rollback', c_A5d_rollback),
                    ('failure_preserves_plan', c_failure_preserves_plan)):
        res, exc = attempt(lambda fn=fn: fn(LIVE))
        g[key] = {'exception': exc, 'observed': res}

    def G(key, name, pred):
        row = g[key]
        ok = row['exception'] is None and row['observed'] is not None and pred(row['observed'])
        chk('G', name, ok, {'exception': row['exception'], 'observed': row['observed']})

    G('A2_no_certifier', 'G1  A-2 no certifier -> refused BEFORE sampling, zero installs',
      lambda r: r['n_installs'] == 0 and r['returned'] is False
      and r['n_pre_sampling_refusals'] == 1 and r['last_event']['reason'] == 'no_certifier'
      and r['last_event']['refused_before_sampling'] is True)
    G('A2_identity_mismatch', 'G2  A-2 certified plan A cannot be swapped for plan B',
      lambda r: r['returned'] is False and r['n_installs'] == 0
      and r['last_event']['reason'] == 'certified_plan_identity_mismatch' and r['hashes_differ'])
    G('A2_deadline', 'G3  A-2 decision deadline blown -> refused, zero installs',
      lambda r: r['returned'] is False and r['n_installs'] == 0
      and r['last_event']['reason'] == 'decision_deadline_exceeded')
    G('A1_short', 'G4  A-1a short score -> clean fail-closed score_len (no silent prefix)',
      lambda r: r['fr'] == 'score_len' and r['chosen'] == []
      and r['diag_score_contract']['len_scores'] == 1)
    G('A1_long', 'G5  A-1b over-long score -> clean fail-closed score_len in BOTH branches',
      lambda r: r['inside_fr'] == 'score_len' and r['beyond_fr'] == 'score_len'
      and r['inside_exception'] is None and r['beyond_exception'] is None
      and r['inside_chosen'] == [])
    G('A1_2d', 'G6  A-1c 2-D score -> clean fail-closed score_shape, never reaches selection',
      lambda r: r['adapter_fr'] == 'score_shape' and r['adapter_exception'] is None
      and r['adapter_diag']['ndim'] == 2)
    G('A1d_geo', 'G7  A-1d truncated GEO vector -> clean fail-closed geo_len (no silent miss)',
      lambda r: r['fr'] == 'geo_len' and r['exception'] is None
      and r['diag_geo_contract']['len_geo'] == 2 and r['diag_geo_contract']['n_legal'] == 3)
    G('A1d_vid', 'G8  A-1d short vehicle_target_vid -> clean fail-closed geo_len (was IndexError)',
      lambda r: r['fr'] == 'geo_len' and r['exception'] is None)
    G('A4b_identity', 'G9  A-4b install is identity-bound and immutable under external mutation',
      lambda r: r['returned'] is True and r['n_installs'] == 1
      and r['receipt_plan_hash'] == r['expected_hash'] == r['certified_plan_hash']
      and r['ledger_entry_stable_under_external_mutation'] is True
      and r['receipt_stable_under_external_mutation'] is True
      and r['mutation_was_real'] is True)
    G('A5b_idempotent', 'G10 A-5b second finish -> refused already_finished, exactly one install',
      lambda r: r['first'] is True and r['second'] is False and r['n_installs'] == 1
      and r['last_event']['reason'] == 'already_finished')
    G('A5d_rollback', 'G11 A-5d three-state rollback: unverified != restored',
      lambda r: r['unbacked_flagged'] is True and r['unbacked_restored_flag'] is False
      and r['backed_event']['committed_state_restored'] is True
      and r['backed_event']['rollback_unverified'] is False
      and r['backed_rollback_fn_calls'] == 1 and r['backed_n_state_restorations'] == 1)
    G('failure_preserves_plan', 'G12 acc#4 every failure path: sampler returns the ORIGINAL plan, '
      'transaction installs ZERO times, source plan unmutated, no crash',
      lambda r: r['all_failed_closed'] and r['all_sampler_returned_original']
      and r['all_tx_zero_installs'] and r['source_plan_unchanged'] and r['no_exception'])

    # ---------------- block E: legal-input behaviour unchanged --------------------------------
    eq_live, exc_l = attempt(lambda: eq_battery(LIVE))
    eq_pre, exc_p = attempt(lambda: eq_battery(prefix))
    chk('E', 'E1 no exception in either module\'s equivalence battery',
        exc_l is None and exc_p is None, {'live': exc_l, 'prefix': exc_p})
    chk('E', 'E2 GEO selector: identical steps + failure reason (legal input)', eq_live == eq_pre
        if (eq_live and eq_pre) else False,
        {'live': (eq_live or {}).get('geo'), 'prefix': (eq_pre or {}).get('geo')})
    chk('E', 'E3 historical selector: identical steps (projected) + failure reason',
        (eq_live or {}).get('historical') == (eq_pre or {}).get('historical'),
        {'live': (eq_live or {}).get('historical'), 'prefix': (eq_pre or {}).get('historical')})
    chk('E', 'E4 F first-maximum primitive identical on the whole battery',
        (eq_live or {}).get('primitive') == (eq_pre or {}).get('primitive'),
        {'live': (eq_live or {}).get('primitive'), 'prefix': (eq_pre or {}).get('primitive')})
    chk('E', 'E5 F first-maximum semantics themselves (no tolerance leaked in)',
        (eq_live or {}).get('primitive') == [0, 1, 0, 0, 0],
        {'battery': '[[2,2,1],[0,1e-13,0],[1,1],[3],[-1,-2,-2]] -> first maximum',
         'got': (eq_live or {}).get('primitive')})

    # ---------------- block R: residuals this batch does NOT own ------------------------------
    r5, e5 = attempt(lambda: r_A5c_no_persistence(LIVE))
    chk('R', 'R1 A-5c durable P_after_install receipt STILL ABSENT (G0-Evidence batch 2)',
        e5 is None and (r5 or {}).get('persistence_api_surface') == [],
        {'exception': e5, 'observed': r5})
    r3, e3 = attempt(lambda: r_A3b_no_commitment_guard(LIVE))
    chk('R', 'R2 A-3b commitment guard STILL OUTSIDE this adapter and STILL OPEN',
        e3 is None and (r3 or {}).get('adapter_accepts_committed_next') is False,
        {'exception': e3, 'observed': r3})
    chk('R', 'R3 no same-named fake field was added to "fix" A-3b',
        e3 is None and (r3 or {}).get('adapter_mentions_committed_next') is False,
        (r3 or {}).get('adapter_mentions_committed_next'))

    # ---------------- report ------------------------------------------------------------------
    blocks = {}
    for row in RESULTS:
        b = blocks.setdefault(row['block'], [0, 0])
        b[0] += int(row['ok'])
        b[1] += 1
    for row in RESULTS:
        print('  %s %-3s %s%s' % ('PASS' if row['ok'] else 'FAIL', row['block'], row['name'],
                                  '' if row['ok'] else ('   <- ' + json.dumps(
                                      row['detail'], ensure_ascii=False, default=str)[:220])))
    print('\nblocks:', json.dumps({k: '%d/%d' % (v[0], v[1]) for k, v in blocks.items()}))
    all_ok = all(row['ok'] for row in RESULTS)
    print('chain_complete:', all_ok)
    out = {
        'stage': 'D3_G0_Green_Phase_batch1_differential',
        'production_source_modified': True,
        'scientific_data_used': False, 'checkpoint_loaded': False,
        'prefix_base_commit': freeze['base_commit'],
        'prefix_sha256': want, 'prefix_sha256_observed': got,
        'results': RESULTS, 'blocks': {k: {'passed': v[0], 'total': v[1]}
                                       for k, v in blocks.items()},
        'chain_complete': all_ok,
        'note': ('block P proves the pre-fix defect is real and detectable; block G proves the '
                 'fix satisfies the invariant WITHOUT substituting a crash for fail-closed; '
                 'block E proves legal-input decisions are unchanged; block R declares the '
                 'residuals this batch does not own (A-3b, A-5c, contract/state identity).'),
    }
    dest = (repo_root() / 'research' / 'review_3080_20261010' / 'd3_g0_green_phase_root_v1' /
            'D3_G0_GREEN_PHASE_MATRIX.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
