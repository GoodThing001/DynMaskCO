#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-2 ERRATA CONTROLS -- the clock upper bound, the callback scope and the install gate.

An independent review raised four scope questions about the first G-2 report.  This file answers them
with executable controls and does NOT modify the earlier report or the earlier 21-control artifact;
the errata is additive and both artifacts stay readable side by side.

    E-01  does `_force_return_all` hand the executor a clock ABOVE the horizon?  (synthetic control)
    E-02  does the corrected clock basis change the numbers in the registered domain, and can it
          change them elsewhere?
    E-03  the advance bound without the unproven time-side tightening
    E-04  the callback scope: which extension points can advance the cold chain, and what witnesses
          the count when they are open
    E-05  `run_repair_transaction(certify_fn=None)` is FAIL-CLOSED, not "an uncertified install"
    E-06  the horizon witness is offline-only; the online path cannot read hidden orders
    E-07  the errata is derivation-only in the registered domain (numbers reproduce)

Usage: python scripts/tests/test_q3_g2b_clock_scope.py --out <json path>
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))

import coldchain_contract as CC                                            # noqa: E402
import q3_g2_time_identity_experimental as G2                              # noqa: E402
import q3_g2b_clock_scope_experimental as G2B                              # noqa: E402

ENV_SRC = EXT / 'scripts' / 'simulation' / 'strict_online_env.py'
ADAPTER_SRC = EXT / 'scripts' / 'simulation' / 'd3_geo_adapter.py'
ROWS = []
LIMITS = ('read-only static analysis + two tiny synthetic episodes + one refusal-path call into the '
          'install entry; no training, no GPU, no formal rollout, no final-test seed, no production '
          'certification wiring')


def ck(cid, name, ok, detail=None, conclusion='', status=None):
    st = status or ('PASS' if ok else 'FAIL')
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-6s %s' % (st, cid, name[:94]))


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_identity():
    """Committed-blob digests (the portable identity; a worktree digest is not, under autocrlf)."""
    import subprocess
    out = {}
    for rel in G2.AUDITED_FILES:
        blob = subprocess.run(['git', 'show', 'HEAD:%s/%s' % (EXT.name, rel)],
                              cwd=str(EXT.parent), capture_output=True).stdout
        if blob:
            out[rel] = hashlib.sha256(blob).hexdigest()
        else:
            out[rel] = sha256_file(EXT / rel)
    return out


# ---------------------------------------------------------------------------------------------
# a 2-node episode whose only vehicle ends 'ready' at the horizon, so `_force_return_all` fires
# ---------------------------------------------------------------------------------------------
class ServeOne(object):
    """Assigns the single customer once, then nothing: the vehicle stays 'ready' at its node."""

    def __init__(self):
        self.done = False

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        free = [c for c in visible_ids if not served_mask[c]]
        for v in vehicles:
            if v.status not in ('idle', 'ready'):
                continue
            if replan_ids is not None and v.vehicle_id not in replan_ids:
                continue
            v.mutable_suffix = [] if self.done or not free else [int(free[0])]
        if free:
            self.done = True


def run_force_return_episode(n_max=20003):
    """Serve late, then let the loop run out of events so the forced return leg is taken."""
    import numpy as np
    from strict_online_env import StrictOnlineEnv
    contract = CC.default_pilot_contract()
    data = {
        'coords': np.array([[[0.0, 0.0], [0.9, 0.0]]], dtype=np.float32),
        'demands': np.array([[0.0, 5.0]], dtype=np.float32),
        'tw_start': np.array([[0.0, 23.5]], dtype=np.float32),
        'tw_end': np.array([[24.0, 24.0]], dtype=np.float32),
        'service_time': np.array([[0.0, 0.4]], dtype=np.float32),
        'reveal_time': np.zeros((1, 2), dtype=np.float32),
        'temp_class': np.array([[0, 0]], dtype=np.int32),
        'initial_quality': np.ones((1, 2), dtype=np.float32),
    }
    env = StrictOnlineEnv(data, capacity=contract.operational.shared_vehicle_capacity,
                          num_vehicles=1, replanner=ServeOne(), coldchain_contract=contract)
    monitor = G2B.attach_advance_monitor(env, n_max)
    clocks = []
    env.snapshot_hook = lambda *a, **k: clocks.append(float(a[2]))
    env.run(0)
    return {'horizon': 24.0, 'travel_one_leg': 0.9, 'monitor_counts': monitor['counts'],
            'max_clock_seen_by_executor': monitor['max_clock'](),
            'event_clocks': clocks, 'n_events': int(env.event_id)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    contract = CC.default_pilot_contract()
    census = G2.call_graph_census(ENV_SRC.read_text(encoding='utf-8'))
    src_now = source_identity()

    # ============================================================ E-01 clock above the horizon
    ep = run_force_return_episode()
    supra = ep['max_clock_seen_by_executor']
    ok1 = (supra is not None and supra > ep['horizon'] + 1e-9)
    ck('E-01', 'the horizon does NOT bound every timestamp: `_force_return_all` commits a return leg '
       'and advances the cold chain to `depart + travel(current, 0)`, which lands ABOVE `tw_end[0]` '
       '-- so the ulp budget must be taken over a clock bound, not over the event-loop horizon',
       ok1,
       {'episode': ep, 'excess_over_horizon': (None if supra is None else supra - ep['horizon']),
        'mechanism': ('strict_online_env.py:507 -> _force_return_all (:658) -> _commit(...,0) sets '
                      'return_finish = depart + travel(:273) -> _advance_fleet(finish) -> '
                      '_advance_coldchain_to(:304)'),
        'reading': ('the first G-2 report asserted "every clock in the episode <= T_max" (L1-6); '
                    'that assertion is refuted here by execution, not by argument'),
        'grading': 'PROVED (by counterexample in the corpus own code path)'},
       conclusion='the horizon is not a clock bound')

    # ============================================================ E-02 the corrected basis
    registered = G2B.clock_upper_bound(24.0, math.sqrt(2.0), 'coordinate_box_diagonal')
    cmp_reg = G2B.basis_comparison(24.0, registered['T_clock_sup'], 20003)
    covers = (supra is not None and supra <= registered['T_clock_sup'] + 1e-12)
    crossing_h, crossing_travel = 1048575.5, 1.0
    crossing = G2B.clock_upper_bound(crossing_h, crossing_travel, 'caller_declared_with_witness')
    cmp_cross = G2B.basis_comparison(crossing_h, crossing['T_clock_sup'], 20003)
    from fractions import Fraction as F
    ulp_doubled = (F(crossing['ulp_ceiling_clock']) == 2 * F(crossing['ulp_ceiling_horizon']))
    ok2 = (covers and registered['same_binade'] is True
           and cmp_reg['numbers_differ'] is False
           and crossing['same_binade'] is False
           and ulp_doubled
           and cmp_cross['numbers_differ'] is True
           and cmp_cross['ratio_clock_over_horizon'] > 1.9)
    ck('E-02', 'the corrected clock basis covers the observed timestamp and is numerically IDENTICAL '
       'in the registered domain (24 and 24+sqrt(2) share a binade), yet it doubles the slack when '
       'the bound crosses a power of two -- so the old derivation was wrong even where it agreed',
       ok2,
       {'registered_domain': {'bound': registered, 'comparison': cmp_reg,
                              'observed_clock': supra, 'covers_observed': covers},
        'binade_crossing_case': {'horizon': crossing_h, 'travel': crossing_travel,
                                 'bound': crossing, 'comparison': cmp_cross},
        'grading': 'PROVED_CONDITIONAL (the travel bound rests on the coordinate-box domain '
                   'invariant)'},
       conclusion='same numbers here, different derivation, and it matters elsewhere')

    # ============================================================ E-03 advance bound, max_iter only
    adopted = G2B.advance_budget_max_iter(census['max_iter'], census['pre_loop'],
                                          census['per_iteration'], census['termination'],
                                          n_iter_current=1001)
    ok3 = (adopted['n_max_per_vehicle'] == 20003
           and adopted['not_adopted']['time_side_tightening'] == 'NOT_ADOPTED'
           and adopted['not_adopted']['n_would_be'] == 2005)
    ck('E-03', 'the ADOPTED advance bound uses `max_iter` alone (20003); the tighter form '
       '`1 + floor(T_max/eps_event)` is recorded as NOT ADOPTED because it needs "every iteration '
       'moves the clock by more than eps_event", which the source never asserts',
       ok3,
       {'adopted': adopted,
        'earlier_report_C6': 'reported the time side as the binding constraint for small horizons; '
                             'that row is superseded by this one (derivation only, not numbers)',
        'grading': 'PROVED_CONDITIONAL on max_iter being the loop cap'},
       conclusion='the number did not change; the justification did')

    # ============================================================ E-04 callback scope
    open_sinks = [s for s in G2B.CALLBACK_SINKS if s[3] == 'OPEN']
    # who in the tree can call the advance?  AST scan of every non-test module under scripts/
    callers = {}
    for path in sorted((EXT / 'scripts').rglob('*.py')):
        rel = path.relative_to(EXT).as_posix()
        if '/tests/' in rel or rel.endswith('_experimental.py'):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:                                                # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == '_advance_coldchain_to'):
                callers.setdefault(rel, []).append(node.lineno)
    expected_callers = {'scripts/simulation/strict_online_env.py',
                        'scripts/simulation/d3_execution_authority.py',
                        'scripts/simulation/d3_write_exclusion.py'}
    unexpected_callers = {k: v for k, v in callers.items() if k not in expected_callers}
    in_tree_ok = not unexpected_callers
    # the monitor must CATCH an over-count: run the same episode with n_max = 1
    caught = None
    try:
        run_force_return_episode(n_max=1)
    except G2B.Refusal as exc:
        caught = exc.code
    except Exception as exc:                                               # noqa: BLE001
        caught = 'error:%s' % type(exc).__name__
    ok4 = (len(open_sinks) == 5 and in_tree_ok and caught == 'COUNT_BOUND_VIOLATED')
    # The AST scan sees attribute calls only.  Two modules REPLACE the method and forward to the
    # original through a subscript dispatch, which the scan cannot see; they are declared by hand
    # rather than quietly left out of the census.
    delegating_wrappers = [
        {'file': 'scripts/simulation/d3_execution_authority.py', 'lines': '196-198, 230',
         'what': 'replaces env._advance_coldchain_to and forwards to the original'},
        {'file': 'scripts/simulation/d3_write_exclusion.py', 'lines': '102-104, 140',
         'what': 'same pattern for the write-exclusion guard'},
    ]
    ck('E-04', 'the callback scope is enumerated rather than assumed: five extension points are OPEN '
       '(they receive `env` and may call any env method), no module in the tree outside the env and '
       'its two wrappers advances the chain, and the runtime monitor catches an advance that a '
       'callback issues -- so the static bound is bonded to a per-episode witness',
       ok4,
       {'callback_sinks': [{'sink': s[0], 'where': s[1], 'scope': s[3], 'note': s[4]}
                           for s in G2B.CALLBACK_SINKS],
        'open_sinks': [s[0] for s in open_sinks],
        'advance_callers_in_tree': callers, 'unexpected_callers': unexpected_callers,
        'delegating_wrappers_not_visible_to_the_scan': delegating_wrappers,
        'scan_limitation': ('the census counts `self._advance_coldchain_to(...)` attribute calls; a '
                            'wrapper that swaps the method and forwards through a subscript is '
                            'invisible to it, so those two files are declared explicitly'),
        'monitor_caught': caught,
        'consequence': ('the static bound 20003 holds for the CLOSED core; for an arbitrary installed '
                        'callback it holds only together with the monitor, which raises '
                        'COUNT_BOUND_VIOLATED')},
       conclusion='bound + monitor, never bound alone')

    # ============================================================ E-05 install gate is fail-closed
    gate = G2B.install_gate_witness(ADAPTER_SRC.read_text(encoding='utf-8'))
    import d3_geo_adapter as GEO
    installs = {'n': 0}

    def install_fn(plan):
        installs['n'] += 1
        return True

    ledger = GEO.InstallLedger(install_fn=install_fn)
    result = GEO.run_repair_transaction(ledger=ledger, plans={'v0': [0]}, certify_fn=None)
    ledger_state = {'n_installs': ledger.n_installs,
                    'n_pre_sampling_refusals': ledger.n_pre_sampling_refusals,
                    'n_rollbacks': ledger.n_rollbacks,
                    'installed_plans': list(ledger.installed_plans),
                    'events': list(ledger.events), 'returned': result}
    ok5 = (gate['fails_closed'] and result is False and ledger.n_installs == 0
           and ledger.n_pre_sampling_refusals == 1 and installs['n'] == 0
           and ledger.installed_plans == []
           and GEO.INSTALL_REQUIRES_CERTIFICATION is True)
    ck('E-05', 'the install entry is FAIL-CLOSED when no certifier is supplied: '
       '`run_repair_transaction(certify_fn=None)` refuses BEFORE sampling '
       '(failure_reason=no_certifier), installs nothing, and the install callback is never invoked. '
       '"There is no certifier" must NOT be written as "an uncertified plan was installed"',
       ok5,
       {'source_witness': gate, 'ledger': ledger_state,
        'install_requires_certification': GEO.INSTALL_REQUIRES_CERTIFICATION,
        'correction_of_earlier_wording': ('the first G-2 report wrote "certify_fn=None skips '
                                          'certification" without distinguishing the transaction '
                                          '(fail-closed) from `stepwise_sample_d3` (which does run '
                                          'uncertified when called directly, and is why V3 is '
                                          'NOT_RUN in the two-arm protocol)'),
        'residual_gap': ('the M1-b certifier still has NO production caller, so no path produces a '
                         'certified install -- that is a wiring gap, not a silent-install defect')},
       conclusion='no uncertified install exists on this path')

    # ============================================================ E-06 offline/online stage split
    import numpy as np
    tw = np.array([[24.0, 12.0, 30.0]], dtype=np.float32)
    rt = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    off = G2B.offline_horizon_witness(tw[0], rt[0], stage='offline')
    online_refused = None
    try:
        G2B.offline_horizon_witness(tw[0], rt[0], stage='online')
    except G2B.Refusal as exc:
        online_refused = exc.code
    # the online projection must refuse a parameter set that has no qualified clock bound
    ident_g2 = G2.build_identity(source={'files': src_now, 'basis': 'committed blob'},
                                 census=census, contract_hash=contract.contract_hash,
                                 contract=contract,
                                 horizon_witness=G2.classify_horizon(tw[0], rt[0]),
                                 max_iter=census['max_iter'], domain={'probe': True})
    g2_missing = None
    try:
        G2B.online_consumable(ident_g2)
    except G2B.Refusal as exc:
        g2_missing = exc.detail.get('missing')
    ident_errata = dict(ident_g2)
    ident_errata['clock_bound'] = {'T_clock_sup': registered['T_clock_sup']}
    ident_errata['advance_budget'] = dict(ident_g2['advance_budget'], n_max_per_vehicle=20003)
    online_view = G2B.online_consumable(ident_errata)
    ok6 = (off['reads_hidden_data'] is True and off['witness_equal'] is False
           and online_refused == 'ONLINE_MUST_NOT_READ_HIDDEN_DATA'
           and 'T_clock_sup' in (g2_missing or [])
           and 'T_clock_sup' in online_view
           and 'hidden_customer_max' not in online_view
           and online_view['certifiable'] is False)
    ck('E-06', 'the horizon witness is OFFLINE-ONLY: computing it with stage="online" is refused, the '
       'online projection exposes only already-qualified scalars (no hidden-order fields), and the '
       'first G-2 identity object is itself rejected as an online parameter set because it carries no '
       'clock bound',
       ok6,
       {'offline_witness': off, 'online_refusal': online_refused,
        'g2_identity_missing_fields': g2_missing,
        'online_view': online_view,
        'reading': ('a certificate may not become pass/fail by reading unrevealed orders; the '
                    'witness is computed once, in the qualification stage, and bound into the '
                    'identity')},
       conclusion='the witness never becomes an online gate')

    # ============================================================ E-07 derivation-only in this domain
    old_slack = G2.slack_budget(24.0, 20003)['total_slack']
    new_slack = G2B.slack_budget(registered['T_clock_sup'], 20003)['total_slack']
    ok7 = (old_slack == new_slack)
    ck('E-07', 'the errata is DERIVATION-ONLY in the registered domain: the slack computed from the '
       'horizon and from the corrected clock bound are the same number, so no earlier numeric '
       'conclusion is retracted -- only the reason it was ever valid is',
       ok7,
       {'slack_from_horizon': old_slack, 'slack_from_clock_bound': new_slack,
        'supersedes': {
            'L1-6': '"every clock in the episode <= T_max" -- refuted by E-01',
            'C1': 'the primary control compared H_max only; it never claimed the clock bound, but the '
                  'slack it reported rested on L1-6 -- the number stands, the derivation is replaced',
            'C6': 'the time-side iteration bound is NOT ADOPTED (E-03); the bound used is max_iter',
            'unchanged': ['C3 horizon witness', 'C4/C4b call-graph census', 'C5 inherited-reading '
                          'refusal', 'C7 unit scale', 'C8 tolerances', 'C9 identity digest',
                          'C10 the nine negative controls', 'C11 sources untouched']},
        'grading': 'PROVED (an equality of two exact rationals)'},
       conclusion='the numbers survive; the reasoning is corrected')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    verdict = 'G2_ERRATA_FAIL' if n_fail else 'G2_ERRATA_PASS'
    payload = {
        'schema': 'q3_g2b_clock_scope_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'supersedes': {
            'additive_to': ['q3_g2_time_identity controls (21/21, kept byte-intact)',
                            'G2_时间参数身份与strict_online来源资格审查_2026-10-17.md'],
            'inside_the_module': ['q3_g2_time_identity_experimental.slack_budget -- the ulp basis is '
                                  'the clock bound, not the horizon',
                                  'row C6 -- the time-side iteration tightening is NOT ADOPTED'],
            'declared_in': 'G2_独立审查勘误_时钟上界与回调作用域_2026-10-17.md section 7',
        },
        'source_identity': src_now,
        'contract_hash': contract.contract_hash,
        'census_digest': census['census_digest'],
        'verdict': verdict, 'n_rows': len(ROWS), 'n_pass': n_pass, 'n_fail': n_fail,
        'clock_bound': registered, 'basis_comparison': cmp_reg,
        'advance_bound_adopted': adopted,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True, 'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED', 'L2_NOT_ESTABLISHED': True,
            'A3_NOT_ESTABLISHED': True, 'stable_online_gain': False,
            'HISTORICAL_BACKBONE_UNVERIFIED': True, 'C0': 'AWAITING_USER_BUDGET_RULING',
            't_min': 'always 0', 'final_test_seeds': 'SEALED'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d errata controls: %d PASS, %d FAIL -> %s ===' % (len(ROWS), n_pass, n_fail,
                                                                   verdict))
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
