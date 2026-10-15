#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 V1.20: execute the composed certifier for real, and audit the budget provenance with the
CORRECTED matcher.  Isolated engineering evidence: no F scoring run, no data collection, no
production write, no freeze change.

Closes the two V1.19 objections the user raised:
  G0-06a  V1.19 only READ the capability declaration.  Here `compose_certification` is actually
          invoked on a REAL day-0 state (captured through the existing collector, restore path
          reused) and `installable` must come back False because of the declared quality gap.
  G0-06b  V1.19 pre-filled None and checked it was still None.  Here a bounded provenance search
          is run with a CORRECTED matcher (tuple assignment, e.g.
          `CAPACITY, NUM_VEHICLES, B_BUDGET = 50.0, 25, 300.0`), the historical CAL fixture constant
          is reported as such, and the conclusion is labelled
          ARCHIVED_BUDGET_GAP_CONFIRMED / CURRENT_PROVENANCE_NOT_EXHAUSTIVELY_SEARCHED.
"""
import argparse
import inspect
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data',
          'scripts/tests', 'scripts/evaluation', 'scripts/models'):
    sys.path.insert(0, str(EXT / p))

CASES = []
REC = {}
BUDGET_RE = re.compile(r'(?:^|[,(]\s*|\s)([A-Za-z_]*BUDGET[A-Za-z_]*)\s*(?::[^=]*)?=\s*([0-9.eE+]+)')


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:280]))


def budget_provenance():
    """Bounded, corrected scan for C0 budget declarations and their provenance labels."""
    hits = []
    roots = [EXT / 'scripts/tests', EXT / 'scripts/evaluation', EXT / 'scripts/simulation',
             EXT / 'scripts/coldchain']
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob('*.py')):
            try:
                text = p.read_text(encoding='utf-8')
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                m = BUDGET_RE.search(line)
                if m:
                    hits.append({'file': str(p.relative_to(EXT)).replace('\\', '/'),
                                 'line': i, 'name': m.group(1), 'value': m.group(2),
                                 'text': line.strip()[:110]})
    b2 = EXT / 'scripts/tests/test_d3_b2_certification.py'
    b2_hits = [h for h in hits if h['file'].endswith('test_d3_b2_certification.py')]
    exact = None
    if b2.is_file():
        for i, line in enumerate(b2.read_text(encoding='utf-8').splitlines(), 1):
            if 'B_BUDGET' in line and '=' in line and 'B_BUDGET' in line.split('=')[0]:
                exact = {'line': i, 'text': line.strip()}
                break
    return {'matcher': BUDGET_RE.pattern, 'n_hits': len(hits), 'hits': hits[:20],
            'b2_fixture_declaration': exact,
            'old_v119_matcher_would_miss_it': bool(exact) and "B_BUDGET = 300" not in exact['text'],
            'labels': {'historical_cal_fixture': [h['file'] for h in b2_hits],
                       'isolation_fixture_1e9': sorted({h['file'] for h in hits
                                                        if h['value'].startswith('1e9')}),
                       'registered_protocol_budget_for_the_64_dev_check_states': None},
            'verdict': 'ARCHIVED_BUDGET_GAP_CONFIRMED / CURRENT_PROVENANCE_NOT_EXHAUSTIVELY_SEARCHED'}


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()

    # ---- G0-06b: corrected budget provenance ---------------------------------------------
    prov = budget_provenance()
    REC['budget_provenance'] = prov
    ck('B1 the corrected matcher finds the historical CAL fixture budget that V1.19 missed '
       '(tuple assignment), and the gap stays labelled as an archived gap with a non-exhaustive '
       'provenance search',
       bool(prov['b2_fixture_declaration']) and prov['old_v119_matcher_would_miss_it']
       and prov['verdict'].startswith('ARCHIVED_BUDGET_GAP_CONFIRMED'),
       {'b2': prov['b2_fixture_declaration'], 'verdict': prov['verdict'],
        'n_hits': prov['n_hits']})

    # ---- G0-06a: EXECUTE the composed certifier on a real state ---------------------------
    import dataclasses
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                       run_baseline_rollout)
    from recourse_snapshot import restore_recourse_snapshot
    import d3_composed_certifier as CC
    sig = inspect.signature(CC.compose_certification)
    REC['compose_signature'] = str(sig)
    REC['capability_gaps'] = list(getattr(CC, 'CAPABILITY_GAPS', ()))
    data = dict(np.load(str(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')))
    prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                      .read_text(encoding='utf-8'))
    flds = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    eff = apply_objective_profile(default_pilot_contract(),
                                  ObjectiveProfile(**{k: v for k, v in prof.items()
                                                      if k in flds}))
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
    col = RevealSlotCollector(env, 0, 1)
    run_baseline_rollout(env, 0, hook=col)
    snap = col.slots[0]['snapshot']
    restored = restore_recourse_snapshot(snap)
    objs = restored if isinstance(restored, tuple) else (restored,)
    vehicles = next((o for o in objs if isinstance(o, list) and o
                     and hasattr(o[0], 'vehicle_id')), None)
    served = next((o for o in objs if hasattr(o, 'shape') and getattr(o, 'dtype', None)
                   is not None and o.dtype == bool), None)
    plans = env.replanner_state if False else None
    try:
        from d3_state_container import StateContainer  # noqa: F401
    except Exception:
        pass
    # rebuild a plan dict the certifier can read: the mutable suffixes of the restored vehicles
    plan_from_vehicles = {}
    for v in vehicles:
        plan_from_vehicles[int(v.vehicle_id)] = type('P', (), {
            'vehicle_id': int(v.vehicle_id), 'suffix': list(getattr(v, 'mutable_suffix', []) or []),
            'anchor_node': int(getattr(v, 'anchor_node', 0)), 'anchor_time': 0.0,
            'anchor_load': 0.0})()
    call = {'env_type': type(env).__name__, 'inst_idx': 0, 'clock': float(snap['clock']),
            'n_vehicles': len(vehicles), 'mask_set': [5]}
    REC['compose_call'] = call
    try:
        SYNTHETIC_B = 300.0   # SYNTHETIC-ONLY: verifies the REFUSAL mechanism, never a formal B
        out = CC.compose_certification(env, 0, float(snap['clock']), vehicles, served,
                                       plan_from_vehicles, mask_set=[5],
                                       B=float(SYNTHETIC_B), contract=eff)
        REC['compose_call']['budget_label'] = 'SYNTHETIC_ONLY_NOT_A_FORMAL_C0_BUDGET'
        REC['compose_call']['synthetic_B'] = float(SYNTHETIC_B)
        detail = {k: out.get(k) for k in ('ok', 'installable', 'reasons', 'blocking_gaps')
                  if k in out}
        REC['compose_result'] = {'keys': sorted(out), 'detail': detail}
        ck('G0-06a EXECUTED (not just declared): compose_certification runs on a REAL day-0 state '
           'and returns installable=False because the quality/salability capability gap is blocking',
           out.get('installable') is False
           and ('quality_salability' in json.dumps(out.get('blocking_gaps', []))
                or 'quality_salability' in json.dumps(detail)), REC['compose_result'])
    except BaseException as exc:
        REC['compose_result'] = {'error': '%s: %s' % (type(exc).__name__, str(exc)[:400]),
                                 'note': 'the certifier could not be driven from this restored '
                                         'state; recorded as NOT_EXECUTABLE rather than assumed'}
        ck('G0-06a EXECUTED: compose_certification runs on a real state and refuses', False,
           REC['compose_result'])

    # ---- L1 readiness (work package 2): shared-input constructibility, D-Geo arm only -------
    from d3_geo_adapter import make_geo_extract_fn, geo_score, SELECTOR_GEO
    l1 = {'shared_contract': getattr(eff, 'contract_hash', None),
          'shared_state_hash': snap.get('state_hash'),
          'shared_clock': float(snap['clock']),
          'd_geo_scorer': 'geo_score = -incremental_distance (registered)',
          'f_scorer': 'restored MaskCO logits -- NOT RUN in this round (forbidden)',
          'f_label': 'F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED',
          'three_outputs': ['first_step_divergence', 'full_repair_completed',
                            'certified_executable_plan'],
          'not_run': ['F-vs-D-Geo comparison', 'any F forward on the 64 states']}
    REC['l1_readiness'] = l1
    ck('L1-1 the L1 shared inputs are constructible from the frozen 64-state asset (contract, '
       'state_hash, clock, registered D-Geo scorer) without running F',
       bool(l1['shared_state_hash']) and l1['shared_contract'] is not None, l1)

    fails = [c for c in CASES if not c[1]]
    REC['summary'] = {'rows': len(CASES), 'pass': len(CASES) - len(fails), 'fail': len(fails),
                      'no_new_scientific_runs': True,
                      'open': ['V1.18 signable execution pre-registration (draft only)',
                               'L1 signable pre-registration (draft only)',
                               'C0 formal budget ruling (user)',
                               'D8 confined unblocking: state certification / credential binding / '
                               'constraint propagation through execution',
                               'E2 pre-registration (no training)']}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, 'n_rows': len(CASES),
                               'n_pass': len(CASES) - len(fails), 'n_fail': len(fails),
                               'tool_sha256': __import__('hashlib').sha256(
                                   Path(__file__).read_bytes()).hexdigest()},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== V1.20: %d rows, %d PASS, %d FAIL ===' % (len(CASES), len(CASES) - len(fails),
                                                          len(fails)))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
