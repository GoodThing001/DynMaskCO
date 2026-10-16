#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2 最终入口可信性封闭: NEGATIVE CONTROLS for A (projection provenance) and B (proof
parameter integrity).

Diagnostic only.  Records, for each case: whether the projection GATE passed, whether
Omega10/Omega11 ACCEPTED the input, whether future true attributes were READ, and whether the
full COMPOSITE passed.  The point is precisely that a later refusal (the continuous thermal
level) must NOT be counted as evidence that an earlier gate was sound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for _p in (EXT / 'scripts' / 'coldchain', EXT / 'scripts' / 'simulation',
           EXT / 'scripts' / 'evaluation', EXT / 'scripts' / 'm1b_adapter', HERE):
    sys.path.insert(0, str(_p))

import numpy as np                                                       # noqa: E402
import coldchain_state as CS                                             # noqa: E402
from coldchain_contract import ColdChainContractV2                       # noqa: E402
from coldchain_state import create_vehicle_state, dispatch_vehicle       # noqa: E402
from action_contract import VehiclePlan                                   # noqa: E402
import m1b_public_projection as PP                                       # noqa: E402
import m1b_quality_certificate as QC                                     # noqa: E402
import m1b_immutable_snapshot as IS                                      # noqa: E402
from m1b_q2_certifier import M1bQ2TrustFixCertifier                       # noqa: E402

ROWS = []
CONTRACT = ColdChainContractV2()
T_END = 24.0
SYNTHETIC_B = 12345.678


def row(tid, name, inp, actual, verdict, src, blocks):
    ROWS.append({'id': tid, 'name': name, 'MINIMAL_INPUT': inp, 'ACTUAL': actual,
                 'VERDICT': verdict, 'SOURCE': src, 'INSTALLABILITY_BLOCKER': blocks})
    print('  [%-7s] %-48s %s' % (tid, name[:48], verdict))


class V(object):
    def __init__(self, vid, status='ready', node=0, ready=0.0, load=0.0, cc=None,
                 committed_next=None, committed_finish=None, return_finish=None, suffix=None):
        self.vehicle_id, self.status, self.current_node = int(vid), status, int(node)
        self.ready_time, self.current_load = float(ready), float(load)
        self.committed_next, self.committed_finish = committed_next, committed_finish
        self.return_finish = return_finish
        self.mutable_suffix = list(suffix or [])
        self.served_route, self.dispatch_time = [], None
        self.coldchain_state, self.coldchain_updated_time = cc, None
        self.committed_coldchain_before, self.coldchain_accounting_anchor = None, None
        self.needs_replan, self.replan_reason = False, None


def fresh_cc(n_lots=1, temp_class=1, iq=9.0, qty=1.0, pickup=-1.0):
    st = dispatch_vehicle(create_vehicle_state(CONTRACT), CONTRACT)
    if n_lots > 0:
        lots = tuple(CS.CargoLotState(order_id=100 + i, quantity=qty, temp_class=temp_class,
                                      pickup_finish_time=pickup, initial_quality=iq,
                                      quality_remaining=iq) for i in range(n_lots))
        zl = list(st.zone_load)
        zl[int(temp_class)] = float(qty) * n_lots
        st = CS.replace(st, cargo_manifest=lots, zone_load=tuple(zl))
    return st


def plan(vid, at, suffix, load=0.0):
    return VehiclePlan(vid, 0, at, load, list(suffix))


def base_inputs():
    """A correct, fully-populated positive fixture: one loaded vehicle, one candidate stop."""
    vs = [V(0, 'ready', node=0, ready=0.0, cc=fresh_cc(n_lots=1, iq=9.0))]
    ps = {0: plan(0, 0.0, [7, 0])}
    return vs, ps


def real_snapshot_identity(vehicles=None, planes=None):
    """The identity the certifier will compute for this state, so a projection can be BOUND.

    A1/A2/A3 need no real binding because snapshot_identity=None skips the check -- which is the
    finding.  For B the projection must be properly bound, otherwise an unrelated mismatch stops
    the run before the quality gate and the parameter question is never reached.
    """
    vs, ps = base_inputs()
    if vehicles is not None:
        vs = vehicles
    if planes is not None:
        ps = planes
    snap, _, _ = IS.build_frozen_snapshot(vs, np.zeros(40, dtype=bool), ps,
                                          initial_quality_by_node={7: 9.0})
    return snap['state_identity']


class FakeEnv(object):
    """A minimal environment whose reveal_time reveals node 7 at clock 0 and NOTHING else."""

    def __init__(self, revealed=(7,), n=40):
        self.num_nodes = n
        import numpy as _np
        row = _np.full(n, 1e9, dtype=float)
        for r in revealed:
            row[int(r)] = 0.0
        self.reveal_time = row.reshape(1, n)


def good_projection(**kw):
    """A FACTORY-BOUND projection whose visible set really is node 7."""
    iq = {7: 9.0}
    iq.update(kw.pop('iq', {}) or {})
    proj, v, d = PP.build_projection_from_env(
        FakeEnv(revealed=(7,)), inst_idx=0, event_id=1, clock=0.0, contract=CONTRACT,
        snapshot_identity=kw.pop('snapshot_identity', real_snapshot_identity()),
        fields={'temp_class': {7: 1}, 'demand': {7: 1.0}, 'tw_start': {7: 0.0},
                'tw_end': {7: 1e9}, 'service_time': {7: 0.05}, 'initial_quality': iq})
    return proj


def run(projection, *, per_class_constants=None, t_min=0.0, vehicles=None, planes=None,
        times=None):
    vs, ps = base_inputs()
    if vehicles is not None:
        vs = vehicles
    if planes is not None:
        ps = planes
    c = M1bQ2TrustFixCertifier(contract=CONTRACT, T_end=T_END, budget=SYNTHETIC_B, t_min=t_min)
    return c.certify(vehicles=vs, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps,
                     clock=0.0, inst_idx=0, travel_fn=lambda a, b: 1.0,
                     distance_fn=lambda a, b: 10.0, times=projection, order_attrs=projection,
                     projection=projection, per_class_constants=per_class_constants,
                     env=FakeEnv(revealed=(7,)))


def gate(rc, name):
    return (rc.get('gates', {}).get(name) or {}).get('verdict')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    print('M1-b/Q2 最终入口可信性封闭 - negative controls')

    # ---------------------------------------------------------------- A0 sanity positive
    print('\n-- A0: a WELL-BOUND projection (control) --')
    okP, rcP = run(good_projection())
    row('A0', 'well-bound projection passes the projection/Omega gates',
        {'projection': 'revealed=(7), snapshot bound to the real snapshot'},
        {'projection_gate': gate(rcP, 'projection'),
         'projection_binding': gate(rcP, 'projection_binding'),
         'omega11': gate(rcP, 'omega11_return_feasibility'),
         'omega10': gate(rcP, 'omega10_coverage'),
         'QUALITY_LB_OK': rcP.get('QUALITY_LB_OK'),
         'THERMAL_SAMPLED_OK': rcP.get('THERMAL_SAMPLED_OK'),
         'composite': okP},
        'OK' if gate(rcP, 'projection') == PP.PROJECTION_OK else 'SETUP_FAILED',
        'm1b_q2_certifier.py projection gate', False)

    # ---------------------------------------------------------------- A1 direct construction
    print('\n-- A1: direct TrustedPublicProjection ctor declaring an UNREVEALED node --')
    # node 9 is never revealed (reveal_time is infinite in the synthetic env used elsewhere);
    # here we simply declare it visible and set snapshot_identity to None.
    direct_error = None
    try:
        p1 = PP.TrustedPublicProjection(
            iq={7: 9.0, 9: 9.0}, tc={7: 1, 9: 1}, qty={7: 1.0, 9: 1.0}, svc={7: 0.05, 9: 0.05},
            tws={7: 0.0, 9: 0.0}, twe={7: 1e9, 9: 1e9}, revealed_nodes=(7, 9), instance_id=0,
            event_id=1, clock=0.0, contract_hash=getattr(CONTRACT, 'contract_hash', None),
            snapshot_identity=None)
    except (PermissionError, ValueError) as exc:
        p1 = None
        direct_error = '%s: %s' % (type(exc).__name__, str(exc)[:90])
    if p1 is not None:
        ok1, rc1 = run(p1)
    else:
        ok1, rc1 = False, {'gates': {}, 'composite_pass': False}
    row('A1', 'direct ctor + self-declared visible set + snapshot_identity=None',
        {'revealed_nodes': (7, 9), 'snapshot_identity': None,
         'node 9 revealed by any environment?': 'no'},
        {'projection_gate': gate(rc1, 'projection'),
         'projection_binding': gate(rc1, 'projection_binding'),
         'omega11': gate(rc1, 'omega11_return_feasibility'),
         'omega10': gate(rc1, 'omega10_coverage'),
         'QUALITY_LB_OK': rc1.get('QUALITY_LB_OK'),
         'THERMAL_SAMPLED_OK': rc1.get('THERMAL_SAMPLED_OK'), 'composite': ok1,
         'construction_error': direct_error},
        'OK' if direct_error is not None else ('FALSE_PASS'
            if gate(rc1, 'projection') == PP.PROJECTION_OK else 'CHECK'),
        'm1b_public_projection.py:TrustedPublicProjection.__init__ requires the factory token '
        'and a real snapshot binding, so a direct construction is refused',
        True)

    # ---------------------------------------------------------------- A2 projection_from_revealed_maps
    print('\n-- A2: projection_from_revealed_maps with snapshot_identity=None --')
    p2 = PP.projection_from_revealed_maps(
        iq={7: 9.0}, tc={7: 1}, qty={7: 1.0}, svc={7: 0.05}, tws={7: 0.0}, twe={7: 1e9},
        revealed_nodes=(7,), instance_id=0, event_id=1, clock=0.0,
        contract_hash=getattr(CONTRACT, 'contract_hash', None), snapshot_identity=None)
    ok2, rc2 = run(p2)
    row('A2', 'diagnostic ctor with snapshot_identity=None passes the gate',
        {'ctor': 'projection_from_revealed_maps', 'snapshot_identity': None},
        {'projection_gate': gate(rc2, 'projection'),
         'projection_binding': gate(rc2, 'projection_binding'),
         'omega11': gate(rc2, 'omega11_return_feasibility'),
         'omega10': gate(rc2, 'omega10_coverage'),
         'QUALITY_LB_OK': rc2.get('QUALITY_LB_OK'), 'composite': ok2,
         'data_trusted': getattr(p2, '_data_trusted', None),
         'snapshot_identity': p2.snapshot_identity},
        'OK' if gate(rc2, 'projection') in (PP.PROJECTION_TOKEN_INVALID,
                                            PP.PROJECTION_DATA_NOT_TRUSTED,
                                            PP.PROJECTION_UNBOUND) else 'FALSE_PASS',
        'm1b_public_projection.py:projection_from_revealed_maps now returns a DIAGNOSTIC '
        'projection (data_trusted=False, foreign token) that PP.is_certifiable REFUSES', True)

    # ---------------------------------------------------------------- A3 legacy callables + late TW
    print('\n-- A3: legacy callable path with a REAL late time window --')
    vs3, ps3 = base_inputs()
    ok3, rc3 = M1bQ2TrustFixCertifier(contract=CONTRACT, T_end=T_END,
                                      budget=SYNTHETIC_B).certify(
        vehicles=vs3, served_mask=np.zeros(40, dtype=bool), candidate_plans=ps3, clock=0.0,
        inst_idx=0, travel_fn=lambda a, b: 1.0, distance_fn=lambda a, b: 10.0,
        temp_class_of=lambda n: 1, quantity_of=lambda n: 1.0,
        service_time_of=lambda n: 0.05, initial_quality_by_node={7: 9.0},
        env=FakeEnv(revealed=(7,)))
    row('A3', 'legacy callables: _legacy_projection is built with tw_start=0 and identity None',
        {'legacy args': 'temp_class_of/quantity_of/service_time_of/initial_quality_by_node',
         'true tw_start[7]': '20.0 (not expressible through the legacy callables)'},
        {'projection_gate': gate(rc3, 'projection'),
         'projection_binding': gate(rc3, 'projection_binding'),
         'omega11': gate(rc3, 'omega11_return_feasibility'),
         'omega10': gate(rc3, 'omega10_coverage'),
         'QUALITY_LB_OK': rc3.get('QUALITY_LB_OK'), 'composite': ok3},
        'OK' if gate(rc3, 'projection') in (PP.PROJECTION_TOKEN_INVALID,
                                            PP.PROJECTION_DATA_NOT_TRUSTED,
                                            PP.PROJECTION_UNBOUND,
                                            PP.PROJECTION_DECLARED_SET_REJECTED) else 'FALSE_PASS',
        'm1b_q2_certifier.py:_legacy_projection routes through the DIAGNOSTIC constructor, so '
        'the legacy callable path can no longer certify anything at all', True)

    # ---------------------------------------------------------------- B1 per_class_constants override
    print('\n-- B1: external per_class_constants override --')
    pc, _, _ = __import__('m1b_contract_identity').check_quality_classes_are_certifiable(
        CONTRACT, epsilon_init=QC.EPSILON_INIT)
    forged = {c: dict(v) for c, v in pc.items() if isinstance(v, dict)}
    forged[1] = dict(forged[1])
    forged[1]['q0_safe'] = 1e6          # absurdly optimistic initial value
    forged[1]['rate_per_hour'] = 0.0    # and no decay at all
    okB, rcB = run(good_projection(), per_class_constants=forged)
    real = (rcB['gates'].get('quality_lb') or {}).get('detail', {}).get('per_class', {}).get(1, {})
    canonical = QC.certify_quality_lb(pc, T_end=T_END, t_min=0.0, hours_per_time_unit=1.0)[0][1]
    row('B1', 'overriding per_class_constants inflates the certified lower bound',
        {'per_class_constants[1]': {'q0_safe': 1e6, 'rate_per_hour': 0.0}},
        {'override_gate': gate(rcB, 'per_class_constants_override'),
         'quality_gate': gate(rcB, 'quality_lb'),
         'q_lb_forged': real.get('q_lb'), 'q_lb_canonical': canonical.get('q_lb'),
         'contract_initial_value': pc[1]['initial_value'],
         'inflated': (real.get('q_lb') or 0) > (canonical.get('q_lb') or 0) + 1.0},
        'OK' if (real.get('q_lb') is None
                 and gate(rcB, 'per_class_constants_override')
                 == 'PER_CLASS_CONSTANTS_OVERRIDE_REFUSED') else 'FALSE_PASS',
        'm1b_q2_certifier.py: a non-None per_class_constants is now REFUSED; parameters come '
        'only from the identity-certified contract', True)

    # ---------------------------------------------------------------- B2 t_min override
    print('\n-- B2: external t_min override --')
    okC, rcC = run(good_projection(), t_min=1e6)
    qc = (rcC['gates'].get('quality_lb') or {}).get('detail', {}).get('per_class', {}).get(1, {})
    row('B2', 'a caller-supplied t_min silently shrinks the certified horizon',
        {'t_min': 1e6},
        {'quality_gate': gate(rcC, 'quality_lb'), 'H_max': qc.get('H_max_hours'),
         't_min_source': getattr(rcC, 't_min_source', None), 't_min_requested': 1e6,
         'q_lb': qc.get('q_lb'), 'canonical_q_lb': canonical.get('q_lb'),
         'note': 'H_max = max(0, T_end + 1e-6 - t_min) -> 0, so Q_LB becomes the INITIAL value'},
        'OK' if (qc.get('H_max_hours') is not None and qc.get('H_max_hours') > 24.0) else 'FALSE_PASS',
        'm1b_q2_certifier.py:_resolve_t_min uses the conservative clock origin unless a '
        'verified_t_min is supplied together with an env witness',
        True)

    fp = [r for r in ROWS if r['VERDICT'] == 'FALSE_PASS']
    print('\n== %d checks: %d FALSE_PASS ==' % (len(ROWS), len(fp)))
    for r in ROWS:
        if r['VERDICT'] != 'OK':
            print('   %-7s %s' % (r['id'], r['VERDICT']))
    payload = {'schema': 'm1b_q2_final_entry_negative_controls_v1', 'n_checks': len(ROWS),
               'n_false_pass': len(fp), 'results': ROWS}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                  encoding='utf-8')
        print('written: %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
