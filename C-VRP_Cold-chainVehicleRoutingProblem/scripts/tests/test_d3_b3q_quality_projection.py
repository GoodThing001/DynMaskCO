#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-Q acceptance: revised Q1-Q5 + L1-L3 + the C1 bounded falsification search.

The id set and expected outcomes are frozen in `B3Q_COUNTEREXAMPLE_SPEC.json` BEFORE this file was
written (user ruling: freeze the counterexamples first, implement second).  This suite asserts the
exact id set so no row can be dropped.

Scope (user ruling 2026-10-14): approved D-B3A-1 (legality short-circuit), D-B3A-2 (read-only
projection, NO conservative claim), D-B3A-4 (this pre-registration), D-B3A-7 (contract transform
verifier).  D-B3A-3 DEFERRED: a passing projection must NOT change `installable`.  D-B3A-6 DEFERRED:
the 8 bypass call sites are NOT rewritten.
No real install, no `_commit` change, no frozen data, no final seeds.

Run: python scripts/tests/test_d3_b3q_quality_projection.py [--instances 0] [--heavy-cap 4]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'simulation'
for _p in (SIM, HERE.parent, HERE, HERE.parent / 'coldchain', HERE.parent / 'evaluation'):
    sys.path.insert(0, str(_p))

import numpy as np                                                     # noqa: E402
import action_contract as AC                                          # noqa: E402
import d3_state_assembly as A                                          # noqa: E402
import d3_composed_certifier as CC                                     # noqa: E402
import d3_quality_projection as QP                                     # noqa: E402
from strict_online_env import StrictOnlineEnv                          # noqa: E402
from jf1h_repair import make_continuation                              # noqa: E402
from coldchain_contract import (default_pilot_contract,                # noqa: E402
                                apply_objective_profile, ObjectiveProfile)
from dynmaskco_cc_context import mutable_scope                         # noqa: E402

EXT = HERE.parent.parent
CAL_NPZ = EXT / 'data' / 'm0_scale' / 'dcc_50_r1_edod05_cal_teacher.npz'
PROFILE_JSON = EXT / 'results' / 'o0cc' / 'scale_v2' / 'objective_profile.json'
CAPACITY, NUM_VEHICLES, B_BUDGET = 50.0, 25, 300.0
PREREGISTERED = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'L1', 'L2', 'L3', 'C1', 'SUF']
RESULTS = []


def chk(cid, name, ok, detail=None):
    RESULTS.append({'cid': cid, 'name': name, 'ok': bool(ok), 'detail': detail})


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_profile():
    d = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))
    p = ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                         quality_scale=float(d['quality_scale']),
                         energy_scale=float(d['energy_scale']),
                         lambda_quality=float(d['lambda_quality']),
                         lambda_energy=float(d['lambda_energy']),
                         scale_source=d.get('scale_source', 'pilot'),
                         dev_statistics=d.get('dev_statistics'))
    assert p.profile_hash == d.get('profile_hash')
    return p


def make_env(dataset, contract, hook=None):
    env = StrictOnlineEnv(dataset, capacity=CAPACITY, tw_speed=1.0, num_vehicles=NUM_VEHICLES,
                          replanner=make_continuation(), coldchain_contract=contract,
                          booking_horizon=None)
    env.snapshot_hook = hook
    return env


class Grab(object):
    def __init__(self, cap=4):
        self.cap, self.items, self.n_calls = cap, [], 0

    def hook(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces, served_mask,
             all_customers=None):
        self.n_calls += 1
        if len(self.items) >= self.cap:
            return
        snap, _ = A.capture_decision_state(env, inst_idx, clock, event_id, reveal_idx, vehicles,
                                          traces, served_mask, all_customers)
        cv, ct, csm = A.clone_state(snap)
        vis = [int(i) for i in all_customers if env.reveal_time[inst_idx, i] <= clock + 1e-6]
        cv, plans, _rid = A.prepare_clone(env, inst_idx, clock, cv, csm, vis)
        mut = mutable_scope(cv)
        with_suffix = [(len(plans[v].suffix), int(v)) for v in mut
                       if v in plans and len(plans[v].suffix) > 0]
        self.items.append({'env': env, 'inst': int(inst_idx), 'clock': float(clock),
                           'vehicles': cv, 'traces': ct, 'served_mask': csm, 'plans': plans,
                           'mutable_ids': mut, 'visible_ids': vis, 'snapshot': snap,
                           'protected': (frozenset({max(with_suffix)[1]}) if with_suffix
                                         else frozenset())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, nargs='+', default=[0])
    ap.add_argument('--heavy-cap', type=int, default=4)
    args = ap.parse_args()

    profile = load_profile()
    dataset = dict(np.load(CAL_NPZ))
    raw_contract = default_pilot_contract()
    eff = apply_objective_profile(raw_contract, profile)
    g = Grab(cap=args.heavy_cap)
    env0 = make_env(dataset, eff, hook=g.hook)
    env0.run(args.instances[0])
    items = g.items
    chk('PRE', 'real env decision points captured read-only', len(items) >= 2,
        {'n_items': len(items), 'n_calls': g.n_calls})
    base = items[0]
    env, inst, clock = base['env'], base['inst'], base['clock']
    plans, vehicles, sm = base['plans'], base['vehicles'], base['served_mask']
    mut, prot = base['mutable_ids'], base['protected']

    def compose(**over):
        kw = dict(mask_set=[], protected=prot, deferred=frozenset(), B=B_BUDGET,
                  contract=eff, base_plans=plans)
        kw.update(over)
        return CC.compose_certification(env, inst, clock, vehicles, sm, kw.pop('plans', plans),
                                        **kw)

    victim = next((v for v, p in plans.items() if p.suffix), None)
    ref = compose()

    # ================= Q1 stress-schedule admissibility =================
    stress = QP.project_plan_quality(env, inst, clock, vehicles,
                                    {v: list(p.suffix) for v, p in plans.items()}, eff,
                                    schedule=QP.SCHEDULE_LATEST_ADMISSIBLE, served_mask=sm)
    nominal = QP.project_plan_quality(env, inst, clock, vehicles,
                                      {v: list(p.suffix) for v, p in plans.items()}, eff,
                                      schedule=QP.SCHEDULE_NOMINAL, served_mask=sm)
    q1_ok = (stress['schedule_kind'] == QP.SCHEDULE_LATEST_ADMISSIBLE
             and 'inadmissible_stress_vehicles' in stress
             and 'admissibility_violations' in stress['per_vehicle'].get(
                 list(stress['per_vehicle'])[0], {})
             and stress['conservativeness'] == QP.CONSERVATIVENESS_NOT_ESTABLISHED
             and nominal['conservativeness'] == QP.CONSERVATIVENESS_NOT_ESTABLISHED)
    chk('Q1', 'Q1 the stress schedule is CHECKED for admissibility and is never a bound', q1_ok,
        {'stress_kind': stress['schedule_kind'],
         'stress_conservativeness': stress['conservativeness'],
         'inadmissible_vehicles': stress['inadmissible_stress_vehicles'],
         'example_violations': (stress['per_vehicle'][list(stress['per_vehicle'])[0]]
                                ['admissibility_violations'][:2] if stress['per_vehicle'] else []),
         'nominal_state': nominal['projection_state'], 'stress_state': stress['projection_state']})

    # ================= Q2 non-precooled branch =================
    # Needs a decision point where a vehicle is actually carrying cargo, otherwise there is no
    # manifest to flip and the case is simply NOT_VERIFIABLE (reported, never skipped silently).
    import coldchain_state as CS
    route = {v: list(p.suffix) for v, p in plans.items() if p.suffix}
    cargo_item = next((it for it in items
                       if any(getattr(v.coldchain_state, 'cargo_manifest', ())
                              for v in it['vehicles'] if v.coldchain_state is not None)), None)
    if cargo_item is not None:
        cenv, cinst, cclock = cargo_item['env'], cargo_item['inst'], cargo_item['clock']
        cveh = cargo_item['vehicles']
        csm = cargo_item['served_mask']
        croute = {v: list(p.suffix) for v, p in cargo_item['plans'].items() if p.suffix}
        precool = QP.project_plan_quality(cenv, cinst, cclock, cveh, croute, eff,
                                          schedule=QP.SCHEDULE_NOMINAL, served_mask=csm)
        veh2, _t2, _s2 = A.clone_state(cargo_item['snapshot'])
        flipped = 0
        for v in veh2:
            if v.coldchain_state is not None and v.coldchain_state.cargo_manifest:
                flipped += len(v.coldchain_state.cargo_manifest)
                v.coldchain_state = CS.VehicleColdChainState(
                    **{**v.coldchain_state.__dict__,
                       'cargo_manifest': tuple(
                           CS.replace(lot, precooled=False)
                           for lot in v.coldchain_state.cargo_manifest)})
        nopc = QP.project_plan_quality(cenv, cinst, cclock, veh2, croute, eff,
                                       schedule=QP.SCHEDULE_NOMINAL, served_mask=csm)
        pc_q = sorted(r['delivered_quality'] for vv in precool['per_vehicle'].values()
                      for r in vv.get('delivered', []))
        npc_q = sorted(r['delivered_quality'] for vv in nopc['per_vehicle'].values()
                       for r in vv.get('delivered', []))
        pc_by = {(vid, r['order_id']): r['delivered_quality']
                 for vid, vv in precool['per_vehicle'].items() for r in vv.get('delivered', [])}
        np_by = {(vid, r['order_id']): r['delivered_quality']
                 for vid, vv in nopc['per_vehicle'].items() for r in vv.get('delivered', [])}
        diffs = sorted((round(pc_by[k] - np_by[k], 12), k) for k in pc_by if k in np_by)
        q2_ok = bool(pc_by) and len(pc_by) == len(np_by) and any(d > 1e-12 for d, _ in diffs)
        chk('Q2', 'Q2 the non-precooled field-temperature branch materially increases decay',
            q2_ok, {'cargo_item': {'inst': cinst, 'clock': cclock},
                    'n_lots_flipped': flipped, 'n_records': len(pc_by),
                    'largest_quality_drops_precooled_minus_nonprecooled': diffs[-3:],
                    'n_orders_affected': sum(1 for d, _ in diffs if d > 1e-12),
                    'note': ('a sorted-prefix print can look identical; the comparison is '
                             'per (vehicle, order)')})
    else:
        chk('Q2', 'Q2 the non-precooled field-temperature branch materially increases decay',
            False, {'status': 'NOT_VERIFIABLE',
                    'why': 'no captured decision point had a vehicle carrying cargo',
                    'n_items': len(items)})

    # ================= Q3 per-class thresholds applied separately =================
    recs = [r for vv in precool['per_vehicle'].values() for r in vv.get('delivered', [])]
    thr = list(eff.quality.unsalable_quality_threshold)      # indexed by temp_class, not a dict
    per_class_ok = bool(recs) and all(
        abs(r['threshold'] - thr[r['temp_class']]) < 1e-12 for r in recs)
    classes = sorted({r['temp_class'] for r in recs})
    chk('Q3', 'Q3 per-class thresholds/dynamics are applied SEPARATELY (no "class N always fails '
        'first" claim)', per_class_ok,
        {'n_records': len(recs), 'classes_present': classes,
         'thresholds_by_class': {str(c): thr[c] for c in classes},
         'ratios_by_class': {str(c): sorted({round(r['ratio'], 6) for r in recs
                                             if r['temp_class'] == c})[:3] for c in classes},
         'note': 'the ordering claim is deliberately NOT asserted: it depends on initial quality, '
                 'temperature history, rate parameters and threshold together'})

    # ================= Q4 projection == actual execution (detached copy) =================
    cv, ct, csm = A.clone_state(base['snapshot'])
    cv_pp, plans_pp, _r = A.prepare_clone(env, inst, clock, cv, csm, base['visible_ids'])
    proj = QP.project_plan_quality(env, inst, clock, cv_pp,
                                   {v: list(p.suffix) for v, p in plans_pp.items()}, eff,
                                   schedule=QP.SCHEDULE_NOMINAL, served_mask=csm)
    # execute the SAME single-leg route on a fresh detached copy through the production machinery
    from coldchain_state import transition_segment, dispatch_vehicle, create_vehicle_state
    exec_records = []
    speed = eff.units.speed_kmph / eff.units.distance_km_per_unit
    for v in cv_pp:
        p = plans_pp.get(int(v.vehicle_id))
        if p is None or not p.suffix:
            continue
        cex, tex, smex = A.clone_state(base['snapshot'])
        vv = next(x for x in cex if int(x.vehicle_id) == int(v.vehicle_id))
        state = (vv.coldchain_state if vv.coldchain_state is not None
                 else dispatch_vehicle(create_vehicle_state(eff), eff))
        cur = int(vv.current_node)
        t = max(float(vv.ready_time), float(clock))
        for o in [int(x) for x in p.suffix]:
            o = int(o)
            d = float(env.dist_mat[inst, cur, o])
            depart = max(t, float(env.reveal_time[inst, o]))
            arrive = depart + d / speed
            sstart = max(arrive, float(env.tw_start[inst, o]))
            sfinish = sstart + float(env.service_time[inst, o])
            iq = 1.0
            state, _ = transition_segment(
                state, depart_time=depart, arrival_time=arrive, service_finish=sfinish,
                served_customer=o, active_zone_mask=(True,) * 3, contract=eff,
                order_quantity=float(env.demands[inst, o]),
                order_temp_class=int(env.dataset['temp_class'][inst, o]),
                initial_quality=iq, segment_distance_units=d)
            cur, t = o, sfinish
        if env.booking_horizon is not None and env.booking_horizon > t + 1e-9:
            state, _ = transition_segment(
                state, depart_time=t, arrival_time=env.booking_horizon,
                service_finish=env.booking_horizon, served_customer=None,
                active_zone_mask=(True,) * 3, contract=eff, segment_distance_units=0.0)
            t = env.booking_horizon
        d_back = float(env.dist_mat[inst, cur, 0])
        arr_back = t + d_back / speed
        state, _ = transition_segment(
            state, depart_time=t, arrival_time=arr_back, service_finish=arr_back,
            served_customer=None, active_zone_mask=(True,) * 3, contract=eff,
            return_to_depot=True, segment_distance_units=d_back)
        for r in getattr(state, 'delivered_to_depot', ()):
            exec_records.append({'vid': int(v.vehicle_id), 'order_id': int(r.order_id),
                                 'delivered_quality': float(r.delivered_quality),
                                 'salable': (None if r.salable is None else bool(r.salable))})
    proj_records = [{'vid': int(vid), 'order_id': r['order_id'],
                     'delivered_quality': r['delivered_quality'], 'salable': r['salable']}
                    for vid, vv in proj['per_vehicle'].items() for r in vv.get('delivered', [])]
    def key(rs):
        return sorted((r['vid'], r['order_id'], round(r['delivered_quality'], 12), r['salable'])
                      for r in rs)
    q4_ok = bool(proj_records) and key(proj_records) == key(exec_records)
    chk('Q4', 'Q4 the projection equals actual execution of the same route (production machinery '
        'on a detached copy)', q4_ok,
        {'n_projected': len(proj_records), 'n_executed': len(exec_records),
         'projected_head': key(proj_records)[:3], 'executed_head': key(exec_records)[:3]})

    # ================= Q5 + L1 + L3 legality short-circuit =================
    reveal_row = env.reveal_time[inst]
    served_nodes = {c for c in range(1, env.num_nodes) if bool(sm[c])}
    unrevealed_ids = [c for c in range(1, env.num_nodes)
                      if not bool(reveal_row[c] <= clock + 1e-6)]
    cases = {}
    if victim is not None:
        p = plans[victim]
        if unrevealed_ids:
            inj = dict(plans)
            inj[victim] = AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time,
                                         p.anchor_load, tuple(list(p.suffix) + [unrevealed_ids[0]]))
            cases['unrevealed'] = compose(plans=inj)
        if served_nodes:
            s = sorted(served_nodes)[0]
            inj = dict(plans)
            inj[victim] = AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time,
                                         p.anchor_load, tuple(list(p.suffix) + [s]))
            cases['served'] = compose(plans=inj)
        inj = dict(plans)
        inj[victim] = AC.VehiclePlan(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                     tuple(list(p.suffix) + list(p.suffix[:1])))
        cases['duplicate'] = compose(plans=inj)
    inj = dict(plans)
    inj[9999] = AC.VehiclePlan(9999, 0, 0.0, 0.0, ())
    cases['closed_or_absent_vehicle'] = compose(plans=inj)

    gate_of = {'unrevealed': 'revealed_only', 'served': 'no_served_in_plan',
               'duplicate': 'no_duplicate_order', 'closed_or_absent_vehicle': 'plan_vehicle_set'}
    l1_rows = {}
    for nm, res in cases.items():
        gate = gate_of[nm]
        comp = res['components'].get(gate, {})
        down = [res['components'].get(k, {}) for k in
                ('route_feasible', 'energy_budget', 'quality_projection')]
        l1_rows[nm] = {
            'gate_ok': comp.get('ok'), 'gate_named': gate,
            'downstream_all_not_run': all(d.get('not_run_illegal_order_set') is True for d in down),
            'downstream_refused_by': sorted({d.get('refused_by_gate') for d in down}),
            'ok': res['ok'], 'installable': res['installable'],
            'reasons': res['reasons'][:3]}
    chk('L1', 'L1 each legality gate is individually attributable and names itself',
        bool(l1_rows) and all(r['gate_ok'] is False and r['ok'] is False for r in l1_rows.values()),
        l1_rows)
    chk('Q5', 'Q5 no illegal order set reaches the route / energy / QUALITY projection',
        bool(l1_rows) and all(r['downstream_all_not_run'] for r in l1_rows.values()), l1_rows)
    chk('L3', 'L3 illegal input (out-of-range customer / absent vehicle) does not crash',
        'closed_or_absent_vehicle' in cases
        and all('error' not in (cases[n]['components'].get(g, {}) or {})
                for n, g in gate_of.items() if n in cases),
        {n: cases[n]['components'].get(g, {}).get('error') for n, g in gate_of.items()
         if n in cases})

    # ================= L2 legal-input verdicts unchanged =================
    l2 = {'legal_plan_passes_executable_components':
          all(ref['components'][k]['ok'] for k in ('repair_plan', 'full_partition') if k in
              ref['components'])}
    if victim is not None:
        p = plans[victim]
        late = {v: (AC.VehiclePlan(pp.vehicle_id, pp.anchor_node, 1e4, pp.anchor_load, pp.suffix)
                    if v == victim else pp) for v, pp in plans.items()}
        tw_res = compose(plans=late)
        heavy = {v: (AC.VehiclePlan(pp.vehicle_id, pp.anchor_node, pp.anchor_time,
                                    CAPACITY + 10.0, pp.suffix) if v == victim else pp)
                 for v, pp in plans.items()}
        cap_res = compose(plans=heavy)
        l2['tw_still_refused'] = (tw_res['ok'] is False
                                  and tw_res['components']['route_feasible'].get('ok') is False)
        l2['tw_reason'] = tw_res['components']['route_feasible']['per_route'][int(victim)]['reason']
        l2['capacity_still_refused'] = (cap_res['ok'] is False
                                        and cap_res['components']['route_feasible'].get('ok')
                                        is False)
        l2['capacity_reason'] = (cap_res['components']['route_feasible']['per_route']
                                 [int(victim)]['reason'])
        tight = compose(B=0.0)
        l2['energy_still_refused'] = tight['components']['energy_budget']['ok'] is False
        l2['route_ok_while_energy_fails'] = tight['components']['route_feasible']['ok'] is True
    chk('L2', 'L2 the reorder does not change legal-input verdicts (B2 D5/D6 reproduced)',
        all(v is True for v in l2.values() if isinstance(v, bool))
        and l2.get('tw_reason') == 'tw' and l2.get('capacity_reason') == 'capacity',
        l2)

    # ================= C1 bounded admissible-schedule falsification search =================
    search_rows, best = [], None
    base_kind = QP.SCHEDULE_LATEST_ADMISSIBLE
    candidate_bound = stress['projection_state']
    # coarse grid: scale each inter-node departure by (1 - s) i.e. arrive earlier than latest
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        vehicle_route = {v: list(p.suffix) for v, p in plans.items() if p.suffix}
        if not vehicle_route:
            break
        # emulate "earlier than latest" by shrinking the allowed arrival: we do this by asking the
        # projection for the nominal schedule and for the latest, and interpolating the WINDOW
        # through a temporary env copy whose tw_end is scaled toward tw_start.
        ds2 = dict(dataset)
        twe = dataset['tw_end'].copy()
        tws = dataset['tw_start'].copy()
        ds2['tw_end'] = (tws + (1.0 - s) * (twe - tws)).astype(twe.dtype)
        envx = make_env(ds2, eff)
        r = QP.project_plan_quality(envx, inst, clock, vehicles, vehicle_route, eff,
                                    schedule=QP.SCHEDULE_LATEST_ADMISSIBLE, served_mask=sm)
        quals = [x['delivered_quality'] for vv in r['per_vehicle'].values()
                 for x in vv.get('delivered', [])]
        row = {'tw_end_scale': round(1.0 - s, 3), 'admissible_vehicles':
               r['inadmissible_stress_vehicles'], 'n_delivered': len(quals),
               'min_delivered_quality': (min(quals) if quals else None),
               'projection_state': r['projection_state']}
        search_rows.append(row)
        if quals and (best is None or min(quals) < best[0]):
            best = (min(quals), row)
    bound_quals = [x['delivered_quality'] for vv in stress['per_vehicle'].values()
                   for x in vv.get('delivered', [])]
    bound_min = min(bound_quals) if bound_quals else None
    falsified = bool(bound_min is not None and best is not None and best[0] < bound_min - 1e-12)
    # HONEST search-effectiveness assessment: if every grid row produced the SAME minimum, the
    # search did not actually move the binding lot's schedule, so "no counterexample found" is not
    # even weak evidence for the bound.  Reporting this is mandatory -- a vacuous search must not
    # be presented as support.
    distinct = sorted({round(r['min_delivered_quality'], 12) for r in search_rows
                       if r['min_delivered_quality'] is not None})
    effectiveness = ('INEFFECTIVE_SEARCH: all grid rows produced the same minimum, so the search '
                     'did not move the binding lot schedule'
                     if len(distinct) <= 1 else 'VARIED_SEARCH')
    chk('C1', 'C1 bounded admissible-schedule search against the candidate bound (result recorded; '
        'the label STAYS NOT_ESTABLISHED either way)', True,
        {'candidate_bound_min_quality': bound_min, 'search_min': (best[0] if best else None),
         'falsified': falsified, 'distinct_minima_across_grid': distinct,
         'search_effectiveness': effectiveness, 'search_rows': search_rows,
         'conservativeness_label': QP.CONSERVATIVENESS_NOT_ESTABLISHED,
         'note': ('a bounded search is NOT a proof. An INEFFECTIVE search is not even weak '
                  'evidence: no counterexample found here must NOT be read as support for the '
                  'bound. A counterexample, if found, falsifies the candidate bound.')})

    # ================= SUF a passing projection must NOT unlock installation =================
    pass_proj = compose()
    with_quality = compose(project_quality=True)
    chk('SUF', 'SUF a passing quality projection does NOT change installable (D-B3A-3 deferred)',
        pass_proj['installable'] is False and with_quality['installable'] is False
        and pass_proj['blocking_capability_gaps'] == ['quality_salability']
        and pass_proj['quality']['state'] == 'CAPABILITY_GAP_BLOCKING',
        {'installable': pass_proj['installable'],
         'blocking_gaps': pass_proj['blocking_capability_gaps'],
         'quality_state': pass_proj['quality']['state'],
         'projection_component_present': 'quality_projection' in pass_proj['components'],
         'projection_state': pass_proj['components'].get('quality_projection', {}).get(
             'projection_state')})

    # ================= pre-registration integrity =================
    ids = [r['cid'] for r in RESULTS]
    missing = [i for i in PREREGISTERED if i not in ids]
    chk('PREREG', 'every pre-registered id is present; none dropped', not missing,
        {'preregistered': PREREGISTERED, 'covered': ids, 'missing': missing})

    out = {
        'stage': 'D3_G0_Evidence_B3Q_quality_projection',
        'production_state_written': False, 'commit_modified': False,
        'real_writeback_enabled': False, 'conservative_certification_claimed': False,
        'installable_unlocked': False,
        'preregistered_ids': PREREGISTERED, 'covered_ids': ids,
        'identities': {
            'data_npz_sha256': sha256_file(CAL_NPZ),
            'profile_hash': profile.profile_hash,
            'raw_contract_hash': raw_contract.contract_hash,
            'effective_contract_hash': eff.contract_hash,
            'contract_version': QP._contract_version(eff),
            'code': {
                'scripts/simulation/d3_quality_projection.py':
                    sha256_file(SIM / 'd3_quality_projection.py'),
                'scripts/simulation/d3_composed_certifier.py':
                    sha256_file(SIM / 'd3_composed_certifier.py'),
                'scripts/tests/test_d3_b3q_quality_projection.py': sha256_file(Path(__file__)),
            },
        },
        'contract_version_qualification': {
            'applies_to': 'v1 QualityConfig (ratio thresholds) on the D3/M0 line',
            'does_NOT_transfer_to': ('the A-v1 main task fish-salability hard constraint under the '
                                     'v2 absolute sensory metric -- different physics and metric '
                                     'semantics; a v2 switch needs independent proof'),
        },
        'results': RESULTS, 'all_pass': all(r['ok'] for r in RESULTS),
    }
    for r in RESULTS:
        print('  %s %-7s %s' % ('PASS' if r['ok'] else 'FAIL', r['cid'], r['name']))
        if not r['ok']:
            print('        <- %s' % json.dumps(r['detail'], ensure_ascii=False, default=str)[:420])
    print('\nall_pass:', out['all_pass'])
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3Q_PROJECTION_MATRIX.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')
    print('wrote', dest)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
