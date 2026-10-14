#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B2-2: the COMPOSED certifier.

Authorised scope: compose existing certification primitives into ONE gate.  Nothing here installs,
nothing modifies `_commit` / `_plan_and_commit`, and no real write-back is enabled.

Why a composition rather than a single call (B2 static review, measured):
`certify_plan` (run_exp_energy_c0.py:154-199) certifies the C0 ENERGY BUDGET ONLY and is a pure
function.  It does not check time windows or capacity (`action_contract.certify_route`), ownership
or commitment (`dynmaskco_cc_context.validate_full_partition`), or plan reconciliation
(`mpre_policy.validate_repair_plan`).  The repository says so itself.  Four checks are therefore
required, and calling all four is STILL not sufficient -- see the compensations and the capability
gap below.

Compensations for three SILENT behaviours found in the static review:
  F1 `c0_completion_energy:127` does `depart = max(t, float(rev[o]))`, reading the future
     `reveal_time` as a scheduling clamp.  It is inert for revealed orders but it DOES NOT REJECT
     an unrevealed one.  => this module asserts revealedness itself.
  F2 `certify_plan:173` silently DROPS served orders from the route instead of refusing.
     => `validate_full_partition(served_mask=...)` is called and its `served_planned` reported.
  F3 a plan entry for a closed vehicle is silently ignored by `certify_plan`, and
     `build_vehicle_plans` drops closed vehicles.  => the plan's vehicle set is checked against
     the live non-closed set.

Capability gap (user ruling: 不能默认合格): the only decision-time cold-chain constraint is
temperature-class support (`certify_route:178-183`); `projected_energy_kwh` /
`projected_quality_loss` / `projected_thermal_margin` are left as `None` (`:171-173`) and fish
salability is checked only terminally (`coldchain_evaluator_a1:115-117`).  So quality/salability
has NO executable online check, and this module reports `CAPABILITY_GAP_BLOCKING` and refuses
`installable` rather than defaulting to satisfied.  A caller may inject a real decision-time
quality check; without one the gap stands.
"""
from __future__ import annotations

from d3_credential import (QUALITY_CAPABILITY_GAP, QUALITY_FAILED, QUALITY_OK)

CERTIFICATION_CHECKS = ('repair_plan', 'full_partition', 'route_feasible', 'energy_budget')
# D-B3A-1 (authorised 2026-10-14): the legality gates run FIRST and short-circuit the rest.
LEGALITY_GATES = ('plan_vehicle_set', 'revealed_only', 'no_served_in_plan', 'no_duplicate_order')
COMPENSATIONS = LEGALITY_GATES          # kept as an alias for the pre-B3-Q name
CAPABILITY_GAPS = ('quality_salability',)


def _plan_routes(plans):
    """`certify_plan` wants `{vid: not-yet-executed route}`; a FleetPlan suffix is exactly that.

    Verified in the B2 static review: `build_vehicle_plans` sets `anchor_node = committed_next`
    and keeps `suffix = mutable_suffix` minus the trailing 0, so the suffix EXCLUDES the committed
    leg and `certify_plan` prepends it exactly once (no double count).
    """
    return {int(vid): [int(x) for x in p.suffix if int(x) > 0] for vid, p in plans.items()}


def compose_certification(env, inst_idx, clock, vehicles, served_mask, plans, *, mask_set,
                          protected=frozenset(), deferred=frozenset(), B, contract,
                          base_plans=None, mutable_ids=None, quality_check_fn=None,
                          project_quality=True, stress_quality_schedule=False):
    """Run the LEGALITY GATES first, then the four checks, plus the read-only quality projection.

    Returns a dict; `installable` is True only when every check passes AND no blocking capability
    gap exists.  A PASSING QUALITY PROJECTION DOES NOT CHANGE `installable` (D-B3A-3 is deferred
    until the conservative bound is proven).
    """
    from mpre_policy import validate_repair_plan
    from dynmaskco_cc_context import validate_full_partition
    from action_contract import certify_route, build_vehicle_plans
    from run_exp_energy_c0 import certify_plan

    components, reasons = {}, []
    # `base_plans` MUST be the INCUMBENT (the pre-repair plan) when the candidate differs from
    # it.  Defaulting it to `plans` would compare the candidate against itself, making the
    # reconciliation silently vacuous -- so a missing base is a REFUSAL, not a pass.
    base_is_given = base_plans is not None
    if not base_is_given:
        base_plans = plans

    def _guard(name, fn):
        """A gate that CRASHES on a malformed input is worse than one that refuses it.

        Each component is therefore run under a guard: an exception becomes a refusal with the
        exception type recorded, never an escaping error.  (Same principle as
        `d3_geo_adapter._plan_identity` returning None instead of raising.)
        """
        try:
            return fn()
        except BaseException as exc:                                  # noqa: BLE001
            components[name] = {'ok': False, 'error': '%s: %s' % (type(exc).__name__,
                                                                  str(exc)[:160])}
            reasons.append('%s_error:%s' % (name, type(exc).__name__))
            return None

    def _not_run(names, why, gate):
        for nm in names:
            components[nm] = {'ok': False, 'not_run_illegal_order_set': True,
                              'refused_by_gate': gate, 'why': why}
        reasons.append('legality_short_circuit:%s' % gate)

    # =======================================================================================
    # D-B3A-1 LEGALITY-FIRST SHORT-CIRCUIT (authorised 2026-10-14)
    # ---------------------------------------------------------------------------------------
    # Previously the compensations ran AFTER the four checks, so `certify_plan` could be handed a
    # plan containing unrevealed or already-served orders -- and inside it,
    # `c0_completion_energy:127` silently reschedules by `reveal_time` while `certify_plan:173`
    # silently drops served orders.  Now the legality gates run FIRST and, if any fails, the
    # route / energy / quality components are NOT RUN AT ALL.  "Every certification component only
    # ever receives a legality-checked order set" thereby becomes a STRUCTURAL property instead of
    # an accident of call order.
    # =======================================================================================
    DOWNSTREAM = ('route_feasible', 'energy_budget', 'quality_projection')

    live_non_closed = {int(v.vehicle_id) for v in vehicles if v.status != 'closed'}
    extra_vehicles = sorted(set(int(v) for v in plans) - live_non_closed)
    components['plan_vehicle_set'] = {
        'ok': not extra_vehicles, 'plan_vehicles': sorted(int(v) for v in plans),
        'live_non_closed': sorted(live_non_closed),
        'closed_or_absent_vehicles_in_plan': extra_vehicles}

    reveal_row = env.reveal_time[inst_idx]
    unrevealed = sorted({int(x) for p in plans.values() for x in p.suffix
                         if int(x) > 0 and not bool(reveal_row[int(x)] <= clock + 1e-6)})
    components['revealed_only'] = {'ok': not unrevealed, 'unrevealed_in_plan': unrevealed}

    served_seen = {int(x) for p in plans.values() for x in p.suffix if int(x) > 0}
    served_planned = sorted(c for c in served_seen if bool(served_mask[c]))
    components['no_served_in_plan'] = {'ok': not served_planned, 'served_planned': served_planned}

    counts = {}
    for p in plans.values():
        for x in p.suffix:
            if int(x) > 0:
                counts[int(x)] = counts.get(int(x), 0) + 1
    duplicates = sorted(c for c, k in counts.items() if k > 1)
    components['no_duplicate_order'] = {'ok': not duplicates, 'duplicate_suffix': duplicates}

    if extra_vehicles:
        _not_run(DOWNSTREAM, 'plan references a vehicle that is not in the live non-closed fleet',
                 'plan_vehicle_set')
    elif unrevealed:
        _not_run(DOWNSTREAM, 'plan contains an order that is not revealed at this clock',
                 'revealed_only')
    elif served_planned:
        _not_run(DOWNSTREAM, 'plan contains an order that has already been served',
                 'no_served_in_plan')
    elif duplicates:
        _not_run(DOWNSTREAM, 'plan contains a duplicated order', 'no_duplicate_order')
    legality_ok = not (extra_vehicles or unrevealed or served_planned or duplicates)

    # ---- ownership / commitment / service partition (still before the downstream checks) ----
    committed = {int(v.committed_next) for v in vehicles
                 if v.committed_next not in (None, 0)}
    universe = [int(c) for c in range(1, env.num_nodes)
                if not bool(served_mask[c]) and int(c) not in committed
                and bool(env.reveal_time[inst_idx, c] <= clock + 1e-6)]
    if legality_ok:
        got = _guard('full_partition', lambda: validate_full_partition(
            committed, plans, set(deferred), universe, served_mask=served_mask))
        if got is not None:
            ok_part, detail_part = got
            components['full_partition'] = {'ok': bool(ok_part), 'detail': detail_part}
            if not ok_part:
                reasons.append('full_partition_failed')
                _not_run(DOWNSTREAM, 'ownership / commitment partition failed', 'full_partition')
                legality_ok = False
        else:
            detail_part = {'served_planned': [], 'duplicate_suffix': []}
            _not_run(DOWNSTREAM, 'partition check errored', 'full_partition')
            legality_ok = False
    else:
        components['full_partition'] = {'ok': False, 'not_run_illegal_order_set': True,
                                        'refused_by_gate': 'legality_gate',
                                        'why': 'a legality gate already refused'}
        detail_part = {'served_planned': [], 'duplicate_suffix': []}

    # ---- plan reconciliation (needs base_plans; legality already established) ---------------
    if not base_is_given:
        components['repair_plan'] = {
            'ok': False, 'error': 'base_plans_missing',
            'why': ('without the incumbent there is nothing to reconcile the candidate against; '
                    'defaulting to the candidate itself would make this check vacuous')}
        reasons.append('repair_plan_unverifiable')
        detail_repair = {'served_planned': [], 'duplicate_suffix': []}
    elif legality_ok:
        got = _guard('repair_plan', lambda: validate_repair_plan(plans, base_plans,
                                                                 list(mask_set),
                                                                 list(protected)))
        if got is not None:
            ok_repair, detail_repair = got
            components['repair_plan'] = {'ok': bool(ok_repair), 'detail': detail_repair}
            if not ok_repair:
                reasons.append('repair_plan_failed')
        else:
            detail_repair = {'served_planned': [], 'duplicate_suffix': []}
    else:
        components['repair_plan'] = {'ok': False, 'not_run_illegal_order_set': True,
                                     'refused_by_gate': 'legality_gate',
                                     'why': 'a legality gate already refused'}
        detail_repair = {'served_planned': [], 'duplicate_suffix': []}

    # ---- route feasibility (TW / capacity / return / temperature class) --------------------
    # `ok` is DERIVED from the per-route report that is actually returned, never from a parallel
    # flag: two sources of truth for the same verdict let a component report "all routes feasible"
    # while the component itself says not-ok, which is exactly the kind of inconsistency a
    # certification gate must not have.
    # RUNS ONLY IF legality_ok (see the short-circuit above).
    if legality_ok:
        def _routes():
            out = {}
            for vid, p in sorted(plans.items()):
                r = certify_route(env, inst_idx, p.anchor_node, p.anchor_time, p.anchor_load,
                                  tuple(p.suffix))
                out[int(vid)] = {'feasible': bool(r['feasible']), 'reason': r['reason'],
                                 'temperature_compatible': bool(r['temperature_compatible']),
                                 'coldchain_reject_reason': r.get('coldchain_reject_reason'),
                                 # recorded as None by the primitive itself: no decision-time
                                 # energy/quality projection exists in it
                                 'projected_energy_kwh': r.get('projected_energy_kwh'),
                                 'projected_quality_loss': r.get('projected_quality_loss')}
            return out

        got = _guard('route_feasible', _routes)
        if got is not None:
            per_route = got
            infeasible_routes = sorted(vid for vid, v in per_route.items() if not v['feasible'])
            ok_route = not infeasible_routes
            components['route_feasible'] = {'ok': ok_route, 'per_route': per_route,
                                            'infeasible_routes': infeasible_routes,
                                            'n_routes_checked': len(per_route)}
            if not ok_route:
                reasons.append('route_infeasible')

        # ---- C0 energy budget -------------------------------------------------------------
        got = _guard('energy_budget', lambda: certify_plan(env, inst_idx, clock, vehicles,
                                                           served_mask, _plan_routes(plans),
                                                           contract, B))
        if got is not None:
            feasible_energy, total_energy = got
            components['energy_budget'] = {'ok': bool(feasible_energy),
                                           'total_energy_kwh': float(total_energy),
                                           'B': float(B),
                                           'headroom_kwh': float(B) - float(total_energy)}
            if not feasible_energy:
                reasons.append('energy_budget_exceeded')

        # ---- read-only quality PROJECTION (D-B3A-2; NEVER a conservative certificate) ------
        if project_quality:
            def _quality_projection():
                from d3_quality_projection import (SCHEDULE_LATEST_ADMISSIBLE,
                                                   project_plan_quality)
                proj = project_plan_quality(
                    env, inst_idx, clock, vehicles, _plan_routes(plans), contract,
                    schedule=(SCHEDULE_LATEST_ADMISSIBLE if stress_quality_schedule
                              else 'nominal'), served_mask=served_mask)
                return proj
            got = _guard('quality_projection', _quality_projection)
            if got is not None:
                projection = got
                components['quality_projection'] = {
                    'ok': projection['projection_state'] != 'PROJECTED_FAILED',
                    'projection_state': projection['projection_state'],
                    'conservativeness': projection['conservativeness'],
                    'schedule_kind': projection['schedule_kind'],
                    'applicable_contract_version': projection['applicable_contract_version'],
                    'unsalable_orders': projection['unsalable_orders'],
                    'still_loaded_after_return': projection['still_loaded_after_return'],
                    'inadmissible_stress_vehicles': projection['inadmissible_stress_vehicles'],
                }
                if projection['projection_state'] == 'PROJECTED_FAILED':
                    reasons.append('quality_projection_failed')
    else:
        components['route_feasible'] = {'ok': False, 'not_run_illegal_order_set': True,
                                        'refused_by_gate': 'legality_gate'}
        components['energy_budget'] = {'ok': False, 'not_run_illegal_order_set': True,
                                       'refused_by_gate': 'legality_gate'}
        if project_quality:
            components['quality_projection'] = {'ok': False, 'not_run_illegal_order_set': True,
                                                'refused_by_gate': 'legality_gate'}

    # ---- quality / salability: three-state, never a default pass ---------------------------
    # D-B3A-3 is DEFERRED (user ruling): `PROJECTED_SATISFIED_CONSERVATIVE` does not exist yet,
    # because no testable bound Q_cert,i <= Q_i(omega) has been proven.  A PASSING PROJECTION
    # therefore changes NOTHING here -- the capability gap and installable=False STAND.
    if quality_check_fn is None:
        quality = {'state': QUALITY_CAPABILITY_GAP,
                   'why': ('no CONSERVATIVE decision-time quality/salability certification exists. '
                           'A read-only projection is available (component `quality_projection`), '
                           'but the user rejected the worst-case-schedule inference and the bound '
                           'is labelled CONSERVATIVENESS_NOT_ESTABLISHED, so the gap STANDS.')}
        blocking_gaps = list(CAPABILITY_GAPS)
    else:
        qok, qdetail = quality_check_fn(env, inst_idx, clock, vehicles, served_mask, plans)
        quality = {'state': QUALITY_OK if qok else QUALITY_FAILED, 'detail': qdetail}
        blocking_gaps = [] if qok else []
        if not qok:
            reasons.append('quality_failed')

    ok = not reasons
    installable = bool(ok and not blocking_gaps)
    return {
        'certification_schema': 'composed_certification_v2_legality_first',
        'checks': list(CERTIFICATION_CHECKS), 'legality_gates': list(LEGALITY_GATES),
        'components': components,
        'legality_ok': legality_ok,
        'quality': quality, 'blocking_capability_gaps': blocking_gaps,
        'reasons': reasons, 'ok': ok, 'installable': installable,
        'contract_hash_used_for_energy': getattr(contract, 'contract_hash', None),
        'note': ('legality gates run FIRST and, when one fails, the route / energy / quality '
                 'components are NOT RUN (`not_run_illegal_order_set`); `installable` still '
                 'requires that no blocking capability gap stands'),
    }
