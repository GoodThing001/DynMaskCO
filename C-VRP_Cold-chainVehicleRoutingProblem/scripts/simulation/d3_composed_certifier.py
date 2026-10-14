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
COMPENSATIONS = ('revealed_only', 'no_served_in_plan', 'plan_vehicle_set', 'no_duplicate_order')
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
                          base_plans=None, mutable_ids=None, quality_check_fn=None):
    """Run the four checks + the compensations, and report the quality capability state.

    Returns a dict; `installable` is True only when every check and compensation passes AND no
    blocking capability gap exists.
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

    # ---- 1. plan reconciliation (exact-once / non-mask order / protected unchanged) ---------
    if not base_is_given:
        components['repair_plan'] = {
            'ok': False, 'error': 'base_plans_missing',
            'why': ('without the incumbent there is nothing to reconcile the candidate against; '
                    'defaulting to the candidate itself would make this check vacuous')}
        reasons.append('repair_plan_unverifiable')
        detail_repair = {'served_planned': [], 'duplicate_suffix': []}
    else:
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

    # ---- 2. ownership / commitment / service partition --------------------------------------
    committed = {int(v.committed_next) for v in vehicles
                 if v.committed_next not in (None, 0)}
    universe = [int(c) for c in range(1, env.num_nodes)
                if not bool(served_mask[c]) and int(c) not in committed
                and bool(env.reveal_time[inst_idx, c] <= clock + 1e-6)]
    got = _guard('full_partition', lambda: validate_full_partition(
        committed, plans, set(deferred), universe, served_mask=served_mask))
    if got is not None:
        ok_part, detail_part = got
        components['full_partition'] = {'ok': bool(ok_part), 'detail': detail_part}
        if not ok_part:
            reasons.append('full_partition_failed')
    else:
        detail_part = {'served_planned': [], 'duplicate_suffix': []}

    # ---- 3. route feasibility (TW / capacity / return / temperature class) ------------------
    # `ok` is DERIVED from the per-route report that is actually returned, never from a parallel
    # flag: two sources of truth for the same verdict let a component report "all routes feasible"
    # while the component itself says not-ok, which is exactly the kind of inconsistency a
    # certification gate must not have.
    def _routes():
        out = {}
        for vid, p in sorted(plans.items()):
            r = certify_route(env, inst_idx, p.anchor_node, p.anchor_time, p.anchor_load,
                              tuple(p.suffix))
            out[int(vid)] = {'feasible': bool(r['feasible']), 'reason': r['reason'],
                             'temperature_compatible': bool(r['temperature_compatible']),
                             'coldchain_reject_reason': r.get('coldchain_reject_reason'),
                             # recorded as None by the primitive itself -- evidence that no
                             # decision-time energy/quality projection exists
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

    # ---- 4. C0 energy budget ---------------------------------------------------------------
    got = _guard('energy_budget', lambda: certify_plan(env, inst_idx, clock, vehicles, served_mask,
                                                       _plan_routes(plans), contract, B))
    if got is not None:
        feasible_energy, total_energy = got
        components['energy_budget'] = {'ok': bool(feasible_energy), 'total_energy_kwh':
                                       float(total_energy), 'B': float(B),
                                       'headroom_kwh': float(B) - float(total_energy)}
        if not feasible_energy:
            reasons.append('energy_budget_exceeded')

    # ---- compensations ---------------------------------------------------------------------
    def _compensations():
        reveal = env.reveal_time[inst_idx]
        unrevealed = sorted({int(x) for p in plans.values() for x in p.suffix
                             if int(x) > 0 and not bool(reveal[int(x)] <= clock + 1e-6)})
        live_non_closed = {int(v.vehicle_id) for v in vehicles if v.status != 'closed'}
        extra_vehicles = sorted(set(int(v) for v in plans) - live_non_closed)
        return {
            'revealed_only': {'ok': not unrevealed, 'unrevealed_in_plan': unrevealed},
            'no_served_in_plan': {'ok': not detail_part['served_planned'],
                                  'served_planned': detail_part['served_planned']},
            'plan_vehicle_set': {'ok': not extra_vehicles,
                                 'plan_vehicles': sorted(int(v) for v in plans),
                                 'live_non_closed': sorted(live_non_closed),
                                 'closed_or_absent_vehicles_in_plan': extra_vehicles},
            'no_duplicate_order': {'ok': not detail_part['duplicate_suffix'],
                                   'duplicate_suffix': detail_part['duplicate_suffix']},
        }

    comp = _guard('compensations', _compensations)
    if comp is None:
        comp = {name: {'ok': False, 'error': 'compensation_check_error'}
                for name in COMPENSATIONS}
    else:
        components['compensations'] = {'ok': all(v['ok'] for v in comp.values())}
    for name in COMPENSATIONS:
        if not comp[name]['ok']:
            reasons.append('compensation_failed:%s' % name)

    # ---- quality / salability: three-state, never a default pass ---------------------------
    if quality_check_fn is None:
        quality = {'state': QUALITY_CAPABILITY_GAP,
                   'why': ('no executable decision-time quality/salability check exists: '
                           'certify_route leaves projected_quality_loss as None and the only '
                           'salability check is terminal (coldchain_evaluator_a1:115-117)')}
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
        'certification_schema': 'composed_certification_v1',
        'checks': list(CERTIFICATION_CHECKS), 'compensations': list(COMPENSATIONS),
        'components': components, 'compensation_results': comp,
        'quality': quality, 'blocking_capability_gaps': blocking_gaps,
        'reasons': reasons, 'ok': ok, 'installable': installable,
        'contract_hash_used_for_energy': getattr(contract, 'contract_hash', None),
        'note': ('`ok` means every executable check passed; `installable` additionally requires '
                 'that no blocking capability gap stands.  A caller must not treat `ok` alone as '
                 'a licence to install.'),
    }
