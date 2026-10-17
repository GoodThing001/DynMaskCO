#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPERIMENTAL_NOT_FOR_PRODUCTION -- Q3 continuous-temperature INTERVAL bounds.

This module exists to make the CORRECTED Q3 propositions executable and testable:

  P1  U recursion        U_n = psi_{dt_n}(U_{n-1}) + I_n * delta_z   =>  T_n <= U_n
  P2  door-count bound   T_n <= tau_z + N_n^z * delta_z
  P3  minimum segment    if every segment is at least delta_z/margin_z long, T <= tau_z + delta_z
  C1  hard-bound safety  tau_z > low_z AND (P2 budget OR P1 budget OR P3 condition)

It is NOT wired into any certifier, it does not modify any frozen file, and it CANNOT emit a
production-consumable receipt: `production_consumable` is hard-coded False and the verdict key is
`experimental_verdict`, deliberately different from any production receipt field.  The rejected
claim (the old Theorem B band invariant for ARBITRARY segment sequences) is refuted in
docs/.../Q3_定理B反例与更弱正确命题_独立勘误_2026-10-16.md and is NOT implemented as a bound here.

Every refusal is fail-closed: missing contract identity, missing state identity, missing or
non-finite segment data, a closed zone (outside the declared domain), or a plan whose bound crosses
the hard bound all yield REFUSED rather than a silent pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

EXPERIMENTAL_NOT_FOR_PRODUCTION = True
PRODUCTION_CONSUMABLE = False
IMPLEMENTATION_VERSION = 'Q3_THERMAL_INTERVAL_EXPERIMENTAL_V1'
# Outward rounding: exp() is not guaranteed correctly rounded by IEEE-754, so one ulp is not a
# proof-grade guard.  The module takes this many ulps upward at every step and RECORDS the
# assumption; declaring a signed safety margin remains an open obligation (spec section 7.1).
ULP_STEPS = 4


@dataclass(frozen=True)
class ZoneThermal:
    zone: int
    ambient_c: float
    target_c: float
    hard_low_c: float
    hard_high_c: float
    heat_transfer_per_hour: float
    cooling_rate_c_per_hour: float
    door_heat_kwh: float
    thermal_capacity_kwh_per_c: float

    @property
    def delta_door_c(self) -> float:
        return self.door_heat_kwh / self.thermal_capacity_kwh_per_c

    @property
    def drift_margin_c_per_hour(self) -> float:
        """c - h*(A - tau)^+ ; must be > 0 for P3 to apply."""
        return (self.cooling_rate_c_per_hour
                - self.heat_transfer_per_hour * max(0.0, self.ambient_c - self.target_c))

    @property
    def return_time_h(self) -> float:
        """delta / margin: the longest time a worst-case excursion needs to fall back to tau."""
        m = self.drift_margin_c_per_hour
        return float('inf') if m <= 0 else self.delta_door_c / m


def zone_thermals(contract) -> tuple[ZoneThermal, ...]:
    th = contract.thermal
    out = []
    for z, _ in enumerate(th.supported_temp_classes):
        low, high = th.hard_temperature_bounds_c[z]
        out.append(ZoneThermal(
            zone=z, ambient_c=float(th.ambient_temperature_c),
            target_c=float(th.target_temperature_c[z]), hard_low_c=float(low), hard_high_c=float(high),
            heat_transfer_per_hour=float(th.heat_transfer_per_hour[z]),
            cooling_rate_c_per_hour=float(th.cooling_rate_c_per_hour[z]),
            door_heat_kwh=float(th.door_heat_kwh[z]),
            thermal_capacity_kwh_per_c=float(th.thermal_capacity_kwh_per_c[z])))
    return tuple(out)


def _up(x: float) -> float:
    """Round upward by ULP_STEPS ulps (outward rounding for an UPPER bound)."""
    for _ in range(ULP_STEPS):
        x = math.nextafter(x, math.inf)
    return x


def advance_upward(z: ZoneThermal, t0: float, dt_h: float, eps: float = 1e-9) -> float:
    """Executor-consistent advance, rounded UP: the short-circuit branch is reproduced exactly."""
    if dt_h <= eps:
        return t0                                     # executor: no cooling, no clamp (state.py:361)
    natural = z.ambient_c + (t0 - z.ambient_c) * math.exp(-z.heat_transfer_per_hour * dt_h)
    cooled = natural - z.cooling_rate_c_per_hour * dt_h
    return _up(max(z.target_c, cooled))               # executor: max(target, cooled) (state.py:376)


def u_recursion(z: ZoneThermal, segments: Sequence[tuple[float, bool]],
                eps: float = 1e-9) -> list[float]:
    """P1. segments = [(dt_h, opens_this_zone), ...]. Returns the U sequence starting at U_0 = tau."""
    u = z.target_c
    out = [u]
    for dt_h, opens in segments:
        u = advance_upward(z, u, dt_h, eps)
        if opens:
            u = _up(u + z.delta_door_c)
        out.append(u)
    return out


def door_count_bound(z: ZoneThermal, n_doors: int) -> float:
    """P2. tau + N*delta, rounded up."""
    return _up(z.target_c + n_doors * z.delta_door_c)


def band_invariant_condition(z: ZoneThermal, segments: Sequence[tuple[float, bool]],
                             eps: float = 1e-9) -> dict:
    """P3. The band invariant [tau, tau+delta] holds ONLY if every segment is at least delta/margin
    long; a segment at or below eps is the worst case, not a safe one."""
    need = z.return_time_h
    short = [i for i, (dt_h, _o) in enumerate(segments) if dt_h < need]
    return {'return_time_h': need, 'n_segments': len(segments), 'short_segment_indices': short,
            'band_invariant_holds': not short and math.isfinite(need),
            'short_includes_eps_branch': any(dt_h <= eps for dt_h, _ in segments)}


def certify_interval(contract, *, contract_identity, state_identity, segments_by_zone,
                     active_zone_mask, eps: float = 1e-9) -> dict:
    """EXPERIMENTAL interval check. FAIL-CLOSED.

    Refuses when: identity missing; a zone is inactive (out of the declared domain D1); segment data
    missing or non-finite; the resulting bound crosses the hard bound.  Never returns a
    production-consumable verdict.
    """
    refusals = []
    if not contract_identity:
        refusals.append('MISSING_CONTRACT_IDENTITY')
    if not state_identity:
        refusals.append('MISSING_STATE_IDENTITY')
    zones = zone_thermals(contract)
    if len(active_zone_mask) != len(zones):
        refusals.append('ZONE_MASK_LENGTH_MISMATCH')
    elif not all(bool(x) for x in active_zone_mask):
        refusals.append('ZONE_INACTIVE_OUTSIDE_DECLARED_DOMAIN')
    if not segments_by_zone:
        refusals.append('MISSING_SEGMENT_DATA')

    per_zone, worst = [], 'INTERVAL_BOUND_HOLDS'
    for z in zones:
        segs = segments_by_zone.get(z.zone)
        if segs is None:
            refusals.append('MISSING_SEGMENTS_FOR_ZONE_%d' % z.zone)
            continue
        for dt_h, _opens in segs:
            if not isinstance(dt_h, (int, float)) or not math.isfinite(float(dt_h)) or dt_h < 0:
                refusals.append('NON_FINITE_OR_NEGATIVE_SEGMENT_ZONE_%d' % z.zone)
        n_doors = sum(1 for _dt, opens in segs if opens)
        u_seq = u_recursion(z, segs, eps)
        u_max = max(u_seq)
        n_bound = door_count_bound(z, n_doors)
        band = band_invariant_condition(z, segs, eps)
        tight = min(u_max, n_bound)
        safe_low = z.target_c > z.hard_low_c
        safe_high = tight <= z.hard_high_c
        if not (safe_low and safe_high):
            worst = 'INTERVAL_BOUND_EXCEEDS_HARD_BOUND'
        per_zone.append({
            'zone': z.zone, 'target_c': z.target_c, 'delta_door_c': z.delta_door_c,
            'n_doors': n_doors, 'u_max_c': u_max, 'door_count_bound_c': n_bound,
            'tight_bound_c': tight, 'hard_low_c': z.hard_low_c, 'hard_high_c': z.hard_high_c,
            'floor_above_hard_low': safe_low, 'bound_below_hard_high': safe_high,
            'return_time_h': z.return_time_h, 'band_invariant_holds': band['band_invariant_holds'],
            'short_segment_indices': band['short_segment_indices'],
        })

    verdict = 'REFUSED' if refusals else worst
    return {
        'schema': 'q3_thermal_interval_experimental',
        'EXPERIMENTAL_NOT_FOR_PRODUCTION': EXPERIMENTAL_NOT_FOR_PRODUCTION,
        'production_consumable': PRODUCTION_CONSUMABLE,
        'implementation_version': IMPLEMENTATION_VERSION,
        'contract_identity': contract_identity,
        'state_identity': state_identity,
        'refusals': refusals,
        'experimental_verdict': verdict,
        'per_zone': per_zone,
        'ulp_steps_upward': ULP_STEPS,
        'fractional_margin_declared': False,
        'note': 'P1/P2/P3/C1 only. The refuted band invariant for arbitrary segment sequences is NOT '
                'implemented. A REFUSED verdict is fail-closed, not an error.',
    }


def independent_recompute(contract, segments_by_zone, zone: int) -> float:
    """A deliberately separate, minimal recomputation a third party can check by reading the source.

    Plain arithmetic on purpose (no outward rounding), so it must be <= the U bound produced by
    `u_recursion`; the negative controls assert that ordering rather than assuming it.
    """
    z = zone_thermals(contract)[zone]
    t = z.target_c
    for dt_h, opens in segments_by_zone.get(zone, ()):
        if dt_h > 1e-9:
            t = max(z.target_c, z.ambient_c + (t - z.ambient_c)
                    * math.exp(-z.heat_transfer_per_hour * dt_h)
                    - z.cooling_rate_c_per_hour * dt_h)
        if opens:
            t = t + z.delta_door_c
    return t
