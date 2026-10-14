"""Frozen A-v1 deployment rule: clock shadow-price budget-reserving admission.

This is the single deployment rule registered in REGISTRATION.json.  It is a
self-contained transcription of the semantics that the v4 panel actually executed
(verified empirically at run time by reproducing v4's recorded per-day utilities),
so that the registration text and the executed action are the same object.

Decision, evaluated per revealed order, before the frozen base SAA decides:

  1. cost      e_lower(i) = quantity_i * precool_energy_per_unit_per_c
                            * max(0, ambient_c - target_c[class_i])
  2. spent     spent_before = sum of e_lower over the orders currently in the base
               replanner's accepted set (base-SAA acceptances only)
  3. budget    b_rem = B - spent_before
  4. hard pre-check   if e_lower(i) > b_rem: reject, reason 'over_remaining_budget'
  5. bucket    ti = min(max(bisect_right(clock_grid, clock) - 1, 0), len(clock_grid) - 1)
  6. price     lambda = the smallest value in the ascending lambda grid with
               G[lambda][ti] <= b_rem; if none satisfies it, lambda = max(grid)
  7. admit     iff REV[class_i] - lambda * e_lower(i) > 0  (strictly greater)
               otherwise reject, reason 'below_shadow_price'
  8. on admit  the frozen base SAA still decides; the rule can only add rejections

Calibration table G (TRAIN split only, seed 20260925, 200 days):
  G[lambda][ti] = mean over train days of the sum of e_lower over that day's orders whose
  reveal time is at or after clock_grid[ti] and which satisfy REV - lambda*e_lower > 0.
"""
import bisect

REV = {0: 10.0, 1: 20.0, 2: 30.0}


def e_lower_of(class_id, quantity, objective):
    delta = max(0.0, float(objective['ambient_c']) - float(objective['target_c'][int(class_id)]))
    return float(quantity) * float(objective['precool_energy_per_unit_per_c']) * delta


def build_lambda_grid(lo=0.25, hi=8.0, step=0.125):
    grid, x = [], lo
    while x <= hi + 1e-9:
        grid.append(round(x, 6))
        x += step
    return grid


def build_clock_grid(horizon, step=0.5):
    grid, x = [], 0.0
    while x <= horizon + 1e-9:
        grid.append(round(x, 6))
        x += step
    return grid


def calibrate(train_ds, objective, lambda_grid, clock_grid):
    import numpy as np
    dem = np.asarray(train_ds['demands'], dtype=np.float64)
    rev_t = np.asarray(train_ds['reveal_time'], dtype=np.float64)
    cls_t = np.asarray(train_ds['temp_class'], dtype=np.int64)
    n_days = dem.shape[0]
    days = []
    for d in range(n_days):
        mask = dem[d, 1:] > 0
        cls = cls_t[d, 1:][mask]
        q = dem[d, 1:][mask]
        rev = rev_t[d, 1:][mask]
        est = np.asarray([e_lower_of(int(c), float(qq), objective) for c, qq in zip(cls, q)], dtype=np.float64)
        rv = np.asarray([REV[int(c)] for c in cls], dtype=np.float64)
        order = np.argsort(rev, kind='stable')
        days.append((rev[order], est[order], rv[order]))
    clk = np.asarray(clock_grid, dtype=np.float64)
    table = []
    for lam in lambda_grid:
        row = np.zeros(len(clock_grid), dtype=np.float64)
        for rev, est, rv in days:
            contrib = np.where((rv - float(lam) * est) > 0.0, est, 0.0)
            suffix = np.concatenate([np.cumsum(contrib[::-1])[::-1], np.zeros(1)])
            row += suffix[np.searchsorted(rev, clk, side='left')]
        table.append((row / max(n_days, 1)).tolist())
    return table


def clock_bucket(clock, clock_grid):
    return min(max(bisect.bisect_right(clock_grid, float(clock)) - 1, 0), len(clock_grid) - 1)


def price_for(clock, budget_remaining, lambda_grid, clock_grid, g_table):
    ti = clock_bucket(clock, clock_grid)
    for li, lam in enumerate(lambda_grid):
        if g_table[li][ti] <= budget_remaining:
            return lam, ti
    return lambda_grid[-1], ti


def decide(e_lower, revenue, clock, budget_remaining, lambda_grid, clock_grid, g_table):
    """Return (admit, reason, lambda, bucket).  Mirrors the executed v4 semantics exactly."""
    if e_lower > budget_remaining:
        return False, 'over_remaining_budget', None, None
    lam, ti = price_for(clock, budget_remaining, lambda_grid, clock_grid, g_table)
    if (float(revenue) - lam * float(e_lower)) > 0.0:
        return True, 'admit', lam, ti
    return False, 'below_shadow_price', lam, ti
