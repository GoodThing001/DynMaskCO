"""Budget-reserving admission for the A-v1 accept/reject decision.

Pre-registered candidate, see research/review_3080_20261010/pickup/
A1_BUDGET_RESERVING_ADMISSION_PREREGISTRATION.json.  The registered per-order cost
is the unavoidable pre-cooling energy

    e_lower(i) = quantity_i * 0.15 * max(0, 25 - target_c[class_i])

and NOT the cooling-share-based marginal estimator: the cooling share is a constant
per order and would invert the class ordering (it penalises the low-revenue class 0),
whereas the recorded A-v1 day energy is dominated by pre-cooling (about 83 percent
for the observed served set).  The registration fixes e_lower as the cost model.

Mechanism: with p_c=0 the base SAA has no reason to refuse an order, so the served
set is decided by arrival order against the hard energy budget, spending budget on
the worst value per unit of pre-cooling energy first.  This rule imposes a
clock-dependent shadow price on pre-cooling energy: an order is admitted only when
its revenue exceeds lambda(clock) * e_lower(order), where lambda(clock) is the
smallest price on a pre-computed grid whose expected accepted pre-cooling energy over
the remaining horizon still fits the remaining budget.  Calibration uses only the
frozen TRAIN split.  The rule is deterministic, is not learned online, and can only
add rejections relative to the base SAA.
"""
import bisect

REV = {0: 10.0, 1: 20.0, 2: 30.0}


def e_lower_of(class_id, quantity, obj):
    delta = max(0.0, float(obj['ambient_c']) - float(obj['target_c'][int(class_id)]))
    return float(quantity) * float(obj['precool_energy_per_unit_per_c']) * delta


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


def calibrate(train_ds, obj, horizon, lambda_grid, clock_grid):
    """G[li][ti] = expected accepted pre-cooling energy over orders revealed at or after clock_grid[ti].

    Per day the orders are sorted by reveal time once; the accepted-energy suffix sums
    are then read off with a binary search for every clock grid point.
    """
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
        est = np.asarray([e_lower_of(int(c), float(qq), obj) for c, qq in zip(cls, q)], dtype=np.float64)
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


class BudgetReservingAdmission(object):
    """Mixin: adds the reservation pre-filter in front of the base SAA decision."""

    def _reserving_init(self, lambda_grid, clock_grid, g_table, objective):
        self._lam = list(lambda_grid)
        self._clock = list(clock_grid)
        self._g = g_table
        self._obj = dict(objective)
        self.reserve_rejects = 0
        self.reserve_decisions = 0
        self.reserve_log = []

    def _e_lower(self, env, inst_idx, o):
        c = int(env.dataset['temp_class'][inst_idx, o])
        q = float(env.demands[inst_idx, o])
        return e_lower_of(c, q, self._obj), c

    def _spent_e(self, env, inst_idx):
        total = 0.0
        for o in self._accepted:
            total += self._e_lower(env, inst_idx, o)[0]
        return total

    def _lambda_for(self, clock, budget_remaining):
        ti = min(max(bisect.bisect_right(self._clock, float(clock)) - 1, 0), len(self._clock) - 1)
        for li, lam in enumerate(self._lam):
            if self._g[li][ti] <= budget_remaining:
                return lam, ti
        return self._lam[-1], ti

    def _reserving_allows(self, env, inst_idx, clock, o, token):
        self.reserve_decisions += 1
        e, cls = self._e_lower(env, inst_idx, o)
        b_rem = float(self.budget) - self._spent_e(env, inst_idx)
        token.update(est=e, revenue=REV[cls], class_id=cls)
        if e > b_rem:
            self.reserve_rejects += 1
            token['reason'] = 'over_remaining_budget'
            return False
        lam, ti = self._lambda_for(clock, b_rem)
        token['lambda'] = lam
        token['clock_bucket'] = ti
        if (REV[cls] - lam * e) > 0.0:
            token['reason'] = 'admit'
            return True
        self.reserve_rejects += 1
        token['reason'] = 'below_shadow_price'
        return False
