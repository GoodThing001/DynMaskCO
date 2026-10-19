"""P2-C-GAP-01: offline support audit; synthetic-only proposed sampling rule.

No source-day execution, reconstruction, continuation, utility or training entry.
The capacity-conditioned proposal is NOT wired into SAA or the real teacher.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation', 'training'))
import p2_pos01_reference as ref
from coldchain_state import CargoLotState, DeliveredCargoRecord, VehicleColdChainState
from scenario_saa import (CondHistoricalSampler, ScenarioOrder, _scenario_from_day,
                          build_history, make_c0_contract_v2)

EXTENSION = Path(__file__).resolve().parents[2]
BASELINE = '28708ba8a7eab80e326c8cad9bf78ef6fe40c6b8'
EPS = ref.EVENT_TOL


@dataclass(frozen=True)
class PublicPrefix:
    """No raw env, hidden count, source future or branch identity is accepted."""
    clock: float
    booking_horizon: float
    declared_customer_slots: int
    orders: tuple[ScenarioOrder, ...]  # all public customers, INCLUDING served/rejected

    def free_slots(self):
        ids = [o.oid for o in self.orders]
        if (not isinstance(self.declared_customer_slots, int) or
                self.declared_customer_slots < 1 or
                not math.isfinite(self.clock) or not math.isfinite(self.booking_horizon) or
                not 0 <= self.clock <= self.booking_horizon or
                len(set(ids)) != len(ids) or
                any(not 1 <= i <= self.declared_customer_slots for i in ids) or
                any(not math.isfinite(o.reveal) or o.reveal > self.clock+EPS
                    for o in self.orders)):
            raise ValueError('INVALID_PUBLIC_PREFIX')
        return self.declared_customer_slots-len(ids)


class CapacityConditionedProposal:
    """G_HIST conditioned WITHIN its original kNN on full suffix fitting N_free.

    Qualification requires >=2 retained historical roots (an engineering support
    floor, NOT a power calculation). No refill, tail clipping or rejection loop.
    Counts use external TRAIN history, never the source day's hidden orders.
    """
    def __init__(self, history, n_neighbors=10):
        if not history or n_neighbors < 1:
            raise ValueError('EMPTY_HISTORY_OR_INVALID_NEIGHBORS')
        self.history = tuple(tuple(day) for day in history)
        self.base = CondHistoricalSampler(self.history, n_neighbors=n_neighbors)

    def support(self, prefix):
        free = prefix.free_slots()
        # The declared public horizon is a contract, not a learned source cutoff.
        if any(not math.isfinite(o.reveal) or not 0 <= o.reveal < prefix.booking_horizon
               for day in self.history for o in day):
            raise ValueError('HISTORY_OUTSIDE_PUBLIC_HORIZON')
        x = self.base._feat(prefix.orders, prefix.clock)
        h = np.array([self.base._feat(day, prefix.clock) for day in self.history])
        mu, sd = h.mean(0), h.std(0)+1e-6
        distances = (((h-mu)/sd-(x-mu)/sd)**2).sum(1)
        # Match CondHistoricalSampler exactly, including NumPy's tie ordering.
        nearest = np.argsort(distances)[:max(1, min(self.base.n_neighbors, len(h)))]
        counts = [sum(o.reveal > prefix.clock+EPS for o in day) for day in self.history]
        eligible = [int(i) for i in nearest if counts[i] <= free]
        return dict(free_slots=free, suffix_counts=counts,
                    squared_distances=distances.tolist(), nearest=nearest.tolist(),
                    eligible=eligible, all_history_compatible=sum(n <= free for n in counts),
                    retained_mass=len(eligible)/len(nearest),
                    excluded_mass=1-len(eligible)/len(nearest),
                    status=('NO_SUPPORT' if not eligible else 'LOW_SUPPORT'
                            if len(eligible) < 2 else 'REFERENCE_SUPPORT_PASS'))

    def draw(self, prefix, event_rng, external_event_key, *, synthetic=False):
        if not synthetic:
            raise PermissionError('REAL_SCENARIO_DRAW_NOT_AUTHORISED_BY_GAP01')
        receipt = self.support(prefix)
        if receipt['status'] != 'REFERENCE_SUPPORT_PASS':
            raise ValueError(receipt['status'])
        rng = event_rng.generator(external_event_key)
        selected = receipt['eligible'][int(rng.integers(0, len(receipt['eligible'])))]
        future = tuple(_scenario_from_day(self.history[selected], prefix.clock))
        if len(future) > receipt['free_slots']:
            raise AssertionError('SUPPORT_IMPLEMENTATION_MISMATCH')
        return future, dict(receipt, selected_history_index=selected,
                            selected_suffix_count=len(future))


def prefix_from_view(view):
    ref.validate_view(view)
    return PublicPrefix(view.clock, view.booking_horizon, 50, tuple(
        ScenarioOrder(o.oid, o.reveal, o.x, o.y, o.demand, o.temp_class,
                      o.tw_start, o.tw_end, o.service)
        for o in view.orders if o.oid))


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': str(value)}
    return value


def view_from_json(capture):
    """Read public JSON only, never the full/private pickle snapshot."""
    raw = capture['public_view']
    contract = make_c0_contract_v2()
    if _plain(asdict(contract)) != raw['contract']:
        raise ValueError('CONTRACT_IDENTITY_MISMATCH')
    vehicles = []
    for raw_v in raw['vehicles']:
        v = dict(raw_v)
        if v['coldchain_state'] is not None:
            cc = dict(v['coldchain_state'])
            for key in ('compartment_temperature_c', 'zone_load'):
                cc[key] = tuple(cc[key])
            cc['cargo_manifest'] = tuple(CargoLotState(**x) for x in cc['cargo_manifest'])
            cc['delivered_to_depot'] = tuple(DeliveredCargoRecord(**x)
                                             for x in cc['delivered_to_depot'])
            v['coldchain_state'] = VehicleColdChainState(**cc)
        for key in ('mutable_suffix', 'served_route'):
            v[key] = tuple(v[key])
        vehicles.append(ref.Vehicle(**v))
    args = dict(raw, contract=contract, vehicles=tuple(vehicles),
                orders=tuple(ref.Order(**o) for o in raw['orders']))
    for key in ('accepted', 'rejected', 'served', 'replan_ids', 'protected'):
        args[key] = tuple(args[key])
    view = ref.validate_view(ref.PublicView(**args))
    if ref.identity(view) != capture['public_identity']:
        raise ValueError('PUBLIC_IDENTITY_MISMATCH')
    return view


def mask_diagnostic(view, seed=0):
    """Fresh public-only diagnostic, NOT a recovered v2 reconstruction receipt."""
    ref.validate_view(view)
    plans = ref.plans_from_view(view)
    allowed = ref.mutable_ids(view)
    mutable = sorted(c for vid, p in plans.items() if vid in allowed for c in p.suffix
                     if c in view.accepted and c not in view.served)
    masks = ref.mask_pool(view, plans, seed=seed)
    # With exactly two mutable IDs every nonempty double-order proposal MUST be
    # that pair. All four rules are nonempty here because a mutable route exists.
    forced = (len(mutable) == 2 and masks == (tuple(mutable),))
    return dict(accepted=len(view.accepted), served=len(view.served),
                accepted_unserved=sorted(set(view.accepted)-set(view.served)),
                mutable_customers=mutable, mutable_vehicle_ids=sorted(allowed),
                suffixes={str(vid): list(p.suffix) for vid, p in plans.items()},
                conservative_base_routes_pass=all(ref.route_check(view, p) for p in plans.values()),
                structural_pair_upper_bound=math.comb(len(mutable), 2),
                masks=masks, distinct_masks=len(masks),
                distinct_mask_pairs=math.comb(len(masks), 2),
                raw_proposals_forced_by_two_customers=(
                    {name: mutable for name in ('distance', 'resource', 'frontier', 'random')}
                    if forced else None),
                duplicate_raw_proposals=(3 if forced else None),
                multi_mask_eligible=(len(masks) >= 2),
                scope='NEW_READ_ONLY_DIAGNOSTIC_NOT_V2_ACTION_RECEIPT')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    """Fixed 16 HIST + four saved public views. No real scenario draws."""
    r0 = EXTENSION/'configs/p2_pos01c_r0_20261019_v1'
    v2 = EXTENSION/'research/p2_pos01c_smoke_20261019_v2'
    manifest = r0/'data_identity.json'
    days = [d for d in json.loads(manifest.read_text())['days'] if d['role'] == 'HIST_TRAIN']
    if len(days) != 16:
        raise ValueError('EXPECTED_16_HIST_ROOTS')
    # Every old tracked artifact, local R0 asset and relevant code is byte checked.
    protected = []
    for folder in (r0, EXTENSION/'data/p2_pos01c_r0_20261019_v1',
                   EXTENSION/'research/p2_pos01c_smoke_20261019_v1', v2,
                   EXTENSION/'research/p2_c_fix_01_20261019'):
        if folder.exists():
            protected.extend(p for p in folder.rglob('*') if p.is_file())
    protected.extend(p for p in (EXTENSION/'scripts').rglob('p2_pos01*.py')
                     if p.name != Path(__file__).name)
    protected.extend([Path(ref.__file__), EXTENSION/'scripts/evaluation/scenario_saa.py'])
    before = {str(p.relative_to(EXTENSION)): sha(p) for p in sorted(set(protected))}
    history = []
    for d in days:
        path = EXTENSION/d['relative_path']
        if sha(path) != d['file_sha256']:
            raise ValueError('HIST_HASH_MISMATCH')
        with np.load(path, allow_pickle=False) as data:
            history.extend(build_history(data))
    sampler = CapacityConditionedProposal(history, n_neighbors=10)
    states = []
    for i in range(4):
        path = v2/('capture_%d.json' % i)
        capture = json.loads(path.read_text())
        view = view_from_json(capture)
        support = sampler.support(prefix_from_view(view))
        support['nearest_day_ids'] = [days[j]['day_id'] for j in support['nearest']]
        support['eligible_day_ids'] = [days[j]['day_id'] for j in support['eligible']]
        masks = mask_diagnostic(view)
        if _plain(masks['masks']) != capture['legal_masks']:
            raise ValueError('SAVED_MASK_POOL_MISMATCH')
        states.append(dict(index=i, day_id=capture['day_id'], clock=view.clock,
                           public_identity=capture['public_identity'], support=support, masks=masks))
    after = {rel: sha(EXTENSION/rel) for rel in before}
    if before != after:
        raise ValueError('READ_ONLY_VIOLATION')
    return dict(baseline=BASELINE, qualification='REFERENCE_ONLY',
                status='BLOCKED_MULTI_MASK_NOT_IDENTIFIABLE_IN_SAVED_STATES',
                numpy_version=np.__version__, python=sys.version,
                history_day_ids=[d['day_id'] for d in days],
                data_identity_sha256=sha(manifest), states=states,
                immutable_before=before, immutable_after_equal=True,
                real_future_draws=0, real_continuations=0, model_forwards=0, training_steps=0,
                v2_missing_action_receipts='NOT_RECOVERED_NOT_BACKFILLED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refuse overwrite: use a new diagnostic identity')
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
