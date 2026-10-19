"""P2-POS-01A isolated CA-UGMR reference prototype (CPU, synthetic acceptance only).

No production registration, admission policy, batch runner or FULL_CERTIFIED claim.
The public boundary and worker return immutable values; neither can mutate an env.
Timeout kills the persistent worker before returning KEEP, rather than cancelling
a Python thread while it continues to write a plan. A killed session cannot resume.

Acceptance record is in test_p2_pos01_reference.py; real-day/SAA qualification,
continuous-temperature/numerical certification and P2-POS-01B remain unestablished.
"""
from __future__ import annotations

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import copy
import hashlib
import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports(('experimental', 'simulation', 'coldchain', 'models', 'decoding', 'training'))
from action_contract import (VehiclePlan, build_vehicle_plans, enumerate_actions_from_plans,
                             apply_action, plan_hash)
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency
from mpre_policy import validate_repair_plan
from coldchain_state import (VehicleColdChainState, create_vehicle_state,
                             dispatch_vehicle, transition_segment)

CAPACITY_MARGIN = 1e-4
EVENT_TOL = 1e-6
QUALIFICATION = 'REFERENCE_ONLY'


def finite(value):
    if is_dataclass(value):
        return finite(asdict(value))
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(x) for x in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    return True


def identity(value):
    def canonical(x):
        if is_dataclass(x):
            if hasattr(x, 'contract_hash'):
                return {'contract_hash': x.contract_hash}
            return {f.name: canonical(getattr(x, f.name)) for f in fields(x)}
        if isinstance(x, dict):
            return {k: canonical(y) for k, y in x.items()}
        if isinstance(x, (tuple, list)):
            return [canonical(y) for y in x]
        if isinstance(x, np.generic):
            return x.item()
        return x
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Order:
    oid: int
    x: float
    y: float
    demand: float
    tw_start: float
    tw_end: float
    service: float
    temp_class: int
    initial_quality: float
    reveal: float


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    status: str = 'idle'
    current_node: int = 0
    ready_time: float = 0.0
    current_load: float = 0.0
    committed_next: int | None = None
    committed_arrive: float | None = None
    committed_finish: float | None = None
    mutable_suffix: tuple = ()
    served_route: tuple = ()
    coldchain_state: object = None
    coldchain_updated_time: float | None = None


@dataclass(frozen=True)
class PublicView:
    orders: tuple[Order, ...]
    vehicles: tuple[Vehicle, ...]
    accepted: tuple[int, ...]
    rejected: tuple[int, ...]
    served: tuple[int, ...]
    replan_ids: tuple[int, ...]
    protected: tuple[int, ...]
    clock: float
    booking_horizon: float
    capacity: float
    tw_speed: float
    energy_budget: float
    contract: object


def validate_view(v):
    if not isinstance(v, PublicView):
        raise ValueError('invalid PublicView type')
    values = asdict(v)
    contract_values = values.pop('contract')
    # v2's tomato threshold=+inf is an UNUSED schema sentinel for type 'none'.
    # Keep the existing contract unchanged; reject all active nonfinite values.
    quality = contract_values['quality']
    if 'threshold_type' in quality:
        quality['threshold'] = tuple(None if typ == 'none' and val == math.inf else val
            for typ, val in zip(quality['threshold_type'], quality['threshold']))
    if not finite(values) or not finite(contract_values):
        raise ValueError('invalid/non-finite PublicView')
    v.contract.validate()
    if v.contract.schema_version != 'coldchain-contract-v2':
        raise ValueError('only the new v2 reference utility contract is supported')
    orders = {o.oid: o for o in v.orders}
    vids = {x.vehicle_id for x in v.vehicles}
    if len(orders) != len(v.orders) or 0 not in orders or len(vids) != len(v.vehicles):
        raise ValueError('duplicate/missing public identity')
    if v.capacity <= CAPACITY_MARGIN or v.tw_speed <= 0 or v.energy_budget <= 0:
        raise ValueError('invalid public limits')
    if abs(v.capacity-v.contract.operational.shared_vehicle_capacity) > 1e-9:
        raise ValueError('contract capacity mismatch')
    if v.clock < 0 or v.booking_horizon < 0 or orders[0].demand != 0:
        raise ValueError('invalid clock/depot')
    for o in v.orders:
        if (o.oid < 0 or o.reveal > v.clock+EVENT_TOL or o.tw_start > o.tw_end or
                o.service < 0):
            raise ValueError('nonpublic/invalid order')
        if o.oid and (o.demand <= 0 or o.initial_quality <= 0 or
                      o.temp_class not in v.contract.thermal.supported_temp_classes):
            raise ValueError('invalid order resource')
    a, r, s = map(set, (v.accepted, v.rejected, v.served))
    if (len(a) != len(v.accepted) or len(r) != len(v.rejected) or
            len(s) != len(v.served) or a & r or not s <= a or
            a | r != set(orders)-{0}):
        raise ValueError('public acceptance ledger mismatch')
    if not set(v.replan_ids) <= vids or not set(v.protected) <= vids:
        raise ValueError('unknown vehicle scope')
    for x in v.vehicles:
        if x.status not in ('idle', 'ready', 'committed', 'returning', 'closed'):
            raise ValueError('invalid vehicle status')
        referenced = {x.current_node, *x.mutable_suffix, *x.served_route}
        if x.committed_next is not None:
            referenced.add(x.committed_next)
        if not referenced <= set(orders) or not set(x.served_route)-{0} <= s:
            raise ValueError('vehicle references nonpublic/nonserved order')
        if x.current_load < 0 or not capacity_ok(v, x.current_load):
            raise ValueError('invalid current capacity')
        if x.status in ('committed', 'returning') and (
                x.committed_finish is None or x.committed_finish < v.clock or
                x.committed_arrive is None or x.committed_next is None):
            raise ValueError('incomplete committed leg')
        cc = x.coldchain_state
        if cc is not None:
            if not isinstance(cc, VehicleColdChainState):
                raise ValueError('untyped physical coldchain state')
            if x.coldchain_updated_time != v.clock and x.status != 'closed':
                raise ValueError('coldchain state is not at public decision clock')
            if abs(cc.total_load-x.current_load) > 1e-9:
                raise ValueError('manifest/load mismatch')
            if not {lot.order_id for lot in cc.cargo_manifest} <= s:
                raise ValueError('cargo/served ledger mismatch')
        elif x.status != 'idle' or x.current_load != 0:
            raise ValueError('missing physical coldchain state')
    return v


def project_public(records, vehicles, *, accepted, rejected, served, replan_ids,
                   protected=(), clock, booking_horizon, capacity, tw_speed,
                   energy_budget, contract):
    """Boundary only: read reveal to filter, then copy ONLY published fields.

    No raw object, hidden values, true future count or environment survives here.
    Incoming vehicles must already be advanced to clock by the private engine.
    """
    orders = []
    for rec in records:
        if float(rec['reveal']) <= clock+EVENT_TOL:
            orders.append(Order(**{k: rec[k] for k in Order.__dataclass_fields__}))
    frozen = []
    for vehicle in vehicles:
        values = {k: copy.deepcopy(getattr(vehicle, k)) for k in Vehicle.__dataclass_fields__}
        values['mutable_suffix'] = tuple(values['mutable_suffix'])
        values['served_route'] = tuple(values['served_route'])
        if values['status'] in ('idle', 'ready'):
            values['ready_time'] = max(float(clock), values['ready_time'])
        frozen.append(Vehicle(**values))
    return validate_view(PublicView(tuple(sorted(orders, key=lambda o: o.oid)),
        tuple(sorted(frozen, key=lambda x: x.vehicle_id)), tuple(sorted(accepted)),
        tuple(sorted(rejected)), tuple(sorted(served)), tuple(sorted(replan_ids)),
        tuple(sorted(protected)), float(clock), float(booking_horizon), float(capacity),
        float(tw_speed), float(energy_budget), contract))


def capacity_ok(view, load):
    return math.isfinite(float(load)) and float(load) <= view.capacity-CAPACITY_MARGIN


def public_env(view):
    """Compatibility projection for existing route math; contains NO hidden rows.

    Original public IDs remain relation keys. Gaps are NaNs, never real data.
    Model inputs compress visible rows separately. Euclidean geometry only.
    """
    n = max(o.oid for o in view.orders)+1
    if n > 51:
        raise ValueError('public ID exceeds supported sparse-index bound')
    arrays = {k: np.full((1, n), np.nan) for k in
              ('demands', 'tw_start', 'tw_end', 'service_time', 'temp_class',
               'initial_quality', 'reveal_time')}
    coords = np.full((1, n, 2), np.nan)
    for o in view.orders:
        coords[0, o.oid] = [o.x, o.y]
        for key, val in (('demands', o.demand), ('tw_start', o.tw_start),
                         ('tw_end', o.tw_end), ('service_time', o.service),
                         ('temp_class', o.temp_class), ('initial_quality', o.initial_quality),
                         ('reveal_time', o.reveal)):
            arrays[key][0, o.oid] = val
    distances = np.linalg.norm(coords[:, :, None]-coords[:, None, :], axis=-1)
    return SimpleNamespace(**arrays, coords=coords, dist_mat=distances,
                           capacity=view.capacity, tw_speed=view.tw_speed,
                           coldchain_contract=view.contract)


def plans_from_view(view):
    return build_vehicle_plans(public_env(view), 0, view.vehicles)


def mutable_ids(view):
    return {v.vehicle_id for v in view.vehicles if v.status in ('idle', 'ready')
            and v.vehicle_id in view.replan_ids and v.vehicle_id not in view.protected}


def route_check(view, plan):
    """Single conservative predicate used by insertion, replay and acceptance."""
    e = public_env(view)
    load, t, cur = plan.anchor_load, plan.anchor_time, plan.anchor_node
    if not finite((load, t)) or not capacity_ok(view, load):
        return False
    known = {o.oid: o for o in view.orders}
    if cur not in known:
        return False
    for c in plan.suffix:
        if c not in known or c == 0 or c not in view.accepted or c in view.served:
            return False
        o = known[c]
        arrival = t+float(e.dist_mat[0, cur, c])/view.tw_speed
        load += o.demand
        if not capacity_ok(view, load) or arrival > o.tw_end+EVENT_TOL:
            return False
        t, cur = max(arrival, o.tw_start)+o.service, c
    return t+float(e.dist_mat[0, cur, 0])/view.tw_speed <= known[0].tw_end+EVENT_TOL


def legal_actions(view, plans, remaining):
    e, allowed, out = public_env(view), mutable_ids(view), []
    for c in remaining:
        if c not in view.accepted or c in view.served:
            raise ValueError('illegal repair customer')
        candidates, _ = enumerate_actions_from_plans(e, 0, plans, c,
                                                    allowed_vehicle_ids=allowed)
        for cand in candidates:
            if cand.feasible:
                trial = apply_action(plans, cand.action, allowed_vehicle_ids=allowed)
                if all(route_check(view, p) for p in trial.values()):
                    out.append((c, cand.action, cand))
    return out


def mask_pool(view, plans, seed=0):
    if not all(route_check(view, p) for p in plans.values()):
        return ()
    allowed = mutable_ids(view)
    pool = sorted(c for vid, p in plans.items() if vid in allowed for c in p.suffix
                  if c in view.accepted and c not in view.served)
    if len(pool) < 2:
        return ()
    e, orders = public_env(view), {o.oid: o for o in view.orders}
    vmap = {v.vehicle_id: v for v in view.vehicles}
    detour, pressure, first = {}, {}, []
    for vid, p in plans.items():
        route_time = p.anchor_time
        for i, c in enumerate(p.suffix):
            pred = p.anchor_node if i == 0 else p.suffix[i-1]
            arrival = route_time+float(e.dist_mat[0, pred, c])/view.tw_speed
            route_time = max(arrival, orders[c].tw_start)+orders[c].service
            if c not in pool:
                continue
            pred, succ = (p.anchor_node if i == 0 else p.suffix[i-1]), (0 if i+1 == len(p.suffix) else p.suffix[i+1])
            detour[c] = float(e.dist_mat[0, pred, c]+e.dist_mat[0, c, succ]-e.dist_mat[0, pred, succ])
            cc = vmap[vid].coldchain_state
            lots = () if cc is None else cc.cargo_manifest
            ratios = [1-l.quality_remaining/l.initial_quality for l in lots
                      if l.temp_class == orders[c].temp_class]
            slack = orders[c].tw_end-arrival
            pressure[c] = (orders[c].demand/(view.capacity-CAPACITY_MARGIN)+
                1/(1+max(0, slack)/max(orders[0].tw_end, 1e-9))+max(ratios, default=0))
            if i == 0:
                first.append(c)
    distance = sorted(pool, key=lambda c: (-detour[c], c))[:2]
    resource = sorted(pool, key=lambda c: (-pressure[c], c))[:2]
    frontier = []
    if first:
        a = min(first, key=lambda c: (-pressure[c], c))
        b = min((c for c in pool if c != a), key=lambda c: (e.dist_mat[0, a, c], c))
        frontier = [a, b]
    random = np.random.default_rng(seed).choice(pool, 2, replace=False).tolist()
    masks = []
    for pair in (distance, resource, frontier, random):
        m = tuple(sorted(pair))
        if m and m not in masks:
            masks.append(m)
    return tuple(masks)


def remove_mask(plans, mask):
    return {vid: replace(p, suffix=tuple(c for c in p.suffix if c not in mask))
            for vid, p in plans.items()}


def wait_or_return(view, vehicle):
    return () if vehicle.current_node != 0 and view.clock < view.booking_horizon-EVENT_TOL else (0,)


def _advance(cc, view, start, arrival, finish, order=None, distance=0.0, returning=False):
    if cc is None:
        cc = dispatch_vehicle(create_vehicle_state(view.contract), view.contract)
    cc, _ = transition_segment(cc, start, arrival, finish,
        None if order is None else order.oid, (True, True, True), view.contract,
        order_quantity=0.0 if order is None else order.demand,
        order_temp_class=None if order is None else order.temp_class,
        initial_quality=1.0 if order is None else order.initial_quality,
        return_to_depot=returning, segment_distance_units=distance)
    if not finite(cc) or not capacity_ok(view, cc.total_load):
        raise ValueError('reference finite/capacity failure')
    if cc.thermal_violation_count:
        raise ValueError('reference segment-endpoint temperature failure')
    return cc


def reference_check(view, plans, mask=()):
    """Replay known commitments through coldchain_state, never a learned gate.

    Segment endpoint thermal checking is not continuous/F-level certification.
    This completion is a known-work reference check, NOT future online utility.
    """
    try:
        validate_view(view)
        base = plans_from_view(view)
        mutable_customers = {c for vid, p in base.items() if vid in mutable_ids(view)
                             for c in p.suffix}
        if mask and (len(mask) != 2 or len(set(mask)) != 2 or not set(mask) <= mutable_customers):
            raise ValueError('mask outside legal mutable domain')
        fixed = set(base)-mutable_ids(view)
        ok, detail = validate_repair_plan(plans, base, mask, fixed)
        if not ok or not finite([asdict(p) for p in plans.values()]):
            raise ValueError('plan/anchor/prefix scope mismatch: %s' % detail)
        pending = [c for p in plans.values() for c in p.suffix]
        committed = [v.committed_next for v in view.vehicles if v.status == 'committed']
        if (len(pending+committed) != len(set(pending+committed)) or
                set(pending+committed) != set(view.accepted)-set(view.served)):
            raise ValueError('accepted-order exact-once failure')
        if not all(route_check(view, p) for p in plans.values()):
            raise ValueError('conservative route replay failure')
        e, orders, finals = public_env(view), {o.oid: o for o in view.orders}, []
        for v in view.vehicles:
            cc, cur, t = v.coldchain_state, v.current_node, view.clock
            if v.status == 'closed':
                if cc is not None:
                    finals.append(cc)
                continue
            if v.status == 'returning':
                if v.committed_finish > orders[0].tw_end+EVENT_TOL:
                    raise ValueError('committed return misses depot deadline')
                cc = _advance(cc, view, t, v.committed_finish, v.committed_finish,
                              distance=float(e.dist_mat[0, cur, 0]), returning=True)
                finals.append(cc)
                continue
            if v.status == 'committed':
                o = orders[v.committed_next]
                cc = _advance(cc, view, t, max(t, v.committed_arrive), v.committed_finish,
                              o, float(e.dist_mat[0, cur, o.oid]))
                cur, t = o.oid, v.committed_finish
            suffix = plans[v.vehicle_id].suffix
            for c in suffix:
                o = orders[c]
                distance = float(e.dist_mat[0, cur, c])
                arrival = t+distance/view.tw_speed
                finish = max(arrival, o.tw_start)+o.service
                cc = _advance(cc, view, t, arrival, finish, o, distance)
                cur, t = c, finish
            if cc is None:
                continue  # empty undispatched vehicle stays idle
            if cur != 0 and t < view.booking_horizon:
                cc = _advance(cc, view, t, view.booking_horizon, view.booking_horizon)
                t = view.booking_horizon
            arrival = t+float(e.dist_mat[0, cur, 0])/view.tw_speed
            if arrival > orders[0].tw_end+EVENT_TOL:
                raise ValueError('public WAIT completion misses depot deadline')
            cc = _advance(cc, view, t, arrival, arrival,
                          distance=float(e.dist_mat[0, cur, 0]), returning=True)
            finals.append(cc)
        energy = sum(c.cumulative_energy_kwh for c in finals)
        if energy > view.energy_budget:
            raise ValueError('reference energy budget failure')
        deliveries = [r for cc in finals for r in cc.delivered_to_depot]
        if (len(deliveries) != len({r.order_id for r in deliveries}) or
                {r.order_id for r in deliveries} != set(view.accepted)):
            raise ValueError('delivered-order ledger failure')
        if any(r.salable is False for r in deliveries):
            raise ValueError('reference unsalable cargo')
        for record in deliveries:
            o = orders[record.order_id]
            if record.temp_class != o.temp_class or abs(record.quantity-o.demand) > 1e-9:
                raise ValueError('delivery attributes do not match accepted order')
            quality = view.contract.quality
            if quality.threshold_type[record.temp_class] == 'absolute':
                if (record.salable is not True or
                        record.delivered_quality < quality.threshold[record.temp_class]):
                    raise ValueError('independent fish salability check failed')
        return {'ok': True, 'qualification': QUALIFICATION, 'energy_kwh': energy,
                'distance_km': sum(c.cumulative_distance_km for c in finals),
                'coverage': 'KNOWN_WORK_SEGMENT_ENDPOINTS_NOT_FULL_CERTIFIED'}
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {'ok': False, 'qualification': QUALIFICATION, 'reason': str(exc)}


def reconstruct(view, base, mask, backbone, deadline):
    """Existing M-pre logits + action mapping; independent conservative enumeration.

    stepwise_sample cannot accept a legal-action callback. Reuse its deterministic
    global argmax semantics here without monkeypatching its module-global enumerator.
    """
    pool = {c for p in base.values() if p.vehicle_id in mutable_ids(view) for c in p.suffix}
    if len(mask) != 2 or len(set(mask)) != 2 or not set(mask) <= pool:
        return None, 'ILLEGAL_MASK'
    partial, remaining = remove_mask(base, mask), list(mask)
    while remaining:
        if time.monotonic() >= deadline:
            return None, 'DECISION_TIMEOUT'
        actions = legal_actions(view, partial, remaining)
        if not actions:
            return None, 'NO_LEGAL_ACTION'
        scores = np.asarray(backbone.insertion_scores(view, base, partial, actions))
        if scores.shape != (len(actions),) or not np.isfinite(scores).all():
            return None, 'NAN'
        if time.monotonic() >= deadline:
            return None, 'DECISION_TIMEOUT'
        c, action, _ = actions[int(np.argmax(scores))]
        partial = apply_action(partial, action, allowed_vehicle_ids=mutable_ids(view))
        remaining.remove(c)
    return partial, None


def action_signature(view, plans):
    return tuple((v.vehicle_id, (('FROZEN', v.committed_next) if v.status not in ('idle', 'ready')
                  else (('SERVE', plans[v.vehicle_id].suffix[0]) if plans[v.vehicle_id].suffix
                        else ('WAIT' if wait_or_return(view, v) == () else 'RETURN', 0))))
                 for v in view.vehicles if v.status != 'closed')


def commit_view(view, plans, mask):
    check = reference_check(view, plans, mask)
    if not check['ok']:
        raise ValueError(check['reason'])
    allowed = mutable_ids(view)
    vehicles = tuple(replace(v, mutable_suffix=tuple(plans[v.vehicle_id].suffix))
                     if v.vehicle_id in allowed else v for v in view.vehicles)
    return replace(view, vehicles=vehicles)


def decide(view, model, deadline, seed=0):
    base, stamp = plans_from_view(view), identity(view)
    keep = reference_check(view, base)
    if not keep['ok']:
        raise ValueError('KEEP_INVALID: '+keep['reason'])
    masks = mask_pool(view, base, seed)
    trace = {'input_identity': stamp, 'qualification': QUALIFICATION, 'mask_pool': masks,
             'keep_plan': plan_hash(base), 'keep_action': action_signature(view, base),
             'head_status': 'SYNTHETIC_WIRING_NOT_TRAINED', 'candidates': []}
    if not masks:
        return view, {**trace, 'status': 'NO_MASK'}
    try:
        scores = np.asarray(model.score_masks(view, base, masks))
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return view, {**trace, 'status': 'EXCEPTION', 'reason': str(exc)}
    trace['scores'] = scores.tolist()
    if not np.isfinite(scores).all() or scores.shape != (len(masks),):
        return view, {**trace, 'status': 'NAN'}
    idx = int(np.argmax(scores))
    if time.monotonic() >= deadline:
        return view, {**trace, 'status': 'DECISION_TIMEOUT'}
    if scores[idx] <= 0.5:
        return view, {**trace, 'status': 'VALUE_KEEP'}
    mask = masks[idx]
    try:
        candidate, fail = reconstruct(view, base, mask, model.backbone, deadline)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return view, {**trace, 'status': 'RECONSTRUCT_FAIL', 'reason': str(exc)}
    if fail:
        return view, {**trace, 'selected_mask': mask, 'status': fail}
    check = reference_check(view, candidate, mask)
    trace['candidates'].append({'mask': mask, 'repair_plan': plan_hash(candidate),
        'reference_ok': check['ok'], 'accepted_plan': plan_hash(candidate) if check['ok'] else plan_hash(base),
        'predicted_next_action': action_signature(view, candidate if check['ok'] else base)})
    if not check['ok'] or time.monotonic() >= deadline:
        return view, {**trace, 'status': 'REFERENCE_REJECT' if not check['ok'] else 'DECISION_TIMEOUT'}
    return commit_view(view, candidate, mask), {**trace, 'status': 'COMMIT', 'selected_mask': mask}


def _worker(conn, checkpoint, synthetic_head_bias):
    try:
        from p2_pos01_mask_value import FrozenBackbone, MaskValueModel
        backbone = FrozenBackbone(checkpoint)
        model = MaskValueModel(backbone, synthetic_bias=synthetic_head_bias)
        conn.send(('READY', backbone.manifest))
        while True:
            message = conn.recv()
            if message[0] == 'CLOSE':
                return
            _, view, deadline, fault = message
            if fault == 'hang':
                time.sleep(3600)  # synthetic interruption counterexample only
            if fault is not None:
                raise ValueError('unknown fault')
            conn.send(('RESULT', decide(view, model, deadline)))
    except BaseException as exc:
        try:
            conn.send(('ERROR', type(exc).__name__+': '+str(exc)))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        conn.close()


class ReferenceSession:
    """Persistent isolated CPU repair process, never live engine/admission state.

    This deadline covers IPC and the PublicView repair decision. Admission and
    public projection happen outside this API; a full C+repair 10s contract is
    NOT established by the component timeout test.
    """
    def __init__(self, checkpoint, init_seconds=300, *, synthetic_head_bias=0.0):
        self.conn, child = mp.get_context('spawn').Pipe()
        self.process = mp.get_context('spawn').Process(target=_worker,
            args=(child, str(checkpoint), float(synthetic_head_bias)))
        self.process.start()
        child.close()
        if not self.conn.poll(init_seconds):
            self.close()
            raise RuntimeError('STOP: CPU checkpoint initialization timed out')
        status, result = self.conn.recv()
        if status != 'READY':
            self.close()
            raise RuntimeError('STOP: CPU checkpoint initialization failed: '+str(result))
        self.manifest = result

    def decide(self, view, seconds=10, *, synthetic_fault=None):
        if not self.process.is_alive():
            raise RuntimeError('stopped worker cannot be reused')
        # Reserve the last second for termination/receipt: work cannot write live state.
        if seconds <= 1:
            raise ValueError('decision slot must include one second termination reserve')
        t0 = time.monotonic()
        self.conn.send(('DECIDE', view, t0+seconds-1, synthetic_fault))
        if not self.conn.poll(max(0, seconds-1-(time.monotonic()-t0))):
            self.close()
            if time.monotonic()-t0 > seconds:
                raise RuntimeError('STOP: worker not reclaimed within decision slot')
            return view, {'status': 'DECISION_TIMEOUT', 'worker_alive': self.process.is_alive(),
                          'input_identity': identity(view), 'qualification': QUALIFICATION}
        status, result = self.conn.recv()
        if status != 'RESULT':
            self.close()
            raise RuntimeError('STOP: isolated worker failed: '+str(result))
        new_view, log = result
        if time.monotonic()-t0 > seconds:
            self.close()
            return view, {'status': 'DECISION_TIMEOUT', 'input_identity': identity(view),
                          'qualification': QUALIFICATION, 'worker_alive': False}
        if log['input_identity'] != identity(view):
            self.close()
            raise RuntimeError('STOP: decision identity mismatch')
        return new_view, log

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(0.4)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(0.4)
        self.conn.close()
        self.process.join(0)
        if self.process.is_alive():
            raise RuntimeError('STOP: worker survived termination')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
