"""P2-C-ANCHOR-01 synthetic transaction qualification (real engine, unmodified, NOT a source day).

Question this answers (registration requirement 4 of 5):
    A queue placed BEHIND a frozen in-transit anchor must be *consumed* by the real executor
    and must not be silently dropped by the next reveal or service completion.

`strict_online_env.py:528` clears `v.mutable_suffix` for every non-closed vehicle on each
reveal, and `:650-656` pops only ONE element per decision point. So a vehicle that is
mid-flight (status='committed') with a non-empty `mutable_suffix` loses that tail at the
next reveal, and nothing in the engine remembers it existed. A controller that computes a
legal suffix, validates it, and writes it ONCE has therefore not shown that the suffix
survives to execution.

This script drives the UNMODIFIED `StrictOnlineEnv` with a hand-built instance and four
controller arms that differ only in their persistence machinery / reveal timing:

  `maintain`                     private plan store + head reconciliation + tail re-install
                                 whenever the engine holds no suffix while work is owed
                                 -> the CAMR requirement (expected: every order exactly once)
  `write_on_reveal_only`         identical store, but the controller only writes when new
                                 information arrives and never repairs a tail the engine
                                 dropped in between; same reveal timing
                                 -> the static-write hazard (expected: tail lost)
  `write_on_reveal_only_late`    CONTROL: identical controller, later reveals moved outside
                                 the flight window so none lands mid-flight
                                 (expected: every order exactly once)
                                 -> isolates the mid-flight reveal as the CAUSE of the loss
  `maintain_no_reconcile`        re-installs the store on every decision point but never
                                 reconciles it against the executed prefix
                                 -> reconciliation is necessary, not optional
                                 (expected: duplicate serves)

Instance: 1 depot + 5 customers, 2 vehicles, all newly revealed work concentrated on one
vehicle so that it really carries a multi-order queue behind its committed anchor.
Early-reveal timing 0.10 / 0.15 / 0.30; late-reveal control 0.10 / 10.0 / 10.0.

Attribution of a loss: a decision point at which (i) a reveal has occurred since the
previous decision point (`reveal_idx` advanced), (ii) the vehicle is idle/ready, (iii) the
engine holds no suffix, and (iv) the store says the vehicle still owes orders. That is
observable without touching the engine. Separately, a `dropped` record captures the case
that matters most here: a legal, validated suffix computed at a reveal while the vehicle is
mid-flight, which the engine cannot install at all because it only consumes
`mutable_suffix` for idle/ready vehicles.

Nothing is modified; no source day is used; no label or training artefact is produced.

Usage: python synthetic_tx.py [--out synthetic_tx.json]      (refuses to overwrite)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
for _p in ('simulation', 'coldchain', 'evaluation', 'data'):
    sys.path.insert(0, str(ROOT / 'scripts' / _p))

from strict_online_env import StrictOnlineEnv, Replanner        # noqa: E402

LOG_CAP = 60          # keep the artefact readable; counters always cover the full run
# c1,c2 arrive at 0.10; c3 arrives at 0.15 -- i.e. while the vehicle is already in flight
EARLY = (1e6, 0.10, 0.10, 0.15)
# control: identical instance, but c3 arrives at 5.0 when the vehicle is parked and ready
LATE = (1e6, 0.10, 0.10, 5.0)
CUSTOMERS = [1, 2, 3]


def dataset(reveal_times):
    """1 depot + 3 customers; the early timing puts the third reveal inside the first leg."""
    n = len(reveal_times)
    return {
        'coords': np.array([[[0.500, 0.500],
                             [0.620, 0.500],
                             [0.440, 0.640],
                             [0.320, 0.560]]], dtype=np.float32),
        'demands': np.array([[0.0] + [10.0] * (n - 1)], dtype=np.float32),
        'tw_start': np.zeros((1, n), dtype=np.float32),
        'tw_end': np.full((1, n), 22.0, dtype=np.float32),
        'service_time': np.array([[0.0] + [0.05] * (n - 1)], dtype=np.float32),
        'reveal_time': np.array([list(reveal_times)], dtype=np.float32),
        'temp_class': np.zeros((1, n), dtype=np.int32),
    }


class ScriptedCAMR(Replanner):
    """CAMR-style controller: private plan store, head reconciliation, tail re-installation.

    `mode`:
      'maintain'              reconcile + re-install whenever the engine holds no tail
      'write_on_reveal_only'  only write when new information arrives; never repair a tail the
                              engine dropped between reveals
      'maintain_no_reconcile' re-install without ever removing served / in-flight orders
    """

    def __init__(self, mode):
        if mode not in ('maintain', 'write_on_reveal_only', 'maintain_no_reconcile'):
            raise ValueError(mode)
        self.mode = mode
        self.store = {}                 # vid -> desired queue behind the anchor
        self.written = set()            # vids that have received at least one install
        self.n_calls = 0
        self.history = []               # snapshot-hook records (per decision point)
        self.entry = []                 # per plan() call
        self.trims = []
        self.installs = []
        self.skipped = []               # static-write arm declined to repair a dropped tail
        self.dropped = []               # a known-good tail could not be installed mid-flight
        self.maintenance = []           # engine holds no tail while work is still owed

    # ---- snapshot hook (runs before _plan_and_commit) --------------------
    def on_decision(self, env, inst_idx, clock, event_id, reveal_idx, vehicles, traces,
                    served_mask, all_customers):
        # `reveal_idx` lags by one decision point (it is advanced at the top of the NEXT
        # iteration), so a reveal is detected from the visible-order set instead, which is
        # exact and independent of engine internals.
        visible = sorted(int(i) for i in all_customers
                         if float(env.reveal_time[inst_idx, i]) <= float(clock) + 1e-6)
        self.history.append({
            'clock': round(float(clock), 6), 'reveal_idx': int(reveal_idx),
            'visible': visible,
            'vehicles': {int(v.vehicle_id): {'status': v.status,
                                             'committed_next': None if v.committed_next is None
                                             else int(v.committed_next),
                                             'suffix': [int(x) for x in (v.mutable_suffix or ())],
                                             'served_route': [int(x) for x in v.served_route
                                                              if int(x) != 0]}
                         for v in vehicles}})

    def _reveal_since_prev(self):
        if len(self.history) < 2:
            return bool(self.history[-1]['visible']) if self.history else False
        return len(self.history[-1]['visible']) > len(self.history[-2]['visible'])

    def _rec(self, bucket, cap=LOG_CAP, **kw):
        if len(bucket) < cap:
            bucket.append(kw)

    # ---- planner ---------------------------------------------------------
    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        self.n_calls += 1
        clock = round(float(clock), 6)
        after_reveal = self._reveal_since_prev()
        if len(self.entry) < LOG_CAP:
            self.entry.append({
                'clock': clock, 'replan': sorted(int(x) for x in (replan_ids or ())),
                'reveal_since_prev': after_reveal,
                'vehicles': [{'vid': int(v.vehicle_id), 'status': v.status,
                              'committed_next': None if v.committed_next is None
                              else int(v.committed_next),
                              'served_route': [int(x) for x in v.served_route if int(x) != 0],
                              'suffix': [int(x) for x in (v.mutable_suffix or ())]}
                             for v in vehicles]})

        # (1) reconcile the store against what the engine already did
        if self.mode != 'maintain_no_reconcile':
            for v in vehicles:
                q = self.store.get(v.vehicle_id) or []
                if not q:
                    continue
                served = {int(x) for x in v.served_route if int(x) != 0}
                anchor = None if v.committed_next is None else int(v.committed_next)
                keep = [c for c in q if c not in served and c != anchor]
                if keep != q:
                    self.store[v.vehicle_id] = keep
                    self._rec(self.trims, clock=clock, vid=int(v.vehicle_id),
                              removed=[c for c in q if c not in keep], rest=list(keep))

        # (2) adopt newly revealed, not-yet-owned, not-yet-served orders.
        #     CONCENTRATED on the vehicle that already carries most work, so that vehicle
        #     really holds a multi-order queue behind its committed anchor.
        known = {c for q in self.store.values() for c in q}
        known |= {int(v.committed_next) for v in vehicles if v.committed_next not in (None, 0)}
        known |= {int(x) for v in vehicles for x in (v.mutable_suffix or ()) if int(x) != 0}
        new = [int(i) for i in sorted(visible_ids)
               if not served_mask[i] and int(i) not in known]
        if new:
            loads = {v.vehicle_id: len(self.store.get(v.vehicle_id) or []) for v in vehicles}
            target = max(loads, key=lambda k: (loads[k], -k))
            self.store.setdefault(target, []).extend(new)

        # (3) install the tail, or record why a legal tail could not take effect
        for v in vehicles:
            vid = int(v.vehicle_id)
            tail = list(self.store.get(vid) or ())
            if not tail:
                continue
            engine = list(v.mutable_suffix or ())
            if engine == tail:
                continue                       # engine already holds exactly this tail
            if not engine and v.status in ('idle', 'ready'):
                self._rec(self.maintenance, clock=clock, vid=vid, owed=list(tail),
                          after_reveal_since_prev=after_reveal,
                          already_written=vid in self.written)
            if v.status not in ('idle', 'ready'):
                # the engine only consumes mutable_suffix for idle/ready vehicles, so a legal
                # suffix computed at a reveal while the vehicle is already mid-flight cannot
                # be installed at all -- it has to be written again later
                self._rec(self.dropped, clock=clock, vid=vid, tail=list(tail),
                          status=v.status, after_reveal_since_prev=after_reveal)
                continue
            if self.mode == 'write_on_reveal_only' and not after_reveal:
                self._rec(self.skipped, clock=clock, vid=vid, tail=list(tail),
                          status=v.status, after_reveal_since_prev=after_reveal)
                continue
            v.mutable_suffix = list(tail)
            self.written.add(vid)
            self._rec(self.installs, clock=clock, vid=vid, installed=list(tail),
                      after_reveal_since_prev=after_reveal)
        return None


def run_arm(mode, reveal_times):
    env = StrictOnlineEnv(dataset(reveal_times), capacity=50.0, num_vehicles=2, tw_speed=1.5,
                          replanner=ScriptedCAMR(mode), coldchain_contract=None,
                          booking_horizon=None)
    rp = env.replanner
    env.snapshot_hook = rp.on_decision
    traces, _info = env.run(0)

    served = []
    for tr in traces:
        for r in getattr(tr, 'services', []):
            # ServiceRecord.picked_order_id is only populated when a cold-chain contract is
            # installed; `node` is always the served customer == order id in pickup-to-depot.
            served.append({'vid': int(tr.vehicle_id), 'order': int(r.node),
                           'arrival': round(float(r.arrival_time), 6)})
    served.sort(key=lambda d: (d['arrival'], d['vid'], d['order']))
    seq = [d['order'] for d in served]
    counts = {}
    for c in seq:
        counts[c] = counts.get(c, 0) + 1
    by_vehicle = {}
    for d in served:
        by_vehicle.setdefault(d['vid'], []).append(d['order'])

    post_reveal_losses = [m for m in rp.maintenance
                          if m['after_reveal_since_prev'] and m['already_written']]
    return {
        'arm': mode,
        'reveal_times': [float(x) for x in reveal_times],
        'served_sequence': seq,
        'served_by_vehicle': by_vehicle,
        'served_set': sorted(counts),
        'n_served_distinct': len(counts),
        'n_served_records': len(seq),
        'duplicate_serves': sorted(c for c, n in counts.items() if n > 1),
        'missing_orders': sorted(set(CUSTOMERS) - set(counts)),
        'exact_once': sorted(counts) == CUSTOMERS and all(n == 1 for n in counts.values()),
        'n_decision_points_with_replanner': rp.n_calls,
        'n_installations': len(rp.installs),
        'n_skipped_writes': len(rp.skipped),
        'n_dropped_writes': len(rp.dropped),
        'n_head_trims': len(rp.trims),
        'n_maintenance_points': len(rp.maintenance),
        'n_maintenance_points_after_reveal': len(post_reveal_losses),
        'maintenance_log': rp.maintenance,
        'install_log': rp.installs,
        'skipped_write_log': rp.skipped,
        'dropped_write_log': rp.dropped,
        'trim_log': rp.trims,
        'entry_log': rp.entry,
        'log_cap': LOG_CAP,
    }


ARMS = (
    ('maintain', 'maintain', EARLY),
    ('write_on_reveal_only', 'write_on_reveal_only', EARLY),
    ('write_on_reveal_only_late', 'write_on_reveal_only', LATE),
    ('maintain_no_reconcile', 'maintain_no_reconcile', EARLY),
)

EXPECTED = {
    'maintain': 'every order served exactly once, with at least one re-installation that was '
                'required because a reveal had wiped the tail',
    'write_on_reveal_only': 'a legal suffix validated at a reveal while the vehicle is '
                            'mid-flight cannot be installed at all, and the tail the engine '
                            'dropped afterwards is never repaired, so orders are never served',
    'write_on_reveal_only_late': 'CONTROL: with no reveal landing mid-flight the same '
                                 'write-on-reveal-only controller serves every order, so the '
                                 'loss above is caused by the mid-flight reveal and not by the '
                                 'controller',
    'maintain_no_reconcile': 're-installation without head reconciliation duplicates orders: '
                             'reconciliation against the executed prefix is necessary',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'synthetic_tx.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    arms = {name: run_arm(mode, rev) for name, mode, rev in ARMS}
    maintain = arms['maintain']
    static = arms['write_on_reveal_only']
    late = arms['write_on_reveal_only_late']
    static_missing = set(static['missing_orders'])

    def _dropped_orders(arm):
        return {c for d in arm['dropped_write_log'] for c in d['tail']}

    checks = {
        # 1. with maintenance the queue behind the anchor IS consumed, exactly once
        'maintain_is_exact_once': maintain['exact_once'],
        # 2. a mid-flight reveal really did destroy a pending, known-good tail
        'mid_flight_reveal_destroyed_a_pending_tail':
            maintain['n_dropped_writes'] > 0 and static['n_dropped_writes'] > 0,
        # 3. the destroyed tail only reached execution because it was written again later
        'orders_were_recovered_by_reinstall':
            bool(_dropped_orders(maintain)) and maintain['n_served_distinct'] == len(CUSTOMERS),
        # 4. without maintenance the destroyed tail never reaches execution
        'no_maintenance_loses_the_tail': not static['exact_once'],
        'lost_orders_were_in_the_destroyed_tail': static_missing <= _dropped_orders(static),
        # 5. the loss is caused by the mid-flight reveal, not by the controller
        'late_reveal_control_is_exact_once': late['exact_once'],
        'late_reveal_control_drops_no_tail': late['n_dropped_writes'] == 0,
        # 6. re-installation without reconciliation corrupts exact-once
        'no_reconcile_duplicates': arms['maintain_no_reconcile']['duplicate_serves'] != [],
    }
    payload = {
        'schema': 'p2_c_anchor_01_synthetic_tx_v5',
        'engine_modified': False,
        'source_day_used': False,
        'instance': ('hand-built: 1 depot + 5 customers / 2 vehicles; new work is concentrated '
                     'on one vehicle so it carries a multi-order queue behind its anchor'),
        'reveal_timing': {'early': list(EARLY), 'late_control': list(LATE)},
        'mechanism_under_test': (
            'strict_online_env.py:528 clears v.mutable_suffix on every reveal and :650-656 '
            'pops one element per decision point, so a mid-flight vehicle loses its queued '
            'tail unless the controller holds it and re-installs it'),
        'reading': (
            'This qualifies PERSISTENCE MACHINERY, not policy quality. Restoring the stored '
            'tail verbatim is the simplest possible maintenance rule; a real CAMR controller '
            'would recompute the suffix after each reveal. The result does not say the '
            'restored order is good, and it says nothing about energy, quality, or a '
            '289-phase source day.'),
        'expected': EXPECTED,
        'checks': checks,
        'all_checks_pass': all(checks.values()),
        'arms': arms,
    }
    a.out.open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n')

    for name, _mode, _rev in ARMS:
        r = arms[name]
        print('%-26s served=%s dups=%s missing=%s exact_once=%s installs=%d skipped=%d '
              'dropped=%d maintenance=%d(post_reveal=%d)'
              % (name, r['served_set'], r['duplicate_serves'], r['missing_orders'],
                 r['exact_once'], r['n_installations'], r['n_skipped_writes'],
                 r['n_dropped_writes'], r['n_maintenance_points'],
                 r['n_maintenance_points_after_reveal']))
    print('checks', json.dumps(checks, sort_keys=True))
    print('all_checks_pass', payload['all_checks_pass'])


if __name__ == '__main__':
    main()
