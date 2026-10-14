"""Engineering-only observer for the existing pickup evaluator; no model training.

Caller supplies dataset/sampler. This module cannot create a formal gate and does
not change the registered driver. Full private state is written only to audit.
"""
from dataclasses import is_dataclass
from pathlib import Path
import sys
import time
import hashlib
from audit_io import Journal

SCRIPTS = Path(__file__).resolve().parents[1]


def source_identity():
    paths={Path(mod.__file__).resolve() for mod in list(sys.modules.values())
           if getattr(mod,'__file__',None) and str(mod.__file__).endswith('.py')}
    paths={p for p in paths if p.is_relative_to(SCRIPTS)}
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run_observed(*, dataset, contract, budget, capacity, vehicles, cooling_share,
                 sampler, penalties, k, time_limit, arm_seed, output):
    for folder in ('evaluation', 'simulation', 'coldchain', 'models'):
        sys.path.insert(0, str(SCRIPTS / folder))
    from scenario_saa import SaaReplanner
    from strict_online_env import StrictOnlineEnv
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from run_exp_reserve import BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
    target=Path(output).resolve()
    workspace=SCRIPTS.parent.parent
    if target.is_relative_to(workspace) and not target.is_relative_to(
            SCRIPTS.parent/'results/three_line_corrections_20261009/engineering'):
        raise ValueError('Engineering traces must use the isolated engineering result directory')
    journal = Journal(output)
    identity=source_identity()

    class ObservedSaa(SaaReplanner):
        def _saa_decide(self, env, inst_idx, clock, fleet, served, order, scenarios, t0):
            before = self.timeouts
            try:
                return super()._saa_decide(env, inst_idx, clock, fleet, served, order, scenarios, t0)
            finally:
                # Capture after commit so serialization never feeds the actor.
                journal.write('decision', day_id=int(inst_idx), clock=float(clock),
                    order=int(order), deadline_seconds=float(self.time_limit),
                    decision_wall_seconds=time.perf_counter() - t0,
                    timeout_delta=int(self.timeouts - before),
                    accepted=order in self._accepted, rejected=order in self._rejected,
                    plan=self._plan, sampled_scenarios=scenarios)

    rp = ObservedSaa(budget=budget, capacity=capacity, booking_horizon=BOOKING_HORIZON,
        contract=contract, cooling_share=cooling_share, sampler=sampler,
        reject_penalty=penalties, K=k, time_limit=time_limit, arm_seed=arm_seed,
        shadow_mode='greedy', energy_pricing='marginal', future_policy='density', standby_orders=0)
    env = StrictOnlineEnv(dataset, capacity, SPEED_KMH / KM_PER_UNIT, vehicles, rp,
                          contract, BOOKING_HORIZON)

    def snapshot(e, i, clock, event_id, reveal_idx, fleet, traces, served, all_customers):
        journal.write('private_snapshot', day_id=int(i), clock=float(clock),
            event_id=int(event_id), reveal_index=int(reveal_idx), fleet=fleet,
            traces=traces, served=served)

    env.snapshot_hook = snapshot
    journal.write('run_start', qualification='engineering_observation_only',
        formal_eligible=False, seed=arm_seed, contract=contract, dataset=dataset,
        budget=budget, penalties=penalties, k=k, decision_limit=time_limit,
        timing_scope='original soft submission deadline; observer IO measured separately')
    journal.write('source_start', sha256=identity)
    rows = []
    try:
        for i in range(len(dataset['coords'])):
            journal.write('day_start', day_id=i)
            try:
                traces, served = env.run(i)
                row = evaluate_trace_a1(traces, {key: value[i] for key, value in dataset.items()},
                    contract, set(rp._accepted), set(rp._rejected), budget, penalties)
                journal.write('day_end', day_id=i, traces=traces, served=served,
                    accepted=set(rp._accepted), rejected=set(rp._rejected), evaluation=row,
                    timeouts=rp.timeouts)
                rows.append(row)
            except Exception as exc:
                journal.write('day_failure', day_id=i, error=repr(exc))
                raise
        end={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in identity}
        if identity != end:
            raise ValueError('Runtime source changed during engineering observation')
        journal.write('source_end',sha256=end)
        journal.write('run_end', days=len(rows), observer_io_seconds=journal.io_seconds,
                      formal_eligible=False)
    finally:
        journal.close()
    return rows


def replay(path):
    """Independently rebuild typed traces and recompute the existing evaluator."""
    import json
    from audit_io import decode, verify
    for folder in ('evaluation', 'simulation', 'coldchain'):
        sys.path.insert(0, str(SCRIPTS / folder))
    import coldchain_state, coldchain_contract, strict_online_env
    from coldchain_evaluator_a1 import evaluate_trace_a1
    classes = {cls.__module__ + '.' + cls.__name__: cls
        for mod in (coldchain_state, coldchain_contract, strict_online_env)
        for cls in vars(mod).values() if isinstance(cls, type) and is_dataclass(cls)}
    info = verify(path)
    if not info['complete']:
        raise ValueError('Incomplete journal cannot qualify for replay')
    checked = []
    start = None
    source_start = None
    source_end = None
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'run_start':
                start = decode(row['payload'], classes)
            elif row['kind'] == 'source_start':
                source_start = decode(row['payload'])['sha256']
            elif row['kind'] == 'source_end':
                source_end = decode(row['payload'])['sha256']
            elif row['kind'] == 'day_end':
                p = decode(row['payload'], classes)
                ds = {key: val[p['day_id']] for key, val in start['dataset'].items()}
                result = evaluate_trace_a1(p['traces'], ds, start['contract'],
                    p['accepted'], p['rejected'], start['budget'], start['penalties'])
                from audit_io import encode
                if encode(result) != encode(p['evaluation']):
                    raise ValueError('Replayed evaluation differs')
                checked.append(p['day_id'])
    if checked != list(range(len(start['dataset']['coords']))):
        raise ValueError('Missing day in complete journal')
    if not source_start or source_start != source_end:
        raise ValueError('Missing or unequal source identity')
    if any(not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest
           for path,digest in source_start.items()):
        raise ValueError('Replay source differs from recorded runtime')
    return {'days_replayed': checked, 'all_evaluations_equal': True, 'formal_eligible': False}
