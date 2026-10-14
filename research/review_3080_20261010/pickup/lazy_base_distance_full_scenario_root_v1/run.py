"""Pickup: full fixed archived scenario-8 sequence, paired reference vs lazy base reuse.

Arms (same fixed sealed inputs, same K10 scenarios, same seed):
  REF  : the unmodified original serial insertion (EnergyCompleteSAA._insert_u)
  CAND : the registered lazy same-loop base-distance reuse (LazyBaseDistanceSAA)

Both arms are executed interleaved (REF,CAND,REF,CAND) after one untimed warm-up
of each, and only the insertion + local-search + completion section is timed.
Exactness gates compare CAND against REF on the same fixed branch.
"""
import json
import os
import pickle
import copy
import sys
import time
import hashlib
import ctypes
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

import qualify_pickup_joint_policy_transaction_root_v3 as old
import engineering_stage_closure_v1 as seal
import numpy as np
import psutil
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot
from scenario_saa import UncondHistoricalSampler
import scenario_saa
from energy_complete_saa import EnergyCompleteSAA
from lazy_base_distance import LazyBaseDistanceSAA

BASE = HERE.parent / 'independent_rolling_incremental_engineering_root_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(n, x):
    seal.save(HERE / n, x)


def brief(states):
    return {str(v): dict(route=[int(o) for o in s['route']], cur=int(s['cur']),
                         cur_time=float(s['cur_time']).hex(), load=float(s['load']).hex(),
                         frozen=int(s.get('frozen', 0))) for v, s in states.items()}


def make_observer(cls):
    class Observer(cls):
        def _route_dist_u(self, space, cur, route):
            value = super()._route_dist_u(space, cur, route)
            self.distance_log.append(dict(operation=self.operation, cur=int(cur),
                                          route=[int(o) for o in route], value_hex=float(value).hex(),
                                          type=type(value).__module__ + '.' + type(value).__qualname__))
            return value

        def _route_feasible_u(self, space, env, cur, cur_time, load, route, deadline):
            value = super()._route_feasible_u(space, env, cur, cur_time, load, route, deadline)
            self.feasible_log.append(dict(operation=self.operation, cur=int(cur),
                                          cur_time_hex=float(cur_time).hex(), load_hex=float(load).hex(),
                                          route=[int(o) for o in route], deadline_hex=float(deadline).hex(),
                                          feasible=bool(value)))
            return value
    return Observer


def main():
    seal.initialize()
    ctypes.WinDLL('kernel.appcore.dll')
    encode, decode = old.codecs()
    old.pf.digest({'prime': 'digest dependencies'})
    np.savez(io.BytesIO(), prime=np.array([0]))
    if np.__version__ != '1.26.4' or sys.version_info[:3] != (3, 12, 7):
        raise ValueError('Registered runtime mismatch')
    if psutil.virtual_memory().available < .5 * 2 ** 30:
        raise ValueError('No-model resource gate failed before stage')
    reg = read(HERE / 'REGISTRATION.json')
    ctx = decode(read(BASE / 'SIMULATION_CONTEXT.json'))
    p = ctx['packet']
    initial_param_digest = old.pf.digest(p)
    prior = p['snapshot']['replanner_state']
    clock = p['snapshot']['clock']
    fleet, _, served = restore_recourse_snapshot(p['snapshot'])
    with old.HISTORY.open('rb') as f:
        history = pickle.load(f)
    row = next(r for r in ctx['candidates']['reject'] if r['index'] == ctx['selected']['reject'])

    def setup(cls):
        actor = make_observer(cls)(
            budget=p['budget'], capacity=p['capacity'], booking_horizon=p['booking_horizon'],
            contract=p['contract'], cooling_share=0., hist=history['history_times_classes'], k=10,
            sampler=UncondHistoricalSampler(history['history']), reject_penalty=p['penalties'], K=10,
            time_limit=10., arm_seed=7001, anytime_vote=False, shadow_mode='ls',
            energy_pricing='marginal', future_policy='reveal', standby_orders=0, incr_eval=False)
        actor.restore_state(copy.deepcopy(prior))
        env = StrictOnlineEnv(p['dataset'], p['capacity'], p['tw_speed'], p['snapshot']['num_vehicles'],
                              actor, p['contract'], p['booking_horizon'])
        actor._sync_id_map(env, 0, clock)
        observed = actor._build_snapshot(env, 0, clock, fleet, served)
        scenarios = actor.sampler.sample(observed, np.random.default_rng(202610130901), 10)
        space, orders = actor._build_space(env, 0, scenarios)
        lookup = {id(x): space.n_real + j for j, x in enumerate(orders)}
        ids = [[lookup[id(x)] for x in s] for s in scenarios]
        actor.restore_state(copy.deepcopy(row['packet']['snapshot']['replanner_state']))
        states = actor._sim_states(env, 0, clock, fleet, served, {int(v): r for v, r in row['plan'].items()})
        space.D = np.array(space.D, copy=True, order='C')
        space.D.flags.writeable = False
        return actor, env, space, states, ids, observed, scenarios

    def search(actor, space, env, states, sequence, deadline):
        actor.base_computations = 0
        actor.base_reuses = 0
        actor.score_log = []
        actor.distance_log = []
        actor.feasible_log = []
        transitions = []
        original_transition = scenario_saa.transition_segment

        def physical(*args, **kwargs):
            out = original_transition(*args, **kwargs)
            transitions.append(encode(out))
            return out

        scenario_saa.transition_segment = physical
        events = []
        try:
            started = time.perf_counter()
            for j, o in enumerate(sequence):
                actor.operation = 'insert_' + str(j)
                before = brief(states)
                ok = actor._insert_u(space, env, o, states, deadline)
                events.append(dict(kind='insert', order=int(o), accepted=bool(ok),
                                   before=before, after=brief(states)))
            actor.operation = 'local_search_4'
            before = brief(states)
            actor._local_search_u(space, env, states, deadline, max_passes=4)
            events.append(dict(kind='local_search', max_passes=4, before=before, after=brief(states)))
            actor.operation = 'physical_completion'
            energy, distance = actor._complete_u(space, env, 0, clock, states, actor.booking_horizon)
            seconds = time.perf_counter() - started
        finally:
            scenario_saa.transition_segment = original_transition
        return dict(seconds=seconds, events=events, transitions=transitions,
                    distance_log=actor.distance_log, feasible_log=actor.feasible_log,
                    score_log=actor.score_log, base_computations=actor.base_computations,
                    base_reuses=actor.base_reuses,
                    terminal_energy_hex=None if energy is None else float(energy).hex(),
                    terminal_distance_hex=None if distance is None else float(distance).hex())

    # ---- fixed inputs, once ----------------------------------------------------
    ref_actor, ref_env, ref_space, ref_states, ids, observed, scenarios = setup(EnergyCompleteSAA)
    full = sorted(ids[reg['scenario_index']], key=lambda o: ref_space.reveal[o])
    sequence = list(full)
    reveals = [float(ref_space.reveal[o]).hex() for o in full]
    matrix_sha = hashlib.sha256(ref_space.D.tobytes()).hexdigest()
    deadline = float(ref_env.tw_end[0, 0])
    scenario_digest = old.pf.digest(observed)

    extra = tuple(Path(x) for x in reg['sources'])
    start_closure = seal.closure(extra)
    save('SOURCE_START.json', start_closure)

    checks = []

    def ck(n, v):
        checks.append(dict(name=n, pass_=bool(v)))

    runs = {}
    order = []
    # one untimed warm-up per arm, then interleaved timed runs
    for label, cls in (('REF', EnergyCompleteSAA), ('CAND', LazyBaseDistanceSAA)):
        a, e, s, st, idl, obs, scn = setup(cls)
        ck('deterministic_scenarios_and_ids:' + label,
           idl == ids and old.pf.digest(obs) == scenario_digest and idl == ctx['indices']
           and old.pf.digest(scn) == ctx['scenario_sha256'])
        ck('deterministic_sequence:' + label,
           sorted(idl[reg['scenario_index']], key=lambda o: s.reveal[o]) == sequence)
        search(a, s, e, st, sequence, deadline)
    for rep in range(reg['timed_repeats']):
        for label, cls in (('REF', EnergyCompleteSAA), ('CAND', LazyBaseDistanceSAA)):
            a, e, s, st, idl, obs, scn = setup(cls)
            r = search(a, s, e, st, sequence, deadline)
            r['repeat'] = rep
            runs.setdefault(label, []).append(r)
            order.append(label)
            ck('input_matrix_unchanged_after_run:%s:%d' % (label, rep),
               not s.D.flags.writeable and hashlib.sha256(s.D.tobytes()).hexdigest() == matrix_sha)

    ref = runs['REF'][0]
    cand = runs['CAND'][0]

    # ---- exactness gates: CAND must reproduce REF on the same branch -----------
    ck('identical_insertion_choices', ref['events'] == cand['events'])
    ck('identical_feasibility_calls_and_values', ref['feasible_log'] == cand['feasible_log'])
    ck('identical_scored_candidates', ref['score_log'] == cand['score_log'])
    ck('identical_c0_physical_transitions', ref['transitions'] == cand['transitions'])
    ck('identical_terminal_energy', ref['terminal_energy_hex'] == cand['terminal_energy_hex'])
    ck('identical_terminal_distance', ref['terminal_distance_hex'] == cand['terminal_distance_hex'])

    # the candidate log must be exactly the reference log with only repeated
    # same-loop base evaluations deleted, in the original order
    paired = []
    for r in ref['distance_log']:
        pairs = paired.setdefault(r['operation'], [])
        pairs.append(r)
    is_base = {}
    for op, entries in paired.items():
        for n, entry in enumerate(entries):
            is_base[id(entry)] = bool(n % 2)
    deleted, j = [], 0
    sub_ok = True
    for entry in ref['distance_log']:
        if j < len(cand['distance_log']) and cand['distance_log'][j] == entry:
            j += 1
        else:
            deleted.append(entry)
    sub_ok = j == len(cand['distance_log'])
    skipped = len([e for e in deleted if is_base.get(id(e))])
    computed = len([e for e in cand['distance_log'] if not is_base.get(id(e))])
    ck('candidate_log_is_reference_log_with_deletions_in_order', sub_ok)
    ck('every_deleted_entry_is_a_base_call', len(deleted) == skipped)
    ck('candidate_saving_count_matches_log_difference',
       len(ref['distance_log']) - len(cand['distance_log']) == skipped)
    ck('candidate_base_computations_and_reuses',
       cand['base_reuses'] == skipped and cand['base_computations'] > 0)

    # cross-repeat determinism of each arm, by digest, plus full reference logs
    def dg(x):
        return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    run_digests = {label: [dict(repeat=r['repeat'], seconds=r['seconds'],
                                terminal_energy_hex=r['terminal_energy_hex'],
                                terminal_distance_hex=r['terminal_distance_hex'],
                                events=dg(r['events']), feasible=dg(r['feasible_log']),
                                distance=dg(r['distance_log']), score=dg(r['score_log']))
                           for r in runs[label]] for label in ('REF', 'CAND')}
    for label in ('REF', 'CAND'):
        ck('cross_repeat_identical_events:' + label,
           len({d['events'] for d in run_digests[label]}) == 1)
        ck('cross_repeat_identical_distance_log:' + label,
           len({d['distance'] for d in run_digests[label]}) == 1)
        ck('cross_repeat_identical_terminal:' + label,
           len({(d['terminal_energy_hex'], d['terminal_distance_hex']) for d in run_digests[label]}) == 1)

    ck('full_scenario8_reveal_sorted_sequence',
       sequence == sorted(ids[reg['scenario_index']], key=lambda o: ref_space.reveal[o])
       and len(sequence) == len(ids[reg['scenario_index']]))
    ck('full_common_scenarios_and_ids_original',
       ids == ctx['indices'] and old.pf.digest(scenarios) == ctx['scenario_sha256'])
    ck('original_incr_eval_false', not ref_actor.incr_eval)
    ck('all_original_public_parameters_unchanged', old.pf.digest(p) == initial_param_digest)
    ck('terminal_present', ref['terminal_energy_hex'] is not None and ref['terminal_distance_hex'] is not None)

    np.save(HERE / 'DISTANCE_MATRIX.npy', ref_space.D, allow_pickle=False)
    end_closure = seal.closure(extra)
    save('SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))

    result = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks), checks=checks,
                  sequence=[int(o) for o in sequence], scenario8_ids=[int(o) for o in full],
                  scenario8_reveals=reveals, inserted=len(sequence),
                  accepted=int(sum(1 for e in ref['events'][:-1] if e['accepted'])),
                  run_order=order, timed_repeats=reg['timed_repeats'], runs=run_digests,
                  reference_seconds=[r['seconds'] for r in runs['REF']],
                  candidate_seconds=[r['seconds'] for r in runs['CAND']],
                  reference_distance_calls=len(ref['distance_log']),
                  candidate_distance_calls=len(cand['distance_log']),
                  proved_removed_base_calls=skipped,
                  distance_log=ref['distance_log'], candidate_distance_log=cand['distance_log'],
                  events=ref['events'], feasible_log=ref['feasible_log'], score_log=ref['score_log'],
                  cold_transition_outputs=ref['transitions'],
                  terminal_energy_hex=ref['terminal_energy_hex'],
                  terminal_distance_hex=ref['terminal_distance_hex'],
                  immutable_matrix_sha256=matrix_sha, original_parameter_digest=initial_param_digest,
                  scenario_digest=scenario_digest,
                  source_actual_files=len(start_closure['files']), source_actual_maps=len(start_closure['modules']),
                  full_request_budget_qualified=False, formal_strong_dynamic_qualified=False,
                  scientific_data_generated=False, stable_online_gain=False)
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('checks', 'events', 'score_log', 'feasible_log',
                                   'cold_transition_outputs', 'scenario8_reveals', 'runs')}))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
