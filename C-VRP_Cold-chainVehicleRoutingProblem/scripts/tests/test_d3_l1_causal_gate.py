#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1-D3-L1 CAUSAL GATE (user-mandated, pre-collection).

Answers three questions with code-grounded evidence, no rollout of the collection:
  1. Does the JF1-H-F baseline's WAIT/CLOSE decision read real future reveal information?
     -> YES when `booking_horizon` is None (the constructor default), because
        `StrictOnlineEnv.has_future_reveal` (strict_online_env.py:248-258) then scans the real
        `reveal_time`, and `jf1h_repair.py:194` consumes it.  With a public `booking_horizon` set
        the method returns `clock < booking_horizon` and never touches `reveal_time`.
  2. Is the answer to (1) invariant under perturbation of the UNREVEALED future while the
     revealed prefix is held fixed?
     -> with `booking_horizon=None` it is NOT (the answer flips when an invisible order is
        removed); with a horizon it IS.
  3. Are the R1 event slots well defined?  -> NO for `event_id` (it counts physical events:
     service completions and returns too, strict_online_env.py:488-511), YES for `reveal_idx`
     (reveal events are `sorted(set(reveal_time))`, so simultaneous reveals are ONE event).

Also measures, read-only, the dev_check instance/reveal census needed to size R1 x K=4 exactly.

This is a gate, not a collection: nothing is captured, no snapshot is written, no model is used.
Exit 0 = gate rows all PASS.  A FAIL is a finding, never to be explained away.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(HERE))

import numpy as np                                                # noqa: E402
from strict_online_env import StrictOnlineEnv                     # noqa: E402

CASES = []
RECORDS = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def rec(key, value):
    RECORDS[key] = value


class _StandIn:
    """Carries only the attributes `has_future_reveal` reads, so the REAL method is exercised
    without constructing an environment (no dataset, no rollout)."""

    def __init__(self, reveal_time, num_nodes, horizon):
        # the real method indexes reveal_time[inst_idx, i]; keep that 2-D layout
        self.reveal_time = np.asarray(reveal_time, dtype=float).reshape(1, -1)
        self.num_nodes = int(num_nodes)
        self.booking_horizon = horizon


def hfr(reveal_time, num_nodes, horizon, clock, served_mask):
    return bool(StrictOnlineEnv.has_future_reveal(
        _StandIn(reveal_time, num_nodes, horizon), 0, float(clock),
        np.asarray(served_mask, dtype=bool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(EXT / 'data/m0_scale/'
                                          'dcc_50_r1_edod05_dev_check_teacher.npz'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    src_env = (EXT / 'scripts/simulation/strict_online_env.py').read_text(encoding='utf-8')
    src_jf = (EXT / 'scripts/simulation/jf1h_repair.py').read_text(encoding='utf-8')

    # ---- 1. source evidence -------------------------------------------------------------
    rec('source_evidence', {
        'has_future_reveal_default_branch':
            'if self.booking_horizon is not None: return clock < self.booking_horizon - 1e-6'
            in src_env.replace('\n', ' ').replace('    ', ' ').replace('  ', ' ')
            or 'self.booking_horizon is not None' in src_env,
        'reads_real_reveal_time_when_horizon_is_none':
            'self.reveal_time[inst_idx, i] > clock + 1e-6' in src_env,
        'constructor_default_none': 'booking_horizon=None' in src_env,
        'jf1h_repair_calls_it': 'env.has_future_reveal(inst_idx, clock, served_mask)' in src_jf,
        'jf1h_repair_call_line': [i for i, l in
                                  enumerate(src_jf.splitlines(), 1)
                                  if 'has_future_reveal' in l],
        'default_docstring_says_non_causal':
            '旧行为（读真实 reveal_time，仅用于负结果比较，非因果主张）' in src_env})
    ck('G1 JF1-H-F consumes has_future_reveal and the constructor default is booking_horizon=None',
       'booking_horizon=None' in src_env
       and 'env.has_future_reveal(inst_idx, clock, served_mask)' in src_jf,
       RECORDS['source_evidence'])

    # ---- 2. behavioural: leak with None, invariance with a horizon ----------------------
    base = [0.0, 5.0, 30.0]          # node 2 reveals at 30 -> invisible at clock 5
    served = [True, True, False]
    removed = [0.0, 5.0, 0.0]        # node 2 removed from the universe (num_nodes=2 below)
    none_orig = hfr(base, 3, None, 5.0, served)
    none_pert = hfr(removed, 2, None, 5.0, served[:2])
    hz_orig = hfr(base, 3, 16.0, 5.0, served)
    hz_pert = hfr(removed, 2, 16.0, 5.0, served[:2])
    timing_pert = hfr([0.0, 5.0, 300.0], 3, None, 5.0, served)       # timing-only perturbation
    rec('perturbation', {'clock': 5.0,
                         'horizon_None': {'original': none_orig, 'future_removed': none_pert,
                                          'flips': none_orig != none_pert,
                                          'timing_only_perturbation': timing_pert},
                         'horizon_16': {'original': hz_orig, 'future_removed': hz_pert,
                                        'flips': hz_orig != hz_pert},
                         'perturbation_definition': 'remove an order that is INVISIBLE at the '
                         'decision clock while the revealed prefix is bit-identical'})
    ck('G2 with booking_horizon=None the WAIT/CLOSE answer FLIPS when the unrevealed future is '
       'perturbed while the revealed prefix is fixed (the leak the user identified)',
       none_orig is True and none_pert is False,
       RECORDS['perturbation']['horizon_None'])
    ck('G3 with booking_horizon=16.0 the answer is INVARIANT under the same perturbation '
       '(decision-visible future information eliminated by construction)',
       hz_orig is True and hz_pert is True,
       RECORDS['perturbation']['horizon_16'])
    ck('G3b timing-only perturbations do NOT expose the leak at horizon=None (it is sensitivity '
       'to the EXISTENCE of a future reveal, not to its timing), so the horizon is what removes it',
       none_orig == timing_pert is True,
       {'none_orig': none_orig, 'timing_only': timing_pert})
    edges = {str(c): hfr([0.0, 5.0, 30.0], 3, 16.0, c, served) for c in (0.0, 15.9, 16.0, 20.0)}
    rec('horizon_edges', edges)
    ck('G4 the horizon branch is a clean public predicate: True before the booking deadline and '
       'False at/after it',
       edges == {'0.0': True, '15.9': True, '16.0': False, '20.0': False}, edges)

    # ---- 3. R1 event identity ------------------------------------------------------------
    rec('r1_event_identity', {
        'event_id_is_a_PHYSICAL_event_counter':
            'self.event_id += 1' in src_env and
            '下一个物理事件 = min(下一 reveal, 各 committed finish, 各 return finish)' in src_env,
        'reveal_events_deduplicate_simultaneous_reveals':
            'reveal_events = sorted(set(float(self.reveal_time[inst_idx, i])' in src_env,
        'reveal_idx_is_the_reveal_counter':
            'reveal_events[reveal_idx]' in src_env,
        'snapshot_hook_receives_both':
            'int(self.event_id), int(reveal_idx)' in src_env,
        'conclusion': 'R1 must select on reveal_idx (order-reveal events), NEVER on event_id '
                      '(which also counts service completions, returns and the booking end)'})
    ck('G5 R1 slots must be keyed on reveal_idx, not event_id: event_id counts physical events '
       '(service completion / return / booking end) and reveal_events is a de-duplicated set',
       RECORDS['r1_event_identity']['event_id_is_a_PHYSICAL_event_counter']
       and RECORDS['r1_event_identity']['reveal_events_deduplicate_simultaneous_reveals']
       and RECORDS['r1_event_identity']['reveal_idx_is_the_reveal_counter'],
       {k: v for k, v in RECORDS['r1_event_identity'].items() if k != 'conclusion'})

    # ---- 4. dev_check census for R1 x K=4 ------------------------------------------------
    d = np.load(args.data)
    keys = sorted(d.files)
    rt = np.asarray(d['reveal_time'], dtype=float)
    n_inst = int(rt.shape[0])
    per_inst = [int(len(set(np.round(rt[i][rt[i] > 0], 9)))) for i in range(n_inst)]
    K = 4
    slots = [min(K, n) for n in per_inst]
    rec('dev_check_census', {
        'file': Path(args.data).name, 'keys': keys, 'n_instances': n_inst,
        'reveal_time_max': float(rt.max()), 'reveal_time_min_positive': float(rt[rt > 0].min()),
        'distinct_reveal_events_per_instance': per_inst,
        'K': K, 'slots_per_instance': slots, 'total_slots': int(sum(slots)),
        'instances_with_fewer_than_K_events': int(sum(1 for s in slots if s < K)),
        'dataset_carries_a_horizon_field': [k for k in keys if 'horizon' in k.lower()],
        'horizon_16_covers_all_reveals': bool(rt.max() < 16.0)})
    ck('G6 R1 x K=4 slots enumerated on dev_check (slot count is a pre-registered slot budget, '
       'not a promise of K qualified snapshots per instance)',
       n_inst == 16 and all(s >= 1 for s in slots),
       {k: RECORDS['dev_check_census'][k] for k in
        ('n_instances', 'total_slots', 'instances_with_fewer_than_K_events',
         'horizon_16_covers_all_reveals', 'reveal_time_max')})
    H_OLD, H_GEN = 16.0, 24.0
    rec('horizon_candidates', {
        'A_v1_pilot_protocol_constant': H_OLD,
        'data_generator_declared_window': H_GEN,
        'source_of_24_0': 'scripts/data/generate_coldchain_data.py:119 "tw_width, horizon = '
                          '0.2, 24.0" (the declared 50-node window; tw_end[0] = horizon)',
        'reveal_time_max': float(rt.max()),
        'old_constant_would_truncate': bool(rt.max() >= H_OLD),
        'generator_window_covers_all_reveals': bool(rt.max() < H_GEN),
        'why_not_derived_from_data': 'the window is the generator declaration, not a value read '
                                     'back from the last reveal of each day'})
    ck('G7a the A-v1 pilot constant 16.0 is INADMISSIBLE for this frozen data (reveals reach '
       '%.3f): using it would drop late reveals and change the problem' % float(rt.max()),
       bool(rt.max() >= H_OLD), RECORDS['horizon_candidates'])
    ck('G7b the data generator\'s declared public window 24.0 covers every reveal, so a '
       'pre-specified (non-data-derived) horizon exists that removes the leak without truncating '
       'the problem',
       bool(rt.max() < H_GEN), RECORDS['horizon_candidates'])

    fails = [c for c in CASES if not c[1]]
    verdict = 'CAUSAL_GATE_PASS_WITH_MANDATORY_HORIZON' if not fails else 'CAUSAL_GATE_FAILED'
    rec('verdict', {
        'status': verdict,
        'mandatory_condition': 'the L1 collector MUST construct StrictOnlineEnv with an '
                               'explicit, pre-specified public booking_horizon = 24.0 (the data '
                               'generator declaration at generate_coldchain_data.py:119, which '
                               'covers every reveal; the pilot constant 16.0 would truncate this '
                               'data and is rejected); with the None default the baseline '
                               'WAIT/CLOSE decision reads real future reveal data and the state '
                               'may NOT be labelled STRICT_ONLINE_QUALIFIED',
        'collection_may_be_labelled_STRICT_ONLINE_QUALIFIED': not fails,
        'not_run_here': ['no collection', 'no snapshot', 'no model', 'no rollout']})
    print('\n=== VERDICT: %s (%d rows, %d PASS, %d FAIL) ==='
          % (verdict, len(CASES), len(CASES) - len(fails), len(fails)))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'rows': [{'name': n, 'ok': o, 'detail': dd} for n, o, dd in CASES],
             'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
             'n_fail': len(fails)}, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
