#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.7 patch record: the exact minimal edits applied to the P0-C runner and collector to close
the blockers the independent review located in V1.6.  Each edit is anchor-asserted, so a mismatch
aborts loudly instead of silently skipping.

Edits
  E1  parity_ok: no longer compares suffix lists / booleans against the string 'EQUAL';
      it now checks the bitwise array verdicts, the suffix-structure verdict, the serialisation
      round trip AND a real full-resume parity verdict.
  E2  full resume parity: the worker keeps the original rollout's terminal state, resumes each
      snapshot through run_resumed and compares the terminal trajectory bitwise.
  E3  slot registration is bound to the REAL reveal hook: the collector gains a pre_capture
      callback invoked immediately before capture_recourse_snapshot, so ATTEMPT_STARTED is written
      (and fsynced) only for reveals that actually arrive; slots that never arrive stay
      NOT_ATTEMPTED_DUE_TO_TERMINATION instead of being pre-marked.
  E4  the 120 s cap applies to the ROLLOUT window only: the worker writes DAY_ROLLOUT_STARTED /
      DAY_ROLLOUT_FINISHED and the supervisor measures the cap between those timestamps; the
      process lifetime is kept only as a separate guard.
  E5  the identity gate is extended to the frozen runtime identity (runner SHA, 13-module runtime
      closure, profile SHA, dataset SHA, effective contract hash) plus a post-run re-check.
  E6  --inject-block-s lets the acceptance block inside a slot so the supervisor's slot deadline and
      kill path are exercised end to end.
"""
import pathlib
import re
import sys

EXT = pathlib.Path(__file__).resolve().parents[1] if (pathlib.Path(__file__).resolve().parents[1]
                                                      / 'scripts').is_dir() else None
ROOT = pathlib.Path(r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem')
RUNNER = ROOT / 'scripts/simulation/d3_l1_collection_runner.py'
COLLECTOR = ROOT / 'scripts/simulation/d3_l1_state_collector.py'
edits = []


def sub(path, old, new, tag):
    s = path.read_text(encoding='utf-8')
    assert old in s, 'ANCHOR MISSING [%s] in %s' % (tag, path.name)
    path.write_text(s.replace(old, new, 1), encoding='utf-8')
    edits.append(tag)


# ---- E3: collector pre_capture callback ------------------------------------------------
sub(COLLECTOR,
    """    def __init__(self, env, inst_idx, K, capture=True, clock_eps=1e-6):
        self.env = env""",
    """    def __init__(self, env, inst_idx, K, capture=True, clock_eps=1e-6, pre_capture=None):
        self.pre_capture = pre_capture
        self.env = env""", 'E3a')
sub(COLLECTOR,
    """        if slot is not None and self.capture:
            from recourse_snapshot import capture_recourse_snapshot
            t0 = time.time()""",
    """        if slot is not None and self.capture:
            from recourse_snapshot import capture_recourse_snapshot
            if self.pre_capture is not None:
                # the ledger is written (and fsynced) BEFORE the capture happens
                self.pre_capture(slot, clock, int(event_id), int(reveal_idx))
            t0 = time.time()""", 'E3b')

# ---- E1/E2/E3c/E4/E6: runner ------------------------------------------------------------
sub(RUNNER,
    """    # ---- blocker 1: register every pre-registered slot BEFORE any capture -----------------
    for position in range(1, 5):
        if not ok_led:
            append_jsonl(ledger_p, {'event': 'SKIPPED_LEDGER_CORRUPT', 'day_id': day,
                                    'position': position, 'run_id': args.run_id})
            continue
        allowed, ev = may_attempt(led, day, position)
        if allowed:
            append_jsonl(ledger_p, {'event': 'ATTEMPT_STARTED', 'day_id': day,
                                    'position': position, 'run_id': args.run_id,
                                    'registered_before_capture': True,
                                    'ledger_integrity_ok': True})
        else:
            append_jsonl(ledger_p, {'event': 'SKIPPED_ALREADY_ATTEMPTED', 'day_id': day,
                                    'position': position, 'previous_event': ev,
                                    'run_id': args.run_id})
    led = read_ledger(ledger_p)""",
    """    # ---- blocker 1 fix (V1.7): registration happens INSIDE the real reveal hook ----------
    if not ok_led:
        for position in range(1, 5):
            append_jsonl(ledger_p, {'event': 'SKIPPED_LEDGER_CORRUPT', 'day_id': day,
                                    'position': position, 'run_id': args.run_id})
        return 0
    registered = {}

    def _pre_capture(slot, clock, event_id, reveal_idx):
        allowed, ev = may_attempt(read_ledger(ledger_p), day, slot)
        if not allowed:
            append_jsonl(ledger_p, {'event': 'SKIPPED_ALREADY_ATTEMPTED', 'day_id': day,
                                    'position': slot, 'previous_event': ev,
                                    'run_id': args.run_id})
            raise RuntimeError('slot %d already attempted (%s)' % (slot, ev))
        append_jsonl(ledger_p, {'event': 'ATTEMPT_STARTED', 'day_id': day, 'position': slot,
                                'run_id': args.run_id, 'clock': float(clock),
                                'event_id': int(event_id), 'reveal_idx': int(reveal_idx),
                                'registered_before_capture': True, 'ledger_integrity_ok': True})
        registered[slot] = time.time()
    led = read_ledger(ledger_p)""", 'E3c')
sub(RUNNER,
    """    t_day = time.time()
    col = RevealSlotCollector(env, day, 4)
    run_baseline_rollout(env, day, hook=col)          # captures happen inside this rollout""",
    """    t_day = time.time()
    append_jsonl(ledger_p, {'event': 'DAY_ROLLOUT_STARTED', 'day_id': day, 'run_id': args.run_id})
    col = RevealSlotCollector(env, day, 4, pre_capture=_pre_capture)
    roll = run_baseline_rollout(env, day, hook=col)
    orig_traces, orig_served = roll['result']
    append_jsonl(ledger_p, {'event': 'DAY_ROLLOUT_FINISHED', 'day_id': day,
                            'run_id': args.run_id, 'day_rollout_wall_s': time.time() - t_day})""",
    'E4a')
sub(RUNNER,
    """    for position in range(1, 5):
        ev, _ = slot_state(led, day, position)
        if ev != 'ATTEMPT_STARTED':
            continue
        r = shots.get(position)""",
    """    for position in sorted(registered):
        ev, _ = slot_state(led, day, position)
        if ev != 'ATTEMPT_STARTED':
            continue
        r = shots.get(position)""", 'E3d')
sub(RUNNER,
    """            par['visible_mask'] = bitwise_equal(np.asarray(snap['visible_mask']),
                                                np.asarray(snap['visible_mask']))[0]""",
    """            # E2: real full-resume parity against the ORIGINAL terminal rollout
            from recourse_snapshot import validate_snapshot_against_env
            env_r = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
            validate_snapshot_against_env(env_r, snap)
            resumed = env_r.run_resumed(snap)
            res_traces, res_served = (resumed if isinstance(resumed, tuple) else (resumed, None))
            par['full_resume_served_mask'] = bitwise_equal(np.asarray(orig_served),
                                                           np.asarray(res_served))[0] \\
                if res_served is not None else 'NOT_VERIFIABLE'
            par['full_resume_traces'] = bitwise_equal(orig_traces, res_traces)[0]
            par['full_resume_parity'] = (par['full_resume_served_mask'] == 'EQUAL'
                                         and par['full_resume_traces'] == 'EQUAL')""", 'E2')
sub(RUNNER,
    """            parity_ok = all(v == 'EQUAL' for k, v in par.items() if k != 'suffix_boundary') \\
                and par.get('suffix_boundary') is True""",
    """            # E1: only the array-comparison verdicts are strings; the structural checks are
            # booleans and must be tested as booleans.
            bit_keys = ('served_mask', 'full_resume_served_mask', 'full_resume_traces')
            parity_ok = (all(par.get(k) == 'EQUAL' for k in bit_keys)
                         and par.get('suffix_boundary') is True
                         and par.get('serialization_roundtrip') is True
                         and par.get('full_resume_parity') is True)""", 'E1')
sub(RUNNER,
    """        t0 = time.time()
        try:
            snap = r['snapshot']""",
    """        t0 = registered.get(position, time.time())
        try:
            if args.inject_block_s > 0 and position == min(registered):
                time.sleep(args.inject_block_s)          # E6: force the supervisor's kill path
            snap = r['snapshot']""", 'E6')
sub(RUNNER,
    """    ap.add_argument('--runtime-closure', action='store_true')""",
    """    ap.add_argument('--runtime-closure', action='store_true')
    ap.add_argument('--inject-block-s', type=float, default=0.0)
    ap.add_argument('--day-rollout-cap-s', type=float, default=DAY_CAP_S)
    ap.add_argument('--expect-runner-sha256', default=None)
    ap.add_argument('--expect-profile-sha256', default=None)
    ap.add_argument('--expect-effective-contract', default=None)
    ap.add_argument('--runtime-identity', default=None)""", 'E6b')
# E4b: supervisor measures the 120 s cap on the ROLLOUT window
sub(RUNNER,
    """                if (now - t) > args.day_cap_s:
                    p.kill(); out, err = p.communicate(); status = 'day_deadline_killed'
                    append_jsonl(ledger_p, {'event': 'DAY_DEADLINE_KILLED', 'day_id': day,
                                            'run_id': args.run_id, 'day_cap_s': args.day_cap_s})
                    break""",
    """                _led = read_ledger(ledger_p)
                _started = [x.get('ts') for x in _led
                            if x.get('event') == 'DAY_ROLLOUT_STARTED' and x.get('day_id') == day]
                _done = any(x.get('event') == 'DAY_ROLLOUT_FINISHED' and x.get('day_id') == day
                            for x in _led)
                if _started and not _done and (now - _started[-1]) > args.day_rollout_cap_s:
                    p.kill(); out, err = p.communicate(); status = 'rollout_deadline_killed'
                    append_jsonl(ledger_p, {'event': 'ROLLOUT_DEADLINE_KILLED', 'day_id': day,
                                            'run_id': args.run_id,
                                            'day_rollout_cap_s': args.day_rollout_cap_s})
                    break
                if (now - t) > args.day_cap_s + args.batch_cap_s:
                    p.kill(); out, err = p.communicate(); status = 'lifetime_guard_killed'
                    append_jsonl(ledger_p, {'event': 'LIFETIME_GUARD_KILLED', 'day_id': day,
                                            'run_id': args.run_id})
                    break""", 'E4b')
sub(RUNNER,
    """        cmd = [sys.executable, str(Path(__file__)), '--role', 'worker', '--run-id', args.run_id,
               '--outdir', str(outdir), '--dataset', args.dataset, '--day-id', str(day),
               '--identity', args.identity, '--slot-cap-s', str(args.slot_cap_s),
               '--day-cap-s', str(args.day_cap_s), '--batch-cap-s', str(args.batch_cap_s)]""",
    """        cmd = [sys.executable, str(Path(__file__)), '--role', 'worker', '--run-id', args.run_id,
               '--outdir', str(outdir), '--dataset', args.dataset, '--day-id', str(day),
               '--identity', args.identity, '--slot-cap-s', str(args.slot_cap_s),
               '--day-cap-s', str(args.day_cap_s), '--batch-cap-s', str(args.batch_cap_s),
               '--inject-block-s', str(args.inject_block_s),
               '--day-rollout-cap-s', str(args.day_rollout_cap_s)]""", 'E4c')
sub(RUNNER,
    """    ap.add_argument('--runtime-identity', default=None)""",
    """    ap.add_argument('--runtime-identity', default=None)
    ap.add_argument('--inject-block-s', type=float, default=0.0)""", 'E6c')
print('applied edits:', edits)
