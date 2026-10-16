#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 RUN-ENTRY ACCEPTANCE: freeze, synthetic arms, guard refusal, and the full supervised path.

Run against a THROW-AWAY directory with a SYNTHETIC scorer.  It proves the entry works end to end
before a signer ever authorises the real comparison, and it proves the guard refuses a direct start:

  E1 freeze       the L1 identity carries 64 states, the 34-path watch set and the frozen budget
  E2 synthetic    both arms run in-process and the adjudication reports the four V1 buckets with
                  V3 = NOT_RUN and `synthetic = true`
  E3 guard        a direct `--worker` refuses BEFORE loading anything
  E4 supervised   the full path works: authorize -> ONE-SHOT consume -> execution context ->
                  contained worker -> 64 per-state records -> adjudication
  E5 evidence     the supervised path leaves exactly one consumption receipt and a valid execution
                  context, and no artifact claims a real comparison

The scorer here is synthetic, so nothing in this file says anything about MaskCO or D-Geo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_authorization as AUTH                                       # noqa: E402

RUNNER = EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'
PREREG = (REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1'
          '/a1_d3_l1_prereg_root_v1/A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md')
RUN_ID = 'd3l1-l1-two-arm-20261016-l1v2'
ROWS = []


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:220]))


def run(args, timeout_s=900):
    return subprocess.run([sys.executable, str(RUNNER)] + args, capture_output=True, text=True,
                          timeout=timeout_s)


def runner_module():
    spec = importlib.util.spec_from_file_location('l1_entry', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['l1_entry'] = module
    spec.loader.exec_module(module)
    return module


def supervised_ready_seen(ident, run_dir):
    """`(ok, detail)`: did the supervisor SEE the worker's immutable READY record?

    The point of blocker B is that the supervisor must decide "initialization is over" from the
    worker's own durable READY record -- not from a default probe looking for a fixed `READY.json`
    the V1.18+ worker can never publish.  This re-runs the PRODUCTION probe against the finished run
    directory and asserts it reports ready, and separately asserts `run_contained` recorded that
    probe's name and verdict rather than the legacy default.
    """
    detail = {}
    try:
        module = runner_module()
        probe = module.make_l1_ready_probe(run_dir)
        report = probe(run_dir)
        supervision = json.loads((Path(run_dir) / 'L1_SUPERVISION.json').read_text('utf-8'))
        recorded = supervision.get('ready_state') or {}
        detail = {'reprobe_ready': report.get('ready'), 'reprobe_n_records': report.get('n_records'),
                  'reprobe_problems': report.get('problems'),
                  'reprobe_run_id_matches': report.get('run_id') == ident['run_id'],
                  'recorded_probe': recorded.get('probe'),
                  'recorded_ready': recorded.get('ready'),
                  'recorded_published_name': recorded.get('published_name'),
                  'legacy_default_probe_used': recorded.get('probe') == 'default READY.json existence'}
        ok = (bool(report.get('ready')) and int(report.get('n_records') or 0) >= 1
              and report.get('run_id') == ident['run_id']
              and recorded.get('probe') == 'immutable L1_READY record'
              and recorded.get('ready') is True)
        return ok, detail
    except BaseException as exc:                                          # noqa: BLE001
        detail['error'] = '%s: %s' % (type(exc).__name__, exc)
        return False, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix='l1_entry_acc_'))
    run_dir = root / 'run'
    results = {}
    try:
        # E1 ----------------------------------------------------------------------------
        proc = run(['--freeze-identity', '--out-dir', str(run_dir), '--run-id', RUN_ID])
        ident = json.loads((run_dir / 'L1_IDENTITY.json').read_text(encoding='utf-8'))
        results['E1'] = {'returncode': proc.returncode,
                         'run_id': ident.get('run_id'),
                         'n_states': ident['state_list']['n_states'],
                         'n_watch': len(ident['watch']),
                         'n_source_paths': (ident.get('watch_derivation') or {})
                         .get('n_source_paths'),
                         'critical_missing': (ident.get('watch_derivation') or {})
                         .get('critical_missing'),
                         'budget': ident['budget'],
                         'state_binding_sha256': ident['state_binding_sha256']}
        # The watch set is DERIVED, so it is asserted by CONTENT, not by a remembered count: a run
        # whose closure shrinks must not pass merely because the number 34 was written down once.
        watch_paths = [p.replace('\\', '/') for p in ident.get('watch_paths') or []]
        l1_own = [p for p in watch_paths
                  if p.endswith(('d3_l1_l1_formal_run.py', 'd3_l1_real_arm.py',
                                 'd3_l1_two_arm_protocol.py', 'd3_geo_adapter.py',
                                 'd3_l1_v110_input_qualification.py', 'd3_l1_closure.py'))]
        results['E1']['n_l1_own_sources_frozen'] = len(l1_own)
        results['E1']['watch_derivation'] = (ident.get('watch_derivation') or {}).get('basis')
        ck('E1 freeze materialises the L1 identity: 64 states, a DERIVED watch set that covers the '
           'L1 decision modules, the frozen budget and the state binding',
           proc.returncode == 0 and ident['state_list']['n_states'] == 64
           and len(ident['watch']) >= 34 and len(l1_own) == 6
           and (ident.get('watch_derivation') or {}).get('critical_missing') == []
           and ident['budget']['per_arm_total_scorer_calls'] == 128
           and ident['budget']['per_state_wall_clock_s'] == 10.0
           and ident['budget']['initialize_compile_s'] == 60.0
           and ident['budget']['batch_wall_clock_s'] == 180.0
           and ident['state_binding_sha256'].startswith('f5eedd10'), results['E1'])

        # E3 (before E2 so the guard is tested on a clean directory) ----------------------
        proc3 = run(['--worker', '--out-dir', str(run_dir)])
        guard_refused = (proc3.returncode != 0 and 'EXECUTION CONTEXT REFUSED' in proc3.stderr)
        results['E3'] = {'returncode': proc3.returncode,
                         'stderr_tail': proc3.stderr.strip()[-160:],
                         'artifacts': sorted(p.name for p in run_dir.glob('L1_STATE_RECORDS*'))}
        ck('E3 a direct `--worker` start REFUSES before loading anything (no execution context)',
           guard_refused and not (run_dir / 'L1_STATE_RECORDS.jsonl').exists(), results['E3'])

        # E4 ----------------------------------------------------------------------------
        module = runner_module()
        identity = module.authorization_identity_view(ident)
        targets = module.authorization_targets_view(ident)
        cred = AUTH.seal_template(dict(
            AUTH.build_template(identity, targets, preregistration_sha256=sha(PREREG),
                                output_dir=run_dir),
            approved_by='SYNTHETIC ACCEPTANCE FIXTURE',
            approved_utc='2026-10-15T00:00:00Z',
            approval_statement='synthetic acceptance fixture; this is NOT a real human approval'))
        cred_path = root / 'AUTHORIZATION_SYNTHETIC.json'
        cred_path.write_text(json.dumps(cred, indent=2, sort_keys=True, default=repr),
                             encoding='utf-8')
        consumption = root / 'consumption'
        proc4 = run(['--supervise', '--out-dir', str(run_dir), '--authorization', str(cred_path),
                     '--preregistration', str(PREREG), '--consumption-root', str(consumption),
                     '--synthetic-run'])
        adj = json.loads((run_dir / 'L1_ADJUDICATION.json').read_text(encoding='utf-8'))
        records = [json.loads(ln) for ln in
                   (run_dir / 'L1_STATE_RECORDS.jsonl').read_text(encoding='utf-8').splitlines()
                   if ln.strip()]
        supervision = json.loads((run_dir / 'L1_SUPERVISION.json').read_text(encoding='utf-8'))
        # `L1_SUPERVISION.json` strips the trailing identity, and `ready_state` is recorded as its
        # probe NAME plus the `ready` boolean -- the two fields the supervisor decides on.
        ready_probe = supervision.get('ready_state') or {}
        ready_ok, ready_detail = supervised_ready_seen(ident, run_dir)
        phases = sorted((run_dir / 'phases').glob('phase_*.json'))
        phase_recs = [json.loads(p.read_text(encoding='utf-8')) for p in phases]
        stage_counts = {}
        for rec in phase_recs:
            stage_counts[rec.get('stage')] = stage_counts.get(rec.get('stage'), 0) + 1
        # The V1.16 timeline audit over the L1 journal: every `slot` must be opened and closed for
        # each of the 64 states, and the journal must end with `completed`.
        slot_pairs = sum(1 for r in phase_recs if r.get('stage') == 'slot_finished')
        slot_opens = sum(1 for r in phase_recs if r.get('stage') == 'slot')
        last_stage = phase_recs[-1].get('stage') if phase_recs else None
        results['E4'] = {'returncode': proc4.returncode, 'supervision_ok': supervision.get('ok'),
                         'reason': supervision.get('reason'),
                         'ready_state': ready_probe.get('probe'),
                         'ready_check': ready_detail,
                         'n_records': len(records),
                         'n_phases': len(phases), 'phase_stage_counts': stage_counts,
                         'journal_last_stage': last_stage,
                         'timeline_ok': (supervision.get('control_timeline') or {}).get('ok'),
                         'V1_counts': adj['V1_counts'], 'V3': adj['V3'],
                         'denominator': adj['n_states_in_denominator'],
                         'status': adj['status'],
                         'all_ledgers_cross_checked': adj['all_ledgers_cross_checked'],
                         'wall_clock_scope': adj.get('wall_clock_scope'),
                         'worker_control_wall_s': supervision.get('worker_control_wall_s')}
        ck('E4 the full supervised path runs: authorize -> one-shot consume -> execution context -> '
           'contained worker -> 64 per-state records -> adjudication with V3 = NOT_RUN, an immutable '
           'READY the supervisor actually SAW, and a matched per-state phase journal',
           proc4.returncode == 0 and supervision.get('ok') is True and len(records) == 64
           and adj['n_states_in_denominator'] == 64 and adj['status'] == 'ADJUDICATED'
           and ready_probe.get('probe') == 'immutable L1_READY record' and ready_ok
           and slot_opens == 64 and slot_pairs == 64
           and stage_counts.get('initialize') == 1 and stage_counts.get('compile') == 1
           and last_stage == 'completed'
           and set(adj['V1_counts']) == {'DIVERGENT', 'IDENTICAL', 'NOT_COMPARABLE',
                                         'NOT_APPLICABLE_EMPTY_MASK', 'NOT_ATTEMPTED'}
           and adj['V3'].startswith('NOT_RUN') and adj['all_ledgers_cross_checked'] is True
           and adj['synthetic'] is True, results['E4'])

        # E5 ----------------------------------------------------------------------------
        receipts = sorted(consumption.glob('authorization_consumption_*.json'))
        contexts = sorted(run_dir.glob('EXECUTION_CONTEXT_*.json'))
        results['E5'] = {'n_receipts': len(receipts), 'n_contexts': len(contexts),
                         'context_published': any('EXECUTION_CONTEXT_' in p.name
                                                  for p in run_dir.iterdir())}
        ck('E5 the supervised path leaves exactly ONE consumption receipt and a published execution '
           'context (so the single real opportunity is accounted for)',
           len(receipts) == 1 and len(contexts) == 1, results['E5'])

        # E2 ----------------------------------------------------------------------------
        standalone = root / 'standalone'
        run(['--freeze-identity', '--out-dir', str(standalone), '--run-id', RUN_ID])
        proc2 = run(['--synthetic-run', '--out-dir', str(standalone)])
        adj2 = json.loads((standalone / 'L1_SYNTHETIC_ACCEPTANCE.json').read_text(encoding='utf-8'))
        results['E2'] = {'returncode': proc2.returncode,
                         'n_records': len(adj2['records']),
                         'V1_counts': adj2['verdict']['V1_counts'],
                         'scope': adj2['scope'][:80]}
        ck('E2 the standalone synthetic acceptance runs both arms on three states and labels the '
           'record SYNTHETIC',
           proc2.returncode == 0 and len(adj2['records']) == 3
           and adj2['verdict']['synthetic'] is True, results['E2'])

        fails = [r for r in ROWS if not r['ok']]
        payload = {
            'schema': 'd3_l1_run_entry_acceptance',
            'run_id': RUN_ID, 'rows': ROWS, 'n_rows': len(ROWS),
            'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
            'results': results,
            'scorer': 'SYNTHETIC_SCORER_NOT_F -- this acceptance says nothing about MaskCO or D-Geo',
            'not_run': ['the real F-vs-D-Geo comparison', 'training', 'L2', 'L3', 'the 40-day batch',
                        'the final test'],
            'science_ready': False, 'stable_online_gain': False,
            'tool_sha256': sha(Path(__file__)), 'runner_sha256': sha(RUNNER),
            'verdict': ('L1_RUN_ENTRY_ACCEPTED' if not fails else 'L1_RUN_ENTRY_BLOCKED'),
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
        print('\n=== L1 run-entry acceptance: %d rows, %d PASS, %d FAIL ==='
              % (len(ROWS), len(ROWS) - len(fails), len(fails)))
        print('WROTE', out)
        return 0 if not fails else 1
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
