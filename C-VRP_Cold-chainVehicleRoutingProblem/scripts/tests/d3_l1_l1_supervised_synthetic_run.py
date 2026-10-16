#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 SUPERVISED SYNTHETIC RUN: the real entry, the real supervisor, a SYNTHETIC scorer.

The signing package's M10 gate requires evidence that the WHOLE supervised path -- authorize ->
one-shot consume -> execution context -> contained worker -> per-state records -> adjudication --
runs on the frozen L1 identity, not merely that its pieces exist.  This tool performs exactly that,
in a throw-away directory, with the SYNTHETIC acceptance scorer, and writes the evidence.

It proves nothing about MaskCO or D-Geo.  A synthetic scorer is deterministic and
distance-independent, so a "divergence" here is meaningful for the harness and meaningless for
science; the emitted record says so on its face.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

import d3_l1_v118_authorization as AUTH                                       # noqa: E402

RUNNER = EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'
PREREG = (REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1'
          '/a1_d3_l1_prereg_root_v1/A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md')
RUN_ID = 'd3l1-l1-two-arm-20261015-l1v1'


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location('l1_entry_synth', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['l1_entry_synth'] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix='.l1_synth_run_', dir=str(REPO)))
    run_dir = scratch / 'run'
    consumption = scratch / 'consumption'
    try:
        module = load_runner()
        freeze = subprocess.run([sys.executable, str(RUNNER), '--freeze-identity',
                                 '--out-dir', str(run_dir), '--run-id', RUN_ID],
                                capture_output=True, text=True, timeout=900)
        ident = module.load_frozen(run_dir)
        cred = AUTH.seal_template(dict(
            AUTH.build_template(module.authorization_identity_view(ident),
                                module.authorization_targets_view(ident),
                                preregistration_sha256=sha(PREREG), output_dir=run_dir),
            approved_by='SUPERVISED SYNTHETIC RUN FIXTURE',
            approved_utc='2026-10-15T00:00:00Z',
            approval_statement='synthetic acceptance fixture; NOT a real human approval and NOT an '
                               'authorisation to run the real comparison'))
        cred_path = scratch / 'AUTHORIZATION_SYNTHETIC.json'
        cred_path.write_text(json.dumps(cred, indent=2, sort_keys=True, default=repr),
                             encoding='utf-8')
        proc = subprocess.run([sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir),
                               '--authorization', str(cred_path), '--preregistration', str(PREREG),
                               '--consumption-root', str(consumption), '--synthetic-run'],
                              capture_output=True, text=True, timeout=3600)
        supervision = json.loads((run_dir / 'L1_SUPERVISION.json').read_text(encoding='utf-8')) \
            if (run_dir / 'L1_SUPERVISION.json').is_file() else {}
        adj = json.loads((run_dir / 'L1_ADJUDICATION.json').read_text(encoding='utf-8')) \
            if (run_dir / 'L1_ADJUDICATION.json').is_file() else {}
        records = [json.loads(ln) for ln in
                   (run_dir / 'L1_STATE_RECORDS.jsonl').read_text(encoding='utf-8').splitlines()
                   if ln.strip()] if (run_dir / 'L1_STATE_RECORDS.jsonl').is_file() else []
        phase_recs = [json.loads(p.read_text(encoding='utf-8'))
                      for p in sorted((run_dir / 'phases').glob('phase_*.json'))]
        stage_counts = {}
        for rec in phase_recs:
            stage_counts[rec.get('stage')] = stage_counts.get(rec.get('stage'), 0) + 1
        receipts = sorted(consumption.glob('authorization_consumption_*.json'))
        ready = supervision.get('ready_state') or {}
        timeline = supervision.get('control_timeline') or supervision.get('timeline') or {}
        ok = (proc.returncode == 0 and freeze.returncode == 0
              and supervision.get('ok') is True
              and ready.get('probe') == 'immutable L1_READY record' and ready.get('ready') is True
              and len(records) == 64 and adj.get('n_states_in_denominator') == 64
              and adj.get('status') == 'ADJUDICATED'
              and adj.get('all_ledgers_cross_checked') is True
              and stage_counts.get('slot') == 64 and stage_counts.get('slot_finished') == 64
              and len(receipts) == 1)
        payload = {
            'schema': 'd3_l1_l1_supervised_synthetic_run',
            'run_id': RUN_ID,
            'verdict': ('L1_SUPERVISED_SYNTHETIC_RUN_ACCEPTED' if ok
                        else 'L1_SUPERVISED_SYNTHETIC_RUN_BLOCKED'),
            'freeze_returncode': freeze.returncode,
            'supervise_returncode': proc.returncode,
            'supervision_ok': supervision.get('ok'),
            'supervision_reason': supervision.get('reason'),
            'ready_probe': ready.get('probe'), 'ready': ready.get('ready'),
            'ready_published_name': ready.get('published_name'),
            'legacy_default_probe_used': ready.get('probe') == 'default READY.json existence',
            'n_state_records': len(records),
            'n_states_in_denominator': adj.get('n_states_in_denominator'),
            'adjudication_status': adj.get('status'),
            'adjudication_blocking_reasons': adj.get('blocking_reasons'),
            'V1_counts': adj.get('V1_counts'), 'outcome_counts': adj.get('outcome_counts'),
            'all_ledgers_cross_checked': adj.get('all_ledgers_cross_checked'),
            'V3': adj.get('V3'),
            'n_phase_records': len(phase_recs), 'phase_stage_counts': stage_counts,
            'journal_last_stage': phase_recs[-1].get('stage') if phase_recs else None,
            'timeline_ok': timeline.get('ok'), 'timeline_errors': timeline.get('errors'),
            'n_consumption_receipts': len(receipts),
            'per_state_wall_clock_s': ident['budget']['per_state_wall_clock_s'],
            'wall_clock_scope': module.WALL_CLOCK_SCOPE,
            'worker_control_wall_s': supervision.get('worker_control_wall_s'),
            'scorer': 'SYNTHETIC_SCORER_NOT_F -- this run says nothing about MaskCO or D-Geo',
            'not_run': ['the real L1 comparison', 'the real F scorer', 'D-Geo', 'training', 'L2',
                        'L3', 'the 40-day batch', 'the final test'],
            'stderr_tail': proc.stderr.strip()[-400:],
            'science_ready': False, 'stable_online_gain': False,
            'tool_sha256': sha(Path(__file__)), 'runner_sha256': sha(RUNNER),
            'preregistration_sha256': sha(PREREG),
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
        print(json.dumps({k: v for k, v in payload.items()
                          if k not in ('stderr_tail', 'V1_counts', 'phase_stage_counts')},
                         indent=1, default=repr)[:2600])
        print('WROTE', out)
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        if scratch.exists():
            print('SCRATCH KEPT', scratch)


if __name__ == '__main__':
    sys.exit(main())
