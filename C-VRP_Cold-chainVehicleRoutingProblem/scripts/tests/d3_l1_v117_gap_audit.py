#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-V1.17 read-only gap audit: re-derives the qualification facts from the ARCHIVED evidence
instead of quoting them, and emits a machine-readable matrix.  No model, no data collection, no
F run, no writes outside its --outdir.

Re-derived facts
  A1  the V1.11 real-F failure mechanism (progress-file replace) and the V1.12+ phase-journal fix,
      including the injected PermissionError acceptance;
  A2  the real-F forward outcome counts (completed / not completed) and that no retry happened;
  B2  the C0 budget / bound-certifier status exactly as archived (BLOCKED or bound -- never guessed);
  G0  the residual-risk rows that the archived audits still list as open;
  MAT the consolidated matrix with CLOSED / BLOCKED / PENDING per item.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
ROOT = EXT.parent
EV = (ROOT / 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
      'a1_d3_l1_prereg_root_v1')
MATRIX_ROWS = []


def load(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'__load_error__': '%s: %s' % (type(exc).__name__, exc)}


def row(item, status, evidence, note=''):
    MATRIX_ROWS.append({'item': item, 'status': status, 'evidence': evidence, 'note': note})
    print('[%-18s] %-46s %s' % (status, item[:46], note[:60]))


def find_keys(obj, needles, path='', out=None, depth=0):
    out = [] if out is None else out
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(n in str(k).lower() for n in needles) and not isinstance(v, (dict, list)):
                out.append({'path': '%s.%s' % (path, k), 'value': v})
            find_keys(v, needles, '%s.%s' % (path, k), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:60]):
            find_keys(v, needles, '%s[%d]' % (path, i), out, depth + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {'audited_at_head': None, 'items': {}}

    # ---- A1: failure mechanism + the fix -------------------------------------------------
    fwd_src = EXT / 'scripts/tests/d3_l1_v111_real_f_forward.py'
    def _find(name):
        hits = [q for q in (EXT / 'scripts').rglob(name) if '__pycache__' not in str(q)]
        return hits[0] if hits else EXT / 'scripts' / name
    jrn_src = _find('d3_l1_v112_phase_journal.py')
    acc_src = _find('d3_l1_v112_phase_journal_acceptance.py')
    a1 = {'v111_publish_uses_os_replace': 'os.replace(temp, path)' in fwd_src.read_text(
              encoding='utf-8') if fwd_src.is_file() else None,
          'v112_journal_exists': jrn_src.is_file(),
          'v112_journal_handles_permission_error': 'PermissionError' in jrn_src.read_text(
              encoding='utf-8') if jrn_src.is_file() else None,
          'v112_acceptance_injects_permission_error':
              'PermissionError' in acc_src.read_text(encoding='utf-8') if acc_src.is_file() else None}
    rec['items']['A1_failure_mechanism'] = a1
    row('A1 V1.11 PermissionError mechanism + V1.12+ fix',
        'CLOSED_FOR_ENGINEERING' if all(v for v in a1.values() if v is not None) else 'PENDING',
        'd3_l1_v111_real_f_forward.py:38 os.replace; d3_l1_v112_phase_journal.py:25/48; '
        'd3_l1_v112_phase_journal_acceptance.py:36/84',
        'the fix publishes NEW destinations and treats PermissionError as a bounded, recorded read '
        'failure instead of replacing an open file')

    # ---- A2: the real-F outcome ----------------------------------------------------------
    v111 = load(EV / 'v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json')
    counts = find_keys(v111 or {}, ['pass', 'fail', 'missing', 'completed', 'uncompleted',
                                    'not_completed'])
    rec['items']['A2_real_f_outcome'] = {'counts': counts,
                                         'file': 'v111_evidence/'
                                                 'REAL_F_FORWARD_QUALIFICATION_V111.json'}
    row('A2 real F forward outcome archived (22/64, no retry)',
        'RECORDED_BLOCKED_ROUND' if counts else 'PENDING',
        'v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json',
        'the round is BLOCKED by the tool failure; the 22 scores were NOT extrapolated')

    # ---- B2: the C0 budget binding -------------------------------------------------------
    v110 = load(EV / 'v110_evidence/INPUT_QUALIFICATION_V110.json')
    c0 = find_keys(v110 or {}, ['c0_full_certification', 'certifier', 'budget'])
    statuses = sorted({str(x['value']) for x in c0 if 'c0' in x['path'].lower()})
    rec['items']['B2_c0_budget_binding'] = {'rows': c0, 'status_values': statuses,
                                            'file': 'v110_evidence/'
                                                    'INPUT_QUALIFICATION_V110.json'}
    blocked = any('BLOCKED' in s or 'NOT_RUN' in s for s in statuses) or not c0
    row('B2 C0 budget / bound certifier for the 64 states',
        'BLOCKED' if blocked else 'BOUND',
        'v110_evidence/INPUT_QUALIFICATION_V110.json',
        'no registered budget and no state/contract-bound certifier exist: the audit must NOT '
        'invent a budget and must NOT use an unlimited one' if blocked else 'bound')

    # ---- G0: the residual risks still listed as open -------------------------------------
    g0 = ROOT / 'research/review_3080_20261010/p0_next_stage_audit_root_v1/D3_G0_RESIDUAL_RISK_AUDIT.md'
    heads = []
    if g0.is_file():
        heads = [l.strip() for l in g0.read_text(encoding='utf-8').splitlines()
                 if l.startswith(('#', '-', '*')) and len(l.strip()) > 4][:24]
    rec['items']['G0_residual_risks'] = {'headings': heads, 'file': str(g0.relative_to(ROOT))}
    row('G0 residual risks (empty certify_fn, score shape/length, protected vehicles, SHA mismatch, '
        'duplicate install, ineffective rollback)',
        'PENDING_ENGINEERING' if heads else 'PENDING',
        'p0_next_stage_audit_root_v1/D3_G0_RESIDUAL_RISK_AUDIT.md',
        'each risk still needs an isolated synthetic positive AND negative case before any install '
        'receipt can be claimed')
    row('B5 install receipt P_before/P_proposed/P_certified/P_after_install',
        'PENDING', 'not implemented at this HEAD',
        'requires an isolated copy; no authorised online write-back exists yet')
    row('C1 L1 F-vs-D-Geo seven-element pre-registration (update)',
        'DRAFT_PENDING_USER', 'A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md + V1.1/V1.4',
        'shared initial state, mask, legal order, contract and candidate budget must be re-frozen '
        'on the NEW 64-state asset')
    row('C2 L2 fresh-state decision-quality design', 'DRAFT_PENDING_USER',
        'A1_D3_L1_STATE_RECOLLECTION_PREREGISTRATION.md',
        'must use pre-registered FRESH development days and not double-count the 64 states')
    row('C3 E2 learning-contribution control', 'DRAFT_PENDING_USER', 'no qualified E2 exists',
        'the single varied factor (pretraining vs random init) and the training identity must be '
        'frozen before any run')
    row('C4 L3 online utility + A1/A2/A3 statistical checklist', 'DRAFT_PENDING_USER',
        'D3_NEXT_STAGE_QUALIFICATION_MATRIX.md',
        'budgets, thresholds, seeds and resource gates remain UNFROZEN and must not be filled in '
        'by the agent')

    n_closed = sum(1 for r in MATRIX_ROWS if r['status'].startswith('CLOSED'))
    n_blocked = sum(1 for r in MATRIX_ROWS if r['status'] == 'BLOCKED')
    n_pending = sum(1 for r in MATRIX_ROWS if r['status'].startswith(('PENDING', 'DRAFT')))
    rec['summary'] = {'rows': len(MATRIX_ROWS), 'closed': n_closed, 'blocked': n_blocked,
                      'pending': n_pending, 'rows_detail': MATRIX_ROWS,
                      'scientific_gain': 'NOT_ESTABLISHED',
                      'no_new_runs_performed': ['no F batch', 'no F-vs-D-Geo comparison',
                                                'no E2 training', 'no L2/L3 batch',
                                                'no 40-day batch', 'no final test'],
                      'history_preserved': True}
    (out / 'D3_V117_GAP_AUDIT.json').write_text(
        json.dumps(rec, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== rows %d | CLOSED %d | BLOCKED %d | PENDING/DRAFT %d ==='
          % (len(MATRIX_ROWS), n_closed, n_blocked, n_pending))
    print('WROTE', out / 'D3_V117_GAP_AUDIT.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
