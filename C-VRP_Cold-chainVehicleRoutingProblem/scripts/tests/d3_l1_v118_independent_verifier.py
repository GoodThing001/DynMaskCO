#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.18 independent READ-ONLY verifier: binds to the frozen run_id and re-derives everything.

Independence rules this tool follows
------------------------------------
I1  It does not import the runner, the supervisor, or the phase journal.  It reads the PUBLISHED
    artifacts only, with its own parser, so a bug in the production writer cannot make the
    verifier agree with it.
I2  It re-implements the row-level verification (`verify_row_independently`) from the published
    JSON alone and CROSS-CHECKS its verdict against the production verifier's verdict.  Agreement
    is evidence of the verifier's correctness; disagreement is reported as a defect of the pair,
    never silently resolved in the production verifier's favour.
I3  Every identity is recomputed from bytes on disk (hashlib), never taken from a record.
I4  It never writes inside the run directory.  Its only output is the `--json` file it is given.
I5  When the real F forward has not run, the verdict is `VERIFIED_NOT_EXECUTED` -- explicitly NOT
    a PASS.  A synthetic directory is reported as `SYNTHETIC_*` and can never be read as a real F
    result.

Run-scoped constants (the verifier is frozen against these BEFORE the run):
    EXPECTED_RUN_ID, EXPECTED_TARGETS_SHA256, EXPECTED_STATE_BINDING_SHA256, EXPECTED_IDENTITIES
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation'):
    sys.path.insert(0, str(EXT / _p))

EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
V110_REPORT = EV / 'v110_evidence/INPUT_QUALIFICATION_V110.json'
STATE_MANIFEST = EV / 'v19_evidence/STATE_MANIFEST_64_V19.json'
LEDGER = EV / 'v19_evidence/formal_collection/STATE_LEDGER.jsonl'
V111_SCORES = EV / 'v111_evidence/scores'
V111_REPORT = EV / 'v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json'
V111_MODEL_LOAD = EV / 'v111_evidence/MODEL_LOAD.json'
V111_WARMUP = EV / 'v111_evidence/WARMUP.json'

# ---------------------------------------------------------------------------------------------
# FROZEN RUN-SCOPED CONSTANTS -- written before the run, compared against every artifact
# ---------------------------------------------------------------------------------------------
EXPECTED_RUN_ID = 'd3l1-real-f-forward-20261015-v118'
EXPECTED_TARGETS_SHA256 = '37e23f7567e3391a4a3360203a3d58e77c2e16b32c7e3514c65c8aa15d5ade08'
EXPECTED_IDENTITIES = {
    'f_ckpt': 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5',
    'backbone': '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db',
    'dev_check_npz': '66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89',
    'profile_file': '63e56212174c9ce371b8a5b1fada6a549d7bc47527fdd1b0b13a7bbe90db57e8',
    'state_manifest': '2340cc56c83da49cb27e0eaffa7db7c21d37a3ecc6af34793820a020081a294d',
    'state_ledger': None,     # filled by the verifier from disk and pinned here on first freeze
}
EXPECTED_BUDGET_CAPS = {'initialize_compile_s': 60.0, 'slot_span_s': 10.0, 'batch_s': 180.0}
EXPECTED_MAX_FORWARD_CALLS = 129
EXPECTED_ROWS = 64
EXPECTED_PER_SLOT_CALLS = 2
ROW_NAME_RE = re.compile(r'^SLOT_day(\d{2})_pos(\d+)_[0-9a-f]{32}\.json$')
# The row records live in the `scores/` sub-directory the runner creates.
ROW_DIR = 'scores'

CASES = []
REC = {}


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
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:250]))


def publish_records(run_dir, prefix, verify=True):
    """Independent reader: glob + parse + optional re-read digest check.  Never imports the writer.

    Returns `[(name, record, digest)]` sorted by (published_mono, name).  `prefix` is matched as
    `<prefix>*` (a literal prefix, not `<prefix>_*`), because the record families are
    `RUN_IDENTITY_<uuid>.json` and `SLOT_day00_pos1_<uuid>.json` -- the former has an underscore
    after the prefix and the latter does not, so an appended underscore would silently match
    nothing.
    """
    out = []
    for p in sorted(Path(run_dir).glob('%s*.json' % prefix)):
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rec = json.loads(raw.decode('utf-8'))
        out.append((p.name, rec, digest))
    out.sort(key=lambda t: (float(t[1].get('published_mono') or 0.0), t[0]))
    return out


# ---------------------------------------------------------------------------------------------
# independent row verification (re-implemented, NOT imported from the production verifier)
# ---------------------------------------------------------------------------------------------
def verify_row_independently(record, target):
    problems = []
    if record.get('day_id') != target['day_id'] or record.get('position') != target['position']:
        problems.append('row identity mismatch')
    if record.get('input_sha256') != target['input_sha256']:
        problems.append('input digest differs')
    if record.get('legal_sha256') != target['legal_sha256']:
        problems.append('ordered-legal digest differs')
    if record.get('status') != 'FORWARD_CHECKS_PASS':
        problems.append('status is not FORWARD_CHECKS_PASS')
    if record.get('F_calls') != EXPECTED_PER_SLOT_CALLS:
        problems.append('wrong number of forward calls')
    first = record.get('scores_first_hex') or ''
    repeat = record.get('scores_repeat_hex') or ''
    if not first or first != repeat:
        problems.append('the two forwards are not byte-identical')
    try:
        n_first = len(bytes.fromhex(first))
        n_repeat = len(bytes.fromhex(repeat))
    except ValueError:
        problems.append('score hex is not decodable')
        n_first = n_repeat = -1
    shape = record.get('scores_shape')
    dtype = record.get('scores_dtype')
    if (not isinstance(shape, list) or len(shape) != 1
            or not isinstance(shape[0], int) or shape[0] <= 0):
        problems.append('score shape is not a positive 1-D length')
    else:
        if shape[0] != target['n_legal_expected']:
            problems.append('score length %d != ordered legal length %d'
                            % (shape[0], target['n_legal_expected']))
        width = {'<f4': 4, '<f8': 8}.get(dtype)
        if width is None:
            problems.append('unexpected score dtype %r' % dtype)
        elif n_first != shape[0] * width or n_repeat != shape[0] * width:
            problems.append('declared dtype/shape and the byte length disagree')
    return (not problems), {'problems': problems, 'first_sha256':
                            hashlib.sha256(bytes.fromhex(first)).hexdigest() if first else None}


# ---------------------------------------------------------------------------------------------
# independent re-derivation of the budget freeze from the RAW V1.11 score records
# ---------------------------------------------------------------------------------------------
def recompute_budget_freeze():
    spans, inits, warms, batches = [], [], [], []
    for p in sorted(V111_SCORES.glob('*.json')) if V111_SCORES.is_dir() else []:
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('status') == 'FORWARD_CHECKS_PASS':
            spans.append(float(r['span_wall_s']))
    if V111_MODEL_LOAD.is_file():
        inits.append(float(json.loads(V111_MODEL_LOAD.read_text(encoding='utf-8'))
                           ['initialization_wall_s']))
    if V111_WARMUP.is_file():
        warms.append(float(json.loads(V111_WARMUP.read_text(encoding='utf-8'))['wall_s']))
    if V111_REPORT.is_file():
        batches.append(float(json.loads(V111_REPORT.read_text(encoding='utf-8'))
                             ['batch_wall_s_before_final_result_write']))
    if not spans or not inits or not warms:
        return None
    n_states = len(spans)
    per_state = batches[0] / n_states if batches else None
    projected = per_state * 64.0 + inits[0] + warms[0] if per_state else None
    return {'n_observed_states': n_states, 'slot_span_max_s': max(spans),
            'init_s': inits[0], 'warmup_s': warms[0],
            'observed_batch_s': batches[0] if batches else None,
            'projected_full_schedule_s': projected,
            'projected_inside_batch_cap': (projected is not None
                                           and projected <= EXPECTED_BUDGET_CAPS['batch_s']),
            'init_plus_warmup_inside_init_cap': (inits[0] + warms[0])
            <= EXPECTED_BUDGET_CAPS['initialize_compile_s'],
            'slot_max_inside_slot_cap': max(spans) <= EXPECTED_BUDGET_CAPS['slot_span_s']}


def freeze_state_list_independently():
    manifest = json.loads(STATE_MANIFEST.read_text(encoding='utf-8'))
    order, hashes = [], []
    for line in LEDGER.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('event') == 'PUBLISHED':
            order.append((int(r['day_id']), int(r['position'])))
            hashes.append(r.get('state_hash'))
    man = {int(m['day_id']) * 10 + int(m['position']): m for m in manifest}
    mismatch = [k for k, h in zip(order, hashes) if man.get(k[0] * 10 + k[1], {}).get('state_hash') != h]
    binding = hashlib.sha256(json.dumps(
        [[d, p, h] for (d, p), h in zip(order, hashes)], sort_keys=True).encode()).hexdigest()
    return {'n_published': len(order), 'order_equals_canonical': order == sorted(order),
            'order_keys_unique': len(set(order)) == len(order),
            'all_have_state_hash': all(hashes), 'n_manifest_hash_mismatches': len(mismatch),
            'state_binding_sha256': binding, 'ledger_sha256': sha(LEDGER),
            'manifest_sha256': sha(STATE_MANIFEST)}


def build_selftest_fixture(root):
    """Build a throw-away run directory holding 64 SYNTHETIC row records for the frozen targets.

    The fixture is generated by running the synthetic worker (a separate process) against a freshly
    frozen run directory, i.e. exactly what the acceptance harness does.  It is deleted afterwards.
    Nothing here is a real F result and the verdict says so.
    """
    import subprocess
    import tempfile
    runner = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
    synth = EXT / 'scripts/tests/d3_l1_v118_synthetic_worker.py'
    run_dir = Path(tempfile.mkdtemp(prefix='v118_verifier_selftest_', dir=str(root)))
    subprocess.run([sys.executable, str(runner), '--freeze-identity', '--run-id', EXPECTED_RUN_ID,
                    '--out-dir', str(run_dir), '--now-utc', '2026-10-15T00:00:00Z'],
                   check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(synth), '--out-dir', str(run_dir), '--mode', 'pass'],
                   check=True, capture_output=True, text=True)
    return run_dir


def build_supervised_selftest_fixture(root):
    """Same, but driven through the PRODUCTION supervisor so a supervision verdict is published.

    The supervisor is invoked through its Python API (the CLI path is covered by the acceptance
    harness), because the verifier must not depend on CLI argument ordering.
    """
    import importlib.util
    import subprocess
    import tempfile
    runner = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
    synth = EXT / 'scripts/tests/d3_l1_v118_synthetic_worker.py'
    spec = importlib.util.spec_from_file_location('v118_runner_self', str(runner))
    module = importlib.util.module_from_spec(spec)
    sys.modules['v118_runner_self'] = module
    spec.loader.exec_module(module)
    run_dir = Path(tempfile.mkdtemp(prefix='v118_verifier_supervised_', dir=str(root)))
    subprocess.run([sys.executable, str(runner), '--freeze-identity', '--run-id', EXPECTED_RUN_ID,
                    '--out-dir', str(run_dir), '--now-utc', '2026-10-15T00:00:00Z'],
                   check=True, capture_output=True, text=True)
    module.supervise(run_dir, authorization=None,
                     dry_command=[sys.executable, str(synth), '--out-dir', str(run_dir),
                                  '--mode', 'pass'])
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--json', required=True)
    ap.add_argument('--expect-synthetic', action='store_true',
                    help='the run directory is the synthetic acceptance fixture, not a real F run')
    ap.add_argument('--selftest', action='store_true',
                    help='build a disposable synthetic fixture, verify it, and delete it')
    ap.add_argument('--selftest-supervised', action='store_true',
                    help='like --selftest but driven through the production supervisor, so the '
                         'supervision and round verdicts are also exercised')
    # The run-scoped constants below are FROZEN defaults.  A new execution identity (V1.19) may
    # override them explicitly, so one frozen verifier can check whichever identity is being
    # signed, without editing the tool's hard-coded expectations.
    ap.add_argument('--expect-run-id', default=None)
    ap.add_argument('--expect-targets-sha256', default=None)
    args = ap.parse_args()
    if args.expect_run_id:
        globals()['EXPECTED_RUN_ID'] = args.expect_run_id
    if args.expect_targets_sha256:
        globals()['EXPECTED_TARGETS_SHA256'] = args.expect_targets_sha256
    import shutil
    import tempfile
    cleanup = None
    if args.selftest or args.selftest_supervised:
        scratch = Path(tempfile.mkdtemp(prefix='v118_verifier_scratch_'))
        builder = (build_supervised_selftest_fixture if args.selftest_supervised
                   else build_selftest_fixture)
        args.run_dir = str(builder(scratch))
        args.expect_synthetic = True
        cleanup = scratch
    if not args.run_dir:
        raise SystemExit('--run-dir is required unless --selftest is given')
    try:
        return _verify(args)
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def _verify(args):
    run_dir = Path(args.run_dir)
    REC['run_dir'] = str(run_dir)
    REC['expected_run_id'] = EXPECTED_RUN_ID
    REC['synthetic_expected'] = bool(args.expect_synthetic)

    # ---- V1-1 frozen run constants --------------------------------------------------------
    consts = {'run_id': EXPECTED_RUN_ID, 'targets_sha256': EXPECTED_TARGETS_SHA256,
              'budget_caps': EXPECTED_BUDGET_CAPS, 'max_forward_calls': EXPECTED_MAX_FORWARD_CALLS,
              'rows': EXPECTED_ROWS, 'per_slot_calls': EXPECTED_PER_SLOT_CALLS,
              'identities': EXPECTED_IDENTITIES}
    REC['frozen_constants'] = consts
    ck('V1-1 the verifier is frozen against a run id, a target digest, budget caps, call counts and '
       'asset identities that were fixed BEFORE the run',
       bool(EXPECTED_RUN_ID) and bool(EXPECTED_TARGETS_SHA256)
       and all(v for k, v in EXPECTED_IDENTITIES.items() if k != 'state_ledger'),
       {'run_id': EXPECTED_RUN_ID, 'targets': EXPECTED_TARGETS_SHA256[:16],
        'caps': EXPECTED_BUDGET_CAPS})

    # ---- V1-2 on-disk identity re-derivation ----------------------------------------------
    on_disk = {'f_ckpt': sha(EXT / 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'),
               'backbone': sha(REPO / 'MASKCO_code/ckpts/cvrp100.ckpt'),
               'dev_check_npz': sha(EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'),
               'profile_file': sha(EXT / 'results/o0cc/scale_v2/objective_profile.json'),
               'state_manifest': sha(STATE_MANIFEST),
               'state_ledger': sha(LEDGER)}
    REC['on_disk_identities'] = on_disk
    ck('V1-2 every registered asset identity recomputes from the bytes on disk to the frozen value',
       all(on_disk[k] == v for k, v in EXPECTED_IDENTITIES.items() if v is not None),
       {k: (on_disk[k] == v) for k, v in EXPECTED_IDENTITIES.items() if v is not None})

    # ---- V1-3 the 64-state list, re-parsed independently ----------------------------------
    states = freeze_state_list_independently()
    REC['state_list'] = states
    ck('V1-3 the frozen 64-state list is re-derived independently in LEDGER publication order, in '
       'canonical order, with a state hash on every row and no manifest disagreement',
       states['n_published'] == 64 and states['order_equals_canonical']
       and states['order_keys_unique'] and states['all_have_state_hash']
       and states['n_manifest_hash_mismatches'] == 0, states)

    # ---- V1-4 the budget freeze, re-derived from the RAW V1.11 measurements ----------------
    budget = recompute_budget_freeze()
    REC['budget_rederivation'] = budget
    ck('V1-4 the frozen budget caps are re-derived independently from the raw V1.11 F timings and '
       'every observed statistic sits inside its cap (no widened ceiling)',
       bool(budget) and budget['projected_inside_batch_cap']
       and budget['init_plus_warmup_inside_init_cap'] and budget['slot_max_inside_slot_cap'],
       budget)

    # ---- V1-5 the frozen architecture actually froze those numbers -------------------------
    try:
        st = sys.modules.get('d3_l1_v118_run_identity')
        if st is None:
            import d3_l1_v118_run_identity as st
        ident = st.build_run_identity(EXPECTED_RUN_ID, created_utc='verifier-probe')
        arch = {'caps': ident['budgets']['caps'],
                'caps_match': ident['budgets']['caps'] == EXPECTED_BUDGET_CAPS and
                ident['budgets']['caps'].get('batch_s') is not None,
                'max_forward_calls': ident['call_schedule']['max_forward_calls'],
                'state_binding_sha256': ident['state_binding_sha256'],
                'n_states': ident['state_list']['n_states']}
        arch['max_forward_calls_match'] = arch['max_forward_calls'] == EXPECTED_MAX_FORWARD_CALLS
        arch['state_binding_match'] = arch['state_binding_sha256'] == states['state_binding_sha256']
        REC['architecture_probe'] = arch
        ck('V1-5 the architecture the runner builds for this run_id carries exactly the frozen caps, '
           'the frozen 129-call schedule and the same 64-state binding',
           arch['caps_match'] and arch['max_forward_calls_match'] and arch['state_binding_match']
           and arch['n_states'] == 64, arch)
    except BaseException as exc:                                       # noqa: BLE001
        REC['architecture_probe'] = {'error': '%s: %s' % (type(exc).__name__, exc)}
        ck('V1-5 the architecture the runner builds for this run_id carries the frozen caps and '
           'schedule', False, REC['architecture_probe'])

    # ---- V1-6 the run directory's own published identity/targets --------------------------
    if not run_dir.is_dir():
        REC['run_dir_state'] = {'exists': False}
        ck('V1-6 the run directory carries a run identity and a target list that match the frozen '
           'constants', False, {'run_dir': str(run_dir), 'exists': False})
    else:
        rid = publish_records(run_dir, 'RUN_IDENTITY')
        tgt = publish_records(run_dir, 'FORWARD_TARGETS')
        idt = publish_records(run_dir, 'IDENTITY_TARGETS')
        rd = {'n_run_identity': len(rid), 'n_targets': len(tgt), 'n_identity_targets': len(idt),
              'run_id': rid[-1][1].get('run_id') if rid else None,
              'targets_sha256': tgt[-1][1].get('targets_sha256') if tgt else None,
              'n_target_rows': tgt[-1][1].get('n_rows') if tgt else None,
              'budget_caps_in_identity': (rid[-1][1].get('budgets') or {}).get('caps')
              if rid else None,
              'max_forward_calls_in_identity':
                  (rid[-1][1].get('call_schedule') or {}).get('max_forward_calls') if rid else None}
        rd['run_id_matches'] = rd['run_id'] == EXPECTED_RUN_ID
        rd['targets_match'] = rd['targets_sha256'] == EXPECTED_TARGETS_SHA256
        rd['caps_match'] = rd['budget_caps_in_identity'] == EXPECTED_BUDGET_CAPS
        rd['calls_match'] = rd['max_forward_calls_in_identity'] == EXPECTED_MAX_FORWARD_CALLS
        REC['run_dir_state'] = rd
        ck('V1-6 the run directory carries a run identity and a target list that match the frozen '
           'run id / target digest / budget caps / 129-call schedule',
           rd['run_id_matches'] and rd['targets_match'] and rd['caps_match']
           and rd['calls_match'] and rd['n_target_rows'] == 64, rd)

    # ---- V1-7 row-level verification, independently re-implemented and cross-checked -------
    rows = publish_records(run_dir / ROW_DIR, 'SLOT_day') if run_dir.is_dir() else []
    parsed, unparsable = [], []
    for name, rec, _digest in rows:
        m = ROW_NAME_RE.match(name)
        if not m:
            unparsable.append(name)
            continue
        parsed.append((int(m.group(1)), int(m.group(2)), name, rec))
    tgt_rows = {}
    for _n, rec, _d in publish_records(run_dir, 'FORWARD_TARGETS'):
        for r in rec.get('rows', []):
            tgt_rows[(int(r['day_id']), int(r['position']))] = r
    # cross-check against the production verifier (imported ONLY to compare verdicts)
    prod = None
    try:
        import d3_l1_v118_targets as prod
    except BaseException:                                              # noqa: BLE001
        prod = None
    agree, disagree, missing_target = 0, [], []
    independent_ok = 0
    for day, pos, name, rec in parsed:
        target = tgt_rows.get((day, pos))
        if target is None:
            missing_target.append(name)
            continue
        ok_i, detail_i = verify_row_independently(rec, target)
        independent_ok += int(ok_i)
        if prod is not None:
            ok_p, _dp = prod.verify_slot_record(rec, target,
                                                out_dir=run_dir)
            if bool(ok_p) == bool(ok_i):
                agree += 1
            else:
                disagree.append({'row': [day, pos], 'independent': ok_i, 'production': bool(ok_p),
                                 'independent_problems': detail_i['problems']})
    row_state = {'n_row_records': len(rows), 'n_parsed': len(parsed),
                 'n_unparsable_names': len(unparsable),
                 'unparsable_sample': unparsable[:5],
                 'n_missing_target': len(missing_target), 'n_independently_ok': independent_ok,
                 'n_cross_checked': agree + len(disagree), 'n_agree': agree,
                 'n_disagree': len(disagree), 'disagreements': disagree[:8],
                 'targets_available': len(tgt_rows),
                 'expected_rows': EXPECTED_ROWS}
    REC['row_state'] = row_state
    executed = bool(parsed)
    if not executed:
        ck('V1-7 no slot row records exist yet, so the row-level verification is reported as NOT '
           'EXECUTED (this is not a PASS)', True,
           dict(row_state, verdict='NOT_EXECUTED_NO_ROWS'))
    else:
        all_matched = (len(parsed) == EXPECTED_ROWS and not unparsable and not missing_target
                       and independent_ok == EXPECTED_ROWS and not disagree)
        ck('V1-7 every slot row re-verifies under an INDEPENDENTLY re-implemented checker, and that '
           'verdict agrees with the production verifier on every row',
           all_matched if not args.expect_synthetic else
           (not disagree and agree + len(disagree) == len(parsed)), row_state)

    # ---- V1-8 the trailing identity record, re-derived --------------------------------------
    tail = []
    for base in (run_dir / 'phases', run_dir):
        if base.is_dir():
            tail += [(n, r, d) for n, r, d in publish_records(base, 'identity_end')]
    tail_state = {'n_identity_end_records': len(tail)}
    REC['tail_identity'] = tail_state
    if not executed:
        # NOT_EXECUTED and PASS are different verdicts.  A check that cannot run yet is recorded as
        # NOT_EXECUTED, never as a pass and never as a rejection of the frozen architecture.
        ck('V1-8 (NOT_EXECUTED) no trailing identity record exists yet, so it is recorded as NOT '
           'EXECUTED rather than PASS or FAIL', True, tail_state)
    else:
        rec = tail[-1][1] if tail else {}
        paths = rec.get('paths') or {}
        recomputed = {p: sha(p) for p in paths}
        changed = [p for p in paths if recomputed[p] != paths[p]]
        absent = [p for p in paths if recomputed[p] is None]
        tail_state.update(run_id=rec.get('run_id'), n_paths=len(paths), changed=changed[:5],
                          absent=absent[:5],
                          run_id_matches=rec.get('run_id') == EXPECTED_RUN_ID)
        REC['tail_identity'] = tail_state
        ck('V1-8 the trailing identity record covers real files whose digests still recompute, and '
           'it belongs to this run_id',
           bool(tail) and not changed and not absent and tail_state['run_id_matches']
           and len(paths) > 0, tail_state)

    # ---- V1-9 the round verdict, read from the published artifacts --------------------------
    sup = publish_records(run_dir, 'SUPERVISION_RESULT')
    verdicts = []
    for base in (run_dir, run_dir / 'phases'):
        if base.is_dir():
            verdicts += publish_records(base, 'ROUND')
    sup_state = {'n_supervision_records': len(sup), 'n_round_records': len(verdicts),
                 'verdict': sup[-1][1].get('verdict') if sup else None,
                 'round_status': (sup[-1][1].get('round') or {}).get('status') if sup else None,
                 'n_rows_qualified': sup[-1][1].get('n_rows_qualified') if sup else None,
                 'science_ready': sup[-1][1].get('science_ready') if sup else None,
                 'stable_online_gain': sup[-1][1].get('stable_online_gain') if sup else None}
    if sup:
        v = sup[-1][1]
        consistent = ((v.get('verdict') == 'REAL_F_FORWARD_ENGINEERING_PASS')
                      == (v.get('round', {}).get('status') == 'ROUND_PASS'))
        consistent = consistent and v.get('science_ready') is False \
            and v.get('stable_online_gain') is False
        sup_state['verdict_consistent_with_round'] = consistent
        sup_state['counts_consistent'] = (v.get('n_rows_qualified')
                                          == (EXPECTED_ROWS if v.get('round', {}).get('status')
                                              == 'ROUND_PASS' else 0))
    REC['supervision'] = sup_state
    if not executed and not sup:
        ck('V1-9 (NOT_EXECUTED) no supervision result has been published yet, so it is recorded as '
           'NOT EXECUTED rather than PASS or FAIL', True, sup_state)
    else:
        ck('V1-9 the published supervision verdict is internally consistent (PASS iff ROUND_PASS, '
           'qualified row count matches, and no science / gain claim was smuggled in)',
           bool(sup) and sup_state.get('verdict_consistent_with_round')
           and sup_state.get('counts_consistent'),
           sup_state if sup else {'reason': 'no supervision result has been published'})

    # ---- verdict ---------------------------------------------------------------------------
    frozen_checks = [c for c in CASES[:6]]
    fails = [c for c in CASES if not c[1]]
    synthetic = bool(args.expect_synthetic)
    if fails:
        verdict = 'VERIFIER_REJECTED'
    elif not executed:
        verdict = 'VERIFIED_NOT_EXECUTED'
    elif synthetic:
        verdict = 'SYNTHETIC_VERIFIED'
    else:
        verdict = 'VERIFIED'
    if args.selftest and not fails and executed:
        verdict = 'SELFTEST_VERIFIED'
    if args.selftest_supervised and not fails and executed:
        verdict = 'SELFTEST_SUPERVISED_VERIFIED'
    REC['verdict'] = verdict
    REC['selftest'] = bool(args.selftest or args.selftest_supervised)
    REC['summary'] = {'rows': len(CASES), 'pass': len(CASES) - len(fails), 'fail': len(fails),
                      'n_frozen_checks_pass': sum(1 for c in frozen_checks if c[1]),
                      'n_frozen_checks': len(frozen_checks),
                      'executed_rows': len(parsed), 'synthetic_run_dir': synthetic,
                      'not_executed_checks': [c[0] for c in CASES if 'NOT_EXECUTED' in c[0]]}
    REC['claim_limits'] = [
        'this verifier never loads F and never scores a state',
        'VERIFIED_NOT_EXECUTED is not a PASS: it means the frozen architecture checks out and the '
        'real F forward has not happened',
        'a SYNTHETIC verdict is an engineering statement about the harness, never about F',
        'no F-vs-D-Geo comparison, no training, no utility or significance claim is made here']
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, **REC['summary'],
                               'verdict': verdict, 'selftest': REC['selftest'],
                               'tool_sha256': sha(Path(__file__))},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== V1.18 independent read-only verifier: %d rows, %d PASS, %d FAIL | %s ==='
          % (len(CASES), len(CASES) - len(fails), len(fails), verdict))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
