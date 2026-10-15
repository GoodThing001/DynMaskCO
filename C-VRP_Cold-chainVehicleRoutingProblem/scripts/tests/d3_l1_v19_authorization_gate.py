"""Scoped authorization counterexamples. No valid formal worker or rollout is run."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

EXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXT / 'scripts/simulation'))
import d3_l1_collection_runner as R
import d3_l1_p0c_v18 as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    root = Path(ap.parse_args().outdir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for n, data in enumerate(({'n_fail': 0}, {'ok': True})):
        V.write_json(R, root / ('review%d.json' % n), data)
    runtime = root / 'fixture_runtime.json'
    V.write_json(R, runtime, {'synthetic_fixture': True})
    args = SimpleNamespace(authorization=str(root / 'fixture_authorization.json'),
        expect_authorization_sha256=None, runtime_identity=str(runtime), days=16, day_id=0,
        purpose='formal', run_id='SYNTHETIC_TOKEN_ONLY', outdir=str(root / 'not_executed'),
        fault='none', inject_block_s=0., dataset=R.DEFAULT_DATASET, identity=R.DEFAULT_IDENTITY,
        expect_dataset_sha256=R.DEV_CHECK_SHA, day_cap_s=120., slot_cap_s=60.,
        day_rollout_cap_s=120., batch_cap_s=3600.)
    auth = {'authority': 'human_user_conditional_instruction',
            'user_quote': 'SYNTHETIC UNIT FIXTURE, NOT A HUMAN AUTHORIZATION',
            'prior_request_scope': 'synthetic gate test only',
            'authorized_action': 'ONE_F_FREE_16_DAY_4_SLOT_COLLECTION',
            'dataset_sha256': R.DEV_CHECK_SHA, 'parameters': V.FROZEN_PARAMETERS,
            'platform': sys.platform, 'python': sys.version,
            'runtime_identity_sha256': R.sha_file(runtime), 'output_directory': args.outdir,
            'required_reviews': [{'path': 'review0.json', 'sha256': R.sha_file(root / 'review0.json'),
                                  'pass_key': 'n_fail', 'pass_value': 0},
                                 {'path': 'review1.json', 'sha256': R.sha_file(root / 'review1.json'),
                                  'pass_key': 'ok', 'pass_value': True}]}
    V.write_json(R, args.authorization, auth)
    args.expect_authorization_sha256 = R.sha_file(args.authorization)
    rows = []
    def ck(name, ok, detail=None):
        rows.append({'name': name, 'ok': bool(ok), 'detail': detail})
        print(name, bool(ok), flush=True)
    ck('scoped_artifact_structure_positive_unit_only', not V.authorization_errors(R, args))
    for field in ('authority', 'user_quote', 'authorized_action', 'dataset_sha256',
                  'parameters', 'platform', 'python', 'runtime_identity_sha256',
                  'output_directory', 'required_reviews'):
        altered = copy.deepcopy(auth)
        altered[field] = [] if field == 'required_reviews' else {} if field == 'parameters' else ''
        run = copy.copy(args)
        run.authorization = str(root / ('bad_' + field + '.json'))
        V.write_json(R, run.authorization, altered)
        run.expect_authorization_sha256 = R.sha_file(run.authorization)
        errors = V.authorization_errors(R, run)
        ck('reject:' + field, bool(errors), errors)
    run = copy.copy(args)
    run.expect_authorization_sha256 = '0' * 64
    ck('reject:changed_authorization_sha', bool(V.authorization_errors(R, run)))
    for field, value in (('days', 15), ('day_id', 16), ('fault', 'type'), ('inject_block_s', 1.)):
        run = copy.copy(args)
        setattr(run, field, value)
        ck('reject:runtime_' + field, bool(V.authorization_errors(R, run)))
    V.write_json(R, root / 'failed_review.json', {'n_fail': 1})
    failed = copy.deepcopy(auth)
    failed['required_reviews'][0].update(path='failed_review.json',
                                        sha256=R.sha_file(root / 'failed_review.json'))
    run = copy.copy(args)
    run.authorization = str(root / 'review_failed_authorization.json')
    V.write_json(R, run.authorization, failed)
    run.expect_authorization_sha256 = R.sha_file(run.authorization)
    ck('reject:failed_prerequisite_with_valid_sha', bool(V.authorization_errors(R, run)))
    cmd = V.worker_command(R, args, 7, root / 'go')
    ck('worker_inherits_formal_day_range_and_authorization',
       cmd[cmd.index('--days') + 1] == '16' and cmd[cmd.index('--day-id') + 1] == '7'
       and cmd[cmd.index('--authorization') + 1] == args.authorization
       and cmd[cmd.index('--expect-authorization-sha256') + 1] == args.expect_authorization_sha256, cmd)
    V.claim_authorization(R, args, {'runtime_identity_sha256': R.sha_file(runtime)})
    repeated = copy.copy(args)
    repeated.run_id = 'DIFFERENT_RUN_ID'
    try:
        V.claim_authorization(R, repeated, {'runtime_identity_sha256': R.sha_file(runtime)})
        refused = False
    except FileExistsError:
        refused = True
    ck('one_pass:token_refuses_new_run_id', refused)
    # Actual CLI rejection, with real source identity but no authorization. Zero workers.
    actual_runtime = root / 'actual_runtime.json'
    V.write_json(R, actual_runtime, V.execution_identity(R, args))
    out = root / 'actual_cli_without_authorization'
    cmd = [sys.executable, str(R.HERE / 'd3_l1_collection_runner.py'), '--role', 'supervisor',
           '--purpose', 'formal', '--days', '16', '--run-id', 'v19-missing-authorization-negative',
           '--runtime-identity', str(actual_runtime), '--outdir', str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    result = json.loads((out / 'SUPERVISOR_RESULT.json').read_text())
    ck('real_cli_missing_authorization_no_workers', proc.returncode == 2
       and result['verdict'] == 'HARD_STOP_IDENTITY' and result['workers_started'] == 0
       and not list(out.glob('WORKER_DAY*.stdout')), {'command': cmd, 'result': result})
    V.write_json(R, root / 'AUTHORIZATION_GATE.json', {'rows': rows,
        'n_pass': sum(r['ok'] for r in rows), 'n_fail': sum(not r['ok'] for r in rows),
        'formal_workers_started': 0, 'formal_rollouts': 0,
        'classification': 'synthetic_authorization_controls_and_actual_CLI_rejection',
        'argv': sys.argv})
    return int(any(not r['ok'] for r in rows))


if __name__ == '__main__':
    sys.exit(main())
