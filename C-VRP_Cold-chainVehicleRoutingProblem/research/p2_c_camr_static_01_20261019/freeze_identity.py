"""P2-C-CAMR-STATIC-01  T2: run-identity freeze.

Runs AFTER the implementation and AFTER the synthetic regression, and BEFORE the degeneration
gate. Per the authorisation ruling, this file must NOT contain the gate result: the gate has not
run yet. It records

  * the sha256 of the T1 registration it binds to
  * the sha256 of every source file listed at T1 (the list is frozen at T1, the hashes here)
  * the synthetic-regression receipt (sha256 + pass/fail) it binds to
  * the loaded model identity, i.e. the actual parameter digest read out of the checkpoint,
    not just the file hash
  * the interpreter / library versions and the frozen runtime constants

FAIL CLOSED: if any file is missing, any hash is unexpected, the synthetic regression did not
pass, or the model cannot be loaded with the declared identity, nothing is written and the exit
code is non-zero. A source repair requires a fresh T2 with every prerequisite check re-run.

Usage: python freeze_identity.py [--out run_identity.json]
"""
import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import camr_shared as cs                                            # noqa: E402

CHECKPOINT = 'MASKCO_code/ckpts/cvrp100.ckpt'


def git(*args):
    r = subprocess.run(['git'] + list(args), cwd=str(WORKSPACE), capture_output=True)
    if r.returncode != 0:
        raise SystemExit('git %s failed: %s' % (' '.join(args),
                                                r.stderr.decode('utf-8', 'replace')))
    return r.stdout.decode('utf-8', 'replace').strip()


WORKSPACE = cs.WORKSPACE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'run_identity.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    reg = cs.load_json(OUT / 'registration.json')
    failures = []

    # ---- T1 binding -------------------------------------------------------
    reg_sha = cs.sha_file(OUT / 'registration.json')
    head = git('rev-parse', 'HEAD')
    if head != reg['baseline_commit']:
        failures.append('HEAD %s != registration baseline %s' % (head, reg['baseline_commit']))

    # ---- source identity --------------------------------------------------
    source = {}
    for rel in reg['source_file_list']['files']:
        p = WORKSPACE / rel
        source[rel] = cs.sha_file(p) if p.is_file() else None
        if source[rel] is None:
            failures.append('missing source file: %s' % rel)

    # ---- synthetic regression binding ------------------------------------
    syn_path = OUT / 'synthetic_regression.json'
    syn = cs.load_json(syn_path)
    syn_sha = cs.sha_file(syn_path)
    if not syn.get('all_ok'):
        failures.append('synthetic regression did not pass')
    if syn.get('model_loaded') is not False:
        failures.append('synthetic regression claims a model was loaded')

    # ---- model identity (last, and only if the rest is clean) ------------
    manifest = None
    if not failures:
        from p2_pos01_mask_value import FrozenBackbone          # CPU-only import
        ckpt = WORKSPACE / CHECKPOINT
        try:
            bb = FrozenBackbone(str(ckpt))
            manifest = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in bb.manifest.items()}
        except Exception as exc:                                # noqa: BLE001
            failures.append('MODEL_LOAD_FAILED: %s: %s' % (type(exc).__name__, exc))
        else:
            want = reg['model_identity_to_freeze_at_T2']
            if manifest['checkpoint_sha256'] != want['checkpoint_sha256']:
                failures.append('checkpoint sha256 %s != registered %s'
                                % (manifest['checkpoint_sha256'], want['checkpoint_sha256']))
            if manifest['encoder_dim'] != 512 or manifest['decoder_dim'] != 512:
                failures.append('unexpected model dims %s/%s'
                                % (manifest['encoder_dim'], manifest['decoder_dim']))
            if manifest['input_dim'] != 3:
                failures.append('unexpected encoder input dim %s' % manifest['input_dim'])
            if manifest['platform'] != 'cpu':
                failures.append('unexpected platform %s' % manifest['platform'])

    if failures:
        print(json.dumps({'T2': 'FAILED', 'failures': failures}, ensure_ascii=False, indent=2))
        sys.exit(4)

    import jax
    import numpy as np
    import flax

    payload = {
        'schema': 'p2_c_camr_static_01_run_identity_v1',
        'run_id': reg['run_id'],
        'plan_version': reg['plan_version'],
        'T1': {'registration_sha256': reg_sha,
               'units_sha256': reg['units']['sha256'],
               'n_units': reg['units']['n']},
        'baseline_commit': head,
        'source_files': source,
        'synthetic_regression': {'sha256': syn_sha,
                                 'n_checks': syn['n_checks'],
                                 'n_ok': syn['n_ok'],
                                 'all_ok': syn['all_ok'],
                                 'model_loaded': syn['model_loaded']},
        'model': {'checkpoint': CHECKPOINT,
                  'checkpoint_bytes': (WORKSPACE / CHECKPOINT).stat().st_size,
                  'manifest': manifest,
                  'scoring_entry': 'FrozenBackbone.insertion_scores(view, base, partial, actions)'},
        'runtime': {'python': sys.version.split()[0],
                    'platform': platform.platform(),
                    'numpy': np.__version__, 'jax': jax.__version__, 'flax': flax.__version__,
                    'jax_devices': [str(d) for d in jax.devices()],
                    'constants': reg['resources']},
        'degeneration_gate': ('NOT YET RUN. The gate receipt lives in preflight.json and binds to '
                              'this file by run_identity_sha256; it is deliberately not recorded '
                              'here, because T2 freezes before the gate executes.'),
        'gate_result_included': False,
    }
    cs.write_json(a.out, payload)
    print('T2 frozen')
    print(' registration_sha256', reg_sha)
    print(' sources', len(source))
    print(' synthetic_regression', syn_sha[:16], 'ok', syn['all_ok'])
    print(' parameter_sha256', manifest['parameter_sha256'])
    print(' wrote', a.out)


if __name__ == '__main__':
    main()
