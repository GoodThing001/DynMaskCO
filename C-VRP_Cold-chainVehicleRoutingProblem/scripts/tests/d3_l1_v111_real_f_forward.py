"""V1.12 persistence repair of the failed V1.11 harness. No second F execution authorized here.

Original V1.11 bytes are archived in v111_evidence/PREFAILURE_SOURCE_BYTES_V111.zip.
The revised harness remains unexecuted on F; only its immutable phase primitive is stress-tested.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
REPO = EXT.parent
ROOT = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E110 = ROOT / 'v110_evidence'
OUT = ROOT / 'v112_forward_evidence'
MODEL = EXT / 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'
BACKBONE = REPO / 'MASKCO_code/ckpts/cvrp100.ckpt'
LABEL = 'F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED'
for part in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/models',
             'scripts/tests', 'scripts/evaluation'):
    sys.path.insert(0, str(EXT / part))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True).encode()
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'wb') as fh:
        fh.write(raw);fh.flush();os.fsync(fh.fileno())
    os.replace(temp, path)


def stage(name, **extra):
    from d3_l1_v112_phase_journal import PhaseJournal
    global _phase_journal
    if '_phase_journal' not in globals():
        _phase_journal = PhaseJournal(OUT / 'phases')
    _phase_journal.publish(name, **extra)


def worker():
    while not (OUT / 'GO').exists():
        time.sleep(.01)
    started = time.monotonic()
    stage('initialize')
    try:
        import numpy as np
        import jax
        from flax import nnx
        from mtrained_replanner import load_trained_model, mtrained_score_fn
        from mpre_trained import partition
        from d3_l1_v18_readonly_audit import typed_bytes
        assert jax.default_backend() == 'cpu'
        expected = json.loads((OUT / 'IDENTITY_START.json').read_text())
        assert sha(MODEL) == expected['model_sha256'] and sha(BACKBONE) == expected['backbone_sha256']
        meta = pickle.loads(MODEL.read_bytes())
        assert {k: meta[k] for k in ('seed', 'num_steps', 'batch_states', 'K', 'F_NODE', 'F_ACTION_EXPLICIT')} == {
            'seed': 42, 'num_steps': 1000, 'batch_states': 8, 'K': 4, 'F_NODE': 10, 'F_ACTION_EXPLICIT': 20}
        model, _ = load_trained_model(str(BACKBONE), str(MODEL), seed=42)
        _, train, frozen, other = partition(model)
        trained = jax.tree_util.tree_flatten_with_path(train)[0]
        reference = jax.tree_util.tree_flatten_with_path(meta['train_state'])[0]
        equal = len(trained) == len(reference) == 83 and all(
            str(p) == str(q) and typed_bytes(np.asarray(x)) == typed_bytes(np.asarray(y))
            for (p, x), (q, y) in zip(trained, reference))
        nonzero = {}
        for group in ('c_adapter', 'delta_head'):
            arrays = [np.asarray(x) for p, x in trained if group in str(p)]
            nonzero[group] = {'leaves': len(arrays), 'absmax': float(max(np.max(np.abs(a)) for a in arrays))}
        load = {'label': LABEL, 'train_leaves': len(trained), 'restored_exact': equal,
                'other_param_leaves': len(jax.tree_util.tree_leaves(other)), 'groups': nonzero,
                'frozen_leaves': len(jax.tree_util.tree_leaves(frozen)),
                'initialization_wall_s': time.monotonic() - started, 'backend': jax.default_backend(),
                'devices': [str(d) for d in jax.devices()],
                's_TRAIN': float(meta['s_TRAIN']), 'meta': {k: meta[k] for k in
                    ('seed', 'num_steps', 'batch_states', 'K', 'F_NODE', 'F_ACTION_EXPLICIT')}}
        write(OUT / 'MODEL_LOAD.json', load)
        assert equal and load['other_param_leaves'] == 0 and all(v['leaves'] == 6 and v['absmax'] > 0 for v in nonzero.values())
        report = json.loads((E110 / 'INPUT_QUALIFICATION_V110.json').read_text())
        assert report['input_checks_pass'] == 64 and report['input_checks_failed_or_missing'] == 0
        score = mtrained_score_fn(model, Nv_max=30, M_max=26)
        first = report['rows'][0]
        first_payload = pickle.loads((E110 / first['derived_path']).read_bytes())
        stage('compile')
        warm_start = time.monotonic()
        warm = score(first_payload['original']['inputs'])
        write(OUT / 'WARMUP.json', {'input_sha256': first['input_sha256'], 'calls': 1,
              'wall_s': time.monotonic() - warm_start, 'scores_dtype': warm.dtype.str,
              'scores_shape': list(warm.shape), 'scores_hex': warm.tobytes().hex()})
        write(OUT / 'READY.json', {'mono': time.monotonic(), 'initialize_and_compile_s': time.monotonic() - started})
        for row in report['rows']:
            day, pos = row['day_id'], row['position']
            stage('slot', day_id=day, position=pos)
            t0 = time.monotonic()
            raw_path = E110 / row['derived_path']
            assert sha(raw_path) == row['derived_sha256']
            payload = pickle.loads(raw_path.read_bytes())
            inputs = payload['original']['inputs']
            assert hashlib.sha256(typed_bytes(inputs)).hexdigest() == row['input_sha256']
            t1 = time.monotonic();a = score(inputs);s1 = time.monotonic() - t1
            t2 = time.monotonic();b = score(inputs);s2 = time.monotonic() - t2
            expected_shape = (len(payload['original']['legal']),)
            checks = {'shape_matches_ordered_legal': a.shape == b.shape == expected_shape,
                      'all_finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                      'repeat_exact_type_dtype_shape_bytes': typed_bytes(a) == typed_bytes(b),
                      'span_le_10s': time.monotonic() - t0 <= 10.,
                      'first_argmax_unchanged': bool(np.isfinite(a).all()) and a.size > 0
                          and int(np.argmax(a)) == int(np.flatnonzero(a == a.max())[0])}
            raw = OUT / ('scores/day%02d_pos%d.pkl' % (day, pos))
            with open(raw, 'xb') as fh:
                pickle.dump({'scores_first': a, 'scores_repeat': b, 'input_sha256': row['input_sha256'],
                             'legal_sha256': row['legal_sha256'], 'label': LABEL}, fh, protocol=4)
                fh.flush();os.fsync(fh.fileno())
            record = {'day_id': day, 'position': pos, 'status': 'FORWARD_CHECKS_PASS' if all(checks.values()) else 'FAILED_CHECKS',
                      'checks': checks, 'raw_path': str(raw.relative_to(OUT)), 'raw_sha256': sha(raw),
                      'input_sha256': row['input_sha256'], 'legal_sha256': row['legal_sha256'],
                      'scores_dtype': a.dtype.str, 'scores_shape': list(a.shape),
                      'scores_first_hex': a.tobytes().hex(), 'scores_repeat_hex': b.tobytes().hex(),
                      'forward_first_wall_s': s1, 'forward_repeat_wall_s': s2, 'span_wall_s': time.monotonic() - t0,
                      'label': LABEL, 'F_calls': 2}
            write(OUT / ('scores/day%02d_pos%d.json' % (day, pos)), record)
            if not all(checks.values()):
                raise RuntimeError('fixed-input forward checks failed; no retry')
        inventory = {}
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, '__file__', None)
            if not f:
                continue
            p = Path(f).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(REPO):
                key = p.relative_to(REPO).as_posix()
                inventory[name] = {'path': key, 'sha256_at_start': expected['source_files'].get(key),
                    'sha256_at_end': sha(p), 'origin': str(p),
                    'spec_origin': getattr(getattr(mod, '__spec__', None), 'origin', None)}
        write(OUT / 'RUNTIME_ORIGINS.json', inventory)
        assert all(r['sha256_at_start'] == r['sha256_at_end'] for r in inventory.values())
        assert sha(MODEL) == expected['model_sha256'] and sha(BACKBONE) == expected['backbone_sha256']
        stage('completed')
        return 0
    except Exception as exc:
        from d3_l1_v112_phase_journal import latest
        write(OUT / 'WORKER_FAILURE.json', {'type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'stage': latest(OUT / 'phases')})
        return 1


def supervise():
    OUT.mkdir(exist_ok=False)
    (OUT / 'scores').mkdir()
    assert sha(MODEL) == 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5'
    assert sha(BACKBONE) == '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
    source_files = {p.relative_to(REPO).as_posix(): sha(p) for base in (EXT / 'scripts', REPO / 'MASKCO_code')
                    for p in base.rglob('*.py') if '__pycache__' not in p.parts}
    write(OUT / 'IDENTITY_START.json', {'source_files': source_files, 'model_sha256': sha(MODEL),
        'backbone_sha256': sha(BACKBONE), 'label': LABEL,
        'prereg_sha256': sha(ROOT / 'A1_D3_L1_REAL_F_FORWARD_QUALIFICATION_PREREGISTRATION_V1_11.md'),
        'input_independent_audit_sha256': sha(E110 / 'INDEPENDENT_READONLY_AUDIT_V110.json'),
        'resources_sha256': sha(E110 / 'RESOURCES_BEFORE_F.json')})
    from d3_l1_p0c_v18 import ProcessTree
    from d3_l1_v112_phase_journal import latest
    env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONIOENCODING='utf-8')
    start = time.monotonic();reason = None;cleanup = None
    with open(OUT / 'WORKER.stdout', 'wb') as stdout, open(OUT / 'WORKER.stderr', 'wb') as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
            stdout=stdout, stderr=stderr, env=env, start_new_session=(os.name != 'nt'))
        tree = ProcessTree(process)
        try:
            (OUT / 'GO').write_bytes(b'contained')
            while process.poll() is None:
                now = time.monotonic()
                try:
                    current = latest(OUT / 'phases')
                except (OSError, ValueError):
                    reason = 'phase_journal_unreadable_or_corrupt'
                    cleanup = tree.terminate()
                    break
                if now - start > 180.:
                    reason = 'batch_deadline_180s'
                elif not (OUT / 'READY.json').exists() and now - start > 60.:
                    reason = 'initialization_compile_deadline_60s'
                elif current.get('stage') == 'slot' and now - current['mono'] > 10.:
                    reason = 'slot_span_deadline_10s'
                if reason:
                    cleanup = tree.terminate();break
                time.sleep(.05)
            process.wait(timeout=5)
            active_after = tree.active()
        finally:
            if process.poll() is None:
                cleanup = tree.terminate()
            tree.close()
    rows = []
    for day in range(16):
        for pos in range(1, 5):
            path = OUT / ('scores/day%02d_pos%d.json' % (day, pos))
            rows.append(json.loads(path.read_text()) if path.exists() else {'day_id': day, 'position': pos,
                'status': 'NOT_COMPLETED_AFTER_FAILURE_OR_TIMEOUT', 'reason': reason or 'worker_error'})
    passed = sum(r['status'] == 'FORWARD_CHECKS_PASS' for r in rows)
    report = {'rows': rows, 'denominator': 64, 'pass': passed, 'failed_or_missing': 64 - passed,
        'returncode': process.returncode, 'pid': process.pid, 'cleanup': cleanup, 'tree_active_after': active_after,
        'termination_reason': reason, 'batch_wall_s_before_final_result_write': time.monotonic() - start,
        'label': LABEL, 'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'F_D_Geo_comparisons': 0, 'install_calls': 0, 'training_runs': 0, 'new_rollouts': 0,
        'recorded_F_forward_calls': sum(r.get('F_calls', 0) for r in rows) + int((OUT / 'WARMUP.json').exists()),
        'verdict': 'REAL_INPUT_F_INTERFACE_CHECKS_PASS' if passed == 64 and process.returncode == 0 and active_after == 0
                   else 'REAL_INPUT_F_INTERFACE_BLOCKED'}
    write(OUT / 'REAL_F_FORWARD_QUALIFICATION_V111.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    return int(report['verdict'] != 'REAL_INPUT_F_INTERFACE_CHECKS_PASS')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        sys.exit(worker())
    raise SystemExit('HOLD: V1.11 failed. Revised harness requires a separately frozen execution scope; no automatic retry.')
