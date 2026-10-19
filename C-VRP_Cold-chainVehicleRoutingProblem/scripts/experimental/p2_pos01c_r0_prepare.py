"""P2-C-R0 only: fresh TRAIN identities and reference energy-scale calibration.

No MaskCO, SAA controller, counterfactual labels, utility comparison or training.
Use freeze first, inspect the registration, then calibrate. Never overwrites assets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT, WORKSPACE_ROOT
for sub in ('evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(SCRIPTS / sub))
from run_exp_reserve import generate_dataset
from run_exp_energy_c0 import C0ReserveReplanner, c0_energy_from_traces
from scenario_saa import make_c0_contract_v2
from coldchain_evaluator_a1 import add_v2_initial_quality
from strict_online_env import StrictOnlineEnv

VERSION = 'p2_pos01c_r0_20261019_v1'
CONFIG = EXTENSION_ROOT / 'configs' / VERSION
DATA = EXTENSION_ROOT / 'data' / VERSION
BASE_FIELDS = ('coords', 'demands', 'tw_start', 'tw_end', 'service_time',
               'reveal_time', 'temp_class')
SOURCE_PATHS = (
    'scripts/experimental/p2_pos01c_r0_prepare.py', 'scripts/project_paths.py',
    'scripts/evaluation/run_exp_reserve.py', 'scripts/evaluation/run_exp_energy_c0.py',
    'scripts/evaluation/scenario_saa.py', 'scripts/evaluation/coldchain_evaluator_a1.py',
    'scripts/simulation/strict_online_env.py', 'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(data, fields=None):
    h = hashlib.sha256()
    for key in sorted(data if fields is None else fields):
        a = np.ascontiguousarray(data[key])
        h.update(str((key, a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def source_hashes():
    return {p: file_hash(EXTENSION_ROOT / p) for p in SOURCE_PATHS}


def freeze():
    if CONFIG.exists() or DATA.exists():
        raise RuntimeError('Existing R0 identity: never regenerate or overwrite')
    # Independent root RNG invocation per day, no shared PRNG parent across roles.
    days = []
    for role, count in (('BUDGET_TRAIN_CAL', 8), ('HIST_TRAIN', 16), ('SOURCE_TRAIN', 2)):
        for ordinal in range(count):
            day_id = '%s/%s/%02d' % (VERSION, role, ordinal)
            seed = int.from_bytes(hashlib.sha256(day_id.encode()).digest()[:4], 'big')
            days.append({'day_id': day_id, 'role': role, 'root_seed': seed,
                         'generator_call': 'generate_dataset(1, 50, root_seed)',
                         'instance_index': 0, 'derivation_parent': None})
    seeds = {d['root_seed'] for d in days}
    assert len(seeds) == len(days)
    excluded = {20260925, 20260926, 20260927, 20260928, 20260930, 20261001}
    checked = []
    # Bounded historical seed declarations, not an exhaustive hidden-asset claim.
    for folder, pattern in ((SCRIPTS, '*.py'), (EXTENSION_ROOT / 'configs', '*.json')):
        for p in sorted(folder.rglob(pattern)):
            if p == Path(__file__).resolve():
                continue
            checked.append({'path': p.relative_to(EXTENSION_ROOT).as_posix(), 'sha256': file_hash(p)})
            text = p.read_text(encoding='utf-8-sig', errors='strict')
            excluded.update(int(x) for x in re.findall(r'\b\d{8,10}\b', text))
    assert not seeds & excluded, 'Registered seed collision: stop, do not reseed'
    contract = make_c0_contract_v2()
    registration = {
        'schema': VERSION, 'scope': 'REFERENCE_ONLY_PREPARATION_NOT_SMOKE',
        'user_ruling': 'rho=0.60; independent TRAIN calibration; review B_ref before smoke',
        'baseline_head': '39b517a3e3c6707d673181141647b0ddb5349ce2',
        'rho': 0.60, 'budget_formula': '(1-rho)*arithmetic_mean(all_accept_C0_energy_kwh)',
        'operating_candidate': {'orders': 50, 'capacity': 50.0, 'num_vehicles': 15,
            'booking_horizon_h': 16.0, 'tw_speed_units_per_h': 1.5,
            'units': {'distance_km_per_unit': 20., 'hours_per_time_unit': 1., 'speed_kmph': 30.},
            'rejection_penalty_candidate': [5., 10., 15.],
            'selection_note': '15 vehicles is an explicit R0 reference candidate from the existing calibration driver; not a P2-B fixture inheritance or formal fleet ruling'},
        'calibration_policy': {'class': 'C0ReserveReplanner', 'mode': 'myopic',
            'energy_budget': 'UNBOUNDED_SCALE_CALIBRATION_ONLY', 'cooling_share_placeholder': 2.,
            'history': None, 'k_unused': 10,
            'acceptance_condition': 'all active orders accepted and served exactly once on every calibration day; otherwise STOP, no deletion or replacement',
            'capacity_domain': 'capacity-1e-4',
            'failure_rule': 'first exception, missing trace, nonfinite energy, rejected/unserved order => BLOCKED; retain attempted and unattempted days'},
        'contract_hash': contract.contract_hash, 'contract_manifest': asdict(contract),
        'contract_json_note': 'Nonfinite contract constants use explicit nonfinite_float envelopes; native contract_hash is computed by the unchanged factory',
        'source_sha256': source_hashes(), 'days': days,
        'seed_exclusion': {'scanned': checked, 'excluded_integer_count': len(excluded),
                           'collisions': [], 'scope': 'current scripts/*.py recursively and configs/*.json recursively, plus sealed/viewed seed list'},
        'resource_plan': {'cpu_workers': 1, 'blas_threads': 1, 'calibration_day_slots': 8,
                          'day_timeout_s': 30, 'total_hard_limit_s': 300},
        'no_source_day_execution': True, 'no_HIST_execution': True,
        'environment': {'python': sys.version, 'numpy': np.__version__,
                        'platform': platform.platform(), 'machine': platform.machine(),
                        'executable': sys.executable, 'cpu_count': os.cpu_count()}}
    write_json(CONFIG / 'registration.json', registration)  # before data or energy observed
    DATA.mkdir(parents=True)
    for d in days:
        ds = add_v2_initial_quality(generate_dataset(1, 50, d['root_seed']), contract)
        assert all(np.isfinite(a).all() for a in ds.values())
        assert ds['coords'].shape == (1, 51, 2)
        path = DATA / ('%s_%02d.npz' % (d['role'], int(d['day_id'].split('/')[-1])))
        with path.open('xb') as f:
            np.savez_compressed(f, **ds)
    seal_identities(registration)


def seal_identities(registration):
    """Seal already generated files; never selects roots or regenerates data."""
    effective = source_hashes()
    changed = [p for p, h in effective.items() if h != registration['source_sha256'][p]]
    if changed:
        attempt = json.loads((CONFIG / 'freeze_attempt1.json').read_text(encoding='utf-8'))
        assert changed == ['scripts/experimental/p2_pos01c_r0_prepare.py']
        assert attempt['all_matches_inside_new_R0'] and attempt['energy_calibrations_started'] == 0
        write_json(CONFIG / 'preparation_amendment.json', {
            'reason': 'Windows case-insensitive TRAIN glob included the newly generated R0 directory; explicitly exclude this directory from historical comparison',
            'original_registration_sha256': file_hash(CONFIG / 'registration.json'),
            'changed_source': changed, 'effective_source_sha256': effective,
            'roots_data_formula_and_operating_candidate_unchanged': True,
            'energy_calibrations_started': 0})
    identities = []
    contract = make_c0_contract_v2()
    for d in registration['days']:
        path = DATA / ('%s_%02d.npz' % (d['role'], int(d['day_id'].split('/')[-1])))
        with np.load(path, allow_pickle=False) as z:
            ds = {k: z[k] for k in z.files}
        expected = add_v2_initial_quality(generate_dataset(1, 50, d['root_seed']), contract)
        assert digest(ds) == digest(expected), 'Generated root differs from registration'
        identities.append(dict(d, relative_path=path.relative_to(EXTENSION_ROOT).as_posix(),
            file_sha256=file_hash(path), dataset_sha256=digest(ds),
            root_content_sha256=digest(ds, BASE_FIELDS), bytes=path.stat().st_size))
    assert len({d['root_content_sha256'] for d in identities}) == len(registration['days'])
    # Exact feature-content exclusion against locally available historical TRAIN NPZs.
    legacy = []
    wanted = {d['root_content_sha256'] for d in identities}
    for p in sorted((EXTENSION_ROOT / 'data').rglob('*train*.npz')):
        if p.is_relative_to(DATA):
            continue  # New R0 roots are compared pairwise above, never against themselves.
        with np.load(p, allow_pickle=False) as z:
            if not all(k in z.files for k in BASE_FIELDS) or z['coords'].shape[1:] != (51, 2):
                legacy.append({'path': p.relative_to(EXTENSION_ROOT).as_posix(), 'comparison': 'different_shape_or_schema'})
                continue
            arr = {k: z[k] for k in BASE_FIELDS}
            hashes = [digest({k: a[i:i+1] for k, a in arr.items()}, BASE_FIELDS)
                      for i in range(len(arr['coords']))]
            assert not wanted & set(hashes), 'Historical content overlap: stop'
            legacy.append({'path': p.relative_to(EXTENSION_ROOT).as_posix(),
                           'file_sha256': file_hash(p), 'instances_compared': len(hashes), 'collisions': 0})
    write_json(CONFIG / 'data_identity.json', {'registration_sha256': file_hash(CONFIG / 'registration.json'),
        'effective_source_sha256': effective,
        'preparation_amendment_sha256': file_hash(CONFIG / 'preparation_amendment.json') if changed else None,
        'days': identities, 'pairwise_root_seed_intersections': 0,
        'pairwise_root_content_intersections': 0, 'historical_TRAIN_comparisons': legacy,
        'availability': 'NPZs retained locally under extension data/; reproduce with this frozen source and environment; no remote asset availability claim'})
    print(json.dumps({'status': 'IDENTITIES_FROZEN', 'days': len(identities), 'bytes': sum(d['bytes'] for d in identities)}))


def verify():
    reg = json.loads((CONFIG / 'registration.json').read_text(encoding='utf-8'))
    manifest = json.loads((CONFIG / 'data_identity.json').read_text(encoding='utf-8'))
    assert source_hashes() == manifest['effective_source_sha256'], 'Source drift'
    changed = [p for p, h in source_hashes().items() if h != reg['source_sha256'][p]]
    if changed:
        amendment = CONFIG / 'preparation_amendment.json'
        assert file_hash(amendment) == manifest['preparation_amendment_sha256']
        assert changed == ['scripts/experimental/p2_pos01c_r0_prepare.py']
        assert json.loads(amendment.read_text(encoding='utf-8'))['effective_source_sha256'] == source_hashes()
    assert make_c0_contract_v2().contract_hash == reg['contract_hash'], 'Contract drift'
    assert file_hash(CONFIG / 'registration.json') == manifest['registration_sha256']
    for d in manifest['days']:
        p = EXTENSION_ROOT / d['relative_path']
        assert file_hash(p) == d['file_sha256']
        with np.load(p, allow_pickle=False) as z:
            assert digest({k: z[k] for k in z.files}) == d['dataset_sha256']
    return reg, manifest


def calibrate():
    import multiprocessing as mp
    reg, manifest = verify()
    result_path = CONFIG / 'calibration.json'
    if result_path.exists():
        raise RuntimeError('Calibration already attempted: no retries')
    rows = [dict(day_id=d['day_id'], status='NOT_ATTEMPTED', energy_kwh=None)
            for d in manifest['days'] if d['role'] == 'BUDGET_TRAIN_CAL']
    start = time.monotonic()
    ctx = mp.get_context('spawn')
    for row in rows:
        descriptor = next(d for d in manifest['days'] if d['day_id'] == row['day_id'])
        parent, child = ctx.Pipe(duplex=False)
        worker = ctx.Process(target=calibration_worker, args=(child, descriptor, reg))
        worker.start()
        child.close()
        try:
            if parent.poll(min(30., max(0., 300.-(time.monotonic()-start)))):
                row.update(parent.recv())
            else:
                row.update(status='TIMEOUT')
        except Exception as exc:
            row.update(status='EXCEPTION', error=type(exc).__name__+': '+str(exc))
        finally:
            worker.join(.2)
            if worker.is_alive():
                worker.terminate()
                worker.join(2.)
            if worker.is_alive():
                worker.kill()
                worker.join(2.)
            parent.close()
        if row['status'] != 'OK':
            break
    ok = all(r['status'] == 'OK' for r in rows)
    mean = math.fsum(r['energy_kwh'] for r in rows)/len(rows) if ok else None
    orders = sum(r.get('served_count', 0) for r in rows)
    vehicles = sum(r.get('dispatched_vehicles', 0) for r in rows)
    contract = make_c0_contract_v2()
    cooling = sum(contract.thermal.cooling_power_kw)*16./max(orders/max(vehicles, 1), 1) if ok else None
    result = {'qualification': 'REFERENCE_ONLY', 'status': 'R0_PREPARED_REVIEW_REQUIRED' if ok else 'BLOCKED',
        'registration_sha256': file_hash(CONFIG / 'registration.json'),
        'data_identity_sha256': file_hash(CONFIG / 'data_identity.json'),
        'rho': .60, 'mean_all_accept_energy_kwh': mean, 'B_ref_kwh': .40*mean if ok else None,
        'cooling_share_candidate_kwh_per_order': cooling, 'calibration_rows': rows,
        'elapsed_s': time.monotonic()-start, 'source_days_executed': 0, 'HIST_days_executed': 0,
        'counterfactual_rollouts': 0, 'labels': 0, 'smoke_authorised': False,
        'forbidden_model_modules_loaded': [m for m in ('jax', 'flax', 'torch', 'mpre', 'p2_pos01_teacher') if m in sys.modules]}
    assert not result['forbidden_model_modules_loaded']
    verify()
    write_json(result_path, result)
    print(json.dumps({k: result[k] for k in ('status', 'B_ref_kwh', 'mean_all_accept_energy_kwh', 'elapsed_s')}))


def calibration_worker(conn, descriptor, reg):
    started = time.monotonic()
    try:
        contract = make_c0_contract_v2()
        with np.load(EXTENSION_ROOT / descriptor['relative_path'], allow_pickle=False) as z:
            ds = {k: z[k] for k in z.files}
        assert digest(ds) == descriptor['dataset_sha256']
        policy = C0ReserveReplanner(float('inf'), 50., 16., contract, 2., 'myopic', None, 10)
        env = StrictOnlineEnv(ds, 50., 1.5, num_vehicles=15, replanner=policy,
                             coldchain_contract=contract, booking_horizon=16.)
        traces, served = env.run(0)
        universe = set(np.flatnonzero(ds['demands'][0] > 0).tolist())
        sequence = [s.node for t in traces for s in t.services]
        energy = c0_energy_from_traces(traces)
        complete = (policy._accepted == universe and not policy._rejected and
                    set(sequence) == universe and len(sequence) == len(universe) and
                    all(t.final_coldchain_state is not None and t.final_coldchain_state.closed
                        for t in traces if t.dispatch_time is not None))
        conn.send({'status': 'OK' if complete and math.isfinite(energy) and energy > 0 else 'NOT_ALL_ACCEPT',
            'energy_kwh': energy, 'active_orders': len(universe), 'accepted_ids': sorted(policy._accepted),
            'rejected_ids': sorted(policy._rejected), 'served_ids': sequence, 'served_count': len(sequence),
            'dispatched_vehicles': sum(t.dispatch_time is not None for t in traces),
            'elapsed_s': time.monotonic()-started})
    except Exception as exc:
        conn.send({'status': 'EXCEPTION', 'energy_kwh': None,
                   'error': type(exc).__name__+': '+str(exc), 'elapsed_s': time.monotonic()-started})
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'seal', 'verify', 'calibrate'))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'seal':
        seal_identities(json.loads((CONFIG / 'registration.json').read_text(encoding='utf-8')))
    elif args.mode == 'calibrate':
        calibrate()
    else:
        verify()
        print('FROZEN_IDENTITIES_VERIFIED')
