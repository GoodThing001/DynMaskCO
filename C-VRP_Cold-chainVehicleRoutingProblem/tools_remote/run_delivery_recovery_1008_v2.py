"""Engineering QA then the identical fresh six-arm pilot in a new frozen tree."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    project = Path(__file__).resolve().parents[1]
    old = project / 'results/delivery_pilot_1008_v1'
    out = project / 'results/delivery_pilot_1008_v2'
    audit = json.loads((project / 'results/delivery_provider_audit_1008_v2/report.json').read_text())
    assert audit['old_reserve_alone_blocks_10s_loop']
    assert audit['checkpoint_is_not_original_config_initialization']
    old_manifest = json.loads((old / 'SOURCE_MANIFEST.json').read_text())
    assert all(sha(old / 'src' / p) == h for p, h in old_manifest.items())
    old_result = json.loads((old / 'pilot/result.json').read_text())
    assert len(old_result['rows']) == 32
    assert all(row['arms'][arm]['cycles'] == 0 for row in old_result['rows']
               for arm in ['maskco_single', 'maskco_iterative', 'maskco_random_single'])
    out.mkdir(exist_ok=False)
    root = out / 'src'
    ext = root / 'C-VRP_Cold-chainVehicleRoutingProblem'
    ext.mkdir(parents=True)
    with zipfile.ZipFile(project / 'tools_remote/delivery_pilot_1008_v2.zip') as archive:
        for name in archive.namelist():
            if ext.resolve() not in (ext / name).resolve().parents:
                raise ValueError('Unsafe archive path')
        archive.extractall(ext)
    expected = json.loads((ext / 'SOURCE_MANIFEST.json').read_text())
    assert all(sha(ext / p) == h for p, h in expected.items())
    upstream = root / 'MASKCO_code'
    # Reuse trusted original source bytes, never edit the original repository.
    for p in (old / 'src/MASKCO_code').rglob('*.py'):
        q = upstream / p.relative_to(old / 'src/MASKCO_code')
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
    manifest = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
    dump(out / 'SOURCE_MANIFEST.json', manifest)
    cp = project.parent / 'MASKCO_code/ckpts/cvrp100.ckpt'
    cp_sha = sha(cp)
    assert cp_sha == '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
    protocol = ext / 'configs/delivery_20261008_v1/protocol.json'
    assert sha(protocol) == '8e40c705e57a6b6f2ed2244fc9c73108b43db7757596ff8ed03283f110cc3642'
    shutil.copyfile(protocol, out / 'protocol.json')
    shutil.copyfile(ext / 'ENGINEERING_RECOVERY_RECORD.json', out / 'ENGINEERING_RECOVERY_RECORD.json')
    dump(out / 'WEIGHT_START.json', {'path': str(cp), 'sha256': cp_sha})
    os.sched_setaffinity(0, {54})
    env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for p in root.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    for p in sorted([p for p in root.rglob('*') if p.is_dir()], reverse=True):
        p.chmod(0o555)
    root.chmod(0o555)
    dump(out / 'pipeline_state.json', {'stage': 'engineering_activation_QA',
        'time': str(datetime.datetime.now()), 'cpu_affinity': [54], 'gpu_allocated': False,
        'protocol_unchanged': True})

    def run(args, name, limit):
        with (out / (name + '.log')).open('w') as log:
            rc = subprocess.run([sys.executable] + list(map(str, args)), env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=limit).returncode
        (out / (name + '.exit')).write_text(str(rc))
        if rc:
            raise RuntimeError(name + ' failed; retain evidence and do not advance')

    try:
        run([ext / 'scripts/tests/test_delivery_precompile_reserve.py'], 'reserve_qa', 60)
        run([ext / 'scripts/tests/test_delivery_maskco.py'], 'mask_qa', 60)
        run([ext / 'scripts/tests/validate_delivery_runtime_activation.py', '--checkpoint', cp,
             '--upstream', upstream, '--out', out / 'ACTUAL_RUNTIME_QA.json'], 'actual_runtime_qa', 600)
        assert all(sha(root / p) == h for p, h in manifest.items()) and sha(cp) == cp_sha
        dump(out / 'QA_PASS.json', {'actual_51node_runtime_activation': True,
             'source_files_checked': len(manifest), 'source_stable': True,
             'development_data_not_used_by_QA': True, 'online_gain_established': False})
        dump(out / 'pipeline_state.json', {'stage': 'fresh_identical_six_arm_pilot',
             'time': str(datetime.datetime.now()), 'protocol_unchanged': True})
        scripts = ext / 'scripts/delivery'
        run([scripts / 'run_delivery_pilot.py', '--protocol', protocol, '--upstream', upstream,
             '--checkpoint', cp, '--out', out / 'pilot',
             '--manifest-path', out / 'SOURCE_MANIFEST.json'], 'run', 7200)
        assert all(sha(root / p) == h for p, h in manifest.items()) and sha(cp) == cp_sha
        dump(out / 'SOURCE_END.json', {'checked': len(manifest), 'source_stable': True, 'weights_stable': True})
        run([scripts / 'adjudicate_delivery_pilot.py', '--result', out / 'pilot/result.json',
             '--protocol', protocol, '--out', out / 'PILOT_ADJUDICATION.json'], 'adjudication', 300)
        result = json.loads((out / 'PILOT_ADJUDICATION.json').read_text())
        dump(out / 'COMPLETED.json', {'formal_pilot_adjudicable': result['formal_pilot_adjudicable'],
             'active_method_comparison_qualified': result['active_method_comparison_qualified'],
             'pilot_pass': result['pilot_pass'], 'online_gain_established': False})
        dump(out / 'pipeline_state.json', {'stage': 'route_pilot_adjudicated',
             'pilot_pass': result['pilot_pass'], 'time': str(datetime.datetime.now())})
        (out / 'pipeline.exit').write_text('0')
    except Exception as exc:
        dump(out / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        (out / 'pipeline.exit').write_text('1')
        raise


if __name__ == '__main__':
    main()
