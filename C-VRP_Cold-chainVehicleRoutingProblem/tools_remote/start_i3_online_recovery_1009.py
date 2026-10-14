"""Freeze and resume only the pre-data I3 online import failure."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import zipfile

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
ORIGIN = ROOT / 'results/i3_registered_1008_v5'
OUT = ROOT / 'results/i3_online_recovery_1009_v1'
EXT = OUT / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
PY = '/home/hzeng/miniconda3/envs/MASKCO_env/bin/python'
NAME = 'i3_online_recovery_1009_v1'
PROTOCOL_SHA = '4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    with Path(p).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def verify(p):
    result = subprocess.run(['sha256sum', '-c', str(p)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return len(result.stdout.splitlines())


def main():
    if OUT.resolve().parent != (ROOT / 'results').resolve() or OUT.exists():
        raise RuntimeError('Recovery must use its exact new empty directory')
    if subprocess.run(['tmux', 'has-session', '-t', NAME], capture_output=True).returncode == 0:
        raise RuntimeError('Recovery session already exists; do not duplicate')
    old_exit = (ORIGIN / 'pipeline.exit').read_text()
    failure_log = (ORIGIN / 'online_20260926/run.log').read_text()
    if ('RUN_EXIT=1' not in old_exit or 'SEAL_EXIT=0' not in old_exit
            or "No module named 'coldchain_contract'" not in failure_log
            or set(p.name for p in (ORIGIN / 'online_20260926').iterdir()) != {'run.log', 'run.exit'}):
        raise RuntimeError('Not the proved pre-data import failure; do not rerun numerics')
    processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,psr,pcpu,stat,args'], text=True)
    conflicts = []
    for row in processes.splitlines()[1:]:
        cells = row.split(None, 5)
        if len(cells) != 6:
            continue
        pid, _, _, _, _, command = cells
        if int(pid) == os.getpid():
            continue
        if (str(OUT) in command or str(ORIGIN / 'pipeline.py') in command
                or (str(ORIGIN) in command and
                    ('run_i3_online_compare.py' in command or 'train_i3_scenario.py' in command))):
            conflicts.append(row)
    if conflicts:
        raise RuntimeError('An actual owned process is still running: ' + repr(conflicts))
    if verify(ORIGIN / 'SOURCE_MANIFEST.sha256') != 337 or sha(ORIGIN / 'protocol.json') != PROTOCOL_SHA:
        raise RuntimeError('Original source/protocol qualification changed')
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 50:
        raise RuntimeError('Original CPU allocation unavailable')
    OUT.mkdir()
    EXT.mkdir(parents=True)
    old_scripts = ORIGIN / 'src/C-VRP_Cold-chainVehicleRoutingProblem/scripts'
    shutil.copytree(old_scripts, EXT / 'scripts')
    # The sole source change is the standalone entry import closure.
    entry = EXT / 'scripts/evaluation/run_i3_online_compare.py'
    original = entry.read_bytes()
    needle = b"for folder in ('training', 'evaluation', 'models'):"
    if original.count(needle) != 1:
        raise RuntimeError('Unexpected original entry; refuse a broad patch')
    newline = b'\r\n' if b'\r\n' in original else b'\n'
    replacement = (b'# Resolve transitive cold-chain/simulator imports before loading the model.' + newline +
        b"# A subprocess entry cannot depend on the parent pipeline's sys.path." + newline +
        b"for folder in ('training', 'evaluation', 'models', 'coldchain', 'simulation'):")
    patched = original.replace(needle, replacement, 1)
    entry.chmod(entry.stat().st_mode | 0o200)
    entry.write_bytes(patched)
    if patched.replace(replacement, needle, 1) != original:
        raise RuntimeError('Unexpected source changes')
    upstream = ROOT.parent / 'MASKCO_code'
    os.symlink(str(upstream), str(OUT / 'src/MASKCO_code'))
    old_hash = {p.relative_to(old_scripts).as_posix(): sha(p) for p in old_scripts.rglob('*.py')}
    new_hash = {p.relative_to(EXT / 'scripts').as_posix(): sha(p) for p in (EXT / 'scripts').rglob('*.py')}
    changed = [k for k in new_hash if old_hash.get(k) != new_hash[k]]
    if set(old_hash) != set(new_hash) or changed != ['evaluation/run_i3_online_compare.py']:
        raise RuntimeError('Change exceeds isolated entry import fix')
    stage = OUT / 'engineering_package'
    with zipfile.ZipFile(ROOT / 'tools_remote/i3_online_recovery_1009_v1.zip') as archive:
        for item in archive.namelist():
            if Path(item).is_absolute() or '..' in Path(item).parts:
                raise RuntimeError('Unsafe package path')
        archive.extractall(stage)
    package = json.loads((stage / 'release.json').read_text())
    for rel, digest in package.items():
        if sha(stage / rel) != digest:
            raise RuntimeError('Engineering package hash mismatch')
    shutil.copyfile(stage / 'i3_online_recovery_1009.py', OUT / 'pipeline.py')
    shutil.copyfile(stage / 'start_i3_online_recovery_1009.py', OUT / 'launcher.py')
    shutil.copyfile(ORIGIN / 'protocol.json', OUT / 'protocol.json')
    save(OUT / 'SOURCE_DIFF.json', {'origin': str(ORIGIN), 'engineering_only': True,
        'changed_script': changed, 'unchanged_script_count': len(old_hash) - 1,
        'entry_before_sha256': sha(old_scripts / changed[0]), 'entry_after_sha256': sha(entry),
        'protocol_unchanged': True, 'models_unchanged': True,
        'states_or_training_repeated': False, 'online_days_previously_computed': 0})
    save(OUT / 'resources.json', {'time': str(datetime.datetime.now()),
        'available_cpus': allowed, 'registered_first_seed_worker_cpus': allowed[24:32],
        'registered_robustness_worker_cpus': allowed[32:48], 'engineering_audit_cpus': allowed[48:50],
        'gpu_used': False, 'load_average': os.getloadavg(), 'process_snapshot': processes,
        'team_processes_changed': False})
    input_paths = [ORIGIN / 'TEACHER_AUDIT_ACCEPTED.json', ORIGIN / 'teacher_merged/private.json',
        ORIGIN / 'teacher_merged/report.json', ORIGIN / 'online_config_20260926.json',
        ORIGIN / 'maskco_training_config.json', ORIGIN / 'reconstruction_control_training_config.json',
        ORIGIN / 'SOURCE_MANIFEST.sha256', upstream / 'ckpts/cvrp100.ckpt']
    barrier = json.loads((ORIGIN / 'TEACHER_AUDIT_ACCEPTED.json').read_text())
    input_paths.append(Path(barrier['audit_path']))
    for arm in ('maskco', 'reconstruction_control', 'warmstart'):
        input_paths += [ORIGIN / arm / 'model.bin', ORIGIN / arm / 'config.json']
        if arm != 'warmstart':
            input_paths += [ORIGIN / arm / f for f in
                           ('input_warmstart.bin', 'train_report.json', 'progress.jsonl')]
    for p in (EXT / 'scripts').rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'))
    ast.parse((OUT / 'pipeline.py').read_text())
    save(OUT / 'LAUNCH_RECORD.json', {'original_failure_preserved': True,
        'new_run': str(OUT), 'origin': str(ORIGIN), 'protocol_sha256': PROTOCOL_SHA,
        'reuse_only_completed_fixed_last_step_models': True,
        'no_training_or_collection_repeat': True, 'no_numeric_result_seen_before_recovery': True,
        'preconditions': 'Actual independent weight reload and isolated old-fails/new-passes regression',
        'remote_documents_uploaded': False})
    paths = sorted(set(list((EXT / 'scripts').rglob('*.py')) + input_paths +
        [OUT / x for x in ('pipeline.py', 'launcher.py', 'protocol.json', 'SOURCE_DIFF.json',
                           'resources.json', 'LAUNCH_RECORD.json')]))
    manifest = OUT / 'SOURCE_MANIFEST.sha256'
    manifest.write_text(''.join('%s  %s\n' % (sha(p), p) for p in paths), encoding='utf-8')
    for p in paths + [manifest]:
        if p.is_relative_to(OUT):
            p.chmod(p.stat().st_mode & ~0o222)
    for folder in [EXT / 'scripts', *[p for p in (EXT / 'scripts').rglob('*') if p.is_dir()]]:
        folder.chmod(folder.stat().st_mode & ~0o222)
    count = verify(manifest)
    if verify(ORIGIN / 'SOURCE_MANIFEST.sha256') != 337:
        raise RuntimeError('Original frozen tree changed')
    args = ' '.join(shlex.quote(str(x)) for x in [OUT / 'pipeline.py', OUT, ROOT, ORIGIN])
    f = shlex.quote(str(OUT))
    wrapper = ('#!/usr/bin/env bash\nset -uo pipefail\n'
        'export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
        'unset PYTHONPATH\n'
        'date > {f}/started_at.txt\n'
        'timeout 280800s {py} {args} > {f}/pipeline.log 2>&1\nrc=$?\n'
        'sha256sum -c {f}/SOURCE_MANIFEST.sha256 > {f}/SOURCE_END.txt 2>&1\nseal=$?\n'
        "printf 'RUN_EXIT=%s\\nSEAL_EXIT=%s\\nDONE=1\\n' \"$rc\" \"$seal\" > {f}/pipeline.exit\n"
        'date > {f}/ended_at.txt\n').format(f=f, py=shlex.quote(PY), args=args)
    (OUT / 'run.sh').write_text(wrapper, encoding='utf-8')
    subprocess.run(['tmux', 'new-session', '-d', '-s', NAME,
                    'bash ' + shlex.quote(str(OUT / 'run.sh'))], check=True)
    subprocess.run(['tmux', 'has-session', '-t', NAME], check=True)
    print(json.dumps({'started': NAME, 'source_count': count, 'protocol_sha256': PROTOCOL_SHA,
        'stage': 'independent_models_and_import_qa_before_online', 'new_directory': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
