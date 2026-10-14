"""Deploy the approved sampling-only amendment from the trusted frozen v4 tree."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import zipfile

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
OLD = ROOT / 'results/i3_registered_1008_v4'
OUT = ROOT / 'results/i3_registered_1008_v5'
EXT = OUT / 'src/C-VRP_Cold-chainVehicleRoutingProblem'
PY = '/home/hzeng/miniconda3/envs/MASKCO_env/bin/python'
PROTOCOL_SHA = '4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def verify(manifest):
    result = subprocess.run(['sha256sum', '-c', str(manifest)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return len(result.stdout.splitlines())

def seal(paths, manifest):
    paths = sorted(set(map(Path, paths)))
    manifest.write_text(''.join('%s  %s\n' % (sha(p), p) for p in paths), encoding='utf-8')
    for p in paths + [manifest]:
        p.chmod(p.stat().st_mode & ~0o222)

def start(name, script):
    subprocess.run(['tmux', 'new-session', '-d', '-s', name, 'bash ' + shlex.quote(str(script))], check=True)
    subprocess.run(['tmux', 'has-session', '-t', name], check=True)

def finite_script(folder, entry, seconds, exit_name='run.exit', prefix='', log_name='run.log'):
    return ('#!/usr/bin/env bash\nset -uo pipefail\n'
        'export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
        'date > {folder}/started_at.txt\n{prefix}'
        'timeout {seconds}s {py} {entry} > {folder}/{log_name} 2>&1\nrc=$?\n'
        'sha256sum -c {folder}/SOURCE_MANIFEST.sha256 > {folder}/verify_end.txt 2>&1\nseal=$?\n'
        "printf 'RUN_EXIT=%s\\nSEAL_EXIT=%s\\nDONE=1\\n' \"$rc\" \"$seal\" > {folder}/{exit_name}\n"
        'date > {folder}/ended_at.txt\n').format(folder=shlex.quote(str(folder)),
            prefix=prefix, seconds=seconds, py=shlex.quote(PY), entry=entry, exit_name=exit_name, log_name=log_name)

def main():
    assert OUT.resolve().parent == (ROOT / 'results').resolve()
    names = ['i3_registered_1008_v5', 'i3_release_qa_1008_v5',
             'i3_teacher_data_audit_1008_v3', 'i3_teacher_risk_1008_v3']
    for name in names:
        if subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode == 0:
            raise RuntimeError('Existing owned session; do not duplicate: ' + name)
    companions = [ROOT / 'results' / name for name in names[2:]]
    assert not OUT.exists() and all(not p.exists() for p in companions)
    assert verify(OLD / 'SOURCE_MANIFEST.sha256') == 332
    old_protocol = json.loads((OLD / 'protocol.json').read_text())
    assert sha(OLD / 'protocol.json') == '345604e3c3ed0ff2c40e049ded3c4a5a6784fcde1f464fe221dbc98850d8f5a5'
    EXT.mkdir(parents=True)
    shutil.copytree(OLD / 'src/C-VRP_Cold-chainVehicleRoutingProblem/scripts', EXT / 'scripts')
    for p in (EXT / 'scripts').rglob('*'):
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    (EXT / 'scripts').chmod((EXT / 'scripts').stat().st_mode | stat.S_IWUSR)
    stage = OUT / 'overlay'
    with zipfile.ZipFile(ROOT / 'tools_remote/i3_sampling_release_1008_v1.zip') as archive:
        for name in archive.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        archive.extractall(stage)
    release = json.loads((stage / 'release.json').read_text())
    manifest = release.get('files', release.get('manifest'))
    if manifest is None:
        manifest = release
    for rel, digest in manifest.items():
        assert sha(stage / rel) == digest, rel
    revision = json.loads((stage / 'sampling_revision.json').read_text())
    shutil.copytree(stage / 'scripts', EXT / 'scripts', dirs_exist_ok=True)
    new_protocol = json.loads((stage / 'protocol.json').read_text())
    assert sha(stage / 'protocol.json') == PROTOCOL_SHA
    assert new_protocol['execution_authorized'] and new_protocol['sampling_amendment']['human_approval']
    for key in old_protocol:
        if key != 'teacher_collection':
            assert old_protocol[key] == new_protocol[key], key
    for key in old_protocol['teacher_collection']:
        if key not in ('capture_ordinals', 'version'):
            assert old_protocol['teacher_collection'][key] == new_protocol['teacher_collection'][key], key
    old_scripts = OLD / 'src/C-VRP_Cold-chainVehicleRoutingProblem/scripts'
    old_hash = {p.relative_to(old_scripts).as_posix(): sha(p) for p in old_scripts.rglob('*.py')}
    new_hash = {p.relative_to(EXT / 'scripts').as_posix(): sha(p) for p in (EXT / 'scripts').rglob('*.py')}
    removed = sorted(set(old_hash) - set(new_hash))
    changed = sorted('scripts/' + k for k in new_hash if old_hash.get(k) != new_hash[k])
    assert not removed and set(changed) <= set(revision['changed_or_added_script_paths']), changed
    assert sha(EXT / 'scripts/evaluation/scenario_saa.py') == '36e08e06ce7b8e44eee2e6ff778904c1f26bd3145eb7d2072fb7360f33732318'
    top = ['pipeline.py', 'protocol.json', 'teacher_data_audit.py', 'teacher_data_audit_chain.py',
        'teacher_risk_diagnostic.py', 'teacher_risk_chain.py', 'teacher_risk_addendum.json',
        'i3_collection_cpu_leases_1008.py', 'test_i3_collection_cpu_leases_1008.py',
        'sampling_revision.json', 'APPROVAL_AND_CHANGE_RECORD.json', 'release.json']
    for name in top:
        shutil.copyfile(stage / name, OUT / name)
    shutil.copytree(OLD / 'warmstart', OUT / 'warmstart')
    assert sha(OUT / 'warmstart/model.bin') == new_protocol['warmstart_sha256']
    upstream = ROOT.parent / 'MASKCO_code'
    assert sha(upstream / 'ckpts/cvrp100.ckpt') == new_protocol['cvrp_sha256']
    os.symlink(str(upstream), str(OUT / 'src/MASKCO_code'))
    save(OUT / 'SOURCE_DIFF.json', {'approved_sampling_change_only': True,
        'changed_or_added': changed, 'removed': removed, 'unchanged_script_count':
        sum(old_hash.get(k) == v for k, v in new_hash.items()),
        'old_source_manifest_sha256': sha(OLD / 'SOURCE_MANIFEST.sha256'),
        'new_protocol_sha256': PROTOCOL_SHA, 'actor_saa_c0_training_online_unchanged': True})
    shutil.copyfile(OLD / 'resources.json', OUT / 'resources.json')
    for p in (EXT / 'scripts').rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'))
    sealed = list((EXT / 'scripts').rglob('*.py')) + [OUT / n for n in top]
    sealed += list((OUT / 'warmstart').glob('*')) + [upstream / 'ckpts/cvrp100.ckpt', OUT / 'SOURCE_DIFF.json']
    # The upstream checkpoint is already frozen; never change its permissions.
    source = OUT / 'SOURCE_MANIFEST.sha256'
    source.write_text(''.join('%s  %s\n' % (sha(p), p) for p in sorted(set(sealed))), encoding='utf-8')
    for p in sealed + [source]:
        if p.is_relative_to(OUT):
            p.chmod(p.stat().st_mode & ~0o222)
    for folder in [EXT / 'scripts', *[p for p in (EXT / 'scripts').rglob('*') if p.is_dir()]]:
        folder.chmod(folder.stat().st_mode & ~0o222)
    seal_count = verify(source)
    qa_entry = ' '.join(map(shlex.quote, [str(EXT / 'scripts/tests/validate_i3_registered_release.py'),
        '--out', str(OUT), '--project', str(ROOT)]))
    lease = '%s %s > %s/cpu_lease_test.log 2>&1 || exit 1\n' % (shlex.quote(PY),
        shlex.quote(str(OUT / 'test_i3_collection_cpu_leases_1008.py')), shlex.quote(str(OUT)))
    (OUT / 'qa.sh').write_text(finite_script(OUT, qa_entry, 7200, 'qa.exit', lease, 'qa.log'), encoding='utf-8')
    pipeline_entry = ' '.join(map(shlex.quote, [str(OUT / 'pipeline.py'), str(OUT), str(ROOT), str(EXT)]))
    (OUT / 'run.sh').write_text(finite_script(OUT, pipeline_entry, 604800, 'pipeline.exit', log_name='pipeline.log'), encoding='utf-8')
    for folder, typ, seconds in zip(companions, ['audit', 'risk'], [111600, 216000]):
        folder.mkdir()
        pairs = ([('teacher_data_audit.py', 'audit.py'), ('teacher_data_audit_chain.py', 'chain.py')]
            if typ == 'audit' else [('teacher_risk_diagnostic.py', 'diagnostic.py'),
                ('teacher_risk_chain.py', 'chain.py'), ('teacher_risk_addendum.json', 'addendum.json')])
        for src, target in pairs:
            shutil.copyfile(OUT / src, folder / target)
        seal([folder / target for src, target in pairs], folder / 'SOURCE_MANIFEST.sha256')
        entry = ' '.join(map(shlex.quote, [str(folder / 'chain.py'), str(folder), str(ROOT)]))
        (folder / 'run.sh').write_text(finite_script(folder, entry, seconds), encoding='utf-8')
    assert verify(OLD / 'SOURCE_MANIFEST.sha256') == 332
    save(OUT / 'LAUNCH_RECORD.json', {'approved_protocol_sha256': PROTOCOL_SHA,
        'source_count': seal_count, 'source_diff': changed, 'sessions': names,
        'old_states_reused': False, 'start_condition': 'QA plus qualified original A3 failure',
        'sampling_design_amendment_not_numeric_retry': True})
    start(names[1], OUT / 'qa.sh')
    start(names[0], OUT / 'run.sh')
    for folder in companions:
        start(folder.name, folder / 'run.sh')
    print(json.dumps({'started': names, 'source_count': seal_count,
        'protocol_sha256': PROTOCOL_SHA, 'formal_collection': 'waiting_release_qa'}, indent=2))

if __name__ == '__main__':
    main()
