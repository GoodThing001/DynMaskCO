"""Stage/run only this static RRP workspace via the user-provided SFTP config.

Credentials stay in the existing config and are never copied into the payload.
Every run owns a new directory and tmux session; no existing job is stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shlex
import stat
import warnings
import zipfile

warnings.filterwarnings('ignore', category=DeprecationWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
import paramiko

EXT = Path(__file__).resolve().parents[1]
ROOT = EXT.parent
PYTHON = '/home/hzeng/envs/MASKCO_env/bin/python'


def connect(config):
    settings = json.loads(Path(config).read_text(encoding='utf-8-sig'))
    if not settings['remotePath'].startswith('/home/'):
        raise ValueError('This runner intentionally targets the Linux configuration only')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(settings['host'], port=settings.get('port', 22), username=settings['username'],
                   password=settings.get('password'), timeout=15, look_for_keys=False, allow_agent=False)
    return client, settings


def execute(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=30)
    out, err = stdout.read().decode('utf-8', 'replace'), stderr.read().decode('utf-8', 'replace')
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError('Remote operation failed (%d): %s' % (code, err[-2000:]))
    return out


def parents(sftp, directory):
    try:
        entry = sftp.lstat(directory)
        if not stat.S_ISDIR(entry.st_mode):
            raise ValueError('Remote staging ancestor is not an ordinary directory')
    except FileNotFoundError:
        parents(sftp, posixpath.dirname(directory))
        sftp.mkdir(directory)


def locations(settings, label):
    if not re.fullmatch(r'run[0-9]{2}', label):
        raise ValueError('Use an owned run label such as run01')
    remote = posixpath.join(settings['remotePath'].rstrip('/'),
                            'C-VRP_Cold-chainVehicleRoutingProblem/research/static_rrp_20261020', label)
    local = EXT / 'results/static_delivery/remote' / label
    local.mkdir(parents=True, exist_ok=True)
    return remote, local


def prepare(local):
    files = list((EXT / 'scripts/static_delivery').glob('*.py'))
    files += [EXT / 'scripts/project_paths.py']
    files += list((EXT / 'configs/static_delivery').glob('*.json'))
    files += list((ROOT / 'MASKCO_code').rglob('*.py'))
    source = EXT / 'data/public_sources_20261020/refrigerated_routing_rrp'
    files += list((source / 'Datasets').rglob('*.json')) + list((source / 'Solutions').rglob('*.txt'))
    files += [source / 'download_manifest.json', source / 'README.md']
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    payload = local / 'payload.zip'
    if not payload.exists():
        with zipfile.ZipFile(payload, 'x', zipfile.ZIP_DEFLATED) as bundle:
            for p in files:
                bundle.write(p, p.relative_to(ROOT).as_posix())
            bundle.writestr('payload_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(payload) as bundle:
        if (set(bundle.namelist()) != set(manifest) | {'payload_manifest.json'}
                or len(bundle.namelist()) != len(manifest) + 1
                or json.loads(bundle.read('payload_manifest.json')) != manifest
                or any(hashlib.sha256(bundle.read(n)).hexdigest() != h for n, h in manifest.items())):
            raise ValueError('Prepared payload differs from current sources; use a fresh run label')
    record = {'files': len(manifest), 'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
              'credential_values_in_payload': False, 'network_operation_performed': False}
    (local / 'prepared.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return manifest


def stage(client, settings, remote, local):
    manifest = prepare(local)
    payload = local / 'payload.zip'
    sftp = client.open_sftp()
    try:
        parents(sftp, posixpath.dirname(remote))
        sftp.mkdir(remote)  # Refuse to overlay any existing run.
        sftp.put(str(payload), remote + '/payload.zip')
    finally:
        sftp.close()
    extract = ('import pathlib,zipfile,json,hashlib; r=pathlib.Path(' + repr(remote) + '); '
               'z=zipfile.ZipFile(r/"payload.zip"); w=r/"workspace"; w.mkdir(); '
               'assert all(not pathlib.PurePosixPath(n).is_absolute() and ".." not in pathlib.PurePosixPath(n).parts for n in z.namelist()); '
               'z.extractall(w); m=json.loads((w/"payload_manifest.json").read_text()); '
               'assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items()); print(len(m))')
    verified = execute(client, PYTHON + ' -B -c ' + shlex.quote(extract)).strip()
    record = {'remote': remote, 'host': settings['host'], 'files': len(manifest),
              'remote_verified_files': int(verified), 'source_manifest': manifest,
              'cpu_only': True, 'credential_values_in_payload': False}
    (local / 'stage.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k != 'source_manifest'}))


def start(client, remote, local, label, job, model_profile, checkpoint_run=None):
    workspace = remote + '/workspace'
    entry = workspace + '/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_delivery/run_rrp.py'
    output = remote + '/outputs'
    if job == 'audit':
        args = [entry, 'audit', '--out', output + '/audit']
    elif job == 'fleet-check':
        if checkpoint_run is None or not re.fullmatch(r'run[0-9]{2}', checkpoint_run) or checkpoint_run == label:
            raise ValueError('fleet-check requires a distinct completed --checkpoint-run label')
        checkpoint = posixpath.join(posixpath.dirname(remote), checkpoint_run,
                                   'outputs/followup/train/checkpoint')
        args = [entry, 'fleet-solve', '--case', 'D1-n34-np33', '--max-vehicles', '2',
                '--methods', 'local_search', 'maskco_initial', 'maskco', '--checkpoint', checkpoint,
                '--iterations', '3', '--seconds', '45', '--out', output + '/fleet_check']
    else:
        args = [entry, 'fleet-smoke' if job == 'followup' else 'smoke', '--case', 'D1-n4-np33',
                '--steps', '3', '--model-profile', model_profile, '--out', output + '/' + job]
        if job == 'followup':
            args.extend(['--max-vehicles', '2'])
    validate = ('import pathlib,json,hashlib; w=pathlib.Path(' + repr(workspace) + '); '
                'm=json.loads((w/"payload_manifest.json").read_text()); '
                'assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())')
    generate = ('import pathlib,json,hashlib; r=pathlib.Path(' + repr(remote) + '); '
                'm={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/"outputs").rglob("*") if p.is_file()}; '
                '(r/"output_manifest.json").write_text(json.dumps(m,indent=2))')
    command = ' '.join(shlex.quote(a) for a in [PYTHON, '-B', *args])
    script = '\n'.join([
        '#!/bin/bash', 'set -u', 'umask 077', 'ulimit -v 67108864', 'ulimit -c 0',
        'export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4',
        'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"',
        'cd ' + shlex.quote(workspace), 'mkdir -p ' + shlex.quote(output),
        PYTHON + ' -B -c ' + shlex.quote(validate) + ' || exit 91',
        'timeout 900s taskset -c 92-95 ' + command + ' > ' + shlex.quote(remote + '/run.log') + ' 2>&1',
        'result=$?', PYTHON + ' -B -c ' + shlex.quote(validate) + ' || result=92',
        PYTHON + ' -B -c ' + shlex.quote(generate) + ' || result=93',
        'printf "%s\\n" "$result" > ' + shlex.quote(remote + '/run.exit'), 'exit "$result"', ''])
    sftp = client.open_sftp()
    try:
        job_path = remote + '/job_' + job + '.sh'
        with sftp.file(job_path, 'wx') as f:
            f.write(script)
    finally:
        sftp.close()
    session = 'rrp_static_20261020_' + label
    execute(client, 'tmux new-session -d -s ' + shlex.quote(session) + ' bash ' + shlex.quote(job_path))
    print(json.dumps({'started': session, 'job': job, 'model_profile': model_profile,
                      'checkpoint_run': checkpoint_run, 'cpu_cores': [92, 93, 94, 95], 'timeout_s': 900}))


def status_or_pull(client, remote, local, pull):
    sftp = client.open_sftp()
    try:
        status = None
        try:
            with sftp.file(remote + '/run.exit') as f:
                status = f.read().decode().strip()
        except FileNotFoundError:
            pass
        try:
            with sftp.file(remote + '/run.log') as f:
                size = f.stat().st_size
                f.seek(max(0, size - 4000))
                tail = f.read().decode('utf-8', 'replace')
        except FileNotFoundError:
            tail = ''
        print(json.dumps({'exit': status, 'tail': tail}))
        if not pull or status is None:
            return
        with sftp.file(remote + '/output_manifest.json') as f:
            manifest = json.loads(f.read())
        for relative, expected in manifest.items():
            rel = Path(relative)
            if rel.is_absolute() or '..' in rel.parts or not relative.startswith('outputs/'):
                raise ValueError('Unsafe output manifest path')
            target = local / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote + '/' + relative, str(target))
            with target.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest != expected:
                raise ValueError('Downloaded output SHA mismatch')
        for name in ['run.log', 'run.exit', 'output_manifest.json']:
            sftp.get(remote + '/' + name, str(local / name))
        print(json.dumps({'downloaded_verified_files': len(manifest), 'local': str(local)}))
    finally:
        sftp.close()


def probe(client, local):
    # Isolated, CPU-affined, bounded diagnostic; never touches other users' jobs.
    code = '''import json,os,resource,subprocess
from pathlib import Path
result={"affinity":sorted(os.sched_getaffinity(0)),"nproc_limits":resource.getrlimit(resource.RLIMIT_NPROC),"address_limits":resource.getrlimit(resource.RLIMIT_AS)}
for name in ("pids.max","pids.current","memory.max","memory.current"):
 p=Path("/sys/fs/cgroup")/name
 if p.exists(): result[name]=p.read_text().strip()
for limit in (16777216,67108864):
 command="ulimit -v %d; export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; taskset -c 92-95 %s -B -c 'import jax; import jax.numpy as j; print(jax.devices()); print(j.ones(1).block_until_ready())'" % (limit,''' + repr(PYTHON) + ''')
 try:
  p=subprocess.run(["bash","-c",command],capture_output=True,text=True,timeout=25)
  result[str(limit)]={"exit":p.returncode,"out":p.stdout[-1000:],"err":p.stderr[-1000:]}
 except subprocess.TimeoutExpired: result[str(limit)]={"timeout_s":25}
print(json.dumps(result))'''
    output = execute(client, PYTHON + ' -B -c ' + shlex.quote(code))
    (local / 'resource_probe.json').write_text(output, encoding='utf-8')
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'stage', 'start', 'status', 'pull', 'probe'])
    p.add_argument('--label', required=True)
    p.add_argument('--config', type=Path, default=ROOT / '.vscode/sftp.json')
    p.add_argument('--job', choices=['audit', 'smoke', 'followup', 'fleet-check'], default='audit')
    p.add_argument('--checkpoint-run', help='Completed followup run used read-only by fleet-check')
    p.add_argument('--model-profile', choices=['smoke_cpu', 'maskco_256'], default='smoke_cpu')
    args = p.parse_args()
    if args.action == 'prepare':
        if not re.fullmatch(r'run[0-9]{2}', args.label):
            raise ValueError('Use an owned run label such as run06')
        local = EXT / 'results/static_delivery/remote' / args.label
        local.mkdir(parents=True, exist_ok=True)
        manifest = prepare(local)
        print(json.dumps({'prepared_files': len(manifest), 'local': str(local), 'network_operation_performed': False}))
        return
    client, settings = connect(args.config)
    try:
        remote, local = locations(settings, args.label)
        if args.action == 'stage':
            stage(client, settings, remote, local)
        elif args.action == 'start':
            start(client, remote, local, args.label, args.job, args.model_profile, args.checkpoint_run)
        elif args.action == 'probe':
            probe(client, local)
        else:
            status_or_pull(client, remote, local, args.action == 'pull')
    finally:
        client.close()


if __name__ == '__main__':
    main()
