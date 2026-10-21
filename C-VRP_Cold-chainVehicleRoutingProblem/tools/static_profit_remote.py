"""Isolated, bounded profit-algorithm development on the authorized SFTP host.

Credentials are consumed in place by the existing connector, never copied.
Uses its own namespace/tmux, bounded CPU affinity and optional explicit GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shlex
import zipfile

import static_rrp_remote as ssh

EXT, ROOT = ssh.EXT, ssh.ROOT
PYTHON = ssh.PYTHON


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def resource_probe(client):
    code = '''import os,json,subprocess,psutil,importlib.util,platform
z={"affinity":sorted(os.sched_getaffinity(0)),"available_memory_GB":psutil.virtual_memory().available/2**30,"cpu_percent":psutil.cpu_percent(interval=0.5,percpu=True)}
z['python']=platform.python_version()
import importlib.metadata as md
z['versions']={name:md.version(name) for name in ['jax','jaxlib','flax','optax']}
z['global_module_availability']={name:importlib.util.find_spec(name) is not None for name in ['alns','matplotlib','jax','flax']}
p=subprocess.run(["nvidia-smi","--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True)
z["gpu_summary"]=p.stdout
print(json.dumps(z))'''
    return json.loads(ssh.execute(client,PYTHON+' -B -c '+shlex.quote(code)))


def paths(settings,label):
    if not re.fullmatch(r'dev[0-9]{2}',label):
        raise ValueError('use an owned label such as dev01')
    remote = posixpath.join(settings['remotePath'].rstrip('/'),
        'C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021',label)
    local = EXT/'results/static_delivery/algorithm_dev_20261021/remote'/label
    local.mkdir(parents=True,exist_ok=True)
    return remote,local


def gpu_probe(client,local,index):
    if index not in (0,1):
        raise ValueError('explicit GPU index 0 or 1 required')
    code = '''import jax,jax.numpy as jnp,json
assert jax.default_backend()=='gpu', 'CUDA execution is required'
f=jax.jit(jax.value_and_grad(lambda w:jnp.mean((w@w)**2)))
y,g=f(jnp.eye(256)); g.block_until_ready()
print(json.dumps(dict(backend=jax.default_backend(),devices=[dict(name=d.device_kind,id=d.id,platform=d.platform) for d in jax.devices()],loss=float(y),finite=bool(jnp.all(jnp.isfinite(g))))))'''
    command = ('CUDA_VISIBLE_DEVICES='+str(index)+' JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false '
        'OMP_NUM_THREADS=1 timeout 60s '+PYTHON+' -B -c '+shlex.quote(code))
    result = json.loads(ssh.execute(client,command))
    result['physical_gpu_index'] = index
    save(local/'gpu_runtime_probe.json',result)
    print(json.dumps(result))


def inspect_job(client,local):
    session = 'profit_20261021_'+local.name
    code = '''import subprocess,json,psutil,time
p=subprocess.run(['tmux','list-panes','-t',%r,'-F','#{pane_pid}'],text=True,capture_output=True)
rows=[]
if p.returncode==0:
 for line in p.stdout.splitlines():
  for proc in psutil.Process(int(line)).children(recursive=True):
   try:
    t=proc.cpu_times()
    rows.append(dict(pid=proc.pid,name=proc.name(),elapsed_s=time.time()-proc.create_time(),cpu_seconds=t.user+t.system,rss_GB=proc.memory_info().rss/2**30,threads=proc.num_threads(),status=proc.status()))
   except psutil.NoSuchProcess: pass
g=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
print(json.dumps(dict(processes=rows,gpu=g.stdout)))''' % session
    result = json.loads(ssh.execute(client,PYTHON+' -B -c '+shlex.quote(code)))
    save(local/'last_process_probe.json',result)
    print(json.dumps(result))


def stage(client,remote,local):
    lockpath = EXT/'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json'
    lock = json.loads(lockpath.read_text(encoding='utf-8'))
    files = [EXT/r['path'] for r in lock['files']]+[lockpath,lockpath.with_suffix('.sha256')]
    files += list((EXT/'scripts/static_profit').glob('*.py'))
    files += list((ROOT/'MASKCO_code').rglob('*.py'))
    alns = EXT/'comparisons/static_coldchain/upstream/ALNS'
    files += list((alns/'alns').rglob('*.py'))+list(alns.glob('LICENSE*'))
    files += [alns/'SNAPSHOT_PROVENANCE.json']
    files += [EXT/'scripts/project_paths.py',EXT/'configs/static_delivery/algorithm_candidate_v0.json']
    for relative in ['configs/static_delivery/diagnostic_prereg_v1.json',
                     'configs/static_delivery/diagnostic_prereg_v1.sha256',
                     'results/static_delivery/algorithm_diagnostics_20261021/historical_validation_budget_v2.json',
                     'results/static_delivery/algorithm_diagnostics_20261021/diagnostic_decision.json',
                     'results/static_delivery/algorithm_diagnostics_20261021/diagnostic_audit.json']:
        if (EXT/relative).exists():files.append(EXT/relative)
    for relative in ['results/static_delivery/algorithm_dev_20261021/remote/dev04/outputs/train/teachers.json',
                     'results/static_delivery/algorithm_dev_20261021/remote_3080/dev03/outputs/teacher_upgrade/teachers.json']:
        teacher_source=EXT/relative
        if teacher_source.exists():files.append(teacher_source)
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    payload = local/'payload.zip'
    with zipfile.ZipFile(payload,'x',zipfile.ZIP_DEFLATED) as z:
        for relative in manifest:
            z.write(ROOT/relative,relative)
        z.writestr('payload_manifest.json',json.dumps(manifest,indent=2))
    sftp = client.open_sftp()
    try:
        try:
            sftp.lstat(remote)
        except FileNotFoundError:
            ssh.parents(sftp,remote)
        else:
            raise ValueError('remote run directory already exists; use a new label')
        sftp.put(str(payload),remote+'/payload.zip')
    finally:
        sftp.close()
    code = '''import pathlib,zipfile,json,hashlib
r=pathlib.Path(%r); w=r/'workspace'; w.mkdir()
with zipfile.ZipFile(r/'payload.zip') as z:
 assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
 z.extractall(w)
m=json.loads((w/'payload_manifest.json').read_text())
assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())
print(len(m))''' % remote
    verified = int(ssh.execute(client,PYTHON+' -B -c '+shlex.quote(code)).strip())
    save(local/'stage.json',dict(files=len(manifest),remote_verified_files=verified,remote=remote,
        manifest=manifest,credentials_in_payload=False))
    print(json.dumps(dict(staged_files=verified,label=local.name)))


def start(client,remote,local,profile,checkpoint_parent=None,backend='cpu',gpu_index=None,job='smoke',steps=240,autotune=4,predictor='eager',precision='default'):
    info = resource_probe(client)
    idle = [i for i in info['affinity'] if i not in range(92,96) and info['cpu_percent'][i]<10]
    if len(idle)<4 or info['available_memory_GB']<8:
        raise RuntimeError('insufficient idle CPU/memory for this bounded development run')
    cores = idle[-1:] if job=='diagnostic' else idle[-4:]
    threads=1 if job=='diagnostic' else 4
    selected = None
    if backend == 'cuda':
        if gpu_index is None:
            raise ValueError('CUDA requires an explicit --gpu-index')
        rows = [line.split(', ') for line in info['gpu_summary'].strip().splitlines()]
        selected = next((r for r in rows if int(r[0])==gpu_index),None)
        if selected is None or '5090' not in selected[1] or float(selected[4])<16000 or float(selected[5])>30:
            raise RuntimeError('requested RTX 5090 is absent or insufficiently idle/free')
    workspace = remote+'/workspace'
    entry = workspace+'/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py'
    args = [PYTHON,'-B',entry,'smoke','--fold','0','--case','r101','--steps','3',
            '--profile',profile,'--backend',backend,'--threads','4','--iterations','3',
            '--seconds','15','--out',remote+'/outputs/smoke']
    if checkpoint_parent is not None:
        if not re.fullmatch(r'dev[0-9]{2}',checkpoint_parent) or checkpoint_parent == local.name:
            raise ValueError('checkpoint parent must be a distinct owned development label')
        args[3] = 'replay'
        args += ['--checkpoint-root',posixpath.join(posixpath.dirname(remote),checkpoint_parent,'outputs/smoke')]
    commands = [args]
    timeout = 900
    total_timeout=None
    if job in ('profile','profile-panel'):
        if checkpoint_parent is None:
            raise ValueError('profile requires read-only training checkpoint parent')
        commands = [[PYTHON,'-B',entry.replace('run.py','profile_runtime.py'),
            '--checkpoint-root',posixpath.join(posixpath.dirname(remote),checkpoint_parent,'outputs/train'),
            '--backend',backend,'--predictor',predictor,'--matmul-precision',precision,'--out',remote+'/outputs/profile']]
        if job=='profile-panel':
            commands=[]
            for engine,repair in [('eager','classic'),('compiled','classic'),('compiled','vectorized')]:
                commands.append([PYTHON,'-B',entry.replace('run.py','profile_runtime.py'),
                    '--checkpoint-root',posixpath.join(posixpath.dirname(remote),checkpoint_parent,'outputs/train'),
                    '--backend',backend,'--predictor',engine,'--matmul-precision','highest',
                    '--repair-implementation',repair,'--out',remote+'/outputs/'+engine+'_'+repair])
    if job == 'panel':
        if checkpoint_parent is not None:
            raise ValueError('panel trains fresh paired development models')
        folds = json.loads((EXT/'results/static_delivery/publication_protocol_v1_preflight_20261021/geometry_folds.json').read_text(encoding='utf-8'))
        fold = folds['folds'][0] if isinstance(folds,dict) else folds[0]
        variants = 'maskco_base,quality_only,directed_only,quality_directed'
        common = [PYTHON,'-B',entry]
        controls = ['--fold','0','--seed','1000','--profile',profile,'--backend',backend,'--threads','4','--variants',variants]
        commands = [common+['train']+controls+['--cases',','.join(fold['train']),'--steps',str(steps),
                    '--teacher-seconds','0.2','--out',remote+'/outputs/train']]
        for case in fold['validation']:
            commands.append(common+['replay']+controls+['--role','validation','--case',case,
                '--checkpoint-root',remote+'/outputs/train','--seconds','10','--iterations','100000',
                '--out',remote+'/outputs/validation/'+case])
        timeout = 1800
    if job=='training-study':
        folds=json.loads((EXT/'results/static_delivery/publication_protocol_v1_preflight_20261021/geometry_folds.json').read_text(encoding='utf-8'))
        fold=folds['folds'][0]
        commands=[]
        variants='maskco_base,directed_only'
        common=[PYTHON,'-B',entry,'train','--fold','0','--seed','1000','--profile',profile,
                '--backend',backend,'--threads','4','--variants',variants,
                '--cases',','.join(fold['train']),'--steps',str(steps),'--sample-order','shuffled']
        sources={
            'old_teachers':'results/static_delivery/algorithm_dev_20261021/remote/dev04/outputs/train/teachers.json',
            'upgraded_teachers':'results/static_delivery/algorithm_dev_20261021/remote_3080/dev03/outputs/teacher_upgrade/teachers.json'}
        for arm,relative in sources.items():
            if not (EXT/relative).is_file():raise ValueError('missing pinned training teacher input')
            commands.append(common+['--teacher-file',workspace+'/C-VRP_Cold-chainVehicleRoutingProblem/'+relative,
                '--out',remote+'/outputs/'+arm+'/train'])
        for arm in sources:
            for case in fold['validation']:
                commands.append([PYTHON,'-B',entry,'replay','--fold','0','--seed','1000','--profile',profile,
                    '--backend',backend,'--threads','4','--variants',variants,'--role','validation','--case',case,
                    '--checkpoint-root',remote+'/outputs/'+arm+'/train','--predictor','compiled',
                    '--matmul-precision','highest','--repair-implementation','vectorized',
                    '--seconds','10','--iterations','100000','--out',remote+'/outputs/'+arm+'/validation/'+case])
        timeout=1800
        total_timeout=2400
    if job=='validation-recovery':
        if checkpoint_parent!='dev09' or backend!='cuda':
            raise ValueError('Recovery requires the dev09 parent and explicit CUDA')
        commands=[[PYTHON,'-B',entry.replace('run.py','recover_validation.py'),
                   '--parent',posixpath.join(posixpath.dirname(remote),checkpoint_parent),
                   '--out',remote+'/outputs/recovery']]
        timeout=900
    if job=='diagnostic':
        if checkpoint_parent!='dev09' or backend!='cuda':raise ValueError('Diagnostic uses registered dev09 checkpoints on explicit CUDA')
        commands=[[PYTHON,'-B',entry.replace('run.py','diagnostics.py'),
                   '--parent',posixpath.join(posixpath.dirname(remote),checkpoint_parent),
                   '--out',remote+'/outputs/diagnostics']]
        timeout=2400
    if job=='diagnostic-continuation':
        decision=json.loads((EXT/'results/static_delivery/algorithm_diagnostics_20261021/diagnostic_decision.json').read_text(encoding='utf-8'))
        if checkpoint_parent!='dev09' or backend!='cuda' or not decision['approved_training_variants']:
            raise ValueError('Registered diagnostic gate must authorize at least one continuation method')
        registry=EXT/'results/static_delivery/algorithm_diagnostics_20261021/validation_configuration_registry_v1.json'
        budget=json.loads(registry.read_text(encoding='utf-8'))
        if local.name!='dev12' or registry.with_name('validation_configuration_registry_v2.json').exists():
            raise ValueError('This two-setting continuation is registered once as dev12; no duplicate launch or refunded slots')
        if any(budget['remaining_fold0'][m]<2 for m in decision['approved_training_variants']):
            raise ValueError('Insufficient validation configuration budget')
        common=[PYTHON,'-B',entry.replace('run.py','continue_training.py'),
                '--parent',posixpath.join(posixpath.dirname(remote),'dev09'),
                '--diagnostic-parent',posixpath.join(posixpath.dirname(remote),'dev11')]
        commands=[common+['--phase','train','--out',remote+'/outputs/continuation/train'],
                  common+['--phase','validate','--train-root',remote+'/outputs/continuation/train',
                          '--out',remote+'/outputs/continuation/validation']]
        timeout=2400
        total_timeout=3600
    validate = '''import pathlib,json,hashlib
w=pathlib.Path(%r);m=json.loads((w/'payload_manifest.json').read_text())
assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())''' % workspace
    outputs = '''import pathlib,json,hashlib
r=pathlib.Path(%r);m={}
for p in (r/'outputs').rglob('*'):
 if p.is_file():
  digest=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
  m[str(p.relative_to(r))]=digest.hexdigest()
(r/'output_manifest.json').write_text(json.dumps(m,indent=2))''' % remote
    environment = ['export JAX_PLATFORMS='+backend+' XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS='+str(threads)+' OPENBLAS_NUM_THREADS='+str(threads)+' MKL_NUM_THREADS='+str(threads)]
    if backend == 'cuda':
        environment += ['export CUDA_VISIBLE_DEVICES='+str(gpu_index)+' XLA_PYTHON_CLIENT_MEM_FRACTION=0.35',
            'export XLA_FLAGS='+shlex.quote('--xla_gpu_autotune_level='+str(autotune)),
            'export JAX_COMPILATION_CACHE_DIR='+shlex.quote(remote+'/compile_cache')]
    else:
        environment += ['ulimit -v 67108864','export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"']
    launches=[]
    for command in commands:
        if total_timeout is not None:
            launches+=['remaining=$(('+str(total_timeout)+'-SECONDS))',
                'if [ "$remaining" -le 0 ]; then exit 124; fi',
                'if [ "$remaining" -gt '+str(timeout)+' ]; then remaining='+str(timeout)+'; fi']
        bound='"${remaining}"' if total_timeout is not None else str(timeout)
        launches+=['timeout '+bound+'s nice -n 10 taskset -c '+','.join(map(str,cores))+' '+shlex.join(command)+' || exit $?']
    script = '\n'.join(['#!/bin/bash','set -u','umask 077','ulimit -c 0','SECONDS=0']+environment+[
        'cd '+shlex.quote(workspace),'mkdir -p '+shlex.quote(remote+'/outputs'),
        PYTHON+' -B -c '+shlex.quote(validate)+' || exit 91',
        '(',
        *launches,
        ') > '+shlex.quote(remote+'/run.log')+' 2>&1',
        'result=$?',PYTHON+' -B -c '+shlex.quote(validate)+' || result=92',
        PYTHON+' -B -c '+shlex.quote(outputs)+' || result=93',
        'printf "%s\\n" "$result" > '+shlex.quote(remote+'/run.exit'),'exit "$result"',''])
    (local/'job.sh').write_text(script,encoding='utf-8',newline='\n')
    sftp = client.open_sftp()
    try:
        with sftp.file(remote+'/job.sh','wx') as f:
            f.write(script)
    finally:
        sftp.close()
    session = 'profit_20261021_'+local.name
    ssh.execute(client,'tmux new-session -d -s '+shlex.quote(session)+' bash '+shlex.quote(remote+'/job.sh'))
    record = dict(session=session,cores=cores,profile=profile,timeout_per_command_s=timeout,gpu_used=backend=='cuda',
                  gpu_index=gpu_index,gpu_identity=selected,backend=backend,
                  gpu_autotune_level=autotune if backend=='cuda' else None,
                  total_timeout_s=total_timeout,
                  resource_probe=info,development_only=True,checkpoint_parent=checkpoint_parent,job=job,commands=commands)
    save(local/'started.json',record)
    if job=='diagnostic-continuation':
        for method in decision['approved_training_variants']:
            budget['reserved_or_spent_fold0'][method]+=2
            budget['remaining_fold0'][method]-=2
        budget.update(stage='REGISTERED_CONTINUATION_LAUNCHED',continuation_launch=dict(label=local.name,
            methods=decision['approved_training_variants'],charged_each=2,checkpoints=[1440,2880],branches=['keep20','switch10'],
            previous_registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),previous_registry=registry.name))
        save(registry.with_name('validation_configuration_registry_v2.json'),budget)
    print(json.dumps({k:v for k,v in record.items() if k not in ('resource_probe','commands')}))


def status(client,remote,local,pull=False):
    sftp = client.open_sftp()
    try:
        try:
            with sftp.file(remote+'/run.exit') as f:
                exit_code = int(f.read().decode().strip())
        except FileNotFoundError:
            exit_code = None
        with sftp.file(remote+'/run.log') as f:
            log = f.read().decode('utf-8','replace')
        (local/'run.log').write_text(log,encoding='utf-8')
        progress = {}
        for line in log.splitlines():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError,ValueError):
                continue
            if isinstance(row,dict) and 'variant' in row:
                progress[row['variant']] = row
        print(json.dumps(dict(exit_code=exit_code,latest_variant_progress=progress,tail=log[-1800:])))
        if not pull or exit_code is None:
            return
        with sftp.file(remote+'/output_manifest.json') as f:
            manifest = json.loads(f.read().decode())
        downloaded,retained = [],[]
        for name,expected in manifest.items():
            target = (local/name).resolve()
            if not target.is_relative_to(local.resolve()) or not name.startswith('outputs/'):
                raise ValueError('unsafe output path')
            if not name.endswith(('.json','.py')):
                retained.append(name)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            sftp.get(remote+'/'+name,str(target))
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                raise ValueError('output SHA mismatch')
            downloaded.append(name)
        save(local/'output_manifest.json',manifest)
        training_reports = [json.loads((local/name).read_text(encoding='utf-8'))
                            for name in downloaded if name.endswith('_training.json')]
        resume_verified = bool(training_reports) and all(
            r.get('checkpoint_prediction_identical') is True and r.get('optimizer_resume_identical') is True
            for r in training_reports)
        save(local/'download_verification.json',dict(exit_code=exit_code,
            downloaded_verified=downloaded,large_checkpoints_retained_remote=retained,
            checkpoint_resume_was_verified_in_remote_run=resume_verified,
            checkpoint_resume_evidence='downloaded_training_reports' if resume_verified else 'no_complete_training_restore_evidence'))
        print(json.dumps(dict(verified_report_files=len(downloaded),remote_large_files=len(retained))))
    finally:
        sftp.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['probe','gpu-probe','inspect','stage','start','status','pull'])
    p.add_argument('--label',default='dev01')
    p.add_argument('--profile',choices=['smoke_cpu','maskco_256'],default='maskco_256')
    p.add_argument('--checkpoint-parent',help='read-only reuse of previously trained full-backbone checkpoints')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--gpu-index',type=int)
    p.add_argument('--job',choices=['smoke','panel','profile','profile-panel','training-study','validation-recovery','diagnostic','diagnostic-continuation'],default='smoke')
    p.add_argument('--predictor',choices=['eager','compiled'],default='eager')
    p.add_argument('--matmul-precision',choices=['default','highest'],default='default')
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--gpu-autotune-level',type=int,choices=[0,4],default=4)
    args = p.parse_args()
    if args.steps < 1:
        p.error('positive steps required')
    client,settings = ssh.connect(ROOT/'.vscode/sftp.json')
    try:
        remote,local = paths(settings,args.label)
        if args.action=='probe':
            info = resource_probe(client)
            save(local/'resource_probe.json',info)
            print(json.dumps(dict(available_memory_GB=info['available_memory_GB'],
                idle_cores=sum(info['cpu_percent'][i]<10 for i in info['affinity']),gpu_summary=info['gpu_summary'],
                python=info['python'],versions=info['versions'],global_module_availability=info['global_module_availability'])))
        elif args.action=='gpu-probe':
            gpu_probe(client,local,args.gpu_index)
        elif args.action=='inspect':
            inspect_job(client,local)
        elif args.action=='stage':
            stage(client,remote,local)
        elif args.action=='start':
            start(client,remote,local,args.profile,args.checkpoint_parent,args.backend,args.gpu_index,args.job,args.steps,args.gpu_autotune_level,args.predictor,args.matmul_precision)
        else:
            status(client,remote,local,args.action=='pull')
    finally:
        client.close()


if __name__=='__main__':
    main()
