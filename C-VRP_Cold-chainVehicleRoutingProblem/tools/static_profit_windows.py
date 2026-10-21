"""Bounded training-teacher refinement on the authorized Windows 3080 host."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import posixpath
import re
import warnings
import zipfile
warnings.filterwarnings('ignore',category=DeprecationWarning)
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings('ignore',category=CryptographyDeprecationWarning)
import paramiko

EXT=Path(__file__).resolve().parents[1]
ROOT=EXT.parent
PYTHON='D:/anaconda/envs/cc_compare/python.exe'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--cases',default='all')
    args=parser.parse_args()
    if not re.fullmatch(r'dev[0-9]{2}',args.label):raise ValueError('owned devNN required')
    settings=json.loads((ROOT/'.vscode/sftp.3080.json').read_text(encoding='utf-8-sig'))
    project=settings['remotePath'].rstrip('/')
    if not project.startswith('/C:/Users/zh/project/'):
        raise ValueError('unexpected authorized Windows project root')
    # Keep every extracted path under legacy Windows MAX_PATH without changing
    # machine policy or touching old long-path failure directories.
    remote=project+'/research_static_profit_20261021/'+args.label
    native=remote.lstrip('/')
    local=EXT/'results/static_delivery/algorithm_dev_20261021/remote_3080'/args.label
    local.mkdir(parents=True,exist_ok=False)
    lock=EXT/'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json'
    files=[EXT/r['path'] for r in json.loads(lock.read_text(encoding='utf-8'))['files']]+[lock,lock.with_suffix('.sha256')]
    files+=list((EXT/'scripts/static_profit').glob('*.py'))+list((ROOT/'MASKCO_code').rglob('*.py'))
    files+=[EXT/'scripts/project_paths.py']
    alns=EXT/'comparisons/static_coldchain/upstream/ALNS'
    files+=list((alns/'alns').rglob('*.py'))+list(alns.glob('LICENSE*'))+[alns/'SNAPSHOT_PROVENANCE.json']
    old=EXT/'results/static_delivery/algorithm_dev_20261021/remote/dev04/outputs/train/teachers.json'
    files+=[old]
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(local/'payload.zip','x',zipfile.ZIP_DEFLATED) as bundle:
        for name in manifest:bundle.write(ROOT/name,name)
        bundle.writestr('payload_manifest.json',json.dumps(manifest))
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(settings['host'],port=settings.get('port',22),username=settings['username'],
            password=settings.get('password'),timeout=15,look_for_keys=False,allow_agent=False)
        sftp=c.open_sftp()
        try:
            def parents(path):
                import stat
                try:
                    if not stat.S_ISDIR(sftp.lstat(path).st_mode):raise ValueError('not ordinary directory')
                except FileNotFoundError:
                    parents(posixpath.dirname(path));sftp.mkdir(path)
            parents(posixpath.dirname(remote))
            sftp.mkdir(remote)
            sftp.put(str(local/'payload.zip'),remote+'/payload.zip')
        finally:sftp.close()
        entry=native+'/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/teacher_upgrade.py'
        teachers=native+'/workspace/'+old.relative_to(ROOT).as_posix()
        command=[PYTHON,'-B',entry,'--teachers',teachers,'--cases',args.cases,'--seconds','1','--out',native+'/outputs/teacher_upgrade']
        bootstrap='''import pathlib,zipfile,json,hashlib,subprocess,os,ctypes,time
r=pathlib.Path(%r);w=r/'workspace';w.mkdir()
with zipfile.ZipFile(r/'payload.zip') as z:
 assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
 z.extractall(w)
m=json.loads((w/'payload_manifest.json').read_text())
def verify():
 assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())
verify()
affinity_mask=(1<<14)|(1<<15)
assert ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(affinity_mask))
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
start=time.perf_counter()
with (r/'run.log').open('w',encoding='utf-8') as log:
 child=subprocess.Popen(%r,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
 try: result=child.wait(timeout=900)
 except subprocess.TimeoutExpired:
  child.kill();child.wait();result=124
verify()
outputs={}
for p in (r/'outputs').rglob('*'):
 if p.is_file():outputs[p.relative_to(r).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(r/'output_manifest.json').write_text(json.dumps(outputs))
(r/'run.exit').write_text(str(result))
print(json.dumps(dict(exit_code=result,elapsed_s=time.perf_counter()-start,staged_files=len(m),affinity_mask=affinity_mask)))
''' % (native,command)
        encoded=base64.b64encode(bootstrap.encode()).decode()
        ps="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); & '"+PYTHON+"' -B -c \"import base64;exec(base64.b64decode('"+encoded+"'))\""
        shell='powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand '+base64.b64encode(ps.encode('utf-16le')).decode()
        (local/'started.json').write_text(json.dumps(dict(command=command,remote=remote,manifest=manifest,
            timeout_s=900,windows_hidden=True,cpu_threads=1,affinity_logical_cores=[14,15],gpu_used=False,
            formal_experiment=False,credentials_in_payload=False),indent=2),encoding='utf-8')
        print(json.dumps(dict(status='STARTED_WINDOWS_TEACHER_REFINEMENT',label=args.label,gpu_used=False,cases=args.cases)),flush=True)
        _,stdout,stderr=c.exec_command(shell,timeout=960)
        output=stdout.read().decode('utf-8','replace');error=stderr.read().decode('utf-8','replace')
        if stdout.channel.recv_exit_status():raise RuntimeError(error[-1800:])
        record=json.loads(output)
        sftp=c.open_sftp()
        try:
            with sftp.file(remote+'/output_manifest.json') as stream:outputs=json.loads(stream.read().decode())
            for name,expected in outputs.items():
                path=(local/name).resolve()
                if not name.startswith('outputs/') or not path.is_relative_to(local.resolve()):raise ValueError('unsafe output path')
                path.parent.mkdir(parents=True,exist_ok=True);sftp.get(remote+'/'+name,str(path))
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('download SHA mismatch')
            sftp.get(remote+'/run.log',str(local/'run.log'))
        finally:sftp.close()
        record['downloaded_sha_verified_files']=len(outputs)
        record['output_manifest']=outputs
        (local/'completion.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
        if record['exit_code']!=0:raise SystemExit(record['exit_code'])
    finally:c.close()


if __name__=='__main__':main()
