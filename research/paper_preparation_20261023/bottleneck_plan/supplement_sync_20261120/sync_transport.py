"""Inspect, submit, observe and recover the registered paired comparison."""
import json
import shlex
import sys
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'maskco_masked_reconstruction_20261118/run01'))
from host_access import connect, remote_python, PROJECT, PYTHON, transport
DEST=PROJECT+'/_maskco_supplement_sync_20261120/sync01'

PROBE='''import json,os,psutil,shutil,subprocess
from pathlib import Path
r=Path('/home/hzeng/project/MASKCO-Main')
load=psutil.cpu_percent(interval=2,percpu=True)
groups={}
for i in sorted(os.sched_getaffinity(0)):
 p=Path(f'/sys/devices/system/cpu/cpu{i}/topology')
 key=','.join((p/f).read_text().strip() for f in ('physical_package_id','core_id'))
 groups.setdefault(key,[]).append(i)
statuses={}
for base in ('_maskco_perishable_external_20261119','_maskco_recent5_20261120','_maskco_supplement_sync_20261120'):
 for p in (r/base).glob('*/status.json'):
  d=json.loads(p.read_text())
  if 'fleet' in str(p) or not d.get('end'):statuses[str(p.relative_to(r))]=d
procs=[]
for p in psutil.process_iter(['pid','name','cmdline']):
 try:
  cmd=p.info['cmdline'] or []
  if any(('fleet_' in s or 'development_runner' in s or 'sync_runner' in s or 'recent_runner' in s) for s in cmd) and '-c' not in cmd:
   procs.append(dict(pid=p.pid,name=p.info['name'],cmd=cmd,affinity=p.cpu_affinity()))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(dict(time=subprocess.check_output(['date','-Is'],text=True).strip(),
 ram_gib=psutil.virtual_memory().available/2**30,disk_gib=shutil.disk_usage(r).free/2**30,
 groups=groups,loads=load,statuses=statuses,processes=procs,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines())))
'''

def main(action):
    with connect() as c,c.open_sftp() as s:
        if action=='probe':
            d=json.loads(remote_python(c,PROBE))
            (HERE/'live_resources.json').write_text(json.dumps(d,indent=2),encoding='utf8')
            print(json.dumps(d));return
        if action=='submit':
            cfg=json.loads((HERE/'execution.json').read_text())
            if not cfg['cores']:raise ValueError('No assigned physical cores')
            base=DEST.rsplit('/',1)[0]
            try:s.stat(base)
            except FileNotFoundError:s.mkdir(base)
            s.mkdir(DEST)
            s.put(str(HERE/'sync01.tar.gz'),DEST+'/inputs.tar.gz')
            s.put(str(HERE/'sync_runner.py'),DEST+'/sync_runner.py')
            s.put(str(HERE/'execution.json'),DEST+'/execution.json')
            cores=','.join(map(str,cfg['cores']))
            cmd='cd '+shlex.quote(DEST)+' && taskset -c '+cores+' '+PYTHON+' -B sync_runner.py > controller.log 2>&1'
            transport.remote_exec(c,'tar -xzf '+DEST+'/inputs.tar.gz -C '+DEST+' && tmux new-session -d -s maskco_supplement_sync01 '+shlex.quote(cmd))
            print('submitted sync01',cores);return
        if action in ('status','recover'):
            try:
                with s.open(DEST+'/status.json') as f:d=json.load(f)
            except FileNotFoundError:
                print(transport.remote_exec(c,'tail -30 '+DEST+'/controller.log'));return
            print(json.dumps(d))
            if action=='recover':
                if d.get('end') is None:raise ValueError('Comparison remains active')
                transport.remote_exec(c,'cd '+DEST+' && tar -czf receipts.tar.gz out status.json controller.log execution.json phase.json')
                folder=HERE/'results/sync01';folder.mkdir(parents=True,exist_ok=True)
                s.get(DEST+'/receipts.tar.gz',str(folder/'receipts.tar.gz'))
                with tarfile.open(folder/'receipts.tar.gz') as z:
                    for item in z:
                        p=(folder/item.name).resolve()
                        if not p.is_relative_to(folder.resolve()):raise ValueError(item.name)
                        if item.isdir():p.mkdir(exist_ok=True,parents=True)
                        elif item.isfile():p.write_bytes(z.extractfile(item).read())
            return
        raise ValueError(action)

if __name__=='__main__':main(sys.argv[1])
