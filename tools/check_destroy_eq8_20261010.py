"""Read-only status, actual identity verification and complete exclusive download."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import paramiko
from _srv import _cfg, run

R='/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/research_1010_destroy_eq8_v1'
E=Path(__file__).resolve().parents[1]/'C-VRP_Cold-chainVehicleRoutingProblem'
CODE=r'''
from pathlib import Path
import os,json,hashlib,datetime
r=Path(RUN_DIRECTORY)
out={'time':datetime.datetime.now().isoformat(),'files':{},'progress_count':0}
for n in ['pipeline_state.json','qa.exit.json','QA_GATE.json','PILOT_START.json','pilot.exit.json','pipeline_failure.json','pipeline.exit','COMPLETED.json']:
    p=r/n
    if p.exists():
        out['files'][n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()
for stage in ['qa','pilot']:
    p=r/stage/'result.json'
    if p.exists():
        d=json.loads(p.read_text());out[stage]={'n':len(d.get('rows',[])),'qualified':d.get('fixed_schedule_qualified'),'QA':d.get('actual_engineering_qa_pass'),'paired':d.get('paired')}
    p=r/stage/'progress.jsonl'
    if p.exists():out[stage+'_progress_count']=sum(1 for s in p.open() if s.strip())
start=r/'SOURCE_START.json'
if start.exists():
    manifest=json.loads(start.read_text());bad=[]
    for p,h in manifest.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:bad.append(p)
    out['actual_identity_check']={'n':len(manifest),'mismatches':bad}
end=r/'SOURCE_END.json'
if end.exists():
    terminal=json.loads(end.read_text());bad=[]
    for p,h in terminal.items():
        if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:bad.append(p)
    initial=json.loads(start.read_text())
    added=set(terminal)-set(initial)
    out['independent_actual_end_identity']={'n':len(terminal),'mismatches':bad,
        'original_start_items_unchanged':all(terminal.get(p)==h for p,h in initial.items()),
        'only_pre_data_QA_linked_protocol_added':added=={str(r/'PILOT_PROTOCOL.json')}}
parents={}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        status=(p/'status').read_text();ppid=int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        parents[int(p.name)]=(ppid,cmd)
    except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):continue
roots=[]
for pid,(ppid,cmd) in parents.items():
    if (str(r)+'/pipeline.py' in cmd or str(r)+'/src/scripts/research_20261010/delivery_destroy_eq8.py' in cmd):roots.append(pid)
state=r/'pipeline_state.json'
if state.exists() and json.loads(state.read_text()).get('pid') in parents:roots.append(json.loads(state.read_text())['pid'])
seen=set(roots)
while True:
    new={p for p,(pp,cmd) in parents.items() if pp in seen}-seen
    if not new:break
    seen |= new
out['own_process_tree']={str(p):{'ppid':parents[p][0],'command':parents[p][1]} for p in sorted(seen)}
print(json.dumps(out))
'''

def main(args):
    code=CODE.replace('RUN_DIRECTORY',repr(R))
    encoded=base64.b64encode(code.encode()).decode()
    reply=run('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c "import base64;exec(base64.b64decode(\'%s\'))"'%encoded,timeout=90)
    status=json.loads(reply)
    root=E/'results/research_20261010_continuation_v2'
    i=1
    while (root/('SERVER_CHECK_%02d.json'%i)).exists():i+=1
    (root/('SERVER_CHECK_%02d.json'%i)).write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status))
    if not args.download:return
    if status['files'].get('pipeline.exit','').strip()!='0':raise RuntimeError('Only a completed run may be archived as complete')
    dest=root/'delivery_raw_v1';dest.mkdir(exist_ok=False)
    host,user,password=_cfg();c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=password,timeout=30);s=c.open_sftp()
    inventory=[]
    def get(folder,local):
        local.mkdir(exist_ok=True)
        for a in s.listdir_attr(folder):
            import stat
            remote=folder+'/'+a.filename;target=local/a.filename
            if stat.S_ISDIR(a.st_mode):get(remote,target)
            else:
                s.get(remote,str(target));inventory.append({'path':str(target.relative_to(dest)).replace('\\','/'),'size':a.st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    try:get(R,dest)
    finally:s.close();c.close()
    (root/'DELIVERY_COMPLETE_DOWNLOAD.json').write_text(json.dumps({'files':inventory,'remote':R,'complete_recursive_download':True},indent=2),encoding='utf-8')
    print(json.dumps({'downloaded':len(inventory),'directory':str(dest)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');main(p.parse_args())
