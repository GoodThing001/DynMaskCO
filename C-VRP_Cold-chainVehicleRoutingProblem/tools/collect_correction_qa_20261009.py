"""Read-only download and independent end-identity audit for the isolated QA."""
from pathlib import Path,PurePosixPath
import sys,json,hashlib,shlex,paramiko,stat,argparse
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E.parent/'tools'))
from _srv import _cfg
parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v2','v3'],default='v2');args=parser.parse_args()
R=PurePosixPath('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/correction_validation_1009_'+args.version)
D=E/('results/three_line_corrections_20261009/server_qa_'+args.version)
host,user,password=_cfg();c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,port=22,username=user,password=password,timeout=30);s=c.open_sftp()
entries=[]
def fetch(p,rel):
 with s.open(str(p),'rb') as f:data=f.read()
 out=D/str(rel);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as f:f.write(data)
 entries.append({'path':str(rel),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
def walk(p):
 for a in s.listdir_attr(str(p)):
  sub=p/a.filename;rel=sub.relative_to(R)
  if stat.S_ISDIR(a.st_mode):
   if str(rel) in ('recovered','workspace'):continue
   walk(sub)
  elif sub.suffix in ('.json','.log','.exit','.png') or sub.name=='runner.py':fetch(sub,rel)
try:
 s.stat(str(R/'QA_RESULT.json'));s.stat(str(R/'pipeline.exit'))
 if D.exists():raise ValueError('Local QA archive exists; refuse overwrite')
 walk(R)
 try:figures=s.listdir_attr(str(R/'recovered/negative_result_paper/release/figures'))
 except FileNotFoundError:figures=[]
 for a in figures:
  if a.filename.endswith('.png'):
   fetch(R/'recovered/negative_result_paper/release/figures'/a.filename,PurePosixPath('figures')/a.filename)
 code='''
import json,hashlib,datetime
from pathlib import Path
R=Path(%r)
start=json.loads((R/'SOURCE_START.json').read_text())
actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in start}
upload=json.loads((R/'UPLOAD_MANIFEST.json').read_text())
uploaded=[]
for row in upload:
 p=R/row['path']
 if row['path'].startswith('engineering_code/'):
  p=R/'workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts'/row['path'][len('engineering_code/'):]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:uploaded.append(row['path'])
print(json.dumps({'checked_at':str(datetime.datetime.now()),'file_count':len(actual),
 'mismatch':[p for p in start if actual[p]!=start[p]],'actual_sha256':actual,
 'uploaded_file_count':len(upload),'upload_mismatch':uploaded}))
''' % str(R)
 _,out,err=c.exec_command('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c '+shlex.quote(code),timeout=60)
 data=json.loads(out.read().decode());rc=out.channel.recv_exit_status()
 if rc:raise RuntimeError(err.read().decode())
 with (D/'INDEPENDENT_ACTUAL_END_IDENTITY.json').open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
 with (D/'DOWNLOAD_MANIFEST.json').open('x',encoding='utf-8') as f:json.dump(entries,f,indent=2)
 print(json.dumps({'downloaded_files':len(entries),'actual_files_checked':data['file_count'],
                  'mismatch':data['mismatch'],'upload_mismatch':data['upload_mismatch']}))
finally:s.close();c.close()
