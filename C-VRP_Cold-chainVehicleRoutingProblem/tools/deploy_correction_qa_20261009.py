"""Deploy only a new, bounded engineering QA bundle explicitly authorized by user."""
from pathlib import Path, PurePosixPath
import sys,json,hashlib,shlex,paramiko,argparse
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'tools'))
from _srv import _cfg
parser=argparse.ArgumentParser();parser.add_argument('--fixed-only-recovery',action='store_true');args=parser.parse_args()
version='v3' if args.fixed_only_recovery else 'v2'
R=PurePosixPath('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/correction_validation_1009_'+version)
protocol={'version':1,'scope':'corrective engineering QA only; no P/D2 numerical rerun',
 'user_authorization':'2026-10-09 server inspection, corrections and necessary validation',
 'cpu':[54],'threads':1,'gpu':False,'per_stage_timeout_seconds':300,
 'stages':['18 portable synthetic contract/journal/pickup replay tests',
   '1 calibration instance, 2 fixed decision states, 1 stochastic sample per learned model',
   '1 training instance, 2 exhaustive headroom states','3 figures from archived JSON in a copied tree'],
 'expected_training_sha256':'e88c7d0d0d21c73b6335880a20111ee4d02b0a6693071872c74d217d34beb300',
 'expected_smoke_J0':1.4778266296,'expected_smoke_Mpre_det':1.5277455956,
 'expected_headroom_J0':1.1445566521,'numerical_tolerance':1e-8,
 'qualification':'runtime smoke only; mismatch of original training source remains disqualifying for full reproduction',
 'stopping':'no retries chosen by metrics; retain every stage error and old artifacts',
 'forbidden':['original result overwrite','upstream modification','team GPU/process interference','final pickup holdout','shared MD upload']}
if args.fixed_only_recovery:
 protocol.update(version=2,per_stage_timeout_seconds=900,stage_names=['fixed_state'],
  recovery_reason='v2 engineering cutoff at 300 seconds after first completed state; retain v2 and extend CPU engineering time only',
  prior_run='results/correction_validation_1009_v2',stages=[protocol['stages'][1]])
local=E/'results/three_line_corrections_20261009'
with (local/('SERVER_QA_PROTOCOL_'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(protocol,f,indent=2)
host,user,password=_cfg();c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,port=22,username=user,password=password,timeout=30);s=c.open_sftp()
def mkdir(p):
 try:s.stat(str(p))
 except FileNotFoundError:mkdir(p.parent);s.mkdir(str(p))
def put_bytes(p,data):
 mkdir(p.parent)
 with s.open(str(p),'wx') as f:f.write(data)
entries=[]
try:
 try:s.stat(str(R));raise RuntimeError('QA directory already exists; refuse duplicate launch')
 except FileNotFoundError:s.mkdir(str(R))
 for src,rel in [(E/'tools/server_correction_qa_20261009.py','runner.py'),
  (local/('SERVER_QA_PROTOCOL_'+version+'.json'),'protocol.json'),
  (E/'03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/RECOVERY_MANIFEST.json','recovery_manifest.json')]:
  data=src.read_bytes();put_bytes(R/rel,data);entries.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest()})
 for p in sorted((E/'scripts').rglob('*.py')):
  if '__pycache__' in p.parts:continue
  rel='engineering_code/'+p.relative_to(E/'scripts').as_posix();data=p.read_bytes()
  put_bytes(R/rel,data);entries.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest()})
 put_bytes(R/'UPLOAD_MANIFEST.json',json.dumps(entries,indent=2).encode())
 python='/home/hzeng/miniconda3/envs/MASKCO_env/bin/python'
 command='cd '+shlex.quote(str(R))+' && '+shlex.quote(python)+' runner.py > pipeline.log 2>&1; rc=$?; printf "%s\\n" "$rc" > pipeline.exit'
 _,out,err=c.exec_command('tmux new-session -d -s corrective_qa_1009_'+version+' '+shlex.quote(command),timeout=30)
 rc=out.channel.recv_exit_status();error=err.read().decode()
 if rc:raise RuntimeError(error)
 record={'directory':str(R),'session':'corrective_qa_1009_'+version,'uploaded_files':len(entries),'launch_exit':rc,
         'shared_documents_uploaded':False,'original_experiments_restarted':False}
 with (local/('SERVER_QA_LAUNCH_'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
 print(json.dumps(record))
finally:s.close();c.close()
