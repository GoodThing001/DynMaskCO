"""Exclusive new engineering PY/JSON bundle; no shared MD upload or old writes."""
from pathlib import Path
import sys,json,hashlib,shlex,datetime
import paramiko
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E.parent/'tools'))
from _srv import _cfg

def sha(raw):return hashlib.sha256(raw).hexdigest()

host,user,password=_cfg();ssh=paramiko.SSHClient();ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host,port=22,username=user,password=password,timeout=20)
remote='/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/literature_quality_qa_1009_v1'
sftp=ssh.open_sftp();sftp.mkdir(remote)
names=['literature_quality.py','literature_quality_profiles_v1.json','test_literature_quality.py']
manifest={}
for n in names:
 raw=(E/'scripts/correction_engineering'/n).read_bytes();manifest[n]=sha(raw)
 with sftp.file(remote+'/'+n,'wx') as f:f.write(raw)
 assert sha(sftp.open(remote+'/'+n,'rb').read())==manifest[n]
protocol={'scope':'11 synthetic commodity kinetics and units tests, not online research',
          'created':datetime.datetime.now().isoformat(),'files':manifest,
          'cpu':[54],'gpu':False,'max_seconds':60,'protected_old_runs_modified':False}
with sftp.file(remote+'/PROTOCOL.json','wx') as f:f.write(json.dumps(protocol).encode())
code="""
import json,hashlib,subprocess,datetime,os
from pathlib import Path
R=Path.cwd();p=json.loads((R/'PROTOCOL.json').read_text());h=lambda: {n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in p['files']}
before=h();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
x=subprocess.run(['taskset','-c','54','/home/hzeng/miniconda3/envs/MASKCO_env/bin/python','-m','unittest','test_literature_quality','-v'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
(R/'tests.log').write_text(x.stdout);after=h()
result={'checked_at':str(datetime.datetime.now()),'exit':x.returncode,'source_before':before,'source_after':after,'expected':p['files'],'source_match':before==after==p['files'],'formal_online_gain':False}
with (R/'ACCEPTANCE.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
"""
cmd='cd '+shlex.quote(remote)+' && /home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c '+shlex.quote(code)
stdin,stdout,stderr=ssh.exec_command(cmd,timeout=90);result=json.loads(stdout.read());error=stderr.read().decode();exitcode=stdout.channel.recv_exit_status()
assert exitcode==0,(exitcode,error)
A=E/'results/three_line_corrections_20261009/literature_quality_qa';A.mkdir()
for n in names+['PROTOCOL.json','tests.log','ACCEPTANCE.json']:
 raw=sftp.open(remote+'/'+n,'rb').read()
 with (A/n).open('xb') as f:f.write(raw)
 if n in manifest:assert sha(raw)==manifest[n]
sftp.close();ssh.close()
assert result['exit']==0 and result['source_match']
print(json.dumps({'tests_passed':11,'server':remote,'source_match':True}))
