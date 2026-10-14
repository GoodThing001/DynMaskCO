"""Exclusive new deployment; QA gates new data; no shared Markdown upload."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shlex
import sys
import paramiko
from _srv import _cfg

E = Path(__file__).resolve().parents[1] / 'C-VRP_Cold-chainVehicleRoutingProblem'
LOCAL = E / 'results/research_20261010_continuation_v2/delivery_deployment_v1'
REMOTE = PurePosixPath('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/research_1010_destroy_eq8_v1')
LOCAL.mkdir(exist_ok=False)

def save(p, d):
    with p.open('x', encoding='utf-8') as f:
        json.dump(d, f, indent=2, allow_nan=False)

qa = json.loads((E/'scripts/research_20261010/delivery_destroy_eq8_qa_protocol.json').read_text(encoding='utf-8'))
pilot = json.loads((E/'scripts/research_20261010/delivery_destroy_eq8_pilot_protocol_DRAFT.json').read_text(encoding='utf-8'))
save(LOCAL/'QA_PROTOCOL.json', qa)
save(LOCAL/'PRE_DATA_PILOT_TEMPLATE.json', pilot)
save(LOCAL/'EXECUTION_REGISTRATION.json', {'cpu':54, 'threads':1, 'gpu':False,
    'qa_timeout_s':900, 'pilot_timeout_s':5400, 'outer_timeout_s':8400,
    'parent_program_sha256':'bc8cfcb51dba940af539e23194fad7c7e7caf681a1e98e68a5b296c7286632d5',
    'scope':'new fixed-opportunity destroy information attribution; no online/stable claim',
    'pilot_template_frozen_before_data':True, 'only_QA_hash_filled_after_QA':True,
    'all_cases_retained':True, 'old_RESULTS_immutable':True, 'shared_md_uploaded':False})

PIPELINE = r'''
from pathlib import Path
import os, sys, json, hashlib, subprocess, datetime, traceback
RUN=Path(__file__).resolve().parent
ROOT=Path('/home/hzeng/project/MASKCO-Main')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,d):
    with (RUN/name).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def main():
    os.sched_setaffinity(0,{54})
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    src=RUN/'src';manifest=json.loads((RUN/'SOURCE_MANIFEST.json').read_text())
    actual={str(src/p):sha(src/p) for p in manifest}
    if any(actual[str(src/p)]!=h for p,h in manifest.items()):raise ValueError('Uploaded source mismatch')
    upstream=ROOT/'MASKCO_code'
    for p in sorted(upstream.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.so') and '__pycache__' not in p.parts:actual[str(p)]=sha(p)
    cp=upstream/'ckpts/cvrp100.ckpt';actual[str(cp)]=sha(cp)
    if actual[str(cp)]!='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db':raise ValueError('Actual CVRP checkpoint mismatch')
    for p in ('pipeline.py','QA_PROTOCOL.json','PRE_DATA_PILOT_TEMPLATE.json','EXECUTION_REGISTRATION.json','SOURCE_MANIFEST.json'):actual[str(RUN/p)]=sha(RUN/p)
    save('SOURCE_START.json',actual)
    common=[sys.executable,str(src/'scripts/research_20261010/delivery_destroy_eq8.py'),'--checkpoint',str(cp),'--upstream',str(upstream),'--source-manifest',str(RUN/'SOURCE_MANIFEST.json'),'--source-root',str(src)]
    save('pipeline_state.json',{'stage':'actual_synthetic_QA','started':datetime.datetime.now().isoformat(),'pid':os.getpid(),'cpu':[54]})
    start=datetime.datetime.now().isoformat()
    with (RUN/'qa.log').open('xb') as f:rc=subprocess.run(common+['--protocol',str(RUN/'QA_PROTOCOL.json'),'--out',str(RUN/'qa')],stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
    save('qa.exit.json',{'exit':rc,'started':start,'ended':datetime.datetime.now().isoformat()})
    if rc:raise RuntimeError('Actual new QA failed, no research data generation')
    qa=json.loads((RUN/'qa/result.json').read_text())
    if qa.get('actual_engineering_qa_pass') is not True or qa.get('fixed_schedule_qualified') is not True:raise RuntimeError('Actual eight-attempt QA eligibility failed')
    pilot=json.loads((RUN/'PRE_DATA_PILOT_TEMPLATE.json').read_text())
    pilot['synthetic_qa_artifact_sha256']=sha(RUN/'qa/result.json')
    save('PILOT_PROTOCOL.json',pilot)
    actual[str(RUN/'PILOT_PROTOCOL.json')]=sha(RUN/'PILOT_PROTOCOL.json')
    save('QA_GATE.json',{'pass':True,'actual_qa_sha256':sha(RUN/'qa/result.json'),'not_online_or_scientific_gain':True})
    save('PILOT_START.json',{'stage':'fresh_32_snapshot_pilot','started':datetime.datetime.now().isoformat(),'pid':os.getpid(),'cpu':[54]})
    start=datetime.datetime.now().isoformat()
    with (RUN/'pilot.log').open('xb') as f:rc=subprocess.run(common+['--protocol',str(RUN/'PILOT_PROTOCOL.json'),'--out',str(RUN/'pilot'),'--synthetic-qa-artifact',str(RUN/'qa/result.json')],stdout=f,stderr=subprocess.STDOUT,timeout=5400).returncode
    save('pilot.exit.json',{'exit':rc,'started':start,'ended':datetime.datetime.now().isoformat()})
    if rc:raise RuntimeError('New pilot child failed; keep all cases and failure')
    end={p:sha(p) for p in actual};save('SOURCE_END.json',end)
    if actual!=end:raise RuntimeError('Actual source/weight/protocol changed')
    save('COMPLETED.json',{'source_stable':True,'independent_adjudication_pending':True,'online_gain_established':False,'stable_gain_established':False})
if __name__=='__main__':
    try:main()
    except Exception:
        save('pipeline_failure.json',{'traceback':traceback.format_exc()});raise
'''
(LOCAL/'pipeline.py').write_text(PIPELINE, encoding='utf-8')
files = {p.relative_to(E).as_posix():p for p in sorted((E/'scripts').rglob('*.py')) if '__pycache__' not in p.parts}
manifest = {rel:hashlib.sha256(p.read_bytes()).hexdigest() for rel,p in files.items()}
save(LOCAL/'SOURCE_MANIFEST.json',manifest)
host,user,password = _cfg()
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=password,timeout=30)
s=c.open_sftp()
def mkdir(path):
    try:s.stat(str(path))
    except FileNotFoundError:mkdir(path.parent);s.mkdir(str(path))
def put(path,data):
    mkdir(path.parent)
    with s.open(str(path),'wx') as f:f.write(data)
try:
    try:s.stat(str(REMOTE));raise RuntimeError('Refuse duplicate remote experiment')
    except FileNotFoundError:s.mkdir(str(REMOTE))
    for p in LOCAL.iterdir():put(REMOTE/p.name,p.read_bytes())
    for rel,p in files.items():put(REMOTE/'src'/rel,p.read_bytes())
    cmd='cd '+shlex.quote(str(REMOTE))+' && timeout 8400s /home/hzeng/miniconda3/envs/MASKCO_env/bin/python pipeline.py > pipeline.log 2>&1; rc=$?; printf "%s\\n" "$rc" > pipeline.exit'
    _,o,e=c.exec_command('tmux new-session -d -s research_destroy_eq8_1010_v1 '+shlex.quote(cmd),timeout=30)
    rc=o.channel.recv_exit_status();err=e.read().decode()
    if rc:raise RuntimeError(err)
    record={'directory':str(REMOTE),'session':'research_destroy_eq8_1010_v1','uploaded_python_sources':len(manifest),'cpu':54,'launch_exit':rc,
        'pre_data_template_sha256':hashlib.sha256((LOCAL/'PRE_DATA_PILOT_TEMPLATE.json').read_bytes()).hexdigest(),'shared_md_uploaded':False}
    save(LOCAL/'LAUNCH.json',record)
    print(json.dumps(record))
finally:s.close();c.close()
