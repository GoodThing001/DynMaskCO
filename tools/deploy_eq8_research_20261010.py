"""Seal new extension-only source and launch QA-gated exploratory pilot once."""
from pathlib import Path,PurePosixPath
import hashlib,json,sys,shlex,paramiko
from _srv import _cfg
E=Path(__file__).resolve().parents[1]/'C-VRP_Cold-chainVehicleRoutingProblem'
local=E/'results/research_20261010/delivery_deployment_v1'
local.mkdir(exist_ok=False)
R=PurePosixPath('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/research_1010_d_eq8_v1')
reg={'version':'D-EQ8-v1','dataset_seed':2026101002,'mask_seed':2026101041,'arm_order_seed':2026101099,'bootstrap_seed':2026101077,'bootstrap_repetitions':2000,'random_model_seed':42,'steps':8,'mask_fraction':.3,'guide_weight':1.,'time_limit_s':10.,'instance_ids':list(range(32)),'stage':'synthetic_engineering_qa','scope':'fixed_schedule_snapshot_attribution_no_online_gain','registered_before_data':True,'cpu':55,'threads':1,'gpu':False,'per_stage_timeout_s':{'qa':900,'pilot':5400},'exploratory_mean_relative_threshold':.01,'all_failures_retained':True,'original_D2_immutable':True,'interpretation':'P/R/H same attempts, not consumed compute equality; n32 descriptive clue only'}
with (local/'PRE_DATA_PROTOCOL.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
runner=r'''from pathlib import Path
import os,sys,json,hashlib,subprocess,datetime,traceback
RUN=Path(__file__).resolve().parent
ROOT=Path('/home/hzeng/project/MASKCO-Main')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
    with (RUN/n).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def main():
    os.sched_setaffinity(0,{55})
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    src=RUN/'src';manifest=json.loads((RUN/'SOURCE_MANIFEST.json').read_text())
    start={str(src/p):sha(src/p) for p in manifest}
    if any(start[str(src/p)]!=h for p,h in manifest.items()):raise ValueError('Uploaded source mismatch')
    # Preserve actual immutable upstream source and binary closure as observed;
    # not derived from runner self qualification. Parent checks it after download.
    upstream=ROOT/'MASKCO_code'
    for p in sorted(upstream.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.so') and '__pycache__' not in p.parts:start[str(p)]=sha(p)
    cp=upstream/'ckpts/cvrp100.ckpt';start[str(cp)]=sha(cp)
    if start[str(cp)]!='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db':raise ValueError('CVRP bytes mismatch')
    for p in ('pipeline.py','PRE_DATA_PROTOCOL.json','SOURCE_MANIFEST.json'):start[str(RUN/p)]=sha(RUN/p)
    save('SOURCE_START.json',start)
    common=[sys.executable,str(src/'scripts/research_20261010/delivery_equal_schedule.py'),'--checkpoint',str(cp),'--upstream',str(upstream),'--source-manifest',str(RUN/'SOURCE_MANIFEST.json'),'--source-root',str(src)]
    reg=json.loads((RUN/'PRE_DATA_PROTOCOL.json').read_text())
    began=datetime.datetime.now().isoformat()
    with (RUN/'qa.log').open('xb') as f:rc=subprocess.run(common+['--protocol',str(RUN/'PRE_DATA_PROTOCOL.json'),'--out',str(RUN/'qa')],stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
    save('qa.exit.json',{'exit':rc,'started':began,'ended':datetime.datetime.now().isoformat()})
    if rc:raise RuntimeError('Actual model QA child failure; retain and do not generate research data')
    qa=json.loads((RUN/'qa/result.json').read_text());row=qa['rows'][0]
    valid=len(qa['rows'])==1 and row['instance_id']==-1 and qa['actual_engineering_qa_pass'] is True and qa['fixed_schedule_qualified'] is True and all(a['report']['hard'] is True and a['actual_attempts']==8 and a['complete_candidates']==8 and a['hard_candidates']==8 and a['fixed_schedule_qualified'] is True for a in row['arms'].values()) and all(row['arms'][a]['decode_calls']==8 for a in ('P','R')) and row['arms']['H']['decode_calls']==0
    save('QA_GATE.json',{'pass':valid,'actual_model_QA_sha256':sha(RUN/'qa/result.json'),'not_online_gain':True})
    if not valid:raise RuntimeError('QA fixed-schedule/activation qualification failed; do not generate research data')
    reg.update(stage='fresh_mechanism_pilot',synthetic_qa_artifact_sha256=sha(RUN/'qa/result.json'))
    save('PILOT_PROTOCOL.json',reg)
    began=datetime.datetime.now().isoformat()
    with (RUN/'pilot.log').open('xb') as f:rc=subprocess.run(common+['--protocol',str(RUN/'PILOT_PROTOCOL.json'),'--out',str(RUN/'pilot'),'--synthetic-qa-artifact',str(RUN/'qa/result.json')],stdout=f,stderr=subprocess.STDOUT,timeout=5400).returncode
    save('pilot.exit.json',{'exit':rc,'started':began,'ended':datetime.datetime.now().isoformat()})
    if rc:raise RuntimeError('Pilot child failure retained')
    end={p:sha(p) for p in start};save('SOURCE_END.json',end)
    if start!=end:raise RuntimeError('Source/weight closure changed')
    save('COMPLETED.json',{'source_stable':True,'independent_adjudication_pending':True,'online_gain_established':False,'stable_gain_established':False})
if __name__=='__main__':
    try:main()
    except Exception:
        save('pipeline_failure.json',{'error':traceback.format_exc()});raise
'''
with (local/'pipeline.py').open('x',encoding='utf-8') as f:f.write(runner)
if not (E/'scripts/research_20261010/delivery_equal_schedule.py').is_file():raise ValueError('Candidate not integrated')
files={p.relative_to(E).as_posix():p for p in sorted((E/'scripts').rglob('*.py')) if '__pycache__' not in p.parts}
manifest={p:hashlib.sha256(f.read_bytes()).hexdigest() for p,f in files.items()}
with (local/'SOURCE_MANIFEST.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
host,user,password=_cfg();c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=22,username=user,password=password,timeout=30);s=c.open_sftp()
def mkdir(p):
    try:s.stat(str(p))
    except FileNotFoundError:mkdir(p.parent);s.mkdir(str(p))
def put(p,data):
    mkdir(p.parent)
    with s.open(str(p),'wx') as f:f.write(data)
try:
    try:s.stat(str(R));raise RuntimeError('Refuse duplicate remote launch')
    except FileNotFoundError:s.mkdir(str(R))
    for p in local.iterdir():put(R/p.name,p.read_bytes())
    for rel,p in files.items():put(R/'src'/rel,p.read_bytes())
    cmd='cd '+shlex.quote(str(R))+' && timeout 8400s /home/hzeng/miniconda3/envs/MASKCO_env/bin/python pipeline.py > pipeline.log 2>&1; rc=$?; printf "%s\\n" "$rc" > pipeline.exit'
    _,o,e=c.exec_command('tmux new-session -d -s research_d_eq8_1010_v1 '+shlex.quote(cmd),timeout=30)
    rc=o.channel.recv_exit_status();err=e.read().decode()
    if rc:raise RuntimeError(err)
    record={'directory':str(R),'session':'research_d_eq8_1010_v1','uploaded_python_sources':len(manifest),'launch_exit':rc,'shared_md_uploaded':False,'qa_gates_data_generation':True,'protocol_sha256':hashlib.sha256((local/'PRE_DATA_PROTOCOL.json').read_bytes()).hexdigest()}
    with (local/'LAUNCH.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps(record))
finally:s.close();c.close()
