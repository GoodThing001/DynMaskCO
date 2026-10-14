"""Independent raw identity replay of the import-only 3080 runtime census."""
import hashlib,json,time
from pathlib import Path
import paramiko
from _srv import _cfg
from _srv_exec_v2 import build,run_capture
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/review_3080_20261010/delivery/training_runtime_preflight_root_v4'
REMOTE_DIR='C:/Users/zh/project/MASKCO-Main/research/review_3080_20261010/training_runtime_preflight_root_v4'
REMOTE_AUDIT=r'''
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import hashlib,importlib.metadata,json,sys
from pathlib import Path
import psutil
psutil.Process().cpu_affinity([15])
root=Path(__file__).parent
input_path=root/'AUDIT_INPUT.json'
data=json.loads(input_path.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
actual={};missing=[]
for s in data['source_start']:
    p=Path(s)
    if not p.is_file():missing.append(s)
    else:actual[s]=sha(p)
versions={n:importlib.metadata.version(n) for n in data['packages']}
import jax,jax.numpy as jnp,numpy as np
v,g=jax.value_and_grad(lambda x:jnp.dot(x,x))(jnp.asarray([1.,2.],dtype=jnp.float32))
print(json.dumps({'actual_hashes':actual,'missing':missing,'packages':versions,'executable':sys.executable,
    'python':sys.version,'cpu':psutil.Process().cpu_affinity(),'devices':[str(d) for d in jax.devices()],
    'independent_scalar_value':float(v),'independent_scalar_gradient':np.asarray(g).tolist(),
    'input_sha256':sha(input_path),'input_end_sha256':sha(input_path),'audit_program_sha256':sha(Path(__file__))},allow_nan=False))
'''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    data=json.loads((OUT/'RESULT.json').read_text());reg=json.loads((OUT/'REGISTRATION.json').read_text())
    program=OUT/'INDEPENDENT_REMOTE_AUDIT.py'
    with program.open('x',encoding='utf-8') as f:f.write(REMOTE_AUDIT)
    host,user,password=_cfg('.vscode/sftp.3080.json');client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host,port=22,username=user,password=password,timeout=30);sftp=client.open_sftp()
        with sftp.open(REMOTE_DIR+'/AUDIT_INPUT.json','wx') as f:f.write((OUT/'RESULT.json').read_bytes())
        with sftp.open(REMOTE_DIR+'/independent_runtime_audit.py','wx') as f:f.write(program.read_bytes())
        sftp.close()
    finally:client.close()
    raw=run_capture(build('.vscode/sftp.3080.json',r'C:\Users\zh\project\MASKCO-Main',reg['python'],'-B',
                  [REMOTE_DIR+'/independent_runtime_audit.py'],'windows'),180,'.vscode/sftp.3080.json')
    write(OUT/'INDEPENDENT_REMOTE_CAPTURE.json',raw)
    if raw['exit_code']!=0:raise RuntimeError('Independent runtime check failed; capture retained')
    actual=json.loads(raw['stdout']);checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('registration_identity',sha(OUT/'REGISTRATION.json')==data['registration_sha256'])
    ck('remote_uploaded_program_identity',reg['runtime_program_sha256']==sha(OUT/'REMOTE_CENSUS_PROGRAM.py'))
    ck('raw_begin_end_exact_file_set',set(data['source_start'])==set(data['source_end']))
    for name,expected in data['source_start'].items():
        ck('file:'+name,expected==data['source_end'].get(name)==actual['actual_hashes'].get(name))
    ck('no_actual_missing',not actual['missing']);ck('no_unsealed_new_files',not data['new_loaded_files'])
    ck('no_changed_begin_end_files',not data['changed_or_missing_files'])
    ck('independent_actual_packages',data['packages']==actual['packages'])
    ck('independent_interpreter',data['executable']==actual['executable'] and data['python']==actual['python'])
    ck('actual_cpu15',data['actual_cpu_affinity']==actual['cpu']==[15])
    ck('no_gpu_backend',all('cpu' in x.lower() for x in actual['devices']) and all('cpu' in x.lower() for x in data['synthetic_after']['backend_devices']))
    ck('synthetic_before_after',data['synthetic_before']==data['synthetic_after'])
    ck('independent_tiny_autodiff',actual['independent_scalar_value']==5. and actual['independent_scalar_gradient']==[2.,4.])
    ck('remote_input_exact',actual['input_sha256']==actual['input_end_sha256']==sha(OUT/'RESULT.json'))
    ck('actual_auditor_exact',actual['audit_program_sha256']==sha(program))
    ck('census_source_in_actual_inventory',data['source_start'].get(str(Path(REMOTE_DIR+'/runtime_census.py')))==sha(OUT/'REMOTE_CENSUS_PROGRAM.py'))
    ck('scientific_training_not_claimed',data['scientific_training_start_end_qualified'] is False and data['full_model_gradient_qualified'] is False)
    ck('no_new_data_or_training_claim',data['new_scientific_data_generated'] is False and data['model_training_performed'] is False)
    ck('no_gain_claim',data['stable_online_gain'] is False)
    failed=sum(not x['pass'] for x in checks)
    write(OUT/'INDEPENDENT_RUNTIME_ADJUDICATION.json',{'checks':checks,'checks_count':len(checks),'failed':failed,
        'actual_files':len(actual['actual_hashes']),'result_sha256':sha(OUT/'RESULT.json'),'registration_sha256':sha(OUT/'REGISTRATION.json'),
        'independent_remote_capture_sha256':sha(OUT/'INDEPENDENT_REMOTE_CAPTURE.json'),'import_runtime_qualified':failed==0,
        'full_actual_training_qualified':False,'new_science_family_registered':False,'stable_MaskCO_online_gain':False,
        'scope':'Independent remote byte/version/backend replay and tiny algebra only; actual full training must seal its own start/end and actual gradients'})
    print(json.dumps({'checks':len(checks),'failed':failed,'files':len(actual['actual_hashes']),'failed_names':[x['name'] for x in checks if not x['pass']]}))
    return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
