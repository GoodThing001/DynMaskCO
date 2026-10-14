"""Freeze read-only upstream dependencies and run finite actual decoder QA on CPU54."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))

def main():
    project=Path(__file__).resolve().parents[1]
    out=project/'results/delivery_provider_1008_v1';out.mkdir(exist_ok=False)
    root=out/'src';ext=root/'C-VRP_Cold-chainVehicleRoutingProblem';ext.mkdir(parents=True)
    with zipfile.ZipFile(project/'tools_remote/delivery_provider_1008_v1.zip') as z:
        for info in z.infolist():
            if ext.resolve() not in (ext/info.filename).resolve().parents:raise ValueError('Unsafe archive entry')
        z.extractall(ext)
    upstream=root/'MASKCO_code'
    for p in (project.parent/'MASKCO_code').rglob('*.py'):
        if any(x in ('__pycache__','.git') for x in p.parts):continue
        q=upstream/p.relative_to(project.parent/'MASKCO_code');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    manifest={str(p.relative_to(root)):sha(p) for p in root.rglob('*.py')}
    expected=json.loads((ext/'SOURCE_MANIFEST.json').read_text())
    assert all(sha(ext/p)==h for p,h in expected.items())
    checkpoint=project.parent/'MASKCO_code/ckpts/cvrp100.ckpt'
    checkpoint_sha=sha(checkpoint)
    assert checkpoint_sha=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
    dump(out/'SOURCE_MANIFEST.json',manifest)
    dump(out/'WEIGHT_START.json',{'sha256':checkpoint_sha,'path':str(checkpoint)})
    for p in root.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in root.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    root.chmod(0o555)
    os.sched_setaffinity(0,{54})
    dump(out/'pipeline_state.json',{'stage':'actual_pretrained_decoder_engineering_qa','time':str(datetime.datetime.now()),
         'cpu_affinity':[54],'gpu_allocated':False,'max_seconds':900})
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        with (out/'qa.log').open('w') as log:
            rc=subprocess.run([sys.executable,str(ext/'scripts/tests/validate_delivery_pretrained.py'),
                '--checkpoint',str(checkpoint),'--upstream',str(upstream),'--out',str(out/'qa')],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        (out/'qa.exit').write_text(str(rc))
        if rc:raise RuntimeError('Actual decoder engineering QA failed; inspect qa.log')
        assert all(sha(root/p)==h for p,h in manifest.items()) and sha(checkpoint)==checkpoint_sha
        dump(out/'SOURCE_END.json',{'checked':len(manifest),'source_stable':True,'weights_stable':True})
        dump(out/'COMPLETED.json',{'actual_route_provider_verified':True,'formal_pilot_started':False,'online_gain_established':False})
        dump(out/'pipeline_state.json',{'stage':'actual_provider_engineering_pass_pilot_not_started','time':str(datetime.datetime.now())})
        (out/'pipeline.exit').write_text('0')
    except Exception as e:
        dump(out/'failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        (out/'pipeline.exit').write_text('1');raise

if __name__=='__main__':main()
