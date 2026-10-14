"""Finite CPU-only delivery pilot, independent from the pickup pipeline."""
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
    out=project/'results/delivery_pilot_1008_v1'
    # Both actual engineering chains must have finished successfully first.
    for r in ['delivery_engineering_1008_v1','delivery_provider_1008_v1']:
        if (project/'results'/r/'pipeline.exit').read_text().strip()!='0':raise ValueError('Missing delivery engineering activation')
    out.mkdir(exist_ok=False)
    root=out/'src';ext=root/'C-VRP_Cold-chainVehicleRoutingProblem';ext.mkdir(parents=True)
    with zipfile.ZipFile(project/'tools_remote/delivery_pilot_1008_v1.zip') as z:
        for info in z.infolist():
            if ext.resolve() not in (ext/info.filename).resolve().parents:raise ValueError('Unsafe archive entry')
        z.extractall(ext)
    expected=json.loads((ext/'SOURCE_MANIFEST.json').read_text())
    assert all(sha(ext/p)==h for p,h in expected.items())
    upstream=root/'MASKCO_code'
    for p in (project.parent/'MASKCO_code').rglob('*.py'):
        if any(x in ('__pycache__','.git') for x in p.parts):continue
        q=upstream/p.relative_to(project.parent/'MASKCO_code');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    manifest={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    dump(out/'SOURCE_MANIFEST.json',manifest)
    cp=project.parent/'MASKCO_code/ckpts/cvrp100.ckpt';cp_sha=sha(cp)
    assert cp_sha=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
    dump(out/'WEIGHT_START.json',{'path':str(cp),'sha256':cp_sha})
    protocol=ext/'configs/delivery_20261008_v1/protocol.json'
    assert sha(protocol)=='8e40c705e57a6b6f2ed2244fc9c73108b43db7757596ff8ed03283f110cc3642'
    shutil.copy2(protocol,out/'protocol.json')
    for p in root.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in root.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    root.chmod(0o555)
    os.sched_setaffinity(0,{54})
    dump(out/'pipeline_state.json',{'stage':'finite_route_pilot_model_QA_then_32_instances','time':str(datetime.datetime.now()),
        'cpu_affinity':[54],'gpu_allocated':False,'max_seconds':7200})
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    scripts=ext/'scripts/delivery'
    try:
        with (out/'run.log').open('w') as log:
            rc=subprocess.run([sys.executable,str(scripts/'run_delivery_pilot.py'),
                '--protocol',str(protocol),'--upstream',str(upstream),'--checkpoint',str(cp),
                '--out',str(out/'pilot'),'--manifest-path',str(out/'SOURCE_MANIFEST.json')],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200).returncode
        (out/'run.exit').write_text(str(rc))
        if rc:raise RuntimeError('Delivery pilot runner failed; preserve all outputs')
        assert all(sha(root/p)==h for p,h in manifest.items()) and sha(cp)==cp_sha
        dump(out/'SOURCE_END.json',{'checked':len(manifest),'source_stable':True,'weights_stable':True})
        with (out/'adjudication.log').open('w') as log:
            rc=subprocess.run([sys.executable,str(scripts/'adjudicate_delivery_pilot.py'),
                '--result',str(out/'pilot/result.json'),'--protocol',str(protocol),'--out',str(out/'PILOT_ADJUDICATION.json')],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        (out/'adjudication.exit').write_text(str(rc))
        if rc:raise RuntimeError('Independent delivery pilot adjudication failed')
        result=json.loads((out/'PILOT_ADJUDICATION.json').read_text())
        dump(out/'COMPLETED.json',{'formal_pilot_adjudicable':result['formal_pilot_adjudicable'],
            'pilot_pass':result['pilot_pass'],'online_gain_established':False,'time':str(datetime.datetime.now())})
        dump(out/'pipeline_state.json',{'stage':'route_pilot_adjudicated','pilot_pass':result['pilot_pass'],'time':str(datetime.datetime.now())})
        (out/'pipeline.exit').write_text('0')
    except Exception as e:
        dump(out/'failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        (out/'pipeline.exit').write_text('1');raise


if __name__=='__main__':main()
