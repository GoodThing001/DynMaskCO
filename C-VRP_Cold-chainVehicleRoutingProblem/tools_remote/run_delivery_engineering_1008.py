"""Single finite, CPU-only QA in an isolated sealed delivery snapshot."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
import zipfile


def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def verify(src, manifest):
    mismatches=[p for p,h in manifest.items() if hashlib.sha256((src/p).read_bytes()).hexdigest()!=h]
    if mismatches:raise RuntimeError('Frozen source mismatch: '+str(mismatches))
    return {'checked':len(manifest),'source_stable':True}


def main():
    project=Path(__file__).resolve().parents[1]
    out=project/'results/delivery_engineering_1008_v1'
    out.mkdir(exist_ok=False)
    src=out/'src'
    src.mkdir()
    archive=project/'tools_remote/delivery_engineering_1008_v1.zip'
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target=(src/info.filename).resolve()
            if src.resolve() not in target.parents:raise ValueError('Archive path outside source')
        z.extractall(src)
    manifest=json.loads((src/'SOURCE_MANIFEST.json').read_text())
    dump(out/'SOURCE_START.json',verify(src,manifest))
    for p in src.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in src.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    src.chmod(0o555)
    os.sched_setaffinity(0,{54})
    import psutil
    resources={'server_time':str(datetime.datetime.now()),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'memory_available_gb':psutil.virtual_memory().available/2**30,
        'cpu54_utilisation_before_percent':psutil.cpu_percent(interval=1,percpu=True)[54],
        'gpu_allocated':False,'max_seconds':300,'online_or_pilot_evidence':False}
    dump(out/'RESOURCE_RECORD.json',resources)
    dump(out/'pipeline_state.json',{'stage':'engineering_qa_running','time':str(datetime.datetime.now())})
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        with (out/'qa.log').open('w') as log:
            result=subprocess.run([sys.executable,str(src/'scripts/tests/test_delivery_contract.py'),
                '--out',str(out/'qa'),'--baselines'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        (out/'qa.exit').write_text(str(result.returncode))
        if result.returncode:raise RuntimeError('Delivery QA failed; inspect qa.log')
        report=json.loads((out/'qa/delivery_engineering_report.json').read_text())
        if not report.get('delivery_contract_pass') or len(report.get('baseline_smoke',{}))!=2:
            raise RuntimeError('Missing real baseline QA')
        dump(out/'SOURCE_END.json',verify(src,manifest))
        dump(out/'COMPLETED.json',{'engineering_qualified':True,'formal_pilot_started':False,
            'online_gain_established':False,'time':str(datetime.datetime.now())})
        dump(out/'pipeline_state.json',{'stage':'engineering_pass_pilot_not_started','time':str(datetime.datetime.now())})
        (out/'pipeline.exit').write_text('0')
    except Exception as e:
        dump(out/'failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        (out/'pipeline.exit').write_text('1')
        raise


if __name__=='__main__':main()
