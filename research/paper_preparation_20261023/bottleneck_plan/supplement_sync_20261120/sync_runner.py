"""Run the registered paired blocks on disjoint physical CPU cores."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import psutil

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'phase.json').read_text())
EXEC=json.loads((HERE/'execution.json').read_text())
OUT=HERE/'out'
now=lambda:datetime.now(timezone.utc).isoformat()

def main():
    cores=EXEC['cores']
    if set(os.sched_getaffinity(0))!=set(cores):raise ValueError('Controller affinity differs')
    groups=[]
    for core in cores:
        p=Path(f'/sys/devices/system/cpu/cpu{core}/topology')
        groups.append(tuple((p/x).read_text().strip() for x in ('physical_package_id','core_id')))
    if len(set(groups))!=len(cores):raise ValueError('Selected CPUs share physical cores')
    project=HERE.parents[1]
    fleet=[]
    for path in (project/'_maskco_perishable_external_20261119').glob('fleet*/status.json'):
        state=json.loads(path.read_text())
        if not state.get('end'):
            for cpu in state['cores']:
                p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
                g=tuple((p/x).read_text().strip() for x in ('physical_package_id','core_id'))
                if g in groups:raise RuntimeError('A selected physical core belongs to an active fleet run')
            fleet.append(str(path))
    if psutil.virtual_memory().available<4.5*2**30*len(cores):raise RuntimeError('Insufficient available RAM')
    OUT.mkdir()
    lock=threading.Lock();work=iter(enumerate(CFG['blocks']))
    state=dict(start=now(),end=None,total=len(CFG['blocks']),total_cells=CFG['total_cells'],
               cores=cores,active={},complete=0,cells=0,failed=[],active_fleet_phases=fleet,
               resource_start=dict(ram_gib=psutil.virtual_memory().available/2**30))
    def save():
        p=HERE/'status.tmp';p.write_text(json.dumps(state));p.replace(HERE/'status.json')
    save()
    def lane(core):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('STATIC_PROFIT_SCOPE_FILE',None)
        while True:
            with lock:
                if state['failed']:return
                item=next(work,None)
                if item is None:return
                index,block=item
            initial=None
            for method in block['methods']:
                with lock:
                    if state['failed']:return
                    state['active'][core]=dict(block=index,case=block['case'],seed=block['seed'],method=method);save()
                started=time.perf_counter();utc=now();error=None
                with (OUT/f'{index}_{method}.log').open('w') as log:
                    try:
                        rc=subprocess.run(['taskset','-c',str(core),sys.executable,'-B',str(HERE/'sync_worker.py'),str(index),method],
                                          env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
                        elapsed=time.perf_counter()-started
                        if rc:raise RuntimeError(f'Worker exit {rc}')
                        p=OUT/f'{index}_{method}.json';row=json.loads(p.read_text())
                        if initial is None:initial=row['initial_routes']
                        if row['initial_routes']!=initial:raise ValueError('Paired initial routes disagree')
                        if row['cpu']!=[core]:raise ValueError('Worker CPU differs from assigned physical core')
                        row.update(controller_process_s=elapsed,controller_started_utc=utc,controller_finished_utc=now())
                        p.write_text(json.dumps(row,ensure_ascii=False,allow_nan=False))
                    except Exception as exc:
                        log.write(traceback.format_exc());error=str(exc)
                with lock:
                    if error:
                        state['failed'].append(dict(block=index,method=method,error=error));state['active'].pop(core,None);save();return
                    state['cells']+=1;save()
            with lock:
                state['active'].pop(core,None);state['complete']+=1;save()
    with ThreadPoolExecutor(max_workers=len(cores)) as pool:list(pool.map(lane,cores))
    state['end']=now();save()

if __name__=='__main__':main()
