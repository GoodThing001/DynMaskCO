"""Run exactly the registered new public-snapshot requests once, no retries."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,digest
from timed_evidence_io_v1 import save
def main():
    import psutil
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if not reg['execution_authorized'] or not reg['independent_timed_request_checker_sealed']:raise ValueError('Exact registered source chain required')
    if (BASE/'REQUESTS_STARTED.json').exists():raise RuntimeError('No duplicate driver or request replay')
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('Registered package:'+n)
    psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Driver8GiB barrier')
    raw=(BASE/'CHAIN_INPUT.json').read_bytes();row=json.loads(raw.decode('utf-8'))
    if digest(row)!=reg['input_digest']:raise ValueError('Original input digest')
    from delivery_score_split_parent_v1 import request
    save(BASE,'REQUESTS_STARTED.json',dict(clock=time.perf_counter(),requests=reg['requests'],optimizer_steps=0))
    receipts=[]
    for spec in reg['requests']:
        stamp=time.perf_counter()
        save(BASE,'%s_fault%d_STARTED.json'%(spec['arm'],spec['fault_stage']),dict(clock=stamp,request=spec))
        # Public serialization and fallback certificate exist before each clock;
        # no target/score/model/route/C0 warmup or old score reuse is performed.
        receipt=request(BASE,sys.executable,spec['arm'],raw,row['initial'],spec['fault_stage'])
        receipts.append(receipt)
    save(BASE,'RESULT.json',dict(receipts=receipts,count=len(receipts),all_reaped=all(x['worker_reaped'] for x in receipts),
        all_within_total_window=all(x['within_total_window'] for x in receipts),
        selected_count=sum(x['selected'] for x in receipts),optimizer_steps=0,
        public_snapshot_only=True,full_request_budget_qualified=False,formal_controls_qualified=False,
        full_scientific_loaded_closure_qualified=False,scientific_family3_started=False,stable_online_gain=False))
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save(BASE,'FAILURE.json',dict(traceback=traceback.format_exc(),qualification=False,no_automatic_retry=True));raise
