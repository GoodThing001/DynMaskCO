import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import json,time,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from parallel_runtime_identity_v1 import closure_parallel
from startup_native_identity_v1 import prime_native_identity
def main():
 import psutil
 native=prime_native_identity();psutil.Process().cpu_affinity([15])
 reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 if psutil.virtual_memory().available<8*2**30:raise RuntimeError('8GiB resource barrier')
 from delivery_score_split_engine_v1 import InitialEngine
 engine=InitialEngine(B,reg,'pretrained_initial')
 initial=dict(backbone=engine.inventory(engine.model),head=engine.inventory(engine.head))
 save(B,'INITIAL_LIVE_PARAMETERS.json',initial)
 pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix='identity_cpu15')
 try:
  # Native/stdlib/thread startup only, never model/sample or file hash warmup.
  barrier=threading.Barrier(4);futures=[pool.submit(barrier.wait,20) for _ in range(4)]
  for f in futures:f.result(timeout=30)
  extras=engine.extras;trees=engine.trees;closure(extras,trees)
  start=closure(extras,trees);save(B,'SOURCE_START.json',start)
  t0=time.perf_counter();candidate=closure_parallel(extras,trees,pool);t1=time.perf_counter()
  save(B,'PARALLEL_SOURCE_COLLECTION.json',candidate)
  t2=time.perf_counter();reference=closure(extras,trees);t3=time.perf_counter()
  save(B,'SEQUENTIAL_REFERENCE_COLLECTION.json',reference)
  p0=time.perf_counter();final=dict(backbone=engine.inventory(engine.model),head=engine.inventory(engine.head));p1=time.perf_counter()
  save(B,'FINAL_LIVE_PARAMETERS.json',final)
  end=closure(extras,trees);d=delta(start,end);save(B,'SOURCE_END.json',dict(**end,**d))
  result=dict(candidate_files=len(candidate['files']),reference_files=len(reference['files']),file_sha_map_exact=candidate['files']==reference['files'],module_map_exact=candidate['module_files']==reference['module_files'],candidate_source_delta=delta(start,candidate),source_qualified=not d['changed'] and not d['new_files'],all_live_parameters_unchanged=initial==final,backbone_leaves=len(initial['backbone']),head_leaves=len(initial['head']),parallel_seconds=t1-t0,sequential_reference_seconds=t3-t2,live_parameter_inventory_seconds=p1-p0,clocks=dict(parallel_start_hex=t0.hex(),parallel_end_hex=t1.hex(),reference_start_hex=t2.hex(),reference_end_hex=t3.hex(),parameter_start_hex=p0.hex(),parameter_end_hex=p1.hex()),io_threads=4,cpu_affinity=psutil.Process().cpu_affinity(),no_sha_cache=True,entire_file_bytes_read=True,original_file_set_or_scope_not_reduced=True,model_forwards=0,C0_solver_calls=0,optimizer_updates=0,scientific_family3_started=False,full_request_budget_qualified=False,stable_online_gain=False,os_cache_fairness_controlled=False)
  save(B,'RESULT.json',result)
  return 0 if result['file_sha_map_exact'] and result['module_map_exact'] and result['all_live_parameters_unchanged'] and result['source_qualified'] else 1
 finally:pool.shutdown(wait=True,cancel_futures=True)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
