"""A missing unused package file failed hash registration before child launch."""
import json,hashlib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent;OLD=P/'exact_route_distance_cache_request_root_v1';OUT=P/'exact_route_distance_cache_request_root_v2';BASE=P/'independent_rolling_incremental_engineering_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def main():
    if any((OLD/n).exists() for n in ('REGISTRATION.json','INPUT.json','PHASE_TRACE.bin','RUN_RECEIPT.json')):raise RuntimeError('Original advanced farther than observed prelaunch stage')
    put(OLD/'PRELAUNCH_FAILURE_OBSERVATION.json',dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tool_chunk='47c0a3',actual_exit_code=1,error='FileNotFoundError: missing prepare.py while building registered source SHA list',source_sha256=sha(OLD/'run.py'),missing_path=str(OLD/'prepare.py'),no_request_input_or_trace_created=True,no_worker_launch_reached=True,raw_console_trace_not_separately_saved=True))
    OUT.mkdir(exist_ok=False)
    for f in OLD.iterdir():
        if f.is_file() and f.suffix=='.py':(OUT/f.name).write_bytes(f.read_bytes())
    (OUT/'prepare.py').write_bytes((BASE/'prepare.py').read_bytes())
    (OUT/'PARENT_IDENTITIES.json').write_bytes((OLD/'PARENT_IDENTITIES.json').read_bytes())
    put(OUT/'DEPENDENT_EXECUTION_REGISTRATION.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal='Same science/method/input/budget; complete missing unused prepare.py source inventory before the single actual request',recovery_source_sha256=sha(__file__),old_dependent_registration_sha256=sha(OLD/'DEPENDENT_EXECUTION_REGISTRATION.json'),old_prelaunch_failure_sha256=sha(OLD/'PRELAUNCH_FAILURE_OBSERVATION.json'),actual_sources={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},algorithm_and_driver_bytes_equal={n:sha(OUT/n)==sha(OLD/n) for n in ('run.py','controller.py','observed_saa.py','exact_distance_cache.py','audit.py','phase_trace.py','supervise_audit.py')},supplement_file_sha256=sha(BASE/'prepare.py'),normal_actual_request_count=1,worker_budget=8,cleanup_reserve=2,total_cap=10,K=10,seed=202610130901,cache_capacity=256,incr_eval=False,scientific_data_generated=False,stable_online_gain=False))
    print(json.dumps(dict(root=str(OUT),registration_sha256=sha(OUT/'DEPENDENT_EXECUTION_REGISTRATION.json'))))
if __name__=='__main__':main()
