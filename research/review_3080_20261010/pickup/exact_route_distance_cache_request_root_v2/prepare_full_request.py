"""Seal one dependent exact-cache K10 request; never repeat local qualification."""
import ast,json,hashlib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'independent_rolling_incremental_engineering_root_v1';OUT=HERE.parent/'exact_route_distance_cache_request_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def main():
    r=json.loads((HERE/'RESULT.json').read_text());a=json.loads((HERE/'INDEPENDENT_AUDIT.json').read_text())
    if r['failed'] or a['failed']:raise ValueError('Local semantic gate not qualified')
    OUT.mkdir(exist_ok=False)
    for n in ('exact_distance_cache.py','phase_trace.py','supervise_audit.py'):
        origin=HERE/n if n=='exact_distance_cache.py' else BASE/n
        (OUT/n).write_bytes(origin.read_bytes())
    c=(BASE/'controller.py').read_text(encoding='utf-8').replace('incr_eval=True','incr_eval=False')
    c=c.replace("space,orders=shadow._build_space(env,0,scenarios);lookup=", "space,orders=shadow._build_space(env,0,scenarios);shadow.bind_distance_space(space);lookup=")
    c=c.replace("    if time.perf_counter()>=deadline:raise TimeoutError('late decision not publishable')", "    shadow.verify_distance_space(space)\n    if time.perf_counter()>=deadline:raise TimeoutError('late decision not publishable')")
    o=(BASE/'observed_saa.py').read_text(encoding='utf-8').replace('from energy_complete_saa import EnergyCompleteSAA','from exact_distance_cache import ExactDistanceCache').replace('class ObservedSAA(EnergyCompleteSAA):','class ObservedSAA(ExactDistanceCache):')
    o=o.replace('self.completed=time.perf_counter();return value','self.verify_distance_space(args[-1]);self.completed=time.perf_counter();return value')
    o=o.replace("elapsed=self.completed-self.started,incr_eval=self.incr_eval)","elapsed=self.completed-self.started,incr_eval=self.incr_eval,cache_stats=dict(calls=self.distance_calls,hits=self.distance_hits,misses=self.distance_misses,evictions=self.distance_evictions,capacity=self.CACHE_CAPACITY))")
    run=(BASE/'run.py').read_text(encoding='utf-8')
    run=run.replace("HERE/'supervise_audit.py',INPUT", "HERE/'supervise_audit.py',HERE/'exact_distance_cache.py',HERE/'prepare_full_request.py',INPUT")
    run=run.replace("diagnostic_goal='candidate existingincr_eval route-summary prefilters and delta-ranking; exact action/terminalC0/float.hex equivalence versus originalfalse is REQUIRED, not assumed; no utility selection or scientific gain'", "diagnostic_goal='One new exact original serial distance-cache K10 request after local6/121 qualification; no delta-ranking or old incremental flag; fixed256 cache and8+2'")
    run=run.replace("prior_registration_sha256='61d71415311a784bdab5787e30920e32955040baa3890ad7766f20a1e713a474'", "prior_registration_sha256="+repr(sha(HERE/'REGISTRATION.json')))
    run=run.replace('candidate_incr_eval=True','candidate_incr_eval=False,exact_distance_cache=True,cache_capacity=256')
    aud=(BASE/'audit.py').read_text(encoding='utf-8').replace("saved=read(path);j=saved['index']","saved=decode(read(path));j=saved['index']").replace("reg['candidate_incr_eval'] and not reg['exact_reference_incr_eval']","not reg['candidate_incr_eval'] and not reg['exact_reference_incr_eval'] and reg['exact_distance_cache']").replace("saved['incr_eval'] is True","saved['incr_eval'] is False")
    for n,x in [('controller.py',c),('observed_saa.py',o),('run.py',run),('audit.py',aud)]:ast.parse(x);(OUT/n).write_text(x,encoding='utf-8',newline='\n')
    (OUT/'prepare_full_request.py').write_bytes(Path(__file__).read_bytes())
    prior={str(p):sha(p) for p in HERE.iterdir() if p.is_file()}
    put(OUT/'PARENT_IDENTITIES.json',dict(prior_files=prior))
    put(OUT/'DEPENDENT_EXECUTION_REGISTRATION.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),qualification_gate=dict(local_production_checks=6,local_independent_checks=121,failures=0),local_registration_sha256=sha(HERE/'REGISTRATION.json'),sources={str(p):sha(p) for p in OUT.iterdir() if p.is_file()},normal_request_count=1,worker_budget=8,cleanup_reserve=2,total_cap=10,K=10,seed=202610130901,cache_capacity=256,incr_eval=False,all_D_guards_and_source_evidence_within_corresponding_request=True,failure_policy='No selection or installation if all20/source/deadline gates incomplete; retain prior certified incumbent and current undecided',scientific_data_generated=False,stable_online_gain=False))
    print(json.dumps(dict(root=str(OUT),registration_sha256=sha(OUT/'DEPENDENT_EXECUTION_REGISTRATION.json'))))
if __name__=='__main__':main()
