"""Read-only independent masks/state/route/cost replay; never reruns the solver."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
D = ROOT / 'research/review_3080_20261010/delivery'
OUT = D / 'multistep_insertable_teacher_engineering_root_v1'
for sub in ('delivery', 'coldchain', 'research_20261010'): sys.path.insert(0, str(E / 'scripts' / sub))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import reconstruct

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def edges(routes):
    out = set()
    for r in routes:
        chain = [0] + r + [0]
        out.update(tuple(sorted((a,b))) for a,b in zip(chain, chain[1:]) if a != b)
    return out

def main():
    reg = json.loads((OUT / 'ENGINEERING_REGISTRATION.json').read_text())
    raw = json.loads((OUT / 'RESULT.json').read_text())
    records = [json.loads(x) for x in (OUT / 'STATES.jsonl').read_text().splitlines()]
    fixture = D / 'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl'
    old = [json.loads(x) for x in fixture.read_text().splitlines()]
    checks, summaries = [], []
    def ck(k,v): checks.append({'name':k,'pass':bool(v)})
    def near(a,b): return a == b if a is None or b is None else math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
    ck('registered_exact_eight_state_keys', [(r['instance_id'],r['step']) for r in records] == [(i,s) for i in reg['ids'] for s in range(4)])
    ck('registered_fixture_unchanged', sha(fixture) == reg['fixture_sha256'])
    ck('registration_and_raw_bound', raw['registration_sha256']==sha(OUT/'ENGINEERING_REGISTRATION.json') and raw['states_sha256']==sha(OUT/'STATES.jsonl'))
    ck('execution_no_failures', raw['failed']==0 and raw['worker_reaped'])
    start = json.loads((OUT/'WORKER_SOURCE_START.json').read_text()); end = json.loads((OUT/'WORKER_SOURCE_END.json').read_text())
    ck('no_lazy_unsealed_dependency', not end['new_loaded_files'])
    ck('interpreter_and_versions_sealed', all(k in start['packages'] for k in ('numpy','ortools','protobuf')) and start['executable'] in start['files'])
    for p,h in start['files'].items(): ck('actual_dependency:'+p, Path(p).is_file() and sha(p)==h==end['closure']['files'].get(p))
    contract=DeliveryContract(); total=0; changed_states=0
    for i in reg['ids']:
        f=next(x for x in old if x['instance_id']==i and x['mask_ordinal']==0)
        incumbent=f['incumbent_routes']; n=len(f['data']['demands'])
        # Regenerate the registered masks without calling production make_schedule.
        rng=np.random.default_rng(np.random.SeedSequence([reg['mask_seed'],i]))
        masks=[sorted(map(int,rng.choice(np.arange(1,n),math.ceil(reg['mask_fraction']*(n-1)),replace=False))) for _ in range(4)]
        for r,mask in zip([r for r in records if r['instance_id']==i], masks):
            q=r['request']; a=r['answer']; t=a['teacher']; key=str((i,r['step']))
            ck(key+':registered_mask',q['mask']==mask)
            ck(key+':same_observed_data',q['data']==f['data'])
            ck(key+':actual_H_rollout_prior',q['initial']==incumbent)
            changed_states+=int(q['initial']!=f['incumbent_routes'])
            removed=set(mask); required={e for e in edges(incumbent) if not removed.intersection(e)}
            partial=np.zeros((n,n),np.float32)
            for u,v in required:partial[u,v]=partial[v,u]=1
            ck(key+':exact_retained_original_edges',np.array_equal(partial,np.asarray(a['partial'])) and np.array_equal(partial,np.asarray(t['partial_adjacency'])))
            kept=[[o for o in route if o not in removed] for route in incumbent]
            def skeleton(routes):return len(routes)==contract.vehicles and [[o for o in rr if o not in removed] for rr in routes]==kept
            initial=evaluate(q['data'],incumbent,contract);best=incumbent;cost=initial['cost']
            ck(key+':own_incumbent_hard',initial['hard'] and initial['complete'])
            ck(key+':actual_request_in_10s',0<=r['wall_s']<10 and r['received']<q['deadline'] and math.isclose(q['deadline']-r['started'],10.,abs_tol=1e-8))
            ck(key+':teacher_kept_skeleton',t['fixed_unmasked_vehicle_order']==kept)
            for j,c in enumerate(t['all_candidates']):
                total+=1;ev=evaluate(q['data'],c['routes'],contract);name=key+':'+str(j)
                ck(name+':original_vehicle_order_including_empty',skeleton(c['routes']))
                ck(name+':all_retained_edges',required<=edges(c['routes']))
                ck(name+':complete_physical_flags',all(ev[k]==c['report'][k] for k in ('hard','complete','errors')))
                for k in ('cost','distance_km','energy_kwh','quality_loss_value'):ck(name+':'+k,near(ev[k],c['report'][k]))
                accept=ev['hard'] and required<=edges(c['routes']) and ev['cost']<cost-1e-12
                ck(name+':fullC0_selection',c['accepted']==accept)
                if accept:best,cost=c['routes'],ev['cost']
            ck(key+':all_candidate_best_replay',best==t['routes'] and near(cost,t['report']['cost']))
            # Independent invocation of actual H deployment to verify state law,
            # not an independent physical implementation or a new solver run.
            import time
            routes,remaining,insertions,status=reconstruct(q['data'],incumbent,contract,removed,partial,None,0.,time.perf_counter()+10.,time.perf_counter)
            report=evaluate(q['data'],routes,contract)
            accept=not remaining and report['hard'] and report['cost']<initial['cost']
            ck(key+':actual_H_candidate_replay',routes==a['deployment_candidate'] and remaining==a['remaining'] and insertions==a['insertions'] and status==a['status'])
            ck(key+':deployment_acceptance',bool(accept)==a['deployment_accepted'])
            incumbent=routes if accept else incumbent
            ck(key+':next_incumbent_exact',incumbent==a['incumbent_after'])
            final=evaluate(q['data'],t['routes'],contract)
            ck(key+':teacher_complete_fullC0_hard',final['hard'] and final['complete'])
            summaries.append({'instance':i,'step':r['step'],'candidate_count':len(t['all_candidates']),'different_from_original_incumbent':q['initial']!=f['incumbent_routes'],'engineering_only':True})
    failed=sum(not x['pass'] for x in checks)
    out={'checks':checks,'checks_count':len(checks),'failed':failed,'all_states_retained':len(records)==8,'candidates':total,'evolved_states':changed_states,'summaries':summaries,
         'qualification':'two observed fixtures / four nonlearning deployment states each / bounded teacher and byte environment closure',
         'registered_environment_closure_for_this_engineering_only':failed==0,'all_model_rollout_states_covered':False,'new_training_qualified':False,'stable_online_gain':False,
         'limits':['Uses original fullC0 evaluator, not physical calibration or separate physical implementation','H-generated rollout only; learned model rollout distribution remains untested','No all-state greedy-logit target reachability proof','Actual science training/JAX and server execution closures require separate registration']}
    with (OUT/'INDEPENDENT_MULTISTEP_REPLAY.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ('checks_count','failed','candidates','evolved_states','stable_online_gain')}));return int(failed>0)

if __name__=='__main__':raise SystemExit(main())
