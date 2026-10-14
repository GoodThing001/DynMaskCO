from pathlib import Path
import ast, hashlib, importlib.util, json, sys

ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT/'research/review_3080_20261010'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(262144),b''): h.update(b)
    return h.hexdigest()

def save(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2); f.write('\n')

def main():
    lease=TREE/'delivery/local_source_lease_boundary_root_v1'
    diag=TREE/'pickup/cache_saved_boundary_diagnosis_root_v1'
    for folder in (lease,diag):
        for p in folder.glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    enc=Path(sys.prefix)/'Lib/encodings'; sources=[]
    for n in ('utf_8','cp437'):
        p=enc/(n+'.py'); sources.extend([p,Path(importlib.util.cache_from_source(str(p)))])
    assert all(p.is_file() for p in sources), 'Only existing original source and pyc inputs allowed'
    save(lease/'REGISTRATION.json',{'schema':'owned-copy-lease-engineering-v1','environment':'local original Python3.12.7/CPU7,no model, no server/GPU','source_files':{str(p.resolve()):sha(p) for p in sources},'programs':{p.name:sha(p) for p in sorted(lease.glob('*.py'))},'helper_sha256':sha(ROOT/'tools/engineering_stage_closure_v1.py'),'source_count':4,'actual_cases':['six blocked writes/mutations','directory_add_allowed_manifest_reject','released_write','owner_held_write','owner_abnormal_exit17_release'],'locks_only_owned_copies':True,'no_old_QA_rerun':True,'full_runtime_contract_replacement_enabled':False,'full_request_budget_qualified':False,'stable_online_gain':False})
    request=TREE/'pickup/exact_route_distance_cache_request_root_v2'; local=TREE/'pickup/exact_route_distance_cache_engineering_root_v1'
    inputs=[request/'PHASE_TRACE.bin',request/'exact_distance_cache.py',request/'controller.py',local/'RESULT.json']+[request/('ROLLOUT_reject_%02d.json'%j) for j in range(3)]
    save(diag/'REGISTRATION.json',{'schema':'saved-cache-boundary-diagnosis-v1','request_root':request.relative_to(ROOT).as_posix(),'local_root':local.relative_to(ROOT).as_posix(),'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},'programs':{p.name:sha(p) for p in sorted(diag.glob('*.py'))},'helper_sha256':sha(ROOT/'tools/engineering_stage_closure_v1.py'),'scope':'new byte parser/exact rational phase arithmetic and duplicate-key distribution, no model/route/C0/old parser executions','inner_cache_time_attribution_enabled':False,'science_enabled':False})
    print('Both new local engineering registrations sealed before execution')

if __name__=='__main__': main()
