"""Prepare fixed server teacher fixtures before generation; no scientific data."""
import copy,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools';D=ROOT/'research/review_3080_20261010/delivery'
source=D/'multistep_insertable_teacher_engineering_root_v2/STATES.jsonl'
row=json.loads(source.read_text(encoding='utf-8').splitlines()[1]);old=row['request'];selected=sorted(set(o for r in old['initial'][:2] for o in r));ids=[0]+selected;lookup={o:j for j,o in enumerate(ids)}
data={k:[v[i] for i in ids] for k,v in old['data'].items()};two=[[lookup[o] for o in r] for r in old['initial'][:2]];n=len(ids);initial=[[],two[0],[],two[1],[]]
requests=[{'key':'observed_second_H_state','data':old['data'],'initial':old['initial'],'mask':old['mask']},
          {'key':'derived_empty_slots_partial','data':data,'initial':initial,'mask':list(range(1,n,2))},
          {'key':'derived_empty_slots_full_mask','data':data,'initial':initial,'mask':list(range(1,n))},
          {'key':'derived_empty_slots_zero_mask','data':data,'initial':initial,'mask':[]}]
requests_raw=json.dumps(requests,indent=2,allow_nan=False).encode()
def create(name,body):
    compile(body,name,'exec')
    with (T/name).open('x',encoding='utf-8',newline='\n') as f:f.write(body)
deploy=(T/'deploy_delivery_multistep_full_gradient_root_v2.py').read_text()
deploy=deploy.replace("OUT=D/'multistep_full_gradient_root_v2';REL='research/review_3080_20261010/multistep_full_gradient_root_v2'", "OUT=D/'server_teacher_boundary_root_v1';REL='research/review_3080_20261010/server_teacher_boundary_root_v1'")
deploy=deploy.replace("str(base/'full_gradient.py')", "str(base/'teacher.py')")
a=deploy.index("    program=(ROOT/'tools/");b=deploy.index("    host,user,password=_cfg",a)
block="""    program=(ROOT/'tools/qualify_delivery_server_teacher_boundary_program_root_v1.py').read_bytes()
    teacher=(D/'multistep_insertable_teacher_engineering_root_v2/retained_gls.py').read_bytes()
    requests=(ROOT/'tools/server_teacher_boundary_REQUESTS_root_v1.json').read_bytes()
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Actual server teacher and H deployment on one previously observed second-step state and three declared old20-order empty-slot derived boundaries; no scientific sample or resampling','cpu':[15],'max_s':1200,'minimum_ram_gib':8,'program_sha256':sha(program),'inputs':{'retained_gls.py':sha(teacher),'REQUESTS.json':sha(requests)},'prior_states_sha256':sha((D/'multistep_insertable_teacher_engineering_root_v2/STATES.jsonl').read_bytes()),'contract_hash':'ba40cd0d08fc21ef09d418c7df2bff940294ee2cfecebd2a4ba9ddde35a8268a','request_cap_s':10.,'teacher_search_cap_s':3.,'derived_rule':'second old H state; retain first2 original vehicle routes/orders, subset and reindex all fields, place original routes in slots1/3, empty0/2/4; masks odd-index/all/none','new_science_family':False,'fresh_data':False,'research_model_eligible':False,'scientific_training_qualified':False,'shared_MD_uploaded':False}
    write(OUT/'ENGINEERING_REGISTRATION.json',reg)
    files={'teacher.py':program,'retained_gls.py':teacher,'REQUESTS.json':requests,'ENGINEERING_REGISTRATION.json':(OUT/'ENGINEERING_REGISTRATION.json').read_bytes(),'supervisor.py':SUPERVISOR.encode(),'launcher.py':LAUNCHER.encode(),'status.py':STATUS.encode()}
    for n,b in files.items():
        if n!='ENGINEERING_REGISTRATION.json':(OUT/n).write_bytes(b)
    write(OUT/'UPLOAD_MANIFEST.json',{n:sha(b) for n,b in files.items()})
"""
deploy=deploy[:a]+block+deploy[b:]
create('deploy_delivery_server_teacher_boundary_root_v1.py',deploy)
with (T/'server_teacher_boundary_REQUESTS_root_v1.json').open('xb') as f:f.write(requests_raw)
inspect=(T/'inspect_delivery_multistep_optimizer_audit_root_v2.py').read_text().replace('multistep_optimizer_audit_root_v2','server_teacher_boundary_root_v1')
inspect=inspect.replace("'INDEPENDENT_OPTIMIZER_ADJUDICATION.json','AUDIT_SOURCE_START.json','AUDIT_SOURCE_END.json'", "'SOURCE_START.json','SOURCE_END.json','ANSWERS.jsonl'")
create('inspect_delivery_server_teacher_boundary_root_v1.py',inspect)
print(json.dumps({'requests':len(requests),'derived_orders':n-1,'derived_route_slots':[len(r) for r in initial],'scientific_data':False}))
