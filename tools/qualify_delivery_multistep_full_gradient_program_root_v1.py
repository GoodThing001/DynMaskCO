"""Engineering-only full MaskCO forward/backward on eight already observed states.

No optimizer step, checkpoint selection, fresh sample or scientific gain test.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,gc,hashlib,importlib,importlib.metadata,json,pickle,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=Path(r'C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\remote_finite_research_root_v2')
EXT=OLD/'extension';UP=OLD/'upstream'
for p in (UP,OLD/'actual_checkpoint_engineering_v1',BASE,EXT/'scripts/delivery',EXT/'scripts/coldchain'):
    sys.path.insert(0,str(p))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name,x):
    with (BASE/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def closure():
    files={Path(__file__).resolve(),BASE/'features_objective.py',BASE/'STATES.jsonl',BASE/'ENGINEERING_REGISTRATION.json',Path(sys.executable).resolve()}
    for m in list(sys.modules.values()):
        for attr in ('__file__','__cached__'):
            n=getattr(m,attr,None)
            if n and Path(n).is_file():files.add(Path(n).resolve())
    for name in ('numpy','jax','jaxlib','flax','optax','chex','scipy','ml_dtypes','etils','ortools','google.protobuf','psutil'):
        m=importlib.import_module(name);path=Path(m.__file__).parent
        files.update(p.resolve() for p in path.rglob('*') if p.is_file())
    for path in (UP,EXT/'scripts/delivery',EXT/'scripts/coldchain',Path(sys.base_prefix)/'DLLs'):
        files.update(p.resolve() for p in path.rglob('*') if p.is_file())
    # Include standard-library source/native files before compile-time lazy imports.
    lib=Path(sys.base_prefix)/'Lib'
    for child in lib.iterdir():
        if child.name=='site-packages':continue
        if child.is_file():files.add(child.resolve())
        elif child.is_dir():files.update(p.resolve() for p in child.rglob('*') if p.is_file())
    files.update(p.resolve() for p in Path(sys.base_prefix).glob('python*.dll'))
    import psutil
    for m in psutil.Process().memory_maps():
        p=Path(m.path)
        if p.is_file():files.add(p.resolve())
    return {str(p):sha(p) for p in sorted(files)}
def main():
    import psutil,numpy as np,jax,jax.numpy as jnp,flax,optax,chex,scipy,ml_dtypes,etils,ortools,google.protobuf
    from flax import nnx
    p=psutil.Process();p.cpu_affinity([15])
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('8GiB actual resource barrier')
    reg=json.loads((BASE/'ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    if sha(__file__)!=reg['program_sha256'] or sha(BASE/'STATES.jsonl')!=reg['states_sha256'] or sha(BASE/'features_objective.py')!=reg['feature_source_sha256']:raise ValueError('Exact registered engineering input')
    from features_objective import make_features,extend_projection,UNITS
    from inspect_checkpoint import load_trusted,inspect
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_maskco import partial_adjacency
    from helpers import sol2adj
    from modules.functional import coord_normalize
    from ortools.constraint_solver import pywrapcp
    pywrapcp.DefaultRoutingSearchParameters()
    contract=DeliveryContract();checks=[];tensor_rows=[]
    def ck(name,val):checks.append({'name':name,'pass':bool(val)})
    rows=[json.loads(s) for s in (BASE/'STATES.jsonl').read_text(encoding='utf-8').splitlines()]
    ck('eight_exact_evolving_keys',[(r['instance_id'],r['step']) for r in rows]==[(i,s) for i in (900000,900001) for s in range(4)])
    ck('actual_contract',contract.identity()==reg['contract_hash'])
    def adj(routes):
        vector=[0]
        for route in routes:vector.extend(route+[0])
        return np.asarray(sol2adj(jnp.asarray([vector],jnp.int32),dtype=jnp.float32,num_nodes=51,is_cvrp=True))[0]
    x,context,timestep,ce,candidate,incumbent,label=[],[],[],[],[],[],[]
    for r in rows:
        q,a=r['request'],r['answer'];data=q['data'];teacher=a['teacher'];start=q['initial'];target=teacher['routes']
        before=evaluate(data,start,contract);after=evaluate(data,target,contract)
        ck(str((r['instance_id'],r['step']))+':full_C0_pair',before['hard'] and before['complete'] and after['hard'] and after['complete'])
        ck(str((r['instance_id'],r['step']))+':old_runtime_cost_replay',abs(before['cost']-teacher['initial_report']['cost'])<=1e-8 and abs(after['cost']-teacher['report']['cost'])<=1e-8)
        part=np.asarray(partial_adjacency(start,set(q['mask']),51),np.float32)
        ck(str((r['instance_id'],r['step']))+':actual_evolving_context',np.array_equal(part,np.asarray(a['partial'],np.float32)))
        selected=target if after['cost']<before['cost']-1e-8 else start
        sign=0 if abs(before['cost']-after['cost'])<=1e-8 else (1 if before['cost']>after['cost'] else -1)
        co=np.asarray(coord_normalize(jnp.asarray(data['coords'],jnp.float32)[None]))[0]
        features=make_features(data,contract,co,UNITS,contract.identity())[0]
        ca,ia,cta=adj(target),adj(start),adj(selected)
        unmasked=set(range(1,51))-set(q['mask'])
        locations={o:(v,k) for v,route in enumerate(target) for k,o in enumerate(route)}
        ck(str((r['instance_id'],r['step']))+':unmasked_original_vehicle_and_rank',all(all(locations[o][0]==v for o in route if o in unmasked) and all(locations[b][1]>locations[a][1] for a,b in zip([o for o in route if o in unmasked],[o for o in route if o in unmasked][1:])) for v,route in enumerate(start)))
        ck(str((r['instance_id'],r['step']))+':retained_edges_in_teacher',bool(np.all(ca[part==1]==1)))
        ck(str((r['instance_id'],r['step']))+':complete_target_degrees',bool(np.all(cta[1:].sum(1)==2)))
        x.append(features);context.append(part);timestep.append(float(part[1:].sum()/100.));ce.append(cta);candidate.append(ca);incumbent.append(ia);label.append(sign)
        tensor_rows.append({'key':[r['instance_id'],r['step']],'feature':features.tolist(),'context':part.tolist(),'timestep':timestep[-1],'ce_target':cta.tolist(),'candidate':ca.tolist(),'incumbent':ia.tolist(),'label':sign,'selected_routes':selected,'before_report':before,'teacher_report':after})
    if any(not c['pass'] for c in checks):save('TENSOR_FAILURE.json',{'checks':checks});raise ValueError('Matched teacher tensors not qualified')
    save('MATCHED_TENSORS.json',{'rows':tensor_rows,'checks':checks,'old_states_sha256':sha(BASE/'STATES.jsonl'),'scientific_data':False})
    checkpoint=Path(reg['checkpoint']);saved,h=load_trusted(checkpoint,UP);info=inspect(saved,h)
    ck('actual_original_checkpoint_full',info['parameter_leaves']==234 and info['parameter_elements']==76259840)
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10
    m=cfg.construct_model();graph=nnx.graphdef(m);params=copy.deepcopy(saved['params'])
    params['init_proj']['kernel'].value=jnp.asarray(extend_projection(params['init_proj']['kernel'].value))
    del saved,m;gc.collect()
    def inventory(ps):return {str(path):{'sha256':hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest(),'shape':list(np.asarray(v.value).shape),'dtype':str(np.asarray(v.value).dtype)} for path,v in ps.flat_state()}
    original=inventory(params);ck('actual_extended_parameter_count',sum(np.asarray(v.value).size for _,v in params.flat_state())==76263424 and len(original)==234)
    arrays=[jnp.asarray(np.asarray(v),jnp.float32) for v in (x,context,timestep,ce,candidate,incumbent,label)]
    # Both coefficients evaluate the SAME actual initial parameters and all eight states.
    def loss(ps,x,context,t,ce,ca,ia,lab,coef):
        model=nnx.merge(graph,ps);logits=model.decode(model.encode(x),t,context);logp=jax.nn.log_softmax(logits,axis=-1)
        c=-(logp[:,1:]*ce[:,1:]).mean(axis=(1,2))*(51/2)
        score=(logp[:,1:]*(ca[:,1:]-ia[:,1:])).sum(axis=(1,2))/100.
        pref=jnp.where(lab==0,0.,jax.nn.softplus(-lab*score))
        return (c+coef*pref).mean(),(c.mean(),pref.mean(),score)
    vg=jax.jit(jax.value_and_grad(loss,has_aux=True))
    versions={n:importlib.metadata.version(n) for n in ('numpy','jax','jaxlib','flax','optax','chex','scipy','ml_dtypes','etils','ortools','protobuf','psutil')}
    start=closure();save('FULL_SOURCE_START.json',{'files':start,'versions':versions,'python':sys.version,'cpu':p.cpu_affinity(),'devices':[str(d) for d in jax.devices()],'checkpoint_sha256':h,'initial_parameters':original})
    records=[]
    for coefficient in (0.,.2):
        began=time.perf_counter();(value,parts),gradient=vg(params,*arrays,jnp.asarray(coefficient,jnp.float32));float(value)
        gi=inventory(gradient);finite=all(np.isfinite(np.asarray(v.value)).all() for _,v in gradient.flat_state())
        ck('full_gradient_finite:'+str(coefficient),finite and len(gi)==234)
        ck('actual_initial_parameters_unchanged:'+str(coefficient),inventory(params)==original)
        rec={'coefficient':coefficient,'loss':float(value),'CE':float(parts[0]),'preference':float(parts[1]),'score':np.asarray(parts[2]).tolist(),'gradient':gi,'initial_digest':digest(original),'seconds':time.perf_counter()-began,'optimizer_steps':0}
        save('GRADIENT_'+str(coefficient)+'.json',rec);records.append(rec);del gradient;gc.collect()
    end=closure();changed=[p for p,h in start.items() if end.get(p)!=h];extra=sorted(set(end)-set(start))
    ck('actual_start_end_no_changed_files',not changed);ck('actual_start_end_no_unsealed_new_files',not extra);ck('original_checkpoint_unchanged',sha(checkpoint)==h)
    save('FULL_SOURCE_END.json',{'files':end,'changed':changed,'new_files':extra})
    failed=sum(not c['pass'] for c in checks)
    result={'checks':checks,'checks_count':len(checks),'failed':failed,'records':[dict(coefficient=r['coefficient'],loss=r['loss'],CE=r['CE'],preference=r['preference'],initial_digest=r['initial_digest'],seconds=r['seconds']) for r in records],
        'source_files':len(start),'actual_full_forward_backward_qualified':failed==0,'optimizer_steps':0,'research_model_eligible':False,'scientific_training_closure_qualified':False,'dynamic_online_qualified':False,'stable_gain_established':False,'all8_labels_positive':all(v==1 for v in label),'runtime_only_engineering':True,'registration_sha256':sha(BASE/'ENGINEERING_REGISTRATION.json')}
    save('RESULT.json',result);print(json.dumps({'checks':len(checks),'failed':failed,'source_files':len(start)}),flush=True);return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save('FAILURE.json',{'traceback':traceback.format_exc(),'scientific_training_closure_qualified':False});raise
