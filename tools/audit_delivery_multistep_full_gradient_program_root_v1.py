"""Independent CPU replay of stored full-model gradients, never optimization."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import copy,gc,hashlib,json,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'multistep_full_gradient_root_v2'
OLD=BASE.parent/'remote_finite_research_root_v2';EXT=OLD/'extension';UP=OLD/'upstream'
for p in (UP,OLD/'actual_checkpoint_engineering_v1',EXT/'scripts/delivery',EXT/'scripts/coldchain'):sys.path.insert(0,str(p))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (BASE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
    import psutil,numpy as np,jax,jax.numpy as jnp
    from flax import nnx
    psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Actual independent 8GiB resource barrier')
    cklist=[]
    def ck(n,v):cklist.append({'name':n,'pass':bool(v)})
    reg=read(SOURCE/'ENGINEERING_REGISTRATION.json');result=read(SOURCE/'RESULT.json');start=read(SOURCE/'FULL_SOURCE_START.json');end=read(SOURCE/'FULL_SOURCE_END.json');tensor=read(SOURCE/'MATCHED_TENSORS.json')
    auditreg=read(BASE/'AUDIT_REGISTRATION.json')
    ck('adjudicator_registered_bytes',sha(__file__)==auditreg['program_sha256'])
    for name,h in auditreg['inputs'].items():ck('bound_input:'+name,sha(SOURCE/name)==h)
    ck('source_start_end_match',start['files']==end['files'] and not end['changed'] and not end['new_files'])
    for p,h in start['files'].items():ck('actual_server_file:'+p,sha(p)==h)
    ck('producer_pass',result['failed']==0 and result['actual_full_forward_backward_qualified'])
    ck('producer_engineering_only',result['optimizer_steps']==0 and not result['research_model_eligible'] and not result['scientific_training_closure_qualified'] and not result['stable_gain_established'])
    ck('registered_source_inputs',sha(SOURCE/'STATES.jsonl')==reg['states_sha256'] and sha(SOURCE/'features_objective.py')==reg['feature_source_sha256'] and sha(SOURCE/'full_gradient.py')==reg['program_sha256'])
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_maskco import partial_adjacency
    from inspect_checkpoint import load_trusted,inspect
    from helpers import sol2adj
    from modules.functional import coord_normalize
    contract=DeliveryContract();physics=contract.physics()
    rows=[json.loads(s) for s in (SOURCE/'STATES.jsonl').read_text(encoding='utf-8').splitlines()]
    ck('all8_old_evolving_keys',[(r['instance_id'],r['step']) for r in rows]==[(i,s) for i in (900000,900001) for s in range(4)])
    ck('all8_tensor_rows',len(rows)==8 and len(tensor['rows'])==8)
    ck('actual_cpu15_backend',psutil.Process().cpu_affinity()==[15] and jax.default_backend()=='cpu')
    def adjacency(routes):
        sequence=[0]
        for route in routes:sequence+=route+[0]
        return np.asarray(sol2adj(jnp.asarray([sequence],jnp.int32),dtype=jnp.float32,num_nodes=51,is_cvrp=True))[0]
    for r,t in zip(rows,tensor['rows']):
        q,a=r['request'],r['answer'];data=q['data'];initial=q['initial'];target=a['teacher']['routes']
        before=evaluate(data,initial,contract);after=evaluate(data,target,contract)
        n=str((r['instance_id'],r['step']))
        ck(n+':complete_physical_pair',before['hard'] and before['complete'] and after['hard'] and after['complete'])
        ck(n+':actual_server_full_C0_exact',digest(before)==digest(t['before_report']) and digest(after)==digest(t['teacher_report']))
        sign=0 if abs(before['cost']-after['cost'])<=1e-8 else 1 if before['cost']>after['cost'] else -1
        selected=target if before['cost']>after['cost']+1e-8 else initial
        ck(n+':independent_labels_selection',sign==t['label'] and selected==t['selected_routes'])
        part=np.asarray(partial_adjacency(initial,set(q['mask']),51),np.float32)
        ck(n+':actual_step_context',np.array_equal(part,np.asarray(t['context'],np.float32)))
        co=np.asarray(coord_normalize(jnp.asarray(data['coords'],jnp.float32)[None]))[0]
        f=np.zeros((51,10),np.float32);f[:,:2]=co;f[:,2]=np.asarray(data['demands'],np.float32)/contract.capacity
        for col,key in enumerate(('tw_start','tw_end','service_time','reveal_time','ready_time'),3):f[:,col]=np.asarray(data[key],np.float32)/contract.horizon_h
        cls=np.asarray(data['temp_class'],int);f[:,8]=np.asarray(physics.thermal.target_temperature_c,np.float32)[cls]/20.;f[:,9]=np.asarray(physics.quality.initial_value,np.float32)[cls]/9.;f[0,3:]=0
        ck(n+':independent10D_feature_formula',np.array_equal(f,np.asarray(t['feature'],np.float32)))
        for name,route in (('ce_target',selected),('candidate',target),('incumbent',initial)):ck(n+':complete_adjacency:'+name,np.array_equal(adjacency(route),np.asarray(t[name],np.float32)))
        ck(n+':timestep_formula',t['timestep']==float(part[1:].sum()/100.))
    saved,h=load_trusted(Path(reg['checkpoint']),UP);info=inspect(saved,h)
    ck('actual_original234parameter_leaves',info['parameter_leaves']==234 and info['parameter_elements']==76259840 and h==reg['checkpoint_sha256'])
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;m=cfg.construct_model();graph=nnx.graphdef(m)
    params=copy.deepcopy(saved['params']);kernel=np.asarray(params['init_proj']['kernel'].value)
    params['init_proj']['kernel'].value=jnp.asarray(np.concatenate([kernel,np.zeros((7,512),kernel.dtype)],axis=0))
    del saved,m;gc.collect()
    def inventory(ps):return {str(path):{'sha256':hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest(),'shape':list(np.asarray(v.value).shape),'dtype':str(np.asarray(v.value).dtype)} for path,v in ps.flat_state()}
    original=inventory(params);ck('actual_original_extended_identity',original==start['initial_parameters'])
    ts=tensor['rows'];batch=[jnp.asarray(np.asarray([t[k] for t in ts]),jnp.float32) for k in ('feature','context','timestep','ce_target','candidate','incumbent','label')]
    replays=[]
    # Fresh full-model replay uses the registered floating-point grouping.
    def registered_grouping(ps,x,context,t,ce,ca,ia,sign,coefficient):
        model=nnx.merge(graph,ps);lp=jax.nn.log_softmax(model.decode(model.encode(x),t,context),axis=-1)
        ce_per=-(lp[:,1:]*ce[:,1:]).mean(axis=(1,2))*(51/2)
        delta=(lp[:,1:]*(ca[:,1:]-ia[:,1:])).sum(axis=(1,2))/100.
        pref=jnp.where(sign==0,0.,jax.nn.softplus(-sign*delta))
        return (ce_per+coefficient*pref).mean(),(ce_per.mean(),pref.mean(),delta)
    exact=jax.jit(jax.value_and_grad(registered_grouping,has_aux=True))
    for coefficient in (0.,.2):
        stored=read(SOURCE/('GRADIENT_'+str(coefficient)+'.json'));began=time.perf_counter()
        (value,parts),g=exact(params,*batch,jnp.asarray(coefficient,jnp.float32));float(value)
        inv=inventory(g);finite=all(np.isfinite(np.asarray(v.value)).all() for _,v in g.flat_state())
        ck('actual_full_gradient_finite:'+str(coefficient),finite and len(inv)==234)
        for p,h in inv.items():ck('actual_gradient:'+str(coefficient)+':'+p,h==stored['gradient'][p])
        ck('loss_exact:'+str(coefficient),float(value)==stored['loss'] and float(parts[0])==stored['CE'] and float(parts[1])==stored['preference'])
        # Closed-form independent loss identity from stored CE/pair scores.
        expected_total=stored['CE']+coefficient*stored['preference']
        ck('independent_objective_decomposition:'+str(coefficient),abs(float(value)-expected_total)<=1e-6)
        ck('actual_parameters_unchanged:'+str(coefficient),inventory(params)==original)
        replays.append({'coefficient':coefficient,'loss':float(value),'gradient_leaves':len(inv),'seconds':time.perf_counter()-began,'all_gradient_hashes_exact':inv==stored['gradient']});del g;gc.collect()
    for n,h in auditreg['inputs'].items():ck('input_end_unchanged:'+n,sha(SOURCE/n)==h)
    for p,h in start['files'].items():ck('actual_server_file_end:'+p,sha(p)==h)
    ck('audit_program_unchanged',sha(__file__)==auditreg['program_sha256'])
    failed=sum(not c['pass'] for c in cklist)
    save('INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json',{'checks':cklist,'checks_count':len(cklist),'failed':failed,'replays':replays,'full_model_forward_backward_qualified':failed==0,'scientific_training_closure_qualified':False,'optimizer_steps':0,'same_task_trained_random_control_qualified':False,'stable_gain_established':False,'independence_scope':'Fresh process, independent tensor/feature/label/C0 recomputation and full234-gradient replay; same registered floating-point loss grouping/shared JAX model and physical evaluator. No optimizer/update or new data.'})
    print(json.dumps({'checks':len(cklist),'failed':failed}));return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('AUDIT_FAILURE.json',{'traceback':traceback.format_exc()});raise
