"""Independent complete fixed-step replay/NumPy SGD/save-reload, OLD data only."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import copy,gc,hashlib,json,pickle,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE.parent/'multistep_optimizer_root_v1';TENSOR=BASE.parent/'multistep_full_gradient_root_v2';OLD=BASE.parent/'remote_finite_research_root_v2';UP=OLD/'upstream'
for p in (UP,OLD/'actual_checkpoint_engineering_v1'):sys.path.insert(0,str(p))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (BASE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    import psutil,numpy as np,jax,jax.numpy as jnp
    from flax import nnx
    from inspect_checkpoint import load_trusted,inspect
    psutil.Process().cpu_affinity([15])
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Actual 8GiB independent resource barrier')
    ar=read(BASE/'AUDIT_REGISTRATION.json');reg=read(SOURCE/'ENGINEERING_REGISTRATION.json');raw=read(SOURCE/'RESULT.json');start=read(SOURCE/'TRAINING_SOURCE_START.json');end=read(SOURCE/'TRAINING_SOURCE_END.json');init=read(SOURCE/'INITIAL_PARAMETERS.json');checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('independent_program_registered',sha(__file__)==ar['program_sha256'])
    for n,h in ar['inputs'].items():ck('actual_bound_input:'+n,sha(SOURCE/n)==h)
    ck('producer_complete_qualified',raw['failed']==0 and raw['engineering_training_path_qualified'] and raw['fixed_steps_each']==2 and raw['optimizer_steps']==6)
    ck('research_ineligible_old_data',raw['research_model_eligible'] is False and raw['scientific_training_closure_qualified'] is False and raw['stable_gain_established'] is False and reg['fresh_data'] is False and reg['new_science_family'] is False)
    ck('actual_runtime_start_end',start['files']==end['files'] and not end['changed'] and not end['new_files'])
    for p,h in start['files'].items():ck('actual_server_source_start:'+p,sha(p)==h)
    ck('fixed_registered_schedule',reg['schedule']==[[0,1,2,3],[4,5,6,7]] and reg['optimizer']=={'name':'sgd_without_momentum','learning_rate':.00001} and reg['random_initialization_seed']==888)
    for n,h in reg['inputs'].items():ck('bound_prior_tensor_input:'+n,sha(TENSOR/n)==h)
    ts=read(TENSOR/'MATCHED_TENSORS.json')['rows'];ck('all8_keys',[t['key'] for t in ts]==[[i,k] for i in (900000,900001) for k in range(4)])
    arrays=[jnp.asarray(np.asarray([t[k] for t in ts]),jnp.float32) for k in ('feature','context','timestep','ce_target','candidate','incumbent','label')]
    original,h=load_trusted(Path(reg['checkpoint']),UP);info=inspect(original,h);ck('actual_original_checkpoint',h==reg['checkpoint_sha256'] and info['parameter_leaves']==234 and info['parameter_elements']==76259840)
    cfg3=copy.deepcopy(original['model_config']);cfg3.dtype='float32';cfg3.encoder_input_dim=3;cfg3.rngs=888
    random_model=cfg3.construct_model();random_initial=copy.deepcopy(nnx.state(random_model,nnx.Param));del random_model
    cfg=copy.deepcopy(original['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;constructed=cfg.construct_model();graph=nnx.graphdef(constructed);pre_initial=copy.deepcopy(original['params']);del original,constructed
    for ps in (pre_initial,random_initial):
        a=np.asarray(ps['init_proj']['kernel'].value);ps['init_proj']['kernel'].value=jnp.asarray(np.vstack((a,np.zeros((7,512),dtype=a.dtype))))
    def inventory(ps):return {str(p):{'sha256':hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest(),'shape':list(np.asarray(v.value).shape),'dtype':str(np.asarray(v.value).dtype)} for p,v in ps.flat_state()}
    ck('fresh_pretrained_initialization',inventory(pre_initial)==init['pretrained']);ck('fresh_random_factory_initialization',inventory(random_initial)==init['random'])
    ck('architecture_control_is_same',all(init['pretrained'][k]['shape']==init['random'][k]['shape'] for k in init['pretrained']) and len(init['pretrained'])==len(init['random'])==234)
    def loss(ps,x,context,t,ce,ca,ia,sign,coef):
        m=nnx.merge(graph,ps);lp=jax.nn.log_softmax(m.decode(m.encode(x),t,context),axis=-1)
        ce_loss=-(lp[:,1:]*ce[:,1:]).mean(axis=(1,2))*(51/2);delta=(lp[:,1:]*(ca[:,1:]-ia[:,1:])).sum(axis=(1,2))/100.
        pref=jnp.where(sign==0,0.,jax.nn.softplus(-sign*delta));return (ce_loss+coef*pref).mean(),(ce_loss.mean(),pref.mean())
    vg=jax.jit(jax.value_and_grad(loss,has_aux=True));outcomes={}
    for arm,initial,coef in (('pretrained_CE',pre_initial,0.),('pretrained_paired',pre_initial,.2),('random_task_paired',random_initial,.2)):
        records=[json.loads(s) for s in (SOURCE/(arm+'_STEPS.jsonl')).read_text(encoding='utf-8').splitlines()];ck(arm+':all2_steps',len(records)==2)
        ps=copy.deepcopy(initial);t0=time.perf_counter()
        for step,idx in enumerate(reg['schedule'],1):
            rec=records[step-1];ck(arm+':actual_registered_keys:'+str(step),rec['step']==step and rec['indices']==idx and rec['state_keys']==[ts[i]['key'] for i in idx] and rec['coefficient']==coef and rec['learning_rate']==.00001)
            before=inventory(ps)
            for p,h in before.items():ck(arm+':before:'+str(step)+':'+p,h==rec['before'][p])
            batch=[a[jnp.asarray(idx)] for a in arrays];(total,parts),grad=vg(ps,*batch,jnp.asarray(coef,jnp.float32));float(total)
            ck(arm+':objective_exact:'+str(step),float(total)==rec['total_loss'] and float(parts[0])==rec['CE_loss'] and float(parts[1])==rec['preference_loss'])
            gi=inventory(grad);ck(arm+':full234_grad_finite:'+str(step),len(gi)==234 and all(np.isfinite(np.asarray(v.value)).all() for _,v in grad.flat_state()))
            for p,h in gi.items():ck(arm+':gradient:'+str(step)+':'+p,h==rec['gradient'][p])
            # Deliberately apply the update with independent NumPy float32 arithmetic.
            ps=jax.tree.map(lambda p,g:jnp.asarray(np.asarray(p)-np.float32(.00001)*np.asarray(g)),ps,grad)
            after=inventory(ps)
            for p,h in after.items():ck(arm+':independent_numpy_SGD:'+str(step)+':'+p,h==rec['after'][p])
            del grad;gc.collect()
        card=read(SOURCE/(arm+'_MODEL_CARD.json'));saved_path=SOURCE/(arm+'_ENGINEERING_FINAL.pkl');ck(arm+':actual_final_file_sha',sha(saved_path)==card['checkpoint_sha256'])
        with saved_path.open('rb') as f:loaded=pickle.load(f)
        final=inventory(ps);reload=inventory(loaded['params']);ck(arm+':engineering_metadata',loaded['fixed_step']==2 and loaded['research_model_eligible'] is False and loaded['coefficient']==coef and loaded['registration_sha256']==sha(SOURCE/'ENGINEERING_REGISTRATION.json'))
        for p,h in final.items():ck(arm+':independent_final_reload:'+p,h==reload[p] and h==card['final_parameters'][p])
        ck(arm+':actual_final_card',digest(final)==card['final_identity'] and card['initial_identity']==digest(inventory(initial)))
        test=[a[jnp.asarray(reg['schedule'][0])] for a in arrays];val,_=loss(loaded['params'],*test,jnp.asarray(coef,jnp.float32));ck(arm+':actual_reloaded_model_objective',float(val)==card['post_reload_first_batch_objective'])
        outcomes[arm]={'fixed_steps':2,'all234_final_hashes_equal':final==reload==card['final_parameters'],'independent_replay_seconds':time.perf_counter()-t0,'research_model_eligible':False};del ps,loaded;gc.collect()
    for n,h in ar['inputs'].items():ck('actual_input_end_unchanged:'+n,sha(SOURCE/n)==h)
    for p,h in start['files'].items():ck('actual_server_source_end:'+p,sha(p)==h)
    ck('original_checkpoint_end',sha(reg['checkpoint'])==reg['checkpoint_sha256']);ck('independent_program_end',sha(__file__)==ar['program_sha256'])
    failed=sum(not c['pass'] for c in checks)
    save('INDEPENDENT_OPTIMIZER_ADJUDICATION.json',{'checks':checks,'checks_count':len(checks),'failed':failed,'outcomes':outcomes,'engineering_training_path_qualified':failed==0,'scientific_training_closure_qualified':False,'scientific_random_control_qualified':False,'research_model_eligible':False,'stable_gain_established':False,'scope':'Fresh full-model gradient replay + independent float32 NumPy SGD + all fixed-final saved parameters/reloaded objective; shared JAX model and previously accepted OLD tensors; two engineering steps are not science training or gain.'})
    print(json.dumps({'checks':len(checks),'failed':failed}));return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('AUDIT_FAILURE.json',{'traceback':traceback.format_exc()});raise
