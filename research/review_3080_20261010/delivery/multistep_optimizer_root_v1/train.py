"""Two fixed SGD steps/arm on OLD fixtures; every weight is research-ineligible."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import copy,gc,hashlib,importlib.metadata,json,pickle,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE.parent/'multistep_full_gradient_root_v2';OLD=BASE.parent/'remote_finite_research_root_v2';UP=OLD/'upstream';EXT=OLD/'extension'
for p in (UP,OLD/'actual_checkpoint_engineering_v1',SOURCE,EXT/'scripts/delivery',EXT/'scripts/coldchain'):sys.path.insert(0,str(p))
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
    from full_gradient import closure
    psutil.Process().cpu_affinity([15]);reg=read(BASE/'ENGINEERING_REGISTRATION.json');checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    if psutil.virtual_memory().available<8*1024**3:raise RuntimeError('8GiB RAM floor')
    if reg['optimizer']!={'name':'sgd_without_momentum','learning_rate':.00001} or reg['schedule']!=[[0,1,2,3],[4,5,6,7]]:raise ValueError('Fixed engineering optimizer/schedule changed')
    if sha(__file__)!=reg['program_sha256']:raise ValueError('Registered program mismatch')
    for n,h in reg['inputs'].items():
        if sha(SOURCE/n)!=h:raise ValueError('Prior actual accepted input changed:'+n)
    prior=read(SOURCE/'RESULT.json');accepted=read(BASE.parent/'multistep_full_gradient_audit_root_v1/INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json')
    ck('prior_actual_gradient_accepted',prior['failed']==0 and accepted['failed']==0 and accepted['full_model_forward_backward_qualified'] and sha(BASE.parent/'multistep_full_gradient_audit_root_v1/INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json')==reg['prior_independent_sha256'])
    ts=read(SOURCE/'MATCHED_TENSORS.json')['rows'];ck('all8_fixed_keys',[t['key'] for t in ts]==[[i,k] for i in (900000,900001) for k in range(4)])
    arrays=[jnp.asarray(np.asarray([t[k] for t in ts]),jnp.float32) for k in ('feature','context','timestep','ce_target','candidate','incumbent','label')]
    saved,h=load_trusted(Path(reg['checkpoint']),UP);info=inspect(saved,h)
    ck('actual_original_checkpoint',info['parameter_leaves']==234 and info['parameter_elements']==76259840 and h==reg['checkpoint_sha256'])
    cfg3=copy.deepcopy(saved['model_config']);cfg3.dtype='float32';cfg3.encoder_input_dim=3;cfg3.rngs=reg['random_initialization_seed']
    random_model=cfg3.construct_model();random_initial=copy.deepcopy(nnx.state(random_model,nnx.Param));del random_model
    cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10
    model=cfg.construct_model();graph=nnx.graphdef(model);pre_initial=copy.deepcopy(saved['params']);del model,saved
    for ps in (pre_initial,random_initial):
        kernel=np.asarray(ps['init_proj']['kernel'].value);ps['init_proj']['kernel'].value=jnp.asarray(np.concatenate([kernel,np.zeros((7,512),kernel.dtype)],axis=0))
    def inventory(ps):return {str(p):{'sha256':hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest(),'shape':list(np.asarray(v.value).shape),'dtype':str(np.asarray(v.value).dtype)} for p,v in ps.flat_state()}
    pre_inv=inventory(pre_initial);random_inv=inventory(random_initial)
    ck('same_architecture_and_all_parameters',len(pre_inv)==len(random_inv)==234 and all(pre_inv[k]['shape']==random_inv[k]['shape'] for k in pre_inv) and sum(np.asarray(v.value).size for _,v in pre_initial.flat_state())==76263424)
    ck('original_pretrained_initialization',pre_inv==read(SOURCE/'FULL_SOURCE_START.json')['initial_parameters'])
    ck('random_constructor_initialization_differs',digest(pre_inv)!=digest(random_inv))
    ck('same_extra7zero_projection_rows',bool(np.all(np.asarray(pre_initial['init_proj']['kernel'].value)[3:]==0) and np.all(np.asarray(random_initial['init_proj']['kernel'].value)[3:]==0)))
    save('INITIAL_PARAMETERS.json',{'pretrained':pre_inv,'random':random_inv,'random_seed':reg['random_initialization_seed'],'random_construction':'fresh3Dmodel_then_same7zero_extension','extended_parameter_elements':76263424})
    def loss(ps,x,context,t,ce,ca,ia,sign,coef):
        m=nnx.merge(graph,ps);lp=jax.nn.log_softmax(m.decode(m.encode(x),t,context),axis=-1)
        c=-(lp[:,1:]*ce[:,1:]).mean(axis=(1,2))*(51/2);delta=(lp[:,1:]*(ca[:,1:]-ia[:,1:])).sum(axis=(1,2))/100.
        pref=jnp.where(sign==0,0.,jax.nn.softplus(-sign*delta));return (c+coef*pref).mean(),(c.mean(),pref.mean())
    vg=jax.jit(jax.value_and_grad(loss,has_aux=True))
    def all_sources():
        out=closure()
        for p in (Path(__file__).resolve(),BASE/'ENGINEERING_REGISTRATION.json',SOURCE/'MATCHED_TENSORS.json',BASE.parent/'multistep_full_gradient_audit_root_v1/INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json'):out[str(p)]=sha(p)
        return out
    start=all_sources();save('TRAINING_SOURCE_START.json',{'files':start,'python':sys.version,'cpu':psutil.Process().cpu_affinity(),'devices':[str(d) for d in jax.devices()],'checkpoint_sha256':h,'versions':{n:importlib.metadata.version(n) for n in ('numpy','jax','jaxlib','flax','optax','ortools','protobuf')}})
    arms={};began=time.perf_counter()
    for arm,initial,coefficient in (('pretrained_CE',pre_initial,0.),('pretrained_paired',pre_initial,.2),('random_task_paired',random_initial,.2)):
        ps=copy.deepcopy(initial);initial_identity=inventory(ps);records=[]
        with (BASE/(arm+'_STEPS.jsonl')).open('x',encoding='utf-8') as stream:
            for step,idx in enumerate(reg['schedule'],1):
                batch=[a[jnp.asarray(idx)] for a in arrays];before=inventory(ps);t0=time.perf_counter();(total,parts),grad=vg(ps,*batch,jnp.asarray(coefficient,jnp.float32));float(total)
                gradient_inventory=inventory(grad);finite=all(np.isfinite(np.asarray(v.value)).all() for _,v in grad.flat_state())
                ck(arm+':full234_finite_gradient:'+str(step),finite and len(gradient_inventory)==234)
                ck(arm+':finite_objective:'+str(step),bool(np.isfinite(float(total)) and all(np.isfinite(float(x)) for x in parts)))
                newps=jax.tree.map(lambda p,g:p-.00001*g,ps,grad);after=inventory(newps)
                # Independent elementwise float32 arithmetic checks the actual SGD application.
                oldflat=dict(ps.flat_state());gflat=dict(grad.flat_state())
                arithmetic=True
                for path,value in newps.flat_state():
                    expected=np.asarray(oldflat[path].value)-np.float32(.00001)*np.asarray(gflat[path].value)
                    arithmetic=arithmetic and np.array_equal(expected,np.asarray(value.value))
                ck(arm+':SGD_float32_formula_exact:'+str(step),arithmetic)
                ck(arm+':all_updated_parameters_finite:'+str(step),all(np.isfinite(np.asarray(v.value)).all() for _,v in newps.flat_state()))
                rec={'step':step,'indices':idx,'state_keys':[ts[i]['key'] for i in idx],'coefficient':coefficient,'total_loss':float(total),'CE_loss':float(parts[0]),'preference_loss':float(parts[1]),'before':before,'gradient':gradient_inventory,'after':after,'seconds':time.perf_counter()-t0,'optimizer':'sgd_without_momentum','learning_rate':.00001}
                stream.write(json.dumps(rec,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno());records.append(rec)
                ps=newps;del grad,oldflat,gflat,newps;gc.collect()
        final=inventory(ps);changed=[k for k in final if final[k]['sha256']!=initial_identity[k]['sha256']]
        ck(arm+':encoder_and_decoder_actual_update',any(k.startswith("('encoder',") for k in changed) and any(k.startswith("('decoder',") for k in changed))
        destination=BASE/(arm+'_ENGINEERING_FINAL.pkl')
        with destination.open('xb') as f:pickle.dump({'params':ps,'model_config':cfg,'fixed_step':2,'arm':arm,'initialization':'pretrained' if arm.startswith('pretrained') else 'random_seed'+str(reg['random_initialization_seed']),'coefficient':coefficient,'registration_sha256':sha(BASE/'ENGINEERING_REGISTRATION.json'),'research_model_eligible':False},f)
        with destination.open('rb') as f:loaded=pickle.load(f)
        ck(arm+':full_save_reload_identity',inventory(loaded['params'])==final and loaded['fixed_step']==2 and loaded['research_model_eligible'] is False)
        test=[a[jnp.asarray(reg['schedule'][0])] for a in arrays]
        before_value,before_parts=loss(ps,*test,jnp.asarray(coefficient,jnp.float32));reload_value,reload_parts=loss(loaded['params'],*test,jnp.asarray(coefficient,jnp.float32))
        ck(arm+':save_reload_objective_exact',float(before_value)==float(reload_value) and all(float(x)==float(y) for x,y in zip(before_parts,reload_parts)))
        arms[arm]={'fixed_steps':2,'coefficient':coefficient,'initial_identity':digest(initial_identity),'final_identity':digest(final),'final_parameters':final,'changed_leaves':len(changed),'steps_sha256':sha(BASE/(arm+'_STEPS.jsonl')),'checkpoint_sha256':sha(destination),'checkpoint_bytes':destination.stat().st_size,'post_reload_first_batch_objective':float(reload_value),'research_model_eligible':False}
        save(arm+'_MODEL_CARD.json',arms[arm]);del ps,loaded;gc.collect()
    end=all_sources();changed=[p for p,h in start.items() if end.get(p)!=h];extra=sorted(set(end)-set(start))
    ck('actual_start_end_no_changed_files',not changed);ck('actual_start_end_no_unsealed_files',not extra);ck('original_checkpoint_unchanged',sha(reg['checkpoint'])==h)
    ck('pretrained_exact_same_initialization',arms['pretrained_CE']['initial_identity']==arms['pretrained_paired']['initial_identity'])
    ck('task_random_same_schedule_objective_and_steps',arms['random_task_paired']['coefficient']==arms['pretrained_paired']['coefficient'] and all(a['fixed_steps']==2 for a in arms.values()))
    save('TRAINING_SOURCE_END.json',{'files':end,'changed':changed,'new_files':extra});failed=sum(not c['pass'] for c in checks)
    result={'checks':checks,'checks_count':len(checks),'failed':failed,'arms':arms,'wall_s':time.perf_counter()-began,'all8_old_fixture_states':True,'fixed_steps_each':2,'optimizer_steps':6,'source_files':len(start),'engineering_training_path_qualified':failed==0,'research_model_eligible':False,'scientific_training_closure_qualified':False,'scientific_random_control_qualified':False,'dynamic_online_qualified':False,'stable_gain_established':False}
    save('RESULT.json',result);print(json.dumps({'checks':len(checks),'failed':failed,'source_files':len(start),'arms':list(arms)}),flush=True);return int(failed>0)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:save('FAILURE.json',{'traceback':traceback.format_exc(),'research_model_eligible':False});raise
