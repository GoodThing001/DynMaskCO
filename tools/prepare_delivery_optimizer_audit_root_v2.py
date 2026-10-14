"""Append corrected exact arithmetic audit; do not rerun or edit old training."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools'
def create(name,text):
    compile(text,name,'exec')
    with (T/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
code=(T/'audit_delivery_multistep_optimizer_program_root_v1.py').read_text()
code=code.replace("for p in (UP,OLD/'actual_checkpoint_engineering_v1')", "for p in (UP,OLD/'actual_checkpoint_engineering_v1',TENSOR)")
code=code.replace("    from inspect_checkpoint import load_trusted,inspect", "    from inspect_checkpoint import load_trusted,inspect\n    from full_gradient import closure")
code=code.replace("    ck('producer_complete_qualified',raw['failed']==0 and raw['engineering_training_path_qualified'] and raw['fixed_steps_each']==2 and raw['optimizer_steps']==6)", """    known=['pretrained_CE:SGD_float32_formula_exact:1','pretrained_paired:SGD_float32_formula_exact:1']
    ck('original_failure_preserved_exactly',raw['failed']==2 and not raw['engineering_training_path_qualified'] and [c['name'] for c in raw['checks'] if not c['pass']]==known and raw['fixed_steps_each']==2 and raw['optimizer_steps']==6)
    diagnosis_path=BASE.parent/'optimizer_arithmetic_diagnosis_root_v1/NUMERICAL_DIAGNOSIS.json'
    ck('actual_bound_arithmetic_diagnosis',sha(diagnosis_path)==ar['numerical_diagnosis_sha256'])
    ck('no_change_of_optimizer_or_training_artifacts',ar['training_rerun'] is False and ar['new_checkpoint_written'] is False)
    tiny=np.finfo(np.float32).tiny
    def ftz(x):
        a=np.asarray(x,np.float32)
        return np.where(np.abs(a)<tiny,np.copysign(np.float32(0),a),a).astype(np.float32)
    def update(p,g):
        return jnp.asarray(ftz(ftz(p)-ftz(np.float32(.00001)*ftz(g))))
    numerical=jnp.asarray([tiny/2.,-tiny/2.,tiny,0],jnp.float32)*jnp.float32(.5)
    ck('actual_cpu_subnormal_probe',np.array_equal(np.asarray(numerical),np.asarray([0.,-0.,0.,0.],np.float32)))""")
code=code.replace("    vg=jax.jit(jax.value_and_grad(loss,has_aux=True));outcomes={}", """    vg=jax.jit(jax.value_and_grad(loss,has_aux=True));outcomes={}
    def actual_closure():
        out=closure()
        for path in (Path(__file__),BASE/'AUDIT_REGISTRATION.json',diagnosis_path):out[str(path.resolve())]=sha(path)
        return out
    own_start=actual_closure();save('AUDIT_SOURCE_START.json',{'files':own_start,'cpu':psutil.Process().cpu_affinity(),'python':sys.version})""")
code=code.replace("ps=jax.tree.map(lambda p,g:jnp.asarray(np.asarray(p)-np.float32(.00001)*np.asarray(g)),ps,grad)", "ps=jax.tree.map(update,ps,grad)")
code=code.replace("independent_numpy_SGD", "independent_numpy_FTZ_SGD")
code=code.replace("    failed=sum(not c['pass'] for c in checks)", """    own_end=actual_closure();changed=[p for p,h in own_start.items() if own_end.get(p)!=h];extra=sorted(set(own_end)-set(own_start))
    ck('independent_actual_closure_unchanged',not changed and not extra)
    save('AUDIT_SOURCE_END.json',{'files':own_end,'changed':changed,'new_files':extra})
    failed=sum(not c['pass'] for c in checks)""")
code=code.replace("Fresh full-model gradient replay + independent float32 NumPy SGD", "Fresh full-model gradient replay + independent float32 NumPy SGD modeling measured CPU flush-to-zero")
code=code.replace("'engineering_training_path_qualified':failed==0", "'original_producer_failed':True,'original_failure_preserved':True,'corrected_arithmetic_audit_qualified':failed==0,'engineering_training_path_qualified':failed==0,'audit_source_files':len(own_start)")
create('audit_delivery_multistep_optimizer_program_root_v2.py',code)
deploy=(T/'deploy_delivery_optimizer_arithmetic_diagnosis_root_v1.py').read_text()
deploy=deploy.replace("OUT=D/'optimizer_arithmetic_diagnosis_root_v1'", "OUT=D/'multistep_optimizer_audit_root_v2'")
deploy=deploy.replace("REL='research/review_3080_20261010/optimizer_arithmetic_diagnosis_root_v1'", "REL='research/review_3080_20261010/multistep_optimizer_audit_root_v2'")
deploy=deploy.replace('diagnose_delivery_optimizer_arithmetic_program_root_v1.py','audit_delivery_multistep_optimizer_program_root_v2.py')
deploy=deploy.replace("'random_task_paired_STEPS.jsonl'):", "'random_task_paired_STEPS.jsonl','TRAINING_SOURCE_START.json','TRAINING_SOURCE_END.json','pretrained_CE_MODEL_CARD.json','pretrained_paired_MODEL_CARD.json','random_task_paired_MODEL_CARD.json'):")
deploy=deploy.replace("    reg={'created_utc'", "    diagnostic=D/'optimizer_arithmetic_diagnosis_root_v1/downloaded/NUMERICAL_DIAGNOSIS.json'\n    reg={'numerical_diagnosis_sha256':sha(diagnostic.read_bytes()),'training_rerun':False,'new_checkpoint_written':False,'created_utc'")
deploy=deploy.replace('Two failed first-step OLD-state float32 arithmetic diagnosis; no optimizer update','Independent all6 fixed updates and original saved final weights on OLD states; explicit measured CPU float32 FTZ arithmetic; preserve original two failed checks; no training rerun')
create('deploy_delivery_multistep_optimizer_audit_root_v2.py',deploy)
inspect=(T/'inspect_delivery_optimizer_arithmetic_diagnosis_root_v1.py').read_text().replace('optimizer_arithmetic_diagnosis_root_v1','multistep_optimizer_audit_root_v2')
inspect=inspect.replace("'NUMERICAL_DIAGNOSIS.json'", "'INDEPENDENT_OPTIMIZER_ADJUDICATION.json','AUDIT_SOURCE_START.json','AUDIT_SOURCE_END.json'")
create('inspect_delivery_multistep_optimizer_audit_root_v2.py',inspect)
print('Three new immutable audit programs prepared; old files unchanged.')
