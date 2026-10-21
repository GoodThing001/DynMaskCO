"""Registered 1440->2880 continuation and mask-switch pairing, gated by dev11 audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','validate'],required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--diagnostic-parent',type=Path,required=True)
    p.add_argument('--train-root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    os.environ['JAX_PLATFORMS']='cuda';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    threads=4 if args.phase=='train' else 1
    if args.phase=='validate' and hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]=str(threads)
    from profit_core import ProfitTask,partial_batch,verify_research_freeze,EXTENSION_ROOT,fold_definition
    from profit_neural import load_checkpoint,save_checkpoint,parameter_digest,predict,CompiledPredictor,parameter_count
    from training_data import load_teachers,sample_schedule
    from diagnostics import Provider
    from timed_search import guided_search,single_cpu_guard
    from profit_search import repair,remove_customers
    import numpy as np
    import jax
    decision_path=EXTENSION_ROOT/'results/static_delivery/algorithm_diagnostics_20261021/diagnostic_decision.json'
    decision=read(decision_path)
    audit_path=decision_path.with_name('diagnostic_audit.json')
    assert sha(audit_path)==decision['audit_sha256']
    assert read(audit_path)['status']=='PASS_INDEPENDENT_CONTRIBUTION_DIAGNOSTIC'
    assert (args.diagnostic_parent/'run.exit').read_text().strip()=='0'
    assert sha(EXTENSION_ROOT/'configs/static_delivery/diagnostic_prereg_v1.json')==decision['prereg_sha256']
    variants=decision['approved_training_variants']
    assert variants and set(variants)<=set(('maskco_base','directed_only'))
    args.out.mkdir(parents=True,exist_ok=False)
    source=args.out/'source_at_run';source.mkdir()
    pins={}
    for path in Path(__file__).parent.glob('*.py'):
        (source/path.name).write_bytes(path.read_bytes());pins[path.name]=sha(path)
    identity=dict(phase=args.phase,scientific_freeze=verify_research_freeze(),decision_sha256=sha(decision_path),
        prereg_sha256=decision['prereg_sha256'],variants=variants,native_sources=pins,
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=threads,backend=jax.default_backend(),
        parent=str(args.parent),formal_experiment=False,outer_test_cases_used=0)
    dump(args.out/'identity.json',identity)
    if args.phase=='train':
        assert args.parent.name=='dev09'
        parent=args.parent/'outputs/upgraded_teachers/train'
        parent_manifest=read(args.parent/'output_manifest.json')
        assert sha(parent/'teachers.json')==parent_manifest[(parent/'teachers.json').relative_to(args.parent).as_posix()]
        teachers=load_teachers(parent/'teachers.json',0,fold_definition(0)['train'])
        oldplan=read(parent/'training_plan.json')
        schedule=sample_schedule(len(teachers),2880,1000,'shuffled')
        assert oldplan['teacher_indices']==schedule[:1440]
        rng=np.random.default_rng(1000)
        masks=[tuple(int(i) for i in rng.choice(np.arange(1,101),20,replace=False)) for _ in range(2880)]
        dump(args.out/'pairing_plan.json',dict(seed=1000,parent_step=1440,final_step=2880,
            teacher_indices=schedule,masks20=masks,branches=['keep20','switch10'],
            teacher_source_sha256=sha(parent/'teachers.json'),mask_rule='keep20 uses all20; switch10 uses first10 of same20-customer stream after step1440.'))
        dump(args.out/'teachers.json',read(parent/'teachers.json'))
        for variant in variants:
            for branch in ('keep20','switch10'):
                begin=time.perf_counter()
                trainer,record=load_checkpoint(parent/variant/'checkpoint')
                assert trainer.steps==1440 and record['model']['variant']==variant
                for name in ('weights.npz','optimizer.npz','checkpoint.json'):
                    path=parent/variant/'checkpoint'/name
                    assert sha(path)==parent_manifest[path.relative_to(args.parent).as_posix()]
                before=parameter_digest(trainer.model);metrics=[]
                for index in range(1440,2880):
                    task,routes=teachers[schedule[index]]
                    removed=masks[index] if branch=='keep20' else masks[index][:10]
                    batch,_=partial_batch(task,routes,removed,variant)
                    row=trainer.step(batch)
                    row.update(teacher_index=schedule[index],training_case=task.ins['name'],mask_count=len(removed))
                    metrics.append(row)
                    if trainer.steps%120==0:print(json.dumps(dict(variant=variant,branch=branch,step=trainer.steps,loss=row['loss'])),flush=True)
                assert trainer.steps==2880 and parameter_digest(trainer.model)!=before
                directory=args.out/variant/branch;directory.mkdir(parents=True)
                metadata={**record['metadata'],'source_code_sha256':pins,'continuation_parent_checkpoint_sha256':sha(parent/variant/'checkpoint/checkpoint.json'),
                          'continuation_branch':branch,'development_only':True,'decision_sha256':sha(decision_path)}
                checkpoint=directory/'checkpoint'
                save_checkpoint(checkpoint,trainer,metadata)
                task,routes=teachers[0];batch,_=partial_batch(task,routes,masks[-1] if branch=='keep20' else masks[-1][:10],variant)
                restored,_=load_checkpoint(checkpoint)
                assert np.array_equal(predict(trainer.model,batch),predict(restored.model,batch))
                a=trainer.step(batch);b=restored.step(batch)
                assert a==b and parameter_digest(trainer.model)==parameter_digest(restored.model)
                dump(directory/'training.json',dict(branch=branch,variant=variant,updates=metrics,steps=2880,
                    parent_steps=1440,additional_steps=1440,elapsed_s=time.perf_counter()-begin,parameter_count=parameter_count(trainer.model),
                    checkpoint_prediction_identical=True,optimizer_resume_identical=True,
                    saved_checkpoint_sha256=sha(checkpoint/'checkpoint.json'),additional_resume_verification_steps=1,
                    source_teacher_sha256=sha(parent/'teachers.json')))
        dump(args.out/'summary.json',dict(status='PASS_REGISTERED_CONTINUATION_TRAINING',variants=variants,steps=2880,
            branches=['keep20','switch10'],formal_experiment=False,outer_test_cases_used=0))
        return
    affinity=single_cpu_guard()
    assert args.train_root and read(args.train_root/'summary.json')['status']=='PASS_REGISTERED_CONTINUATION_TRAINING'
    jax.config.update('jax_default_matmul_precision','highest')
    masks=read(args.diagnostic_parent/'outputs/diagnostics/mask_schedule.json')
    for variant in variants:
        for branch in ('keep20','switch10'):
            checkpoint=args.train_root/variant/branch/'checkpoint'
            trainer,record=load_checkpoint(checkpoint)
            assert trainer.steps==2880
            model=CompiledPredictor(trainer.model)
            warm=ProfitTask('r101',fold=0,role='train');batch,_=partial_batch(warm,warm.initial(),tuple(range(1,11)),variant)
            start=time.perf_counter();values=predict(model,batch);eager=predict(trainer.model,batch)
            np.testing.assert_allclose(values,eager,rtol=3e-4,atol=3e-4)
            folder=args.out/variant/branch;folder.mkdir(parents=True)
            dump(folder/'warmup.json',dict(role='train',case='r101',elapsed_s=time.perf_counter()-start,
                max_abs_logit_error=float(np.max(np.abs(values-eager))),checkpoint_sha256=sha(checkpoint/'checkpoint.json'),files=record['files']))
            for case in ('c205','c206','c207','c208'):
                task=ProfitTask(case,fold=0,role='validation')
                provider=Provider('trained_top8',model,model)
                result=guided_search(task,provider,variant,seconds=10,iterations=1000000,seed=1000,remove_count=10)
                result.update(variant=variant,branch=branch,case=case,checkpoint_steps=2880,cpu_affinity=affinity)
                dump(folder/(case+'_wall.json'),result)
                for rep,mask in enumerate(masks[case]):
                    current=ProfitTask(case,fold=0,role='validation');initial=current.initial();removed=tuple(mask[:10])
                    heat=provider.heatmap(current,initial,removed,variant,rep);stats={}
                    routes=repair(current,remove_customers(initial,removed),removed,heatmap=heat,topk=8,
                        implementation='vectorized',fallback=False,stats=stats)
                    dump(folder/(case+f'_fixed_{rep}.json'),dict(case=case,variant=variant,branch=branch,repetition=rep,
                        routes=[list(r) for r in routes],removed=list(removed),initial_routes=[list(r) for r in initial],
                        final=current.evaluate(routes,audit=True),operator_counts=stats))
                print(json.dumps(dict(variant=variant,branch=branch,case=case,net_benefit_U=result['final']['net_benefit_U'])),flush=True)
    dump(args.out/'summary.json',dict(status='CONTINUATION_VALIDATION_COMPLETE_REQUIRES_LOCAL_AUDIT',
        variants=variants,steps=2880,branches=['keep20','switch10'],formal_experiment=False,outer_test_cases_used=0))


if __name__=='__main__':main()
