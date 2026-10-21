"""Synchronized, fixed-workload profiling on original TRAIN cases only."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cases',default='r101,r201,rc101')
    p.add_argument('--variants',default='maskco_base,quality_only,directed_only,quality_directed')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--predictor',choices=['eager','compiled'],default='eager')
    p.add_argument('--matmul-precision',choices=['default','highest'],default='default')
    p.add_argument('--repair-implementation',choices=['classic','vectorized'],default='classic')
    args=p.parse_args()
    os.environ['JAX_PLATFORMS']=args.backend
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    from profit_core import ProfitTask,partial_batch,verify_research_freeze,digest
    from profit_neural import load_checkpoint,predict
    import profit_search as search
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update('jax_default_matmul_precision',args.matmul_precision)
    if args.backend=='cuda' and jax.default_backend()!='gpu':
        raise ValueError('GPU required, no CPU fallback')
    if args.repeats<1:
        raise ValueError('positive repeats required')
    tasks=[ProfitTask(name,fold=0,role='train') for name in args.cases.split(',')]
    args.out.mkdir(parents=True,exist_ok=False)
    def dump(name,value):
        (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    models={}
    for variant in args.variants.split(','):
        checkpoint=args.checkpoint_root/variant/'checkpoint'
        trainer,record=load_checkpoint(checkpoint)
        assert record['model']['variant']==variant
        assert all(t['role']=='train' and t['fold']==0 for t in record['metadata']['training_tasks'])
        model=trainer.model
        graph=state=forward=None
        if args.predictor=='compiled':
            from profit_neural import CompiledPredictor
            forward=CompiledPredictor(model)
        def infer(inputs):
            return model(*inputs) if forward is None else forward.device_forward(*inputs)
        warm=ProfitTask('r101',fold=0,role='train')
        batch,_=partial_batch(warm,warm.initial(),tuple(range(1,11)),variant)
        start=time.perf_counter()
        reference=predict(model,batch)
        inputs=tuple(jnp.asarray(batch[k]) for k in ('features','adjacency','timestep'))
        actual=np.asarray(infer(inputs))[0]
        np.testing.assert_allclose(actual,reference,rtol=3e-4,atol=3e-4)
        warm_s=time.perf_counter()-start
        max_error=float(np.max(np.abs(actual-reference)))
        models[variant]=dict(checkpoint_json_sha256=digest(checkpoint/'checkpoint.json'),
            checkpoint_steps=record['steps'],warmup_s=warm_s,warmup_max_abs_logit_error=max_error)
        for template in tasks:
            task=ProfitTask(template.ins['name'],fold=0,role='train')
            initial=task.initial()
            original_route=task.route
            original_options=search.insertion_options
            rng=np.random.default_rng(1000)
            for rep in range(args.repeats):
                buckets={}
                phase='features'
                def timed_route(route):
                    begin=time.perf_counter()
                    try:
                        return original_route(route)
                    finally:
                        key=phase+'_route_oracle_s'
                        buckets[key]=buckets.get(key,0)+time.perf_counter()-begin
                def timed_options(*a,**kw):
                    begin=time.perf_counter()
                    try:
                        return original_options(*a,**kw)
                    finally:
                        key='insertion_options_inclusive_s'
                        buckets[key]=buckets.get(key,0)+time.perf_counter()-begin
                task.route=timed_route
                search.insertion_options=timed_options
                removed=tuple(int(i) for i in rng.choice(np.arange(1,101),10,replace=False))
                times={}
                start=time.perf_counter()
                batch,partial=partial_batch(task,initial,removed,variant)
                times['features_s']=time.perf_counter()-start
                start=time.perf_counter()
                inputs=tuple(jnp.asarray(batch[k]) for k in ('features','adjacency','timestep'))
                jax.block_until_ready(inputs)
                times['host_to_device_s']=time.perf_counter()-start
                start=time.perf_counter()
                device=infer(inputs)
                device.block_until_ready()
                times['forward_s']=time.perf_counter()-start
                start=time.perf_counter()
                logits=np.asarray(device)[0]
                assert np.all(np.isfinite(logits))
                times['device_to_host_s']=time.perf_counter()-start
                start=time.perf_counter()
                shifted=logits-logits.max(axis=1,keepdims=True)
                heatmap=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
                times['logprob_s']=time.perf_counter()-start
                phase='repair'
                profiler=cProfile.Profile() if rep==0 else None
                start=time.perf_counter()
                if profiler: profiler.enable()
                try:
                    candidate=search.repair(task,partial,removed,heatmap=heatmap,topk=8,implementation=args.repair_implementation)
                finally:
                    if profiler: profiler.disable()
                    search.insertion_options=original_options
                    task.route=original_route
                times['repair_s']=time.perf_counter()-start
                start=time.perf_counter()
                score=task.evaluate(candidate,audit=True)
                times['final_independent_audit_s']=time.perf_counter()-start
                stats=[]
                if profiler:
                    for (filename,line,function),(primitive,calls,self_s,cumulative,callers) in pstats.Stats(profiler).stats.items():
                        stats.append(dict(file=Path(filename).name,line=line,function=function,calls=calls,self_s=self_s,cumulative_s=cumulative))
                    stats=sorted(stats,key=lambda x:-x['self_s'])[:25]
                row=dict(variant=variant,case=task.ins['name'],repetition=rep,predictor=args.predictor,
                    timed_with_cprofile=profiler is not None,removed=list(removed),initial_routes=[list(r) for r in initial],
                    routes=[list(r) for r in candidate],evaluation=score,times=times,inclusive_detail=buckets,
                    top_self_time_functions=stats,cache=original_route.cache_info()._asdict())
                rows.append(row)
            print(json.dumps(dict(variant=variant,case=task.ins['name'],completed=args.repeats,predictor=args.predictor)),flush=True)
    report=dict(status='PASS_TRAIN_ONLY_FIXED_WORKLOAD_PROFILE',scientific_freeze=verify_research_freeze(),
        backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],predictor=args.predictor,
        models=models,rows=rows,matmul_precision=args.matmul_precision,repair_implementation=args.repair_implementation,formal_experiment=False,outer_test_cases_used=0,
        scope='Fixed original training initial routes and removed sets; rep0 cProfile overhead excluded from timing summaries; inclusive detail must not be added to phase totals.')
    dump('runtime_profile.json',report)


if __name__=='__main__':
    main()
