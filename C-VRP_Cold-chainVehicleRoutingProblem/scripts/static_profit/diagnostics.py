"""Registered fold0 neural-ranking contribution diagnostic; no outer tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class Provider:
    def __init__(self,arm,trained,untrained,seed=1000):
        self.arm,self.trained,self.untrained,self.seed=arm,trained,untrained,seed
        self.forward_seconds=0.0

    def sync(self):
        import jax
        model=self.untrained if self.arm=='untrained_top8' else self.trained
        jax.block_until_ready(model.state)

    def heatmap(self,task,routes,removed,variant,step):
        import numpy as np
        from profit_core import partial_batch
        from profit_neural import predict
        if self.arm=='exhaustive':return None
        if self.arm=='uniform_top8':return np.zeros((101,101),np.float32)
        before=time.perf_counter()
        batch,_=partial_batch(task,routes,removed,variant)
        model=self.untrained if self.arm=='untrained_top8' else self.trained
        logits=predict(model,batch)
        shifted=logits-logits.max(axis=1,keepdims=True)
        heat=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
        if self.arm=='shuffled_top8':
            rng=np.random.default_rng(self.seed+100000+step)
            perm=np.r_[0,rng.permutation(np.arange(1,101))]
            heat=heat[np.ix_(perm,perm)]
        self.forward_seconds+=time.perf_counter()-before
        return heat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    os.environ['JAX_PLATFORMS']='cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    from profit_core import ProfitTask,partial_batch,verify_research_freeze,EXTENSION_ROOT
    from profit_neural import load_checkpoint,ProfitMaskCO,CompiledPredictor,predict,parameter_digest
    from profit_search import repair,remove_customers
    from timed_search import guided_search,classical_search,single_cpu_guard
    import numpy as np
    import jax
    jax.config.update('jax_default_matmul_precision','highest')
    affinity=single_cpu_guard()
    config=EXTENSION_ROOT/'configs/static_delivery/diagnostic_prereg_v1.json'
    plan=read(config)
    assert sha(config)==config.with_suffix('.sha256').read_text().split()[0]
    assert plan['fold']==0 and plan['seed']==1000 and plan['outer_test_cases_used']==0
    args.out.mkdir(parents=True,exist_ok=False)
    freeze=verify_research_freeze()
    source=args.out/'source_at_run';source.mkdir()
    pins={}
    for path in Path(__file__).parent.glob('*.py'):
        (source/path.name).write_bytes(path.read_bytes());pins[path.name]=sha(path)
    dump(args.out/'identity.json',dict(prereg_sha256=sha(config),scientific_freeze=freeze,
        cpu_affinity=affinity,native_sources=pins,backend=jax.default_backend(),
        gpu=[dict(name=d.device_kind,id=d.id) for d in jax.devices()],formal_experiment=False,outer_test_cases_used=0,
        parent=str(args.parent),config=plan))
    parent_manifest=read(args.parent/'output_manifest.json')
    trained={};untrained={};warmups={};checkpoints={}
    for variant in plan['variants']:
        checkpoint=args.parent/'outputs/upgraded_teachers/train'/variant/'checkpoint'
        for name in ('weights.npz','optimizer.npz','checkpoint.json'):
            assert sha(checkpoint/name)==parent_manifest[(checkpoint/name).relative_to(args.parent).as_posix()]
        trainer,metadata=load_checkpoint(checkpoint)
        assert trainer.steps==1440 and metadata['model']['variant']==variant
        checkpoints[variant]=dict(checkpoint=str(checkpoint),checkpoint_json_sha256=sha(checkpoint/'checkpoint.json'),files=metadata['files'])
        raw=ProfitMaskCO(variant,'maskco_256',1000)
        trained[variant]=CompiledPredictor(trainer.model);untrained[variant]=CompiledPredictor(raw)
        warm=ProfitTask('r101',fold=0,role='train')
        batch,_=partial_batch(warm,warm.initial(),tuple(range(1,11)),variant)
        for tag,model,reference in [('trained',trained[variant],trainer.model),('untrained',untrained[variant],raw)]:
            start=time.perf_counter();logits=predict(model,batch)
            eager=predict(reference,batch)
            np.testing.assert_allclose(logits,eager,rtol=3e-4,atol=3e-4)
            warmups[variant+'_'+tag]=dict(elapsed_s=time.perf_counter()-start,case='r101',role='train',
                max_abs_error=float(np.max(np.abs(logits-eager))),parameter_sha256=parameter_digest(reference))
        print(json.dumps(dict(variant=variant,warmup='PASS_TRAIN_ONLY')),flush=True)
    dump(args.out/'checkpoints.json',checkpoints);dump(args.out/'warmups.json',warmups)
    # Training-only proof of actual single-CPU classical timing, not new validation tuning.
    classic=args.out/'classical_train';classic.mkdir()
    for method in ('alns','vns'):
        result=classical_search(ProfitTask('r101',fold=0,role='train'),method,seconds=60,iterations=1000000,seed=1000)
        dump(classic/(method+'.json'),result)
    fixed=args.out/'fixed_work';fixed.mkdir();wall=args.out/'wall';wall.mkdir()
    masks={}
    for index,case in enumerate(plan['cases']):
        rng=np.random.default_rng(1000+index)
        masks[case]=[tuple(int(i) for i in rng.choice(np.arange(1,101),20,replace=False)) for _ in range(plan['fixed_repetitions'])]
    dump(args.out/'mask_schedule.json',masks)
    # Fixed work replays identical online-initializer states and masks, not adaptive trajectories.
    for block in plan['fixed_order']:
        variant,case=block['variant'],block['case']
        for rep in range(plan['fixed_repetitions']):
            for arm,removed_count in block['arms']:
                task=ProfitTask(case,fold=0,role='validation')
                start=time.perf_counter();initial=task.initial();init_s=time.perf_counter()-start
                removed=masks[case][rep][:removed_count]
                provider=Provider(arm,trained[variant],untrained[variant]);provider.sync()
                begin=time.perf_counter();heat=provider.heatmap(task,initial,removed,variant,rep)
                stats={};routes=repair(task,remove_customers(initial,removed),removed,heatmap=heat,topk=8,
                    implementation='vectorized',fallback=False,stats=stats)
                validated=time.perf_counter();value=task.evaluate(routes,audit=True)
                final=time.perf_counter()
                result=dict(variant=variant,case=case,arm=arm,remove_count=removed_count,repetition=rep,
                    initial_routes=[list(r) for r in initial],removed=list(removed),routes=[list(r) for r in routes],
                    initial=task.evaluate(initial,audit=True),final=value,operator_counts=stats,
                    timing=dict(initialization_s=init_s,operator_s=validated-begin,independent_audit_s=final-validated,
                                total_s=init_s+final-begin,feature_forward_logprob_s=provider.forward_seconds),
                    route_cache=dict(task.route.cache_info()._asdict()),task=task.identity())
                name=f'{variant}_{case}_{arm}_m{removed_count}_r{rep}.json';dump(fixed/name,result)
        print(json.dumps(dict(fixed_block_complete=block)),flush=True)
    for block in plan['wall_order']:
        variant,case,arm,m,seconds=[block[k] for k in ('variant','case','arm','remove_count','seconds')]
        provider=Provider(arm,trained[variant],untrained[variant])
        task=ProfitTask(case,fold=0,role='validation')
        result=guided_search(task,provider,variant,seconds=seconds,iterations=1000000,seed=1000,remove_count=m)
        result.update(variant=variant,case=case,arm=arm,remove_count=m,feature_forward_logprob_s=provider.forward_seconds,
                      route_cache=task.route.cache_info()._asdict())
        dump(wall/f'{variant}_{case}_{arm}_m{m}.json',result)
        print(json.dumps(dict(wall_complete=block,status=result['status'],net_benefit_U=result['final']['net_benefit_U'] if result['final'] else None)),flush=True)
    dump(args.out/'summary.json',dict(status='DIAGNOSTIC_EXECUTION_COMPLETE_REQUIRES_LOCAL_AUDIT',
        scientific_freeze=verify_research_freeze(),fixed_results=len(list(fixed.glob('*.json'))),wall_results=len(list(wall.glob('*.json'))),
        formal_experiment=False,outer_test_cases_used=0,prereg_sha256=sha(config)))


if __name__=='__main__':main()
