"""Development entry point; formal outer-test execution is intentionally gated."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['smoke','classical','train'])
    p.add_argument('--case',default='r101')
    p.add_argument('--cases',help='comma-separated original training case names; train command requires this')
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--role',choices=['train','validation'],default='train')
    p.add_argument('--variant',choices=['maskco_base','quality_only','directed_only','quality_directed'],default='quality_directed')
    p.add_argument('--profile',choices=['smoke_cpu','maskco_256'],default='smoke_cpu')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--teacher-seconds',type=float,default=0.3)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if args.steps < 1 or args.threads < 1 or args.seconds <= 0 or args.iterations < 1:
        p.error('positive steps, threads, seconds and iterations required')
    if args.command in ('train','smoke') and args.role != 'train':
        p.error('training and training smoke require the train role')
    if args.command == 'train' and not args.cases:
        p.error('explicit --cases required, no automatic test-case fallback')
    # Backend selection happens before any JAX import.
    os.environ['JAX_PLATFORMS'] = args.backend
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    if args.backend == 'cpu':
        os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads='+str(args.threads))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from profit_core import EXTENSION_ROOT, ProfitTask, verify_research_freeze, partial_batch, digest, VARIANTS
    from profit_search import vns_search, alns_search, neural_search
    import numpy as np
    freeze = verify_research_freeze()
    args.out.mkdir(parents=True,exist_ok=False)
    out = args.out.resolve()
    # Store the actual native source bytes used by this development run.
    snapshot = out/'source_at_run'
    snapshot.mkdir()
    native_pins = {}
    for source in Path(__file__).parent.glob('*.py'):
        (snapshot/source.name).write_bytes(source.read_bytes())
        native_pins[source.name] = digest(source)
    dump(out/'run_identity.json',dict(scientific_freeze=freeze,command=args.command,
        fold=args.fold,role=args.role,seed=args.seed,model_profile=args.profile,
        formal_experiment=False,native_sources=native_pins))
    task = ProfitTask(args.case,fold=args.fold,role=args.role)
    if args.command == 'classical':
        results = {}
        for method in ('vns','alns'):
            results[method] = (vns_search if method=='vns' else alns_search)(task,
                seconds=args.seconds,iterations=args.iterations,seed=args.seed)
            dump(out/(method+'.json'),results[method])
            print(json.dumps(dict(method=method,final=results[method]['final'])),flush=True)
        dump(out/'summary.json',dict(status='PASS_CLASSICAL_INTEGRATION_DEV',task=task.identity(),
            methods=list(results),formal_experiment=False))
        return

    from profit_neural import ProfitMaskCO, Trainer, predict, parameter_digest, parameter_count, save_checkpoint, load_checkpoint
    import jax
    cases = [s.lower() for s in args.cases.split(',')] if args.cases else [args.case.lower()]
    conditions = ([(profile,qmin) for profile in task.c['quality']['kinetic_sensitivity'] for qmin in (0.8,0.85)]
                  if args.command=='train' else [('main_Product1',0.8),('fast_Product2',0.85)])
    teachers = []
    for name in cases:
        for profile,minimum in conditions:
            current = ProfitTask(name,fold=args.fold,role='train',profile=profile,minimum_quality=minimum)
            teacher = vns_search(current,seconds=args.teacher_seconds,iterations=1000,seed=args.seed)
            routes = tuple(tuple(r) for r in teacher['routes'])
            teachers.append((current,routes))
    dump(out/'teachers.json',[dict(task=t.identity(),routes=[list(r) for r in routes],
        evaluation=t.evaluate(routes,audit=True),origin='online exact-objective VNS from shared NN on training source only')
        for t,routes in teachers])
    variants = ['maskco_base','quality_directed'] if args.command=='smoke' else [args.variant]
    trained,records = {},{}
    rng = np.random.default_rng(args.seed)
    masks = [tuple(int(i) for i in rng.choice(np.arange(1,101),size=20,replace=False)) for _ in range(args.steps)]
    for variant in variants:
        model = ProfitMaskCO(variant,args.profile,args.seed)
        trainer = Trainer(model)
        before = parameter_digest(model)
        metrics = []
        for step in range(args.steps):
            current,routes = teachers[step%len(teachers)]
            batch,_ = partial_batch(current,routes,masks[step],variant)
            row = trainer.step(batch)
            row['training_case'] = current.ins['name']
            metrics.append(row)
            print(json.dumps(dict(variant=variant,**row)),flush=True)
        if parameter_digest(model)==before or not all(r['updated_parameter_leaves']>0 for r in metrics):
            raise ValueError('actual neural parameters did not update')
        current,routes = teachers[0]
        batch,_ = partial_batch(current,routes,masks[0],variant)
        prediction = predict(model,batch)
        metadata = dict(task=current.identity(),training_tasks=[t.identity() for t,_ in teachers],
                        source_code_sha256=native_pins,development_only=True)
        checkpoint = out/variant/'checkpoint'
        save_checkpoint(checkpoint,trainer,metadata)
        restored,_ = load_checkpoint(checkpoint)
        if not np.array_equal(prediction,predict(restored.model,batch)):
            raise ValueError('checkpoint prediction changed after reloading')
        # Check optimizer restoration, not just weight loading.
        original_next = trainer.step(batch)
        restored_next = restored.step(batch)
        if parameter_digest(trainer.model)!=parameter_digest(restored.model) or original_next!=restored_next:
            raise ValueError('optimizer/weight resume is not identical')
        trained[variant] = restored.model
        records[variant] = dict(parameter_count=parameter_count(model),training=metrics,
            checkpoint_prediction_identical=True,optimizer_resume_identical=True,
            actual_train_steps=args.steps,additional_resume_verification_steps=1)
        dump(out/(variant+'_training.json'),records[variant])
    if args.command == 'train':
        dump(out/'summary.json',dict(status='PASS_DEVELOPMENT_TRAINING',variants=records,
            tasks=[t.identity() for t,_ in teachers],backend=jax.default_backend(),formal_experiment=False))
        return
    # Every search arm receives a fresh oracle/cache and the same online initializer.
    searches = {}
    for variant,model in trained.items():
        current = ProfitTask(args.case,fold=args.fold,role='train')
        batch,_ = partial_batch(current,teachers[0][1],masks[0],variant)
        predict(model,batch)  # Training-only warmup, outside solve timing.
        searches[variant] = neural_search(current,model,variant,seconds=args.seconds,
            iterations=args.iterations,seed=args.seed)
        dump(out/(variant+'_search.json'),searches[variant])
        print(json.dumps(dict(variant=variant,search_final=searches[variant]['final'])),flush=True)
    for method in ('vns','alns'):
        current = ProfitTask(args.case,fold=args.fold,role='train')
        searches[method] = (vns_search if method=='vns' else alns_search)(current,
            seconds=args.seconds,iterations=args.iterations,seed=args.seed)
        dump(out/(method+'_search.json'),searches[method])
    initial_values = [s['initial']['net_benefit_U'] for s in searches.values()]
    if max(initial_values)-min(initial_values)>1e-7:
        raise ValueError('methods did not use the same common initial solution')
    summary = dict(status='PASS_ALGORITHM_INTEGRATION_NOT_PERFORMANCE_EVIDENCE',
        task=task.identity(),scientific_freeze=verify_research_freeze(),backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],variants=records,
        searches={k:{x:s[x] for x in ('initial','final','iterations','elapsed_s')} for k,s in searches.items()},
        outer_test_cases_used=0,formal_experiment=False,method_superiority_established=False)
    dump(out/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],out=str(out))),flush=True)


if __name__ == '__main__':
    main()
