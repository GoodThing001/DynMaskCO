"""Complete only missing dev09 validation cells using its unchanged saved sources/weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    parent=args.parent.resolve()
    if parent.name!='dev09' or (parent/'run.exit').read_text().strip()!='124':
        raise ValueError('This recovery is specific to the preserved dev09 timeout')
    workspace=parent/'workspace'
    manifest=read(workspace/'payload_manifest.json')
    def verify_parent():
        for name,expected in manifest.items():
            path=(workspace/name).resolve()
            if not path.is_relative_to(workspace) or digest(path)!=expected:
                raise ValueError('Parent payload changed: '+name)
    verify_parent()
    source=workspace/'C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit'
    sys.path.insert(0,str(source))
    os.environ['JAX_PLATFORMS']='cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    from profit_core import ProfitTask,verify_research_freeze
    from profit_search import alns_search
    from profit_neural import load_checkpoint,parameter_count
    from run import run_searches,dump
    import jax
    jax.config.update('jax_default_matmul_precision','highest')
    freeze=verify_research_freeze()
    original=read(parent/'output_manifest.json')
    missing=[('c206','alns')]+[(c,m) for c in ('c207','c208') for m in ('maskco_base','directed_only','vns','alns')]
    for case,method in missing:
        name='outputs/upgraded_teachers/validation/'+case+'/'+method+'_search.json'
        if name in original or (parent/name).exists():
            raise ValueError('Refuse duplicate result: '+name)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'recovery_source.py').write_bytes(Path(__file__).read_bytes())
    identity=read(parent/'outputs/upgraded_teachers/validation/c205/run_identity.json')
    for name,expected in identity['native_sources'].items():
        if digest(source/name)!=expected:raise ValueError('Executed source changed')
    recovered=args.out/'upgraded_teachers/validation'
    c206=recovered/'c206';c206.mkdir(parents=True)
    dump(c206/'run_identity.json',dict(**identity,recovery_parent=str(parent),recovery_only_methods=['alns']))
    result=alns_search(ProfitTask('c206',fold=0,role='validation'),seconds=10,iterations=100000,seed=1000)
    dump(c206/'alns_search.json',result)
    models,records={},{}
    for variant in ('maskco_base','directed_only'):
        checkpoint=parent/'outputs/upgraded_teachers/train'/variant/'checkpoint'
        for name in ('weights.npz','optimizer.npz','checkpoint.json'):
            relative=(checkpoint/name).relative_to(parent).as_posix()
            if digest(checkpoint/name)!=original[relative]:raise ValueError('Parent checkpoint changed')
        trainer,metadata=load_checkpoint(checkpoint)
        if trainer.steps!=1440 or metadata['model']['variant']!=variant:raise ValueError('Wrong checkpoint')
        models[variant]=trainer.model
        records[variant]=dict(checkpoint=str(checkpoint),checkpoint_steps=trainer.steps,
            checkpoint_json_sha256=digest(checkpoint/'checkpoint.json'),verified_weights=metadata['files'],
            parameter_count=parameter_count(trainer.model))
    for case in ('c207','c208'):
        out=recovered/case;out.mkdir(parents=True)
        arguments=dict(fold=0,seed=1000,case=case,role='validation',predictor='compiled',
            matmul_precision='highest',repair_implementation='vectorized',seconds=10,iterations=100000)
        current_identity={**identity,'arguments':{**identity['arguments'],**arguments,'out':str(out)},
                          'recovery_parent':str(parent)}
        dump(out/'run_identity.json',current_identity)
        snapshot=out/'source_at_run';snapshot.mkdir()
        for name in identity['native_sources']:(snapshot/name).write_bytes((source/name).read_bytes())
        run_searches(SimpleNamespace(**arguments),out,ProfitTask(case,fold=0,role='validation'),models,records,freeze,jax)
    verify_parent()
    dump(args.out/'recovery_summary.json',dict(status='PASS_MISSING_VALIDATION_RECOVERY',parent=str(parent),
        original_exit_code=124,missing_cells_completed=[dict(case=c,method=m) for c,m in missing],
        training_repeated=False,original_sources_unchanged=True,scientific_freeze=verify_research_freeze(),
        formal_experiment=False,outer_test_cases_used=0))


if __name__=='__main__':main()
