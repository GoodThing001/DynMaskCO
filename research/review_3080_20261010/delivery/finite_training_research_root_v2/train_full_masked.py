"""Fixed-final complete encoder/decoder trainer; research requires three gates."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'features_objective_qa_v1'))
sys.path.insert(0, str(HERE.parent / 'actual_checkpoint_engineering_v1'))
from finite_common import sha, digest, dump, schedule, verify_research_gate, check_dependency_identity
from features_objective import make_features, UNITS, extend_projection


def run(a):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import psutil
    reg = json.loads(a.registration.read_text(encoding='utf-8'))
    check_dependency_identity(reg, a.root, a.upstream)
    if a.mode == 'research':
        verify_research_gate(reg, {'teacher_engineering': a.teacher_gate,
            'actual_10D_objective': a.objective_gate, 'deployment_10s_budget': a.budget_gate})
        steps, batch_size, schedule_seed, expected_states = 128, 4, reg['seeds']['training_schedule'], 256
    else:
        if reg.get('status') != 'REGISTERED_ENGINEERING_ONLY':
            raise ValueError('Unregistered engineering stage')
        steps, batch_size, schedule_seed, expected_states = 2, 4, 202610110073, 8
    if reg.get('optimizer') != {'name': 'sgd_without_momentum', 'learning_rate': .00001}:
        raise ValueError('Registered common optimizer changed')
    if type(reg.get('preference_coefficient')) is not float or reg['preference_coefficient'] != .2:
        raise ValueError('Registered objective coefficient changed')
    archive = a.teacher_dir / 'TEACHER_STATES.jsonl'
    if a.mode == 'research':
        if a.accepted_teacher is None:
            raise ValueError('Fresh256 states require independent complete teacher acceptance')
        accepted = json.loads(a.accepted_teacher.read_text(encoding='utf-8'))
        if (accepted.get('independent_acceptance') is not True or
            accepted.get('registration_sha256') != sha(a.registration) or
            accepted.get('teacher_archive_sha256') != sha(archive) or
            accepted.get('states') != 256 or accepted.get('failed') != 0):
            raise ValueError('Fresh teacher independent stage not qualified')
    summary = json.loads((a.teacher_dir / 'TEACHER_SUMMARY.json').read_text(encoding='utf-8'))
    if summary.get('registration_sha256') != sha(a.registration) or summary['teacher_archive_sha256'] != sha(archive):
        raise ValueError('Teacher registration/archive identity mismatch')
    rows = [json.loads(s) for s in archive.read_text(encoding='utf-8').splitlines()]
    if len(rows) != expected_states or len({tuple(r['state_key']) for r in rows}) != expected_states:
        raise ValueError('All exact unique states required without screening')
    for row in rows:
        expected = row['state_sha256']
        if digest({k:v for k,v in row.items() if k!='state_sha256'}) != expected:
            raise ValueError('Teacher state changed')
    expected_ids = reg['train_instance_ids'] if a.mode=='research' else [900000,900001]
    expected_keys = {(i,j) for i in expected_ids for j in range(4)}
    if {tuple(r['state_key']) for r in rows} != expected_keys:
        raise ValueError('Exact registered64x4 or engineering2x4 keys required')
    expected_data_seed = reg['seeds']['train'] if a.mode=='research' else 202610110071
    expected_mask_seed = reg['seeds']['teacher_mask'] if a.mode=='research' else 202610110072
    if type(summary['dataset_seed']) is not int or summary['dataset_seed'] != expected_data_seed:
        raise ValueError('Teacher dataset seed mismatch')
    if type(summary['mask_seed']) is not int or summary['mask_seed'] != expected_mask_seed:
        raise ValueError('Teacher mask seed mismatch')
    a.out.mkdir(parents=True, exist_ok=False)
    available = psutil.virtual_memory().available
    if available < 8 * 1024**3:
        dump(a.out / 'RESOURCE_FAILURE.json', {'available_ram_bytes': available, 'minimum_gib': 8})
        raise RuntimeError('RAM gate failed')
    p = psutil.Process()
    if a.cpu not in p.cpu_affinity():
        raise ValueError('Requested CPU unavailable')
    p.cpu_affinity([a.cpu])
    start = time.perf_counter()
    dump(a.out / 'TRAINING_START_IDENTITY.json', {'registration_sha256': sha(a.registration),
        'teacher_archive_sha256': sha(archive), 'cpu_affinity': p.cpu_affinity(), 'available_ram_bytes': available,
        'dependency_sources': reg['dependency_sources'], 'steps_each': steps, 'batch_size': batch_size})
    for folder in ('delivery', 'coldchain', 'research_20261010'):
        sys.path.insert(0, str(a.root / 'scripts' / folder))
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_equal_schedule import generate, make_schedule
    contract = DeliveryContract()
    if contract.identity() != reg['contract_hash']:
        raise ValueError('Training contract changed')
    from inspect_checkpoint import load_trusted, inspect
    saved, checkpoint_sha = load_trusted(a.checkpoint, a.upstream)
    inspection = inspect(saved, checkpoint_sha)
    if inspection['parameter_elements'] != 76259840 or inspection['parameter_leaves'] != 234:
        raise ValueError('Actual full original model required')
    if digest(inspection['model_config']) != digest(reg['original_model_config']):
        raise ValueError('Original configuration changed')
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from helpers import sol2adj
    from modules.functional import coord_normalize
    features, contexts, timesteps, CE_targets, pair_candidates, pair_incumbents, labels = [], [], [], [], [], [], []
    def adjacency(routes):
        vector = [0]
        for route in routes:
            vector.extend(route + [0])
        return np.asarray(sol2adj(jnp.asarray([vector], jnp.int32), dtype=jnp.float32,
                                 num_nodes=51, is_cvrp=True))[0]
    for row in rows:
        data = row['data']
        regenerated, original_routes = generate(expected_data_seed, row['instance_id'])
        registered_masks = make_schedule(51,4,expected_mask_seed,row['instance_id'],fraction=.5)
        if digest(regenerated) != row['data_sha256'] or original_routes != row['incumbent_routes']:
            raise ValueError('Actual teacher data/initial independently regenerated mismatch')
        if row['mask'] != registered_masks[row['mask_ordinal']]:
            raise ValueError('Actual registered mask schedule mismatch')
        if digest(data) != row['data_sha256']:
            raise ValueError('Data changed')
        report = evaluate(data, row['CE_target_routes'], contract)
        if report['hard'] is not True or report['complete'] is not True:
            raise ValueError('Full C0 reconstruction target not hard feasible')
        if abs(report['cost']-row['CE_target_certified_report']['cost']) > 1e-8:
            raise ValueError('Full C0 target decomposition changed')
        initial_report = evaluate(data, row['incumbent_routes'], contract)
        candidate_report = evaluate(data, row['candidate_routes'], contract)
        valid = row['teacher_pair_valid'] is True
        if valid and (candidate_report['hard'] is not True or abs(candidate_report['cost']-row['candidate_report']['cost']) > 1e-8):
            raise ValueError('Preference route/trace changed')
        delta = initial_report['cost']-candidate_report['cost'] if valid else None
        label = 0 if delta is None or abs(delta)<=1e-8 else (1 if delta>0 else -1)
        if type(row['preference_sign']) is not int or row['preference_sign'] != label:
            raise ValueError('Preference sign changed')
        target_choice = row['candidate_routes'] if valid and delta>1e-8 else row['incumbent_routes']
        if row['CE_target_routes'] != target_choice:
            raise ValueError('CE target must be lower full C0 route of pair, ties incumbent')
        coordinates = np.asarray(coord_normalize(jnp.asarray(data['coords'], jnp.float32)[None]))[0]
        features.append(make_features(data, contract, coordinates, UNITS, contract.identity())[0])
        partial = np.asarray(row['partial_adjacency'], np.float32)
        if partial.shape != (51,51):
            raise ValueError('Invalid full masked context')
        contexts.append(partial)
        timesteps.append(partial[1:].sum()/100.)
        CE_targets.append(adjacency(row['CE_target_routes']))
        pair_candidates.append(adjacency(row['candidate_routes'] if valid else row['incumbent_routes']))
        pair_incumbents.append(adjacency(row['incumbent_routes']))
        labels.append(label)
    arrays = list(map(lambda x:jnp.asarray(np.asarray(x),jnp.float32),
                      (features,contexts,timesteps,CE_targets,pair_candidates,pair_incumbents,labels)))
    fixed = schedule(expected_states, steps, batch_size, schedule_seed)
    cfg = deepcopy(saved['model_config'])
    cfg.dtype, cfg.encoder_input_dim = 'float32', 10
    model = cfg.construct_model()
    graph = nnx.graphdef(model)
    initial = deepcopy(saved['params'])
    initial['init_proj']['kernel'].value = jnp.asarray(extend_projection(initial['init_proj']['kernel'].value))
    del saved, model
    gc.collect()
    def inventory(params):
        import hashlib
        return {str(path):hashlib.sha256(np.asarray(value.value).tobytes()).hexdigest()
                for path,value in params.flat_state()}
    initial_identity = inventory(initial)
    def loss(params, x, context, timestep, ce, candidate, incumbent, label, coefficient):
        m = nnx.merge(graph, params)
        logits = m.decode(m.encode(x), timestep, context)
        logp = jax.nn.log_softmax(logits,axis=-1)
        CE = -(logp[:,1:]*ce[:,1:]).mean(axis=(1,2))*(51/2)
        score = (logp[:,1:]*(candidate[:,1:]-incumbent[:,1:])).sum(axis=(1,2))/100.
        preference = jnp.where(label==0,0.,jax.nn.softplus(-label*score))
        return (CE+coefficient*preference).mean(), (CE.mean(),preference.mean())
    value_and_grad = jax.jit(jax.value_and_grad(loss,has_aux=True))
    arm_summaries = {}
    for arm,coefficient in (('reconstruction_control',0.),('paired_objective',.2)):
        params = deepcopy(initial)
        before = inventory(params)
        if before != initial_identity:
            raise ValueError('Arms not at exact same initialization')
        path = a.out/(arm+'_STEPS.jsonl')
        with path.open('x',encoding='utf-8') as stream:
            for step,indices in enumerate(fixed,1):
                batch = [array[jnp.asarray(indices)] for array in arrays]
                (total,parts),grads = value_and_grad(params,*batch,jnp.asarray(coefficient,jnp.float32))
                gradient_inventory = inventory(grads)
                actual_before = inventory(params)
                finite = len(gradient_inventory)==234 and all(np.isfinite(np.asarray(g.value)).all() for _,g in grads.flat_state())
                if not finite:
                    raise ValueError('Nonfinite actual full gradient; retained prefix, no checkpoint selection')
                params = jax.tree.map(lambda p,g:p-.00001*g,params,grads)
                del grads
                record = {'step':step,'state_indices':indices,'state_keys':[rows[i]['state_key'] for i in indices],
                    'state_sha256':[rows[i]['state_sha256'] for i in indices], 'coefficient':coefficient,
                    'total_loss':float(total),'CE_loss':float(parts[0]),'preference_loss':float(parts[1]),
                    'all234_parameter_leaves_in_gradient':len(gradient_inventory)==234,'gradient_finite':bool(finite),
                    'actual_gradient_leaf_count':len(gradient_inventory),'actual_gradient_tensor_digest':digest(gradient_inventory),
                    'actual_before_parameter_digest':digest(actual_before),'actual_after_parameter_digest':digest(inventory(params))}
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
        final = inventory(params)
        changes = {g:sum(final[p]!=before[p] for p in before
                    if g=='full' or p.startswith("('"+g+"',")) for g in ('full','encoder','decoder')}
        if changes['encoder']==0 or changes['decoder']==0:
            raise ValueError('Complete model did not update')
        destination = a.out/(arm+'_FIXED_FINAL.pkl')
        with destination.open('xb') as stream:
            pickle.dump({'params':params,'model_config':cfg,'fixed_step':steps,
                         'research_model_eligible':a.mode=='research','registration_sha256':sha(a.registration)},stream)
        with destination.open('rb') as stream:
            loaded = pickle.load(stream)['params']
        if inventory(loaded)!=final:
            raise ValueError('Save/reload complete parameter identity failed')
        arm_summaries[arm] = {'fixed_steps':steps,'coefficient':coefficient,'initial_identity':digest(before),
            'final_identity':digest(final),'parameter_change_counts':changes,'checkpoint_sha256':sha(destination),
            'steps_archive_sha256':sha(path),'save_reload_params_equal':True}
        del params,loaded
        gc.collect()
    check_dependency_identity(reg,a.root,a.upstream)
    if sha(a.checkpoint)!=checkpoint_sha:
        raise ValueError('Original checkpoint changed during training')
    result = {'mode':a.mode,'registration_sha256':sha(a.registration),'teacher_archive_sha256':sha(archive),
        'actual_states':expected_states,'batch_size':batch_size,'fixed_schedule':fixed,
        'fixed_steps_each':steps,'arms':arm_summaries,'actual_full_model_parameter_elements':76263424,
        'original_checkpoint_unchanged':sha(a.checkpoint)==checkpoint_sha,
        'source_unchanged':True,'wall_s':time.perf_counter()-start,
        'snapshot_only':True,'dynamic_online_qualified':False,'stable_gain_established':False,
        'research_model_eligible':a.mode=='research', 'engineering_pass':True}
    dump(a.out/'TRAINING_RESULT.json',result)
    print(json.dumps({'engineering_pass':True,'fixed_steps_each':steps,'wall_s':result['wall_s']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('root','upstream','checkpoint','registration','teacher-dir','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--accepted-teacher',type=Path)
    ap.add_argument('--cpu',type=int,required=True)
    ap.add_argument('--mode',choices=('engineering','research'),required=True)
    ap.add_argument('--teacher-gate',type=Path)
    ap.add_argument('--objective-gate',type=Path)
    ap.add_argument('--budget-gate',type=Path)
    args=ap.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out/'FAILURE.json').exists():
            dump(args.out/'FAILURE.json',{'exception':repr(error),'traceback':traceback.format_exc(),
                 'pass':False,'all_prefix_records_preserved':True})
        raise
