"""Seal integration-only source and pure fixed-first input; no model execution."""
import ast, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research/review_3080_20261010'; D=R/'delivery'
O=D/'actual_score_chain_preparation_root_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    actual=D/'actual_action_head_execution_root_v1'
    scope=read(actual/'ACTUAL_EXECUTION_SCOPE.json')
    if not scope['qualified'] or scope['production_records']!=60:
        raise ValueError('Initial scoring component prerequisite')
    previous=read(actual/'EXECUTION_REGISTRATION.json')
    observations=D/'family2_remaining_action_diagnosis_root_v1/OBSERVED_ACTION_RECORDS.json'
    if sha(observations)!=previous['source_inputs'][str(observations)]:
        raise ValueError('Original observed source identity')
    pure=read(actual/'downloaded/INPUTS.json')
    observed=read(observations)
    x=observed[0]; first=pure[0]; req=x['request']
    key=[x['instance_id'],x['step']]
    if first['key']!=key or first['data']!=req['data']:
        raise ValueError('Fixed original first request, not selected by outcome')
    before=[[i for i in route if i not in req['mask']] for route in req['initial']]
    if before!=first['before'] or sorted(req['mask'])!=first['remaining']:
        raise ValueError('Pure current skeleton/remaining identity')
    # partial is required to be exactly the current retained undirected edges.
    n=len(req['data']['demands']); adjacency=[[0.]*n for _ in range(n)]
    for route in before:
        path=[0]+route+[0]
        for a,b in zip(path,path[1:]):
            if a!=b:adjacency[a][b]=adjacency[b][a]=1.
    if adjacency!=x['partial']:
        raise ValueError('Partial context contains more than current skeleton')
    row=dict(key=key,data=first['data'],initial=req['initial'],mask=req['mask'],
             before=before,remaining=first['remaining'],partial=adjacency)
    if any(k in row for k in ('teacher_routes','oracle','target','cost')):
        raise ValueError('Teacher/target/cost exclusion')
    sources=['delivery_actual_score_chain_v1.py','run_delivery_actual_score_chain_v1.py',
             'delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py',
             'delivery_actual_action_head_v1.py','action_head_runtime_v1.py']
    for name in sources:ast.parse((ROOT/'tools'/name).read_text(encoding='utf-8'))
    O.mkdir(exist_ok=False)
    files={}
    for name in sources:
        shutil.copy2(ROOT/'tools'/name,O/name);files[name]=sha(O/name)
    shutil.copy2(actual/'features_objective.py',O/'features_objective.py')
    files['features_objective.py']=sha(O/'features_objective.py')
    create(O/'CHAIN_INPUT.json',row);files['CHAIN_INPUT.json']=sha(O/'CHAIN_INPUT.json')
    frozen=actual/'downloaded'
    artifacts={name:sha(frozen/name) for arm in ['pretrained_initial','random888_initial']
               for name in [arm+'_00.npz',arm+'_00.json',arm+'_PARAMETER_IDENTITY.json']}
    for name in ['EXECUTION_REGISTRATION.json','RESULT.json','INDEPENDENT_ACTION_HEAD_AUDIT.json',
                 'RUN_EXIT.json','AUDIT_RUN_EXIT.json']:
        artifacts[name]=sha(frozen/name)
    identity=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='ACTUAL_SCORE_CHAIN_SOURCE_PREPARATION_ONLY',
        execution_authorized=False,independent_full_chain_checker_sealed=False,package_files=files,
        input_digest=identity,key=key,selection_rule='First of the original30 observed records in immutable order, two initial arms; no teacher label/cost/result selection.',
        source_inputs={str(observations):sha(observations),str(frozen/'INPUTS.json'):sha(frozen/'INPUTS.json'),
                       str(actual/'ACTUAL_EXECUTION_SCOPE.json'):sha(actual/'ACTUAL_EXECUTION_SCOPE.json')},
        initial_component_registration_sha256=sha(actual/'EXECUTION_REGISTRATION.json'),
        initial_component_remote_root=r'C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\actual_action_head_execution_root_v1',
        initial_artifacts=artifacts,old_remote_root=previous['old_remote_root'],checkpoint=previous['checkpoint'],
        checkpoint_sha256=previous['checkpoint_sha256'],contract_hash=previous['contract_hash'],
        normalization_sources=previous['normalization_sources'],head_seed=20261012,random_seed=888,optimizer_steps=0,
        minimum_available_ram_gib=8,cpu_affinity=[15],gpu_used=False,
        proposed_stage_timeout_seconds=900,proposed_reap_seconds=10,
        proposed_workload=dict(chains=2,masked_customers=len(row['remaining']),
            encodes=2,new_decodes_maximum=2*(len(row['remaining'])-1),
            first_step='Reuse immutable original00 NPZ scores only with exact features/actions/context/weights identity; no repeated initial forward.',
            encoder_cache='One new encode per arm inside engineering timer. Fixed10Dfeatures throughout. Fresh decode/embed/final_norm for each changed temporary context; cache equivalence not executed or qualified.',
            action_order='All remaining customerIDs ascending; all original slots/positions ascending. Single full ordered batch for current context.',
            choice='Eligible joint retained-edge/route_ok; highest actual action score, original normalized distance delta secondary, slot/position ties. Regret primary score gap, singleton first, customerID tie.',
            early_stop='No feasible current action stops; preserve all rows/partial chain and fall back to unchanged fullC0 incumbent. No resampling/retry or target fixture.',
            final_C0='Original shared implementation incumbent once/arm, complete candidate once if covered; keep candidate only hard and strict lower contract cost. All candidate reports preserved regardless selection; no scientific gain claim.'),
        exact_checks_required=['Independent source/parameters/headseed/features/current anchors/ordered actions',
            'Fresh fullencode plus changed-context decode/latent exact replay from original checkpoint and seededrandom; no duplicate old firstforward',
            'Independent local route/retained union/all customer regrets/competing transitions and coverage',
            'Original fullC0 incumbent/candidate exact replay and fallback decision, shared original physics limitation explicit',
            'Production/audit full loaded file/native begin/end and module map differences',
            'Actual bounded supervision, new exclusive transport, download SHA/actual process exit'],
        numpy_head_max_abs_error=previous['numpy_head_max_abs_error'],
        model_executed=False,chain_executed=False,C0_executed=False,full_C0_qualified=False,
        cache_equivalence_qualified=False,learned_output_qualified=False,formal_controls_qualified=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,
        outstanding=['Independent complete-chain audit source, exact supervision and dispatch/download package before execution registration.',
            'Resource/CPU15/actual owned tree fresh check before any execution; no GPU/team process termination.',
            'Whole-request10s abortability, complete controls/equal-time iteration not implemented by this900s stage.',
            'Formal trained same-initializationCE/task-trainedrandom, independent strong dynamic admission and all science barriers remain.'])
    create(O/'INTEGRATION_PLAN_PREPARATION.json',plan)
    create(O/'PREPARATION_IDENTITY.json',dict(plan_sha256=sha(O/'INTEGRATION_PLAN_PREPARATION.json'),
        package_files=files,initial_artifacts=artifacts,original_executed_sources_unchanged=True,
        scope='AST/source/byte preparation only; no imports of model, forward, route_ok, solver or C0 execution.'))
    print(json.dumps(dict(directory=str(O),key=key,masked_customers=len(row['remaining']),
        package_files=len(files),plan_sha256=sha(O/'INTEGRATION_PLAN_PREPARATION.json'),execution_authorized=False)))

if __name__=='__main__':main()
