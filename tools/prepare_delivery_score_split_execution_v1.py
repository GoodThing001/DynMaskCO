"""Register a new bounded synchronized decode/head/host timing diagnostic."""
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
PRE = TREE / 'delivery/phase_timed_model_execution_root_v1'
OUT = TREE / 'delivery/score_split_timed_execution_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p, value):
    if isinstance(value, str):
        if p.suffix == '.py':
            ast.parse(value, filename=str(p))
        with p.open('x', encoding='utf-8', newline='\n') as f:
            f.write(value)
    else:
        with p.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    old = json.loads((PRE / 'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    prior = json.loads((PRE / 'STAGE_ADJUDICATION.json').read_text(encoding='utf-8'))
    assert prior['raw_independent_failures'] == 0 and not prior['full_request_budget_qualified']
    for n, h in old['package_files'].items():
        assert sha(PRE / n) == h, n
    OUT.mkdir(exist_ok=False)
    renames = {
        'delivery_phase_actual_engine_v1.py': 'delivery_score_split_engine_v1.py',
        'delivery_phase_shared_worker_v2.py': 'delivery_score_split_worker_v1.py',
        'delivery_phase_shared_parent_v3.py': 'delivery_score_split_parent_v1.py',
        'independent_phase_trace_parser_v1.py': 'independent_score_split_trace_parser_v1.py',
        'run_delivery_phase_timed_request_v1.py': 'run_delivery_score_split_v1.py',
        'audit_delivery_phase_timed_request_v1.py': 'audit_delivery_score_split_v1.py',
        'supervise_delivery_phase_timed_request_v1.py': 'supervise_delivery_score_split_v1.py',
        'phase_timed_request_remote_launcher_v1.py': 'score_split_remote_launcher_v1.py',
        'phase_timed_request_remote_status_v1.py': 'score_split_remote_status_v1.py',
    }
    def rename_refs(source):
        for a, b in renames.items():
            source = source.replace(a, b).replace(a[:-3], b[:-3])
        return source
    block = '''            def score_synchronized():
                latent=self.model.decode(encoded,jnp.asarray(binding['timestep']),jnp.asarray(binding['context']),attn_options={},target='embed')
                if self.model.apply_final_norm:latent=self.model.final_norm(latent)
                scores=self.head(latent,binding)
                jax.block_until_ready((scores,latent))
                return np.asarray(scores),np.asarray(latent)
'''
    new = '''            def score_synchronized():
                # Additional diagnostic barriers change synchronization scheduling,
                # not the original arithmetic expressions or input/action ordering.
                def decode_sync():
                    latent=self.model.decode(encoded,jnp.asarray(binding['timestep']),jnp.asarray(binding['context']),attn_options={},target='embed')
                    jax.block_until_ready(latent)
                    return latent
                latent=m(20,step,decode_sync)
                if self.model.apply_final_norm:
                    def norm_sync():
                        normalized=self.model.final_norm(latent)
                        jax.block_until_ready(normalized)
                        return normalized
                    latent=m(21,step,norm_sync)
                def head_sync():
                    scores=self.head(latent,binding)
                    jax.block_until_ready(scores)
                    return scores
                scores=m(22,step,head_sync)
                return m(23,step,lambda:(np.asarray(scores),np.asarray(latent)))
'''
    for n in old['package_files']:
        if n == 'PHASE_DESIGN.json':
            continue
        target = renames.get(n, n)
        if n.endswith('.py'):
            source = (PRE / n).read_text(encoding='utf-8')
            if n == 'delivery_phase_actual_engine_v1.py':
                assert source.count(block) == 1
                source = source.replace(block, new)
            if n == 'independent_phase_trace_parser_v1.py':
                assert source.count('not 1<=phase<=19') == 1
                source = source.replace('not 1<=phase<=19', 'not 1<=phase<=23')
            if n == 'audit_delivery_phase_timed_request_v1.py':
                anchor = "                ck(tag+':trace_union_bounded:'+role,Fraction(*parsed['exact_operation_union_seconds'])<=Fraction(*parsed['exact_outer_seconds']))"
                extra = '''                if role=='WORKER':
                    records=parsed['records']
                    outers={v['step']:v for v in records if v['phase']==7}
                    for sub in [v for v in records if 20<=v['phase']<=23]:
                        parent=outers.get(sub['step'])
                        right=float.fromhex(parent['operation_end_hex']) if parent and parent['status'] else e
                        left=float.fromhex(parent['began_hex']) if parent else None
                        ck(tag+':nested_score_phase:%d:%d'%(sub['phase'],sub['step']),parent is not None and left<=float.fromhex(sub['began_hex'])<=right and (sub['status']==0 or float.fromhex(sub['operation_end_hex'])<=right))
                    for parent in [v for v in records if v['phase']==7 and v['status']==1]:
                        children={v['phase']:v for v in records if v['step']==parent['step'] and 20<=v['phase']<=23}
                        ck(tag+':four_completed_score_subphases:'+str(parent['step']),set(children)=={20,21,22,23} and all(v['status']==1 for v in children.values()))
'''
                assert source.count(anchor) == 1
                source = source.replace(anchor, extra + anchor)
            put(OUT / target, rename_refs(source))
        else:
            shutil.copy2(PRE / n, OUT / target)
    design = json.loads((PRE / 'PHASE_DESIGN.json').read_text(encoding='utf-8'))
    design.update(new_diagnostic_phases={
        '20': 'Original decode(embed) plus actual completion barrier',
        '21': 'Original final_norm plus actual completion barrier',
        '22': 'Original action head plus actual completion barrier',
        '23': 'Both original NumPy host conversions',
    }, additional_barriers=True, outer_phase7_includes_children=True,
        no_counterfactual_old_timing_equivalence=True, execution_qualification_external=True)
    put(OUT / 'PHASE_DESIGN.json', design)
    tp = ROOT / 'tools/transport_delivery_score_split_v1.py'
    transport = (ROOT / 'tools/transport_delivery_phase_timed_request_v1.py').read_text(encoding='utf-8')
    transport = transport.replace('phase_timed_model_execution_root_v1', 'score_split_timed_execution_root_v1')
    put(tp, rename_refs(transport))
    inspector = (ROOT / 'tools/inspect_delivery_phase_timed_request_v1.py').read_text(encoding='utf-8')
    inspector = inspector.replace('transport_delivery_phase_timed_request_v1', 'transport_delivery_score_split_v1')
    put(ROOT / 'tools/inspect_delivery_score_split_v1.py', rename_refs(inspector))
    names = [renames.get(n, n) for n in old['package_files']]
    reg = {k: v for k, v in old.items() if k not in ('package_files', 'transport_tool_sha256', 'unchanged_original_engine_sha256')}
    reg.update(created_utc=datetime.now(timezone.utc).isoformat(),
        status='PROSPECTIVE_NEW_SCORE_SUBPHASE_DIAGNOSTIC_TWO_INITIAL_ARMS',
        prior_registration_sha256=sha(PRE / 'EXECUTION_REGISTRATION.json'),
        prior_adjudication_sha256=sha(PRE / 'STAGE_ADJUDICATION.json'),
        preparation_tool_sha256=sha(Path(__file__)),
        package_files={n: sha(OUT / n) for n in names}, transport_tool_sha256=sha(tp),
        stage_execution=['run_delivery_score_split_v1.py', 'audit_delivery_score_split_v1.py'],
        remote_directory='research/review_3080_20261010/score_split_timed_execution_root_v1',
        actual_phase_design_sha256=sha(OUT / 'PHASE_DESIGN.json'),
        selection_rule='Exactly one new split-timing request per fixed initial arm; same pure current key12,3, no faults/retries/sample warmup. Not a repeat of old QA: four newly synchronized score subphase observables, independently checked only for the new evidence.',
        additional_synchronization_barriers=True, arithmetic_expression_order_unchanged=True,
        original_phase7_timing_not_counterfactual_equivalent=True,
        phase_range=[1,23], no_whole_request_qualification_from_raw_trace=True,
        inference_goal='Measure decode/final_norm/head/host segments under explicit barriers. Diagnose observed scheduling cost; no utility, training, architecture selection or speed improvement assertion.',
    )
    audit_imports = []
    for node in ast.walk(ast.parse((OUT / 'audit_delivery_score_split_v1.py').read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import): audit_imports.extend(x.name for x in node.names)
        elif isinstance(node, ast.ImportFrom): audit_imports.append(node.module or '')
    assert not any(x.startswith(('delivery_score_split_engine', 'delivery_actual_action_head', 'delivery_actual_score_chain', 'delivery_feasible_action_options', 'bounded_phase_trace')) for x in audit_imports)
    put(OUT / 'EXECUTION_REGISTRATION.json', reg)
    put(OUT / 'LOCAL_EXECUTION_BINDING.json', {'registration_sha256':sha(OUT/'EXECUTION_REGISTRATION.json'),'transport_tool_sha256':sha(tp),'package_files':reg['package_files'],'shared_MD_upload_authorized':False,'model_executed_at_registration':False})
    put(OUT / 'SOURCE_REVIEW_BEFORE_EXECUTION.json', {'independent_imports':audit_imports,'additional_barriers_registered':True,'model_arithmetic_and_parameters_unchanged':True,'source_only':True,'numeric_contract_unchanged':True,'no_training_or_scientific_family3':True})
    print(json.dumps({'root':str(OUT),'registration_sha256':sha(OUT/'EXECUTION_REGISTRATION.json'),'requests':2,'model_executed':False}))


if __name__ == '__main__':
    main()
