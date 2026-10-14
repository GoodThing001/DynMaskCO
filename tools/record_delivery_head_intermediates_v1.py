"""Saved-only qualification boundaries, no arithmetic workload or old QA replay."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
O=R/'research/review_3080_20261010/delivery/head_intermediate_arithmetic_root_v1'
D=O/'downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((D/n).read_text(encoding='utf-8'))
def main():
 manifest=json.loads((O/'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8'))
 if not all((D/x['name']).stat().st_size==x['size'] and sha(D/x['name'])==x['sha256'] for x in manifest):raise ValueError('Archive identity')
 result=load('RESULT.json');audit=load('INDEPENDENT_ARITHMETIC_AUDIT.json')
 with np.load(D/'INTERMEDIATES.npz',allow_pickle=False) as z:
  original=z['saved_original'];old_c=z['saved_old_compiled'];instrumented=z['compiled_output']
  old_difference=dict(bitwise_mismatch_count=int(np.count_nonzero(old_c.view(np.uint32)!=instrumented.view(np.uint32))),max_abs_error=float(np.max(np.abs(old_c.astype(np.float64)-instrumented.astype(np.float64)))))
 start=load('SOURCE_START.json');end=load('SOURCE_END.json');astart=load('AUDIT_SOURCE_START.json');aend=load('AUDIT_SOURCE_END.json')
 obj=dict(registration_sha256=sha(O/'EXECUTION_REGISTRATION.json'),status='NEW_INTERMEDIATE_OBSERVATION_VALID_OLD_COMPILED_SCHEDULE_NOT_REPRODUCED_NO_INSTALL',
  component_check_count=audit['check_count'],component_failure_count=audit['failure_count'],new_intermediate_count=len(result['intermediate_rows']),all_instrumented_vs_eager_intermediates_exact=all(r['exact_equal'] for r in result['intermediate_rows']),
  eager_final_matches_saved_original=result['eager_score_exact_saved_original'],instrumented_final_matches_old_saved_compiled=result['instrumented_score_exact_saved_old_compiled'],instrumented_vs_old_saved_compiled=old_difference,
  first_divergent_new_intermediate=result['first_divergent_intermediate'],old_graph_first_divergence_localized=False,
  insight='For the prospectively fixed first input, returning all intermediate arrays changed the compiled final value relative to the previously saved non-returning compiled graph. The new observed tuple graph is bitwise equal to eager. Observation affects compiled scheduling; this does not prove a unique compiler pass or the cause of all old34 mismatches.',
  nnx_internal_dot_observable_qualified=False,operator_observable_note='linear1_dot and linear2_dot are separately evaluated explicit matmul expressions, not taps inside NNX Linear dot_general. linear1_bias uses actual head.input(x) and output uses actual head.output(t). No NNX internal primitive trace or compiler-pass IR was archived.',
  producer=dict(files_start=len(start['files']),files_end=len(end['files']),modulemap_start=len(start['module_files']),modulemap_end=len(end['module_files']),changed=end['changed'],new_files=end['new_files'],maps_equal=start['module_files']==end['module_files']),
  independent=dict(files_start=len(astart['files']),files_end=len(aend['files']),modulemap_start=len(astart['module_files']),modulemap_end=len(aend['module_files']),changed=aend['changed'],new_files=aend['new_files'],maps_equal=astart['module_files']==aend['module_files']),
  exits={n:load(n) for n in ('RUN_EXIT.json','AUDIT_RUN_EXIT.json','CHAIN_EXIT.json')},
  archive=dict(files=len(manifest),bytes=sum(x['size'] for x in manifest),all_sha_match=True,chunk_size=262144),
  original_strict0_of34_failure_unchanged=True,candidate_installed=False,full_backbone_executions=0,C0_solver_calls=0,optimizer_updates=0,new_scientific_family_started=False,full_request_budget_qualified=False,formal_controls_qualified=False,full_scientific_loaded_closure_qualified=False,stable_online_gain=False,
  next='Do not repeat this fixed intermediate observation or old34 compiled QA. A semantics-preserving scheduling design must separately specify and validate arithmetic observables/outputs without assuming debug tuple return leaves original graph unchanged, then actual whole8+2 request gates. Formal trained CE/taskrandom/strongdynamic/equal-time science remain gated.',
  source_reference=dict(title='JAX FAQ: jit changes the exact numerics of outputs',url='https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs',kind='Official technical documentation; root verified; contextual possibility, not measured pass attribution'))
 with (O/'STAGE_ADJUDICATION.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
 print(json.dumps({k:obj[k] for k in ('status','component_check_count','component_failure_count','new_intermediate_count','instrumented_vs_old_saved_compiled','producer','independent','archive')}))
if __name__=='__main__':main()
