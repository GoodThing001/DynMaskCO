"""Independent saved-byte arithmetic diagnosis; no JAX/head evaluation."""
import json,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save,closure,delta
from startup_native_identity_v1 import prime_native_identity
def main():
 import numpy as np,psutil
 native=prime_native_identity();psutil.Process().cpu_affinity([15])
 reg=json.loads((B/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
 extras=[B/n for n in reg['package_files']]+[B/n for n in ('EXECUTION_REGISTRATION.json','INTERMEDIATES.npz','HEAD_PARAMETERS.npz','RESULT.json','SOURCE_START.json','SOURCE_END.json')]
 # Preinitialize every runtime dependency before collecting beginning identity.
 closure(extras,[]);begin=closure(extras,[]);save(B,'AUDIT_SOURCE_START.json',begin)
 checks=[]
 def check(n,ok,detail=None):checks.append(dict(name=n,ok=bool(ok),detail=detail))
 result=json.loads((B/'RESULT.json').read_text(encoding='utf-8'));source=json.loads((B/'SOURCE_START.json').read_text(encoding='utf-8'));end=json.loads((B/'SOURCE_END.json').read_text(encoding='utf-8'))
 for p,h in source['files'].items():check('production_source:'+p,end['files'].get(p)==h)
 check('production_no_unsealed_loaded_files',not end['new_files']);check('production_start_map_all_hashed',all(p in source['files'] for p in source['module_files'].values()))
 entry=json.loads((B/'FIXED_INPUT.json').read_text(encoding='utf-8'));check('fixed_input_sha',sha(entry['path'])==entry['sha256'])
 with np.load(entry['path'],allow_pickle=False) as z:q={n:z[n].copy() for n in ('latent','actions','route_weights','prefix_weights','suffix_weights','slot_position_length','scores')}
 with np.load(B/'INTERMEDIATES.npz',allow_pickle=False) as z:parts={n:z[n].copy() for n in z.files}
 with np.load(B/'HEAD_PARAMETERS.npz',allow_pickle=False) as z:params={n:z[n].copy() for n in z.files}
 h=q['latent'][0];a=q['actions']
 for n,col in [('customer',0),('predecessor',3),('successor',4)]:check('gather_identity:'+n,np.array_equal(parts['eager_'+n],h[a[:,col]]) and np.array_equal(parts['compiled_'+n],h[a[:,col]]))
 for n in ('eager','compiled'):
  expected=np.concatenate([parts[n+'_'+k] for k in ('customer','predecessor','successor','route_mean','prefix_mean','suffix_mean')]+[q['slot_position_length']],axis=-1)
  check('concat_identity:'+n,np.array_equal(expected,parts[n+'_concat']))
 for row in result['intermediate_rows']:
  e=parts['eager_'+row['name']];c=parts['compiled_'+row['name']];neq=int(np.count_nonzero(e.view(np.uint32)!=c.view(np.uint32)))
  check('reported_exact:'+row['name'],neq==row['bitwise_mismatch_count'] and (neq==0)==row['exact_equal'])
  check('reported_error:'+row['name'],float(np.max(np.abs(e.astype(np.float64)-c.astype(np.float64))))==row['max_abs_error'])
 check('original_final_binding',np.array_equal(parts['eager_output'],q['scores'])==result['eager_score_exact_saved_original'])
 check('compiled_instrumentation_binding',np.array_equal(parts['compiled_output'],parts['saved_old_compiled'])==result['instrumented_score_exact_saved_old_compiled'])
 check('first_divergence_order',next((r['name'] for r in result['intermediate_rows'] if not r['exact_equal']),None)==result['first_divergent_intermediate'])
 # BLAS differences are descriptive diagnostics only, never exact eligibility.
 diagnostics={}
 for side in ('eager','compiled'):
  for output,expected in [('linear1_dot',parts[side+'_concat']@params['k1']),('linear1_bias',parts[side+'_linear1_dot']+params['b1']),('tanh',np.tanh(parts[side+'_linear1_bias'])),('linear2_dot',parts[side+'_tanh']@params['k2']),('output',(parts[side+'_linear2_dot']+params['b2'])[:,0])]:
   diagnostics[side+'_'+output]=float(np.max(np.abs(expected.astype(np.float64)-parts[side+'_'+output].astype(np.float64))))
 finish=closure(extras,[]);d=delta(begin,finish);save(B,'AUDIT_SOURCE_END.json',dict(**finish,**d))
 check('independent_loaded_closure',not d['changed'] and not d['new_files']);check('independent_start_map_hashed',all(p in begin['files'] for p in begin['module_files'].values()))
 output=dict(check_count=len(checks),failure_count=sum(not x['ok'] for x in checks),checks=checks,numpy_operator_diagnostics=diagnostics,first_divergent_intermediate=result['first_divergent_intermediate'],old_strict34_result_unchanged=True,new_gate_relaxed=False,old_model_QA_repeated=False,full_head_science_qualified=False,stable_online_gain=False)
 save(B,'INDEPENDENT_ARITHMETIC_AUDIT.json',output);return int(output['failure_count']>0)
if __name__=='__main__':
 code=1
 try:code=main()
 except BaseException:save(B,'AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()))
 raise SystemExit(code)
