"""Saved-result scope and identity audit; no JAX or prior 34 inputs."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
reg=json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8'))
result=json.loads((HERE/'RESULT.json').read_text(encoding='utf-8'))
checks=[]
def ck(n,v):checks.append({'name':n,'pass':bool(v)})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ck('source_registration_identity',result['source_sha256']['REGISTRATION.json']==sha(HERE/'REGISTRATION.json'))
ck('source_runner_identity',result['source_sha256']['run.py']==sha(HERE/'run.py'))
ck('seven_prespecified_stages',[r['name'] for r in result['rows']]==['route','prefix','suffix','concat','linear1_bias','tanh','linear2_bias'])
ck('all_finite_nonnegative_differences',all(0<=r['max_abs_difference']<float('inf') for r in result['rows']))
ck('reported_first_difference_consistent',result['first_differing_stage']==next((r['name'] for r in result['rows'] if not r['bitwise_equal']),None))
ck('counts_consistent',all((r['different_float32_elements']==0)==r['bitwise_equal'] for r in result['rows']))
ck('fixed_actions',reg['fixed_shapes']['actions']==32)
ck('original_strict_fail_kept',result['original_compiled_head_strict_gate_pass'] is False)
ck('no_original_rerun',result['original_34_cases_rerun'] is False)
ck('no_backbone_or_full_request',result['full_backbone_loaded'] is False and result['full_request_qualified'] is False)
out={'stage':'saved_small_synthetic_reduction_audit','checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),
     'checks':checks,'scope':'Source identity and saved consistency only; does not independently rerun JAX arithmetic.',
     'original_34_gate_still_failed':True,'stable_online_gain':False}
with (HERE/'SAVED_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'checks_count':out['checks_count'],'failed':out['failed']}))
raise SystemExit(int(out['failed']!=0))
