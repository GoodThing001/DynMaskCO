"""Reuse independent route arithmetic, then verify fault and full CPU scope."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
def main(version):
    if version not in ('v2','v3'):raise ValueError('Exact isolated version required')
    out=D/('remote_final_model_budget_bundle_root_'+version)/'downloaded'
    source=D/('final_model_budget_root_'+version)
    text=(ROOT/'tools/audit_actual_model_budget_review.py').read_text()
    text=text.replace("B=R/'remote_model_budget_bundle_root_v1/downloaded'","B=R/'remote_final_model_budget_bundle_root_"+version+"/downloaded'")
    text=text.replace("R/'model_budget_integration_root_v1/BUDGET_ENGINEERING_REGISTRATION.json'","R/'final_model_budget_root_"+version+"/BUDGET_ENGINEERING_REGISTRATION.json'")
    text=text.replace("=='3ba23fe74d0d63dd404281db7a6465818f516fd24b7092d476a0c54c0509428f'",'')
    text=text.replace("R/'remote_finite_training_bundle_v1/downloaded/INDEPENDENT_ACTUAL_SAVED_WEIGHT_SHA.json'",
        "R/'independent_training_tensors_root_v2/INDEPENDENT_WHOLE_SAVED_WEIGHT_BYTES.json'")
    text=text.replace("reg['engineering_weights']","reg['fixed_final_weights']")
    text=text.replace("weights[arm+'_FIXED_FINAL.pkl']","weights[arm]")
    text=text.replace("['full_remote_bytes_read']","['whole_remote_bytes_read']")
    text=text.replace("'No timeout-injection regression in this exact seven-arm wrapper',",'')
    text=text.replace("'Whole weight hashes independently read; no independent per-call saved tensor reload'",
        "'Whole fixed128 weights independently read and separately tensor-reloaded; no per-call gradient reconstruction'")
    marker="    ck('no_online_or_research_upgrade'"
    idx=text.index(marker)
    extra='''    for arm in reg['arms']:
        fault=load(B/(arm+'_TIMEOUT_INJECTION.json'))
        ck(arm+':deliberate_worker_timeout_and_reap',fault['timeout'] is True and fault['worker_reaped'] is True and fault['own_incumbent_fallback'] is True and fault['engineering_only'] is True)
        close(arm+':fault_exact10ms_deadline',fault['deadline']-fault['started'],.01)
        ck(arm+':fault_no_late_submitted_result',fault['observed']>=fault['deadline'] and fault['fault']=='one second worker delay with10ms all-in request deadline')
        data,initial=generate(202610110071,fault['fixture_id'])
        ck(arm+':fault_exact_original_incumbent',fault['fallback_routes']==initial)
        fall=evaluate(data,fault['fallback_routes'],c)
        ck(arm+':fault_complete_hard_fallback',fall['hard'] is True and fall['complete'] is True)
'''
    text=text[:idx]+extra+text[idx:]
    text=text.replace("'formal_training_authorized_by_this_audit':False,","'formal_training_authorized_by_this_audit':False,'deliberate_seven_arm_timeout_regression':True,")
    ast.parse(text)
    ns={'__name__':'independent_final_budget_replay','__file__':str(Path(__file__).resolve())}
    exec(compile(text,str(Path(__file__).resolve()),'exec'),ns)
    code=ns['main']()
    report=json.loads((out/'INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json').read_text())
    actual=json.loads((out/'RESULT.json').read_text())
    cpu=actual.get('actual_charged_parent_cpu_affinity')==[15] and all(r['ready'].get('actual_cpu_affinity')==[15] for r in actual['records'])
    result={'scope':'full charged resource and exact final model engineering acceptance',
        'route_budget_fault_audit_failed':report['failed'],'candidate_routes_replayed':report['candidate_routes_replayed'],
        'final_routes_replayed':report['final_routes_replayed'],'seven_fault_fallbacks_complete_and_reaped':code==0,
        'actual_parent_and_workers_CPU15_qualified':cpu,
        'resource_evidence_gap':None if cpu else 'v2 parent actual0..15 and no exact parent affinity in final record; no full deployment qualification',
        'independent_acceptance':code==0 and cpu,
        'fixed_final_weights_bound':True,'new_development_may_be_registered_and_implemented':code==0 and cpu,
        'fresh_development_already_run':False,'confirmation_may_start':False,'dynamic_online_gain':False}
    with (out/'ACCEPTED_FINAL_MODEL_BUDGET.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
    # v2 is explicitly diagnostic evidence of the gap; acceptance remains false.
    return int(code!=0 or (version=='v3' and not cpu))
if __name__=='__main__':sys.exit(main(sys.argv[1]))
