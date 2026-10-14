"""Narrow initialization-label diagnosis; no admission, SAA or solver rerun."""
import ast,copy,inspect,json
from pathlib import Path
import qualify_pickup_joint_policy_transaction_root_v3 as old
from run_exp_rule_control import UncondDynamicReplanner
from run_exp_energy_c0 import C0ReserveReplanner
import psutil,numpy as np
P=old.P;OUT=P/'live_event_prefix_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=decode(read(OUT/'RESULT.json'));audit=read(OUT/'INDEPENDENT_LIVE_PREFIX.json');reg=read(OUT/'REGISTRATION.json');checks=[];method_evidence=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    expected_failures=['request:'+str(i)+':original_admission:mode' for i in range(14)]+['prefix_final:state:mode']
    ck('original_999_15_fail_retained',audit['checks_count']==999 and audit['failed']==15 and audit['failure_names']==expected_failures)
    ck('original_984_other_checks_pass',sum(c['pass_'] for c in audit['checks'])==984)
    ck('producer_bound_to_original_audit',audit['producer_sha256']==old.sha(OUT/'RESULT.json') and audit['registration_sha256']==old.sha(OUT/'REGISTRATION.json'))
    for p,h in reg['sources'].items():ck('unchanged_selected_identity:'+p,old.sha(p)==h)
    packet=raw['records'][0]['packet'];initial=packet['snapshot']['replanner_state']
    args=dict(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],contract=packet['contract'],cooling_share=0.)
    parent=C0ReserveReplanner(**args);parent._reset_if_new(0)
    reference=UncondDynamicReplanner(**args,hist=[],k=10);reference._reset_if_new(0)
    ck('actual_parent_initial_label_myopic',parent.mode==initial['mode']=='myopic')
    ck('auditor_default_reference_label_knn',reference.mode=='knn')
    original_handler=reference.on_reveal.__func__;original_lookahead=reference._lookahead_lambda.__func__
    reference.restore_state(copy.deepcopy(initial))
    ck('original_state_restoration_keeps_registered_label',reference.mode==initial['mode']=='myopic')
    ck('restoration_does_not_switch_actual_uncond_handlers',reference.on_reveal.__func__ is original_handler and reference._lookahead_lambda.__func__ is original_lookahead)
    # Trace the actual self-method call graph statically. The generic parent
    # on_reveal reads mode, but is overridden by the unconditional admission arm.
    import textwrap
    todo=['on_reveal','plan'];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name);method=getattr(UncondDynamicReplanner,name);source=inspect.getsource(method);tree=ast.parse(textwrap.dedent(source))
        mode_reads=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and n.attr=='mode']
        callees=sorted(set(n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='self' and callable(getattr(UncondDynamicReplanner,n.func.attr,None))))
        ck('actual_handler_graph_no_mode_read:'+name,not mode_reads)
        method_evidence.append(dict(method=name,owner=method.__qualname__,source_sha256=old.sha(inspect.getsourcefile(method)),mode_reads=mode_reads,self_callees=callees));todo.extend(callees)
    parent_tree=ast.parse(textwrap.dedent(inspect.getsource(C0ReserveReplanner.on_reveal)))
    ck('generic_parent_mode_branch_is_not_actual_admission',any(isinstance(n,ast.Attribute) and n.attr=='mode' for n in ast.walk(parent_tree)) and UncondDynamicReplanner.on_reveal is not C0ReserveReplanner.on_reveal)
    auditor_tree=ast.parse((old.ROOT/'tools/audit_pickup_live_prefix_v1.py').read_text(encoding='utf-8'))
    ref_fn=next(n for n in ast.walk(auditor_tree) if isinstance(n,ast.FunctionDef) and n.name=='reference')
    ck('original_auditor_omitted_initial_restore',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='restore_state' for n in ast.walk(ref_fn)))
    for i,r in enumerate(raw['records']):
        n=str(i);mode=r['before']['mode'];response=r['response']
        ck(n+':production_initial_label_preserved',mode==initial['mode']==r['after']['mode'])
        ck(n+':actual_worker_restore_label',response['base_response']['state']['mode']==mode==response['compiled_state'].mode)
        ck(n+':guard_did_not_allow_mode_change',r['installed'] and response['scoring_packet']['snapshot']['replanner_state']['mode']==mode)
    ck('prefix_final_label_from_same_initial_state',raw['final']['state']['mode']==initial['mode'])
    failed=[c['name'] for c in checks if not c['pass_']]
    result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,
                original_audit_sha256=old.sha(OUT/'INDEPENDENT_LIVE_PREFIX.json'),producer_sha256=old.sha(OUT/'RESULT.json'),
                original_checks=999,original_failures=15,original_successful_checks=984,methods=method_evidence,
                scope='Only close reference constructor label omission by original constructor/restoration witnesses, actual handler call graph and serialized policy identity. No admission/scenario/physical rerun, tolerance or production change.',
                initialization_label_gap_closed=not failed,original_failure_report_unchanged=True,
                full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failures=failed,handler_methods=len(seen))));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
