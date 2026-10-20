"""Read-only gate audit; standard library only; imports no implementation code.

No search, geometry, model, runtime probe or source-day deserialization occurs.
The output is a NEW receipt, never an amendment of earlier regression results.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    reg=json.loads((HERE/'registration.json').read_text(encoding='utf-8'))
    case=HERE/'synthetic_artifacts/regression_r2'
    regression=json.loads((case/'regression.json').read_text(encoding='utf-8'))
    checks=[]
    def add(name,ok,detail):
        checks.append(dict(name=name,ok=bool(ok),detail=detail))
    delivery=reg['seven_elements']['identity_and_delivery']
    for group in ('contracts','inputs','existing_source_hashes'):
        mismatches=[p for p,h in delivery[group].items() if sha(WORKSPACE/p)!=h]
        add(group+'_unchanged',not mismatches,dict(n=len(delivery[group]),mismatches=mismatches))
    add('formal_authorization_absent',not reg['authorization']['formal_19_unit_evaluation_authorized'],
        'The explicit human confirmation authorizes phase1 only.')
    expected=set(delivery['acceptance_roster'])-{'source_and_result_independent_recheck'}
    observed={r['name'] for r in regression['checks']}
    add('regression_roster',expected==observed and len(regression['checks'])==29,
        dict(expected=len(expected),observed=len(observed)))
    add('component_regression_receipt',all(r['ok'] for r in regression['checks']),
        '29/29 recorded component checks; does not establish all contract gates.')
    memory=json.loads((case/'memory/result.json').read_text())
    limit=memory['native']['job_limit'];peak=memory['native']['peak_job_commit']
    add('strict_native_memory_ceiling',peak<=limit,
        dict(limit_bytes=limit,native_peak_bytes=peak,excess_bytes=max(0,peak-limit),
             reason='Observed native peak exceeds its configured ceiling. Failure classification is not proof of prevention.'))
    add('memory_failure_not_hidden',memory['failure_code']=='MEMORY_LIMIT_EXCEEDED',
        'Fault receipt retained; not promoted to a normal successful run.')
    mapped=json.loads((case/'mapped/result.json').read_text())
    add('rss_distinct_from_private_commit',mapped['peak_rss_sum']<=mapped['native']['job_limit']
        and mapped['native']['supervisor_hard_ws']['flags']&4,
        dict(rss_sum=mapped['peak_rss_sum'],private_sum=mapped['peak_private_sum'],
             native=mapped['native']))
    for folder,code,cap in [('unit_timeout','UNIT_TIMEOUT',2.),('package_timeout','PACKAGE_TIMEOUT',2.)]:
        d=json.loads((case/folder/'result.json').read_text())
        slots=d['slots']['slots']
        add(folder+'_failure_ledger',d['failure_code']==code and d['elapsed_s']<=cap
            and d['worker_exited'] and len(slots)==19,
            dict(elapsed_s=d['elapsed_s'],slots=len(slots),remaining=[x['failure_code'] for x in slots[1:]]))
    for folder in ('kill_worker','kill_supervisor'):
        d=json.loads((case/folder/'postmortem.json').read_text())
        add(folder+'_durable_postmortem',len(d['slots'])==19 and not d['computation_resumed'],
            dict(first=d['slots'][0],slot_count=len(d['slots'])))
    tree=ast.parse((HERE/'supervisor.py').read_text(encoding='utf-8'))
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    add('integrated_two_phase_runner_qualified',False,
        dict(status='NOT_COMPLETED_AFTER_RESOURCE_STOP',
             reason='Native synthetic probes and the callable closure are separate; no complete supervised two-phase 19-slot pipeline is qualified.',
             supervisor_calls_closure='enumerate_closure' in calls))
    add('new_source_bound_before_regression',False,
        'The final component receipt contains frozen original-definition hashes, but not a pre-run digest of every new module. No retroactive T2 qualification.')
    result=dict(schema='hgeom_independent_readonly_audit_v1',imports_implementation=False,
                source_sha256=sha(Path(__file__)),registration_sha256=sha(HERE/'registration.json'),
                regression_sha256=sha(case/'regression.json'),checks=checks,
                n_checks=len(checks),n_ok=sum(c['ok'] for c in checks),
                status='BLOCKED',formal_evaluation_allowed=False,
                failed=[c['name'] for c in checks if not c['ok']])
    return result


if __name__=='__main__':
    target=HERE/'independent_audit.json'
    if target.exists():raise SystemExit('Refuse overwrite independent audit')
    result=audit()
    target.write_bytes((json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    print(json.dumps(dict(status=result['status'],n_checks=result['n_checks'],
                          n_ok=result['n_ok'],failed=result['failed']),ensure_ascii=False))
