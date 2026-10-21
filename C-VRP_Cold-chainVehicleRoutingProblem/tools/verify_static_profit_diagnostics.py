"""SHA, independent feasibility/money, pairing and strict-anytime audit for dev11."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

EXT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT/'scripts/static_profit'))
from profit_core import ProfitTask,verify_research_freeze


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x):return statistics.mean(x)


def verify_routes(case,routes,score,role='validation'):
    task=ProfitTask(case,fold=0,role=role)
    actual=task.evaluate(routes,audit=True)
    for k in ('net_benefit_U','quality_loss_value_U','distance_km','refrigeration_cost_U','minimum_quality','used_vehicles'):
        assert abs(actual[k]-score[k])<1e-6,(case,k)
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',default='dev11')
    args=p.parse_args()
    assert args.label=='dev11'
    root=EXT/'results/static_delivery/algorithm_dev_20261021/remote'/args.label
    receipt=read(root/'download_verification.json');manifest=read(root/'output_manifest.json')
    assert receipt['exit_code']==0
    for name in receipt['downloaded_verified']:assert sha(root/name)==manifest[name]
    directory=root/'outputs/diagnostics';identity=read(directory/'identity.json')
    plan=read(EXT/'configs/static_delivery/diagnostic_prereg_v1.json')
    assert identity['prereg_sha256']==sha(EXT/'configs/static_delivery/diagnostic_prereg_v1.json')
    assert len(identity['cpu_affinity'])==1 and identity['backend']=='gpu'
    for name,expected in identity['native_sources'].items():assert sha(directory/'source_at_run'/name)==expected
    for w in read(directory/'warmups.json').values():assert w['role']=='train' and w['case']=='r101'
    summary=read(directory/'summary.json')
    fixed=[read(p) for p in sorted((directory/'fixed_work').glob('*.json'))]
    wall=[read(p) for p in sorted((directory/'wall').glob('*.json'))]
    assert len(fixed)==384 and len(wall)==48
    assert summary['fixed_results']==len(fixed) and summary['wall_results']==len(wall)
    assert not summary['formal_experiment'] and summary['outer_test_cases_used']==0
    pairing={};fixed_rows=[];counts=set()
    for r in fixed:
        case=r['case'];assert case in plan['cases'] and r['task']==ProfitTask(case,fold=0,role='validation').identity()
        actual=verify_routes(case,r['routes'],r['final'])
        verify_routes(case,r['initial_routes'],r['initial'])
        key=(r['variant'],case,r['remove_count'],r['repetition'])
        if key not in pairing:pairing[key]=[]
        pairing[key].append(r)
        assert r['operator_counts'].get('fallback_scans',0)==0
        if r['arm']!='exhaustive':counts.add((r['remove_count'],r['operator_counts']['exact_positions']))
        fixed_rows.append(dict(variant=r['variant'],case=case,arm=r['arm'],remove_count=r['remove_count'],repetition=r['repetition'],
            net_benefit_U=actual['net_benefit_U'],improvement_U=actual['net_benefit_U']-r['initial']['net_benefit_U'],
            accepted_improvement_U=max(0,actual['net_benefit_U']-r['initial']['net_benefit_U']),**r['timing'],
            exact_positions=r['operator_counts']['exact_positions']))
    for key,group in pairing.items():
        assert all(g['initial_routes']==group[0]['initial_routes'] and g['removed']==group[0]['removed'] for g in group)
        controlled=[g['operator_counts']['exact_positions'] for g in group if g['arm']!='exhaustive']
        assert len(set(controlled))==1,'Different exact candidate counts in ranking control'
    assert counts=={(10,440),(20,1680)}
    wall_rows=[];event_count=0;failures=[]
    for r in wall:
        for event in r['history']:
            assert event['elapsed_s']<=r['timing']['budget_s']
            verify_routes(r['case'],event['routes'],event['score']);event_count+=1
        assert all(b['elapsed_s']>=a['elapsed_s'] and b['score']['net_benefit_U']>a['score']['net_benefit_U']
                   for a,b in zip(r['history'],r['history'][1:]))
        if r['routes']:verify_routes(r['case'],r['routes'],r['final'])
        assert abs(r['elapsed_s']-r['timing']['total_measured_s'])<1e-9
        assert r['timing']['total_measured_s']>=r['timing']['search_finished_s']
        for t,endpoint in r['anytime'].items():
            if float(t)>r['timing']['budget_s']:assert endpoint['status']=='NOT_REQUESTED'
            else:
                allowed=[e for e in r['history'] if e['elapsed_s']<=float(t)]
                assert endpoint['incumbent']==(allowed[-1] if allowed else None)
        if r['status'] in ('FAILED','INITIALIZATION_TIMEOUT'):failures.append(dict(case=r['case'],variant=r['variant'],arm=r['arm'],status=r['status']))
        wall_rows.append(dict(variant=r['variant'],case=r['case'],arm=r['arm'],remove_count=r['remove_count'],
            net_benefit_10s_U=r['anytime']['10']['incumbent']['score']['net_benefit_U'] if r['anytime']['10']['incumbent'] else None,
            net_benefit_60s_U=r['anytime']['60']['incumbent']['score']['net_benefit_U'] if r['anytime']['60']['incumbent'] else None,
            iterations=r['iterations'],timing=r['timing'],status=r['status']))
    # All adaptive arms must share the exact removal prefix; completed counts need not match.
    for variant in plan['variants']:
        for case in plan['cases']:
            group=[r for r in wall if r['variant']==variant and r['case']==case and r['remove_count']==10]
            n=min(len(r['attempts']) for r in group)
            assert all([a['removed'] for a in r['attempts'][:n]]==[a['removed'] for a in group[0]['attempts'][:n]] for r in group)
    classical=[]
    for method in ('alns','vns'):
        r=read(directory/'classical_train'/(method+'.json'))
        assert len(r['cpu_affinity'])==1 and r['role']=='train'
        for event in r['history']:
            assert event['elapsed_s']<=60
            verify_routes('r101',event['routes'],event['score'],role='train')
        classical.append(dict(method=method,status=r['status'],timing=r['timing'],events=len(r['history'])))
    def fixed_mean(v,c,a,m=10):return mean(r['net_benefit_U'] for r in fixed_rows if (r['variant'],r['case'],r['arm'],r['remove_count'])==(v,c,a,m))
    contrasts=[];decision={}
    for variant in plan['variants']:
        gate=True
        for control in ('shuffled_top8','untrained_top8','uniform_top8','exhaustive'):
            deltas={c:fixed_mean(variant,c,'trained_top8')-fixed_mean(variant,c,control) for c in plan['cases']}
            contrasts.append(dict(variant=variant,contrast='trained-minus-'+control,per_case_U=deltas,mean_U=mean(deltas.values())))
            if control in ('shuffled_top8','untrained_top8'):gate=gate and mean(deltas.values())>1 and sum(x>0 for x in deltas.values())>=3
        decision[variant]=dict(contribution_gate_passed=bool(gate))
    d={c:fixed_mean('directed_only',c,'trained_top8')-fixed_mean('maskco_base',c,'trained_top8') for c in plan['cases']}
    decision['directed_only']['D_vs_base_fixed_per_case_U']=d
    decision['directed_only']['D_continuation_gate_passed']=bool(decision['directed_only']['contribution_gate_passed'] and mean(d.values())>0 and sum(x>0 for x in d.values())>=3)
    selected=['maskco_base'] if decision['maskco_base']['contribution_gate_passed'] else []
    if decision['directed_only']['D_continuation_gate_passed']:selected.append('directed_only')
    report=dict(status='PASS_INDEPENDENT_CONTRIBUTION_DIAGNOSTIC' if not failures else 'AUDITED_WITH_FAILURES',
        scientific_freeze=verify_research_freeze(),fixed_routes_reaudited=len(fixed),wall_results_reaudited=len(wall),incumbents_reaudited=event_count,
        failures=failures,exact_candidate_counts=sorted(counts),fixed_rows=fixed_rows,wall_rows=wall_rows,
        fixed_contrasts=contrasts,D_minus_base_fixed_U=d,classical_train_timing=classical,
        decision=decision,approved_training_variants=selected if not failures else [],
        limits=['Single geometry and development seed; masks are paired measurements, not independent datasets.',
                'Exhaustive arm changes candidate count by design. Adaptive wall trajectories diverge; fixed work isolates the ranking comparison.',
                'Mask10/20 search comparison alone does not establish a training-mask causal effect.'],outer_test_cases_used=0)
    dump=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
    out=EXT/'results/static_delivery/algorithm_diagnostics_20261021'
    dump(out/'diagnostic_audit.json',report)
    dump(out/'diagnostic_decision.json',dict(prereg_sha256=identity['prereg_sha256'],diagnostic_run='dev11',
        audit_sha256=sha(out/'diagnostic_audit.json'),decision=decision,approved_training_variants=report['approved_training_variants'],
        remaining_configurations_per_selected_method=2,outer_test_cases_used=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('fixed_rows','wall_rows')}))


if __name__=='__main__':main()
