"""Audit the registered two-branch continuation; select by strict validation profit."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

EXT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT/'scripts/static_profit'))
from profit_core import verify_research_freeze
from verify_static_profit_diagnostics import verify_routes
from training_data import load_teachers


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='dev12')
    args=parser.parse_args();assert args.label=='dev12'
    results=EXT/'results/static_delivery';root=results/'algorithm_dev_20261021/remote'/args.label
    receipt=read(root/'download_verification.json');manifest=read(root/'output_manifest.json');assert receipt['exit_code']==0
    for name in receipt['downloaded_verified']:assert sha(root/name)==manifest[name]
    directory=root/'outputs/continuation';train=directory/'train';val=directory/'validation'
    decision=read(results/'algorithm_diagnostics_20261021/diagnostic_decision.json')
    variants=decision['approved_training_variants'];assert variants
    assert read(train/'summary.json')['status']=='PASS_REGISTERED_CONTINUATION_TRAINING'
    vi=read(val/'identity.json');assert vi['threads']==1 and len(vi['cpu_affinity'])==1 and vi['backend']=='gpu'
    assert vi['decision_sha256']==sha(results/'algorithm_diagnostics_20261021/diagnostic_decision.json')
    pair=read(train/'pairing_plan.json');assert pair['parent_step']==1440 and pair['final_step']==2880
    assert len(pair['teacher_indices'])==len(pair['masks20'])==2880
    teachers=read(train/'teachers.json')
    load_teachers(train/'teachers.json',0,sorted({r['task']['case'] for r in teachers}))
    rows=[];fixed_rows=[];failures=[];events=0
    for variant in variants:
        for branch in ('keep20','switch10'):
            folder=train/variant/branch
            metrics=read(folder/'training.json');checkpoint=read(folder/'checkpoint/checkpoint.json')
            assert metrics['checkpoint_prediction_identical'] and metrics['optimizer_resume_identical']
            assert metrics['steps']==checkpoint['steps']==2880 and len(metrics['updates'])==1440
            for index,step in enumerate(metrics['updates'],1440):
                assert step['step']==index+1 and step['teacher_index']==pair['teacher_indices'][index]
                assert step['mask_count']==(20 if branch=='keep20' else 10)
                assert step['updated_parameter_leaves']>0
            folder=val/variant/branch
            warm=read(folder/'warmup.json');assert warm['role']=='train' and warm['case']=='r101'
            assert warm['checkpoint_sha256']==sha(train/variant/branch/'checkpoint/checkpoint.json')
            for case in ('c205','c206','c207','c208'):
                r=read(folder/(case+'_wall.json'))
                assert r['checkpoint_steps']==2880 and len(r['cpu_affinity'])==1
                assert r['selected_incumbent_in_budget'] and r['timing']['budget_s']==10
                verify_routes(case,r['routes'],r['final'])
                for event in r['history']:
                    assert event['elapsed_s']<=10;verify_routes(case,event['routes'],event['score']);events+=1
                assert r['anytime']['10']['incumbent']==r['history'][-1]
                assert r['anytime']['60']['status']=='NOT_REQUESTED'
                if r['status']=='FAILED':failures.append((variant,branch,case))
                rows.append(dict(variant=variant,branch=branch,case=case,net_benefit_U=r['anytime']['10']['incumbent']['score']['net_benefit_U'],
                                 iterations=r['iterations'],timing=r['timing']))
                for rep in range(8):
                    fixed=read(folder/(case+f'_fixed_{rep}.json'));verify_routes(case,fixed['routes'],fixed['final'])
                    reference=read(results/'algorithm_dev_20261021/remote/dev11/outputs/diagnostics/fixed_work'/f'{variant}_{case}_trained_top8_m10_r{rep}.json')
                    assert fixed['removed']==reference['removed'] and fixed['initial_routes']==reference['initial_routes']
                    assert fixed['operator_counts']['exact_positions']==reference['operator_counts']['exact_positions']==440
                    fixed_rows.append(dict(variant=variant,branch=branch,case=case,repetition=rep,net_benefit_U=fixed['final']['net_benefit_U']))
    prior=read(results/'algorithm_diagnostics_20261021/diagnostic_audit.json')
    selection={}
    for variant in variants:
        candidates={}
        refs={r['case']:r['net_benefit_10s_U'] for r in prior['wall_rows'] if r['variant']==variant and r['arm']=='trained_top8' and r['remove_count']==10}
        candidates['reference_1440_mask20']=dict(steps=1440,branch='keep20',per_case_U=refs,mean_U=statistics.mean(refs.values()))
        for branch in ('keep20','switch10'):
            percase={r['case']:r['net_benefit_U'] for r in rows if r['variant']==variant and r['branch']==branch}
            candidates[branch+'_2880']=dict(steps=2880,branch=branch,per_case_U=percase,mean_U=statistics.mean(percase.values()))
        winner=max(candidates,key=lambda name:(candidates[name]['mean_U'],-candidates[name]['steps'],candidates[name]['branch']=='keep20'))
        selection[variant]=dict(candidates=candidates,selected=winner,rule='Highest four-case strict10s mean net benefit; exact tie earlier checkpoint, then keep20. CE not used.')
    report=dict(status='PASS_REGISTERED_TRAINING_AND_CHECKPOINT_SELECTION' if not failures else 'AUDITED_WITH_FAILURES',
        scientific_freeze=verify_research_freeze(),training_variants=variants,teachers_reaudited=228,
        final_routes_reaudited=len(rows),fixed_routes_reaudited=len(fixed_rows),incumbents_reaudited=events,
        selection=selection,wall_rows=rows,fixed_rows=fixed_rows,failures=failures,outer_test_cases_used=0,
        limitation='One validation geometry/seed. Only2880 endpoint added; switch10 affects last1440 steps. Shared hardware noise remains; selected recipe is a development choice, not formal method freeze.')
    out=results/'algorithm_diagnostics_20261021/continuation_audit.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('wall_rows','fixed_rows')}))


if __name__=='__main__':main()
