"""Re-score the complete two-arm teacher/training study on validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
EXT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT/'scripts/static_profit'))
from profit_core import ProfitTask,verify_research_freeze


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--allow-partial',action='store_true',help='Audit saved evidence without claiming the interrupted study completed')
    parser.add_argument('--recovery-label',help='Merge disjoint missing-cell recovery evidence, preserving original failure')
    args=parser.parse_args()
    if not re.fullmatch(r'dev[0-9]{2}',args.label):raise ValueError('owned devNN required')
    root=EXT/'results/static_delivery/algorithm_dev_20261021/remote'/args.label
    receipt=read(root/'download_verification.json');manifest=read(root/'output_manifest.json')
    assert receipt['exit_code']==0 or ((args.allow_partial or args.recovery_label) and receipt['exit_code']==124)
    for name in receipt['downloaded_verified']:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest[name]
    recovery=None
    if args.recovery_label:
        assert re.fullmatch(r'dev[0-9]{2}',args.recovery_label) and args.recovery_label!=args.label
        recovery=root.parent/args.recovery_label
        rc=read(recovery/'download_verification.json');rm=read(recovery/'output_manifest.json')
        assert rc['exit_code']==0
        for name in rc['downloaded_verified']:
            assert hashlib.sha256((recovery/name).read_bytes()).hexdigest()==rm[name]
        rs=read(recovery/'outputs/recovery/recovery_summary.json')
        assert rs['status']=='PASS_MISSING_VALIDATION_RECOVERY' and not rs['training_repeated']
        assert rs['original_sources_unchanged'] and rs['original_exit_code']==receipt['exit_code']
    rows=[];plans={};training={};missing=[]
    for arm in ('old_teachers','upgraded_teachers'):
        train=root/'outputs'/arm/'train'
        summary=read(train/'summary.json')
        assert summary['status']=='PASS_DEVELOPMENT_TRAINING' and summary['backend']=='gpu'
        plans[arm]=read(train/'training_plan.json')
        assert plans[arm]['sample_order']=='shuffled'
        teacher_file=read(train/'teachers.json')
        assert len(teacher_file)==228
        for row in teacher_file:
            t=row['task'];assert t['fold']==0 and t['role']=='train'
            task=ProfitTask(t['case'],fold=0,role='train',**t['condition'])
            value=task.evaluate(row['routes'],audit=True)
            assert abs(value['net_benefit_U']-row['evaluation']['net_benefit_U'])<1e-6
        training[arm]={}
        for variant in ('maskco_base','directed_only'):
            checkpoint=read(train/variant/'checkpoint/checkpoint.json')
            metrics=read(train/(variant+'_training.json'))
            assert checkpoint['steps']==len(plans[arm]['teacher_indices'])
            assert checkpoint['model']['profile']=='maskco_256'
            assert metrics['checkpoint_prediction_identical'] and metrics['optimizer_resume_identical']
            assert all(row['updated_parameter_leaves']>0 for row in metrics['training'])
            training[arm][variant]=dict(steps=checkpoint['steps'],parameter_count=metrics['parameter_count'])
        for case in ('c205','c206','c207','c208'):
            directory=root/'outputs'/arm/'validation'/case
            recovered=(recovery/'outputs/recovery'/arm/'validation'/case) if recovery else None
            summary=read(directory/'summary.json') if (directory/'summary.json').exists() else None
            if summary is None and recovered and (recovered/'summary.json').exists():
                summary=read(recovered/'summary.json')
            if summary:
                assert summary['backend']=='gpu' and not summary['formal_experiment']
                assert summary['outer_test_cases_used']==0
            else:
                assert args.allow_partial or recovery
            starts=[]
            for method in ('maskco_base','directed_only','alns','vns'):
                path=directory/(method+'_search.json')
                if recovered and (recovered/path.name).exists():
                    assert not path.exists(), 'Duplicate recovery cell'
                    path=recovered/path.name
                if not path.exists():
                    assert args.allow_partial
                    missing.append(dict(arm=arm,case=case,method=method))
                    continue
                result=read(path)
                task=ProfitTask(case,fold=0,role='validation')
                value=task.evaluate(result['routes'],audit=True)
                for key in ('net_benefit_U','quality_loss_value_U','distance_km','refrigeration_cost_U','minimum_quality'):
                    assert abs(value[key]-result['final'][key])<1e-6
                starts.append(result['initial']['net_benefit_U'])
                rows.append(dict(arm=arm,case=case,method=method,evidence=path.relative_to(root.parent).as_posix(),**value,
                    iterations=result['iterations'],elapsed_s=result['elapsed_s']))
            assert not starts or max(starts)-min(starts)<1e-6
            for variant in ('maskco_base','directed_only'):
                if summary is None:
                    if recovery:
                        cfg=read(directory/'run_identity.json')['arguments']
                        assert cfg['predictor']=='compiled' and cfg['matmul_precision']=='highest' and cfg['repair_implementation']=='vectorized'
                    continue
                cfg=summary['variants'][variant]['inference']
                assert cfg['predictor']=='compiled' and cfg['matmul_precision']=='highest'
                assert cfg['repair_implementation']=='vectorized'
    assert plans['old_teachers']['teacher_indices']==plans['upgraded_teachers']['teacher_indices']
    assert plans['old_teachers']['parent_teacher_file_sha256']!=plans['upgraded_teachers']['parent_teacher_file_sha256']
    means={arm:{method:statistics.mean(r['net_benefit_U'] for r in rows if r['arm']==arm and r['method']==method)
                for method in ('maskco_base','directed_only','alns','vns')}
           for arm in ('old_teachers','upgraded_teachers')}
    common=sorted(set(r['case'] for r in rows if r['arm']=='old_teachers') &
                  {case for case in ('c205','c206','c207','c208') if sum(r['arm']=='upgraded_teachers' and r['case']==case for r in rows)==4})
    paired={m:statistics.mean(next(r['net_benefit_U'] for r in rows if r['arm']=='upgraded_teachers' and r['method']==m and r['case']==c)-
                            next(r['net_benefit_U'] for r in rows if r['arm']=='old_teachers' and r['method']==m and r['case']==c) for c in common)
            for m in ('maskco_base','directed_only')} if common else {}
    report=dict(status=('PARTIAL_EVIDENCE_AUDITED_NOT_COMPLETE' if missing else ('PASS_RECOVERED_TRAINING_STUDY' if recovery else 'PASS_LOCAL_INDEPENDENT_TRAINING_STUDY')),scientific_freeze=verify_research_freeze(),
        training=training,teacher_routesets_reaudited=456,validation_routesets_reaudited=len(rows),rows=rows,
        mean_net_benefit_U=means,mean_coverage={arm:{m:[r['case'] for r in rows if r['arm']==arm and r['method']==m] for m in means[arm]} for arm in means},
        missing=missing,original_exit_code=receipt['exit_code'],recovery_label=args.recovery_label,paired_complete_cases=common,
        new_minus_old_teacher_U=paired,formal_experiment=False,outer_test_cases_used=0,
        limitation='One development seed and validation geometry; no test/generalization or significance claim. Wall budgets include small closing-audit overhead.')
    name='local_partial_training_study_verification.json' if missing else 'local_training_study_verification.json'
    if recovery:name='local_recovered_training_study_verification.json'
    ((recovery or root)/name).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','training')}))


if __name__=='__main__':main()
