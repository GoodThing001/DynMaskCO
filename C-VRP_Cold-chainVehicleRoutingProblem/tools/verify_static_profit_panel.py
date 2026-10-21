"""Independently re-score a completed development panel; never formal test evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT/'scripts/static_profit'))
from profit_core import ProfitTask, fold_definition, verify_research_freeze


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    args = parser.parse_args()
    if not __import__('re').fullmatch(r'dev[0-9]{2}',args.label):
        parser.error('owned development label required')
    root = EXT/'results/static_delivery/algorithm_dev_20261021/remote'/args.label
    receipt = read(root/'download_verification.json')
    assert receipt['exit_code']==0, 'incomplete/failed panel is not a completed experiment'
    manifest = read(root/'output_manifest.json')
    for name in receipt['downloaded_verified']:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest[name]
    variants = ('maskco_base','quality_only','directed_only','quality_directed')
    train = root/'outputs/train'
    training = read(train/'summary.json')
    assert training['status']=='PASS_DEVELOPMENT_TRAINING'
    assert training['backend']=='gpu'
    teachers = read(train/'teachers.json')
    fold = fold_definition(0)
    assert len(teachers)==len(fold['train'])*6
    assert {t['task']['case'] for t in teachers}==set(fold['train'])
    teacher_sha = hashlib.sha256((train/'teachers.json').read_bytes()).hexdigest()
    checkpoint_steps = {}
    for variant in variants:
        record = read(train/variant/'checkpoint/checkpoint.json')
        assert record['model']['variant']==variant and record['model']['profile']=='maskco_256'
        assert record['metadata']['teacher_file_sha256']==teacher_sha
        assert all(t['fold']==0 and t['role']=='train' and t['condition']['value_multiplier']==1
                   for t in record['metadata']['training_tasks'])
        metrics = read(train/(variant+'_training.json'))
        assert metrics['checkpoint_prediction_identical'] and metrics['optimizer_resume_identical']
        assert all(row['updated_parameter_leaves']>0 for row in metrics['training'])
        checkpoint_steps[variant] = record['steps']
    assert len(set(checkpoint_steps.values()))==1
    rows = []
    for case in fold['validation']:
        directory = root/'outputs/validation'/case
        summary = read(directory/'summary.json')
        assert summary['backend']=='gpu' and not summary['formal_experiment']
        assert not summary['method_superiority_established']
        assert summary['outer_test_cases_used']==0
        assert set(summary['searches'])==set(variants)|{'alns','vns'}
        starts = []
        for method in (*variants,'alns','vns'):
            result = read(directory/(method+'_search.json'))
            task = ProfitTask(case,fold=0,role='validation')
            value = task.evaluate(result['routes'],audit=True)
            for key in ('net_benefit_U','quality_loss_value_U','distance_km','refrigeration_cost_U','minimum_quality'):
                assert abs(value[key]-result['final'][key])<1e-6, (case,method,key)
            starts.append(result['initial']['net_benefit_U'])
            rows.append(dict(case=case,method=method,**value,iterations=result['iterations'],elapsed_s=result['elapsed_s']))
        assert max(starts)-min(starts)<1e-6
    report = dict(status='PASS_LOCAL_INDEPENDENT_DEVELOPMENT_PANEL_REEVALUATION',
        scientific_freeze=verify_research_freeze(),fold=0,training_instances=len(fold['train']),
        teacher_conditions=6,checkpoint_steps=checkpoint_steps,validation_instances=fold['validation'],
        rows=rows,outer_test_cases_used=0,formal_experiment=False,superiority_established=False,
        limitation='One development seed, one validation geometry, short budget; no significance or test claim.')
    (root/'local_panel_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],independently_rescored_routesets=len(rows),formal_experiment=False)))


if __name__=='__main__':
    main()
