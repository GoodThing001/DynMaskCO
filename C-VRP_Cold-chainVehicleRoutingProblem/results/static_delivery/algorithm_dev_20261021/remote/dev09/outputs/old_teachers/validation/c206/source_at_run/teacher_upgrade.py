"""Refine pinned training teachers with a monotone full-objective ALNS pass."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teachers',type=Path,required=True)
    p.add_argument('--cases',default='r101,r201,rc101',help='original training names, or all')
    p.add_argument('--seconds',type=float,default=1)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    from profit_core import ProfitTask,digest,verify_research_freeze
    from profit_search import alns_search
    if args.seconds<=0:raise ValueError('positive teacher budget required')
    source=json.loads(args.teachers.read_text(encoding='utf-8'))
    selected=None if args.cases=='all' else set(args.cases.split(','))
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[];teachers=[];seen=set()
    for row in source:
        identity=row['task']
        if selected is not None and identity['case'] not in selected:continue
        if identity['role']!='train':raise ValueError('non-training teacher input')
        condition=identity['condition']
        task=ProfitTask(identity['case'],fold=identity['fold'],role='train',**condition)
        if task.identity()!=identity:raise ValueError('teacher source/config identity mismatch')
        key=(identity['case'],condition['profile'],condition['minimum_quality'])
        if key in seen:raise ValueError('duplicate teacher condition')
        seen.add(key)
        routes=tuple(tuple(r) for r in row['routes'])
        before=task.evaluate(routes,audit=True)
        start=time.perf_counter()
        result=alns_search(task,seconds=args.seconds,iterations=100000,seed=1000,initial_routes=routes)
        after=result['final']
        if after['net_benefit_U']<before['net_benefit_U']-1e-7:raise ValueError('teacher degradation')
        teachers.append(dict(task=identity,routes=result['routes'],evaluation=after,
            origin='training-only ALNS refinement of pinned previous teacher, never validation/test',
            parent_teacher_file_sha256=digest(args.teachers)))
        rows.append(dict(task=identity,before=before,after=after,gain_U=after['net_benefit_U']-before['net_benefit_U'],
            iterations=result['iterations'],elapsed_s=time.perf_counter()-start))
        print(json.dumps(dict(case=identity['case'],condition=condition,gain_U=rows[-1]['gain_U'])),flush=True)
    if not teachers:raise ValueError('no selected original training teachers')
    if selected is not None and {t['task']['case'] for t in teachers}!=selected:raise ValueError('missing requested training case')
    report=dict(status='PASS_TRAIN_ONLY_TEACHER_REFINEMENT',scientific_freeze=verify_research_freeze(),
        parent_teacher_file_sha256=digest(args.teachers),requested_seconds=args.seconds,rows=rows,
        training_cases=sorted({t['task']['case'] for t in teachers}),formal_experiment=False,outer_test_cases_used=0)
    for name,value in [('teachers.json',teachers),('teacher_refinement.json',report)]:
        (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
