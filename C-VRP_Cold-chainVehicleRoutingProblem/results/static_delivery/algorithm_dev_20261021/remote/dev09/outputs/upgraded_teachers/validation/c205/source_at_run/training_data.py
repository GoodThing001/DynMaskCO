"""Audited pinned training teachers and explicit reproducible sample schedules."""
import json
import numpy as np
from profit_core import ProfitTask,fold_definition


def sample_schedule(count,steps,seed,order):
    if count<1 or steps<1 or order not in ('ordered','shuffled'):raise ValueError('invalid sample schedule')
    if order=='ordered':return [i%count for i in range(steps)]
    rng=np.random.default_rng(seed+1)
    sequence=[]
    while len(sequence)<steps:sequence.extend(int(i) for i in rng.permutation(count))
    return sequence[:steps]


def load_teachers(path,fold,cases):
    rows=json.loads(path.read_text(encoding='utf-8'))
    allowed=set(fold_definition(fold)['train'])
    selected=set(cases)
    if not selected<=allowed:raise ValueError('requested non-training teachers')
    seen=set();teachers=[]
    for row in rows:
        identity=row['task']
        if identity['fold']!=fold or identity['role']!='train' or identity['case'] not in allowed:
            raise ValueError('cross-role/fold source teacher')
        condition=identity['condition']
        if condition['value_multiplier']!=1:raise ValueError('non-nominal training value')
        key=(identity['case'],condition['profile'],condition['minimum_quality'])
        if key in seen:raise ValueError('duplicate teacher condition')
        seen.add(key)
        if identity['case'] not in selected:continue
        task=ProfitTask(identity['case'],fold=fold,role='train',**condition)
        if task.identity()!=identity:raise ValueError('teacher source/config identity mismatch')
        routes=tuple(tuple(r) for r in row['routes'])
        audited=task.evaluate(routes,audit=True)
        for key in ('net_benefit_U','quality_loss_value_U','distance_km','refrigeration_cost_U','minimum_quality'):
            if abs(audited[key]-row['evaluation'][key])>1e-6:raise ValueError('teacher money/quality mismatch')
        teachers.append((task,routes))
    expected={(case,profile,minimum) for case in selected
              for profile in ('slow_Product3','main_Product1','fast_Product2') for minimum in (0.8,0.85)}
    actual={(task.ins['name'],task.condition['profile'],task.condition['minimum_quality']) for task,_ in teachers}
    if actual!=expected:raise ValueError('incomplete six-condition training teachers')
    return teachers
