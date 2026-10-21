"""Strict validated-incumbent time accounting. Legacy search records are unchanged."""
import datetime
import math
import os
import time

import numpy as np

from profit_core import partial_batch
from profit_search import BudgetExpired,repair,remove_customers


class SearchClock:
    def __init__(self,task,seconds,*,sync=None,now=time.perf_counter):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive finite budget')
        self.task,self.seconds,self.now=task,float(seconds),now
        self.sync_used=sync is not None
        if sync:sync()  # No pending GPU work may cross into the solve clock.
        self.utc=datetime.datetime.now(datetime.timezone.utc)
        self.start=now();self.deadline=self.start+seconds
        self.events=[];self.rejected_late=0;self.initialization_s=None
        self.fallback=None;self.failure=None;self.initial_score=None

    def elapsed(self):return self.now()-self.start

    def offer(self,routes,iteration):
        # Full frozen hard-constraint and independent audit finishes BEFORE acceptance.
        score=self.task.evaluate(routes,audit=True)
        if iteration==0:self.initial_score=score
        elapsed=self.elapsed()
        if elapsed>self.seconds:
            self.rejected_late+=1
            return False
        if self.events and score['net_benefit_U']<=self.events[-1]['score']['net_benefit_U']+1e-7:return False
        self.events.append(dict(elapsed_s=elapsed,completed_at_utc=(self.utc+datetime.timedelta(seconds=elapsed)).isoformat(),
            iteration=iteration,routes=[list(r) for r in routes],score=score))
        return True

    def initialize(self):
        routes=self.task.initial()
        self.fallback=routes
        self.offer(routes,0)
        self.initialization_s=self.elapsed()
        return routes

    def finish(self,method,iterations,**extra):
        search_end=self.elapsed()
        selected=self.events[-1]['routes'] if self.events else self.fallback
        audit_start=self.now()
        final=self.task.evaluate(selected,audit=True) if selected is not None else None
        closing=self.now()-audit_start
        total=self.elapsed()
        endpoints={}
        for t in (1,10,60):
            eligible=[e for e in self.events if e['elapsed_s']<=t]
            endpoints[str(t)]=dict(status=('NOT_REQUESTED' if t>self.seconds else ('VALID' if eligible else 'NO_VALID_INCUMBENT')),
                incumbent=eligible[-1] if eligible and t<=self.seconds else None)
        status='FAILED' if self.failure else ('INITIALIZATION_TIMEOUT' if not self.events else
                ('BUDGET_EXHAUSTED' if search_end>=self.seconds else 'WORK_LIMIT_COMPLETE'))
        return dict(method=method,status=status,failure=self.failure,routes=[list(r) for r in selected] if selected else None,
            final=final,initial=self.initial_score,
            selected_incumbent_in_budget=bool(self.events),fallback_label=None if self.events else 'ONLINE_INITIALIZER_OUTSIDE_BUDGET_NOT_A_BUDGET_RESULT',
            history=self.events,anytime=endpoints,iterations=iterations,elapsed_s=total,
            timing=dict(budget_s=self.seconds,initialization_s=self.initialization_s,search_finished_s=search_end,
                        closing_audit_s=closing,total_measured_s=total,search_overrun_s=max(0,search_end-self.seconds),
                        rejected_late_candidates=self.rejected_late,clock='perf_counter; independent validation completion timestamps',
                        gpu_sync_before_start=self.sync_used),role=self.task.role,formal_experiment=False,**extra)


def guided_search(task,provider,variant,*,seconds,iterations,seed,remove_count=10,topk=8):
    clock=SearchClock(task,seconds,sync=provider.sync)
    stats={};completed=0;routes=None;attempts=[]
    try:
        routes=clock.initialize();value=task.evaluate(routes)['net_benefit_U']
        rng=np.random.default_rng(seed)
        for step in range(iterations):
            if clock.elapsed()>=seconds:break
            removed=tuple(int(i) for i in rng.choice(np.arange(1,101),size=remove_count,replace=False))
            attempt=dict(iteration=step+1,removed=list(removed),started_s=clock.elapsed(),completed_s=None)
            attempts.append(attempt)
            heatmap=provider.heatmap(task,routes,removed,variant,step)
            partial=remove_customers(routes,removed)
            candidate=repair(task,partial,removed,heatmap=heatmap,topk=topk,regret=True,
                deadline=clock.deadline,implementation='vectorized',stats=stats,fallback=False)
            score=task.evaluate(candidate)['net_benefit_U']
            if score>value+1e-7 and clock.offer(candidate,step+1):routes,value=candidate,score
            attempt.update(completed_s=clock.elapsed(),candidate_net_benefit_U=score,within_budget=clock.elapsed()<=seconds)
            if clock.elapsed()>=seconds:break
            completed+=1
    except BudgetExpired:
        pass
    except Exception as e:
        clock.failure=type(e).__name__+': '+str(e)
    return clock.finish(provider.arm,completed,operator_counts=stats,task=task.identity(),
                        attempts=attempts,
                        removal_stream=dict(seed=seed,remove_count=remove_count,policy='same seeded customer-mask prefix across arms; adaptive routes diverge'))


def single_cpu_guard():
    affinity=list(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    if affinity and len(affinity)!=1:raise ValueError('Classical timing requires exactly one CPU in process affinity')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('Classical BLAS/OpenMP threads must be one')
    return affinity


def classical_search(task,method,*,seconds,iterations,seed):
    """Use pinned legacy operator semantics while recording an independently validated best."""
    affinity=single_cpu_guard()
    if method not in ('alns','vns'):raise ValueError('Unknown classical method')
    # A private wrapper intercepts COMPLETE feasible candidate evaluations. The
    # library may accept a late state internally; only this clock's events count.
    from profit_search import alns_search,vns_search
    clock=SearchClock(task,seconds)
    original_initial,original_evaluate=task.initial,task.evaluate
    offering=False
    def evaluate(routes,*,audit=False):
        nonlocal offering
        value=original_evaluate(routes,audit=audit)
        if not offering and clock.initialization_s is not None:
            if not clock.events or value['net_benefit_U']>clock.events[-1]['score']['net_benefit_U']+1e-7:
                offering=True
                try:clock.offer(routes,-1)
                finally:offering=False
        return value
    def initial():
        nonlocal offering
        routes=original_initial();clock.fallback=routes
        offering=True
        try:clock.offer(routes,0)
        finally:offering=False
        clock.initialization_s=clock.elapsed()
        return routes
    task.initial,task.evaluate=initial,evaluate
    completed=0
    try:
        # Remaining time accounts for imports/initialization inside the original
        # implementation; independently timestamped offers use the outer clock.
        result=(alns_search if method=='alns' else vns_search)(task,seconds=seconds,iterations=iterations,seed=seed)
        completed=result['iterations']
    except Exception as e:clock.failure=type(e).__name__+': '+str(e)
    finally:task.initial,task.evaluate=original_initial,original_evaluate
    return clock.finish(method,completed,cpu_affinity=affinity,task=task.identity(),
        library_accounting='Only independently validated outer-clock events count; all late library states excluded. Legacy internal final audits charged to search overhead.')
