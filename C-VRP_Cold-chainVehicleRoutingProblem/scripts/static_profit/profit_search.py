"""Shared exact-profit reconstruction; objective-adapted neural, ALNS and VNS."""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from profit_core import EXTENSION_ROOT, partial_batch


class BudgetExpired(Exception):
    pass


def check_time(deadline):
    if time.perf_counter() >= deadline:
        raise BudgetExpired()


def remove_customers(routes, removed):
    removed = set(removed)
    return tuple(part for r in routes if (part := tuple(i for i in r if i not in removed)))


def insertion_options(task, routes, customer, heatmap=None, topk=8, *, deadline=math.inf,stats=None,fallback=True):
    positions = []
    for rid,r in enumerate(routes):
        if sum(task.ins['nodes'][j][3] for j in r)+task.ins['nodes'][customer][3] > 20000:
            continue
        for pos in range(len(r)+1):
            a, b = (r[pos-1] if pos else 0), (r[pos] if pos < len(r) else 0)
            priority = 0 if heatmap is None else float(heatmap[a,customer]+heatmap[customer,b]-heatmap[a,b])
            positions.append((priority,rid,pos))
    if heatmap is not None:
        positions.sort(key=lambda x:(-x[0],x[1],x[2]))
        shortlist = positions[:topk]
    else:
        shortlist = positions
    choices = []
    if stats is not None:stats['ranked_positions']=stats.get('ranked_positions',0)+len(positions)
    def inspect(items):
        for _,rid,pos in items:
            check_time(deadline)
            if stats is not None:stats['exact_positions']=stats.get('exact_positions',0)+1
            old = routes[rid]
            new = old[:pos]+(customer,)+old[pos:]
            score = task.route(new)
            if score.feasible:
                choices.append((score.net_benefit-task.route(old).net_benefit,rid,pos,new))
    inspect(shortlist)
    # When the heatmap has no feasible insertion, scan the rest before opening a route.
    if heatmap is not None and not choices and fallback:
        if stats is not None:stats['fallback_scans']=stats.get('fallback_scans',0)+1
        inspect(positions[topk:])
    single = task.route((customer,))
    if single.feasible and len(routes) < 100:
        choices.append((single.net_benefit,len(routes),0,(customer,)))
    choices.sort(key=lambda v:(-v[0],v[1],v[2]))
    return choices


class InsertionIndex:
    """Shared route edges/loads for one regret step; exact original tie rules."""
    def __init__(self,task,routes,heatmap,topk,deadline=math.inf,stats=None,fallback=True):
        self.task,self.routes,self.heatmap,self.topk=task,routes,heatmap,topk
        self.deadline,self.stats,self.fallback=deadline,stats,fallback
        edges=[(rid,pos,a,b,sum(task.ins['nodes'][i][3] for i in r))
               for rid,r in enumerate(routes)
               for pos,(a,b) in enumerate(zip((0,)+tuple(r),tuple(r)+(0,)))]
        self.edges=np.asarray(edges,dtype=np.int64).reshape((-1,5))

    def options(self,customer):
        task,routes,heatmap=self.task,self.routes,self.heatmap
        demand=task.ins['nodes'][customer][3]
        edges=self.edges[self.edges[:,4]+demand<=20000]
        if len(edges):
            rid,pos,a,b,_=edges.T
            priority=heatmap[a,customer]+heatmap[customer,b]-heatmap[a,b]
            order=np.lexsort((pos,rid,-priority))
            edges=edges[order]
        choices=[]
        if self.stats is not None:self.stats['ranked_positions']=self.stats.get('ranked_positions',0)+len(edges)
        def inspect(entries):
            for rid,pos,_,_,_ in entries:
                check_time(self.deadline)
                if self.stats is not None:self.stats['exact_positions']=self.stats.get('exact_positions',0)+1
                rid,pos=int(rid),int(pos)
                old=routes[rid];new=old[:pos]+(customer,)+old[pos:]
                score=task.route(new)
                if score.feasible:choices.append((score.net_benefit-task.route(old).net_benefit,rid,pos,new))
        inspect(edges[:self.topk])
        if not choices and self.fallback:
            if self.stats is not None:self.stats['fallback_scans']=self.stats.get('fallback_scans',0)+1
            inspect(edges[self.topk:])
        single=task.route((customer,))
        if single.feasible and len(routes)<100:choices.append((single.net_benefit,len(routes),0,(customer,)))
        choices.sort(key=lambda v:(-v[0],v[1],v[2]))
        return choices


def repair(task, partial, removed, *, heatmap=None, topk=8, regret=True, deadline=math.inf,implementation='classic',stats=None,fallback=True):
    if implementation not in ('classic','vectorized'):raise ValueError('unknown repair implementation')
    routes, pending = list(partial), sorted(removed)
    while pending:
        check_time(deadline)
        index=InsertionIndex(task,routes,heatmap,topk,deadline,stats,fallback) if implementation=='vectorized' and heatmap is not None else None
        decisions = []
        for i in pending:
            check_time(deadline)
            choices = insertion_options(task,routes,i,heatmap,topk,deadline=deadline,stats=stats,fallback=fallback) if index is None else index.options(i)
            if not choices:
                raise ValueError('singleton-feasible cohort must permit completion with K=n')
            priority = choices[0][0]-choices[1][0] if regret and len(choices)>1 else choices[0][0]
            decisions.append((priority,-i,i,choices[0]))
            if not regret:
                break
        _,_,i,(_,rid,pos,new) = max(decisions)
        if rid == len(routes):
            routes.append(new)
        else:
            routes[rid] = new
        pending.remove(i)
    routes = tuple(routes)
    task.evaluate(routes)
    return routes


def _result(task, routes, initial, history, start, iterations, method):
    return dict(method=method, routes=[list(r) for r in routes],
                initial=task.evaluate(initial,audit=True), final=task.evaluate(routes,audit=True),
                history=history, elapsed_s=time.perf_counter()-start, iterations=iterations,
                role=task.role, formal_experiment=False)


def neural_search(task, model, variant, *, seconds, iterations, seed, remove_count=10, topk=8,repair_implementation='classic'):
    from profit_neural import predict
    start = time.perf_counter()
    deadline = start+seconds
    initial = routes = task.initial()
    value = task.evaluate(routes)['net_benefit_U']
    history, completed = [dict(iteration=0,net_benefit_U=value)], 0
    rng = np.random.default_rng(seed)
    for step in range(iterations):
        try:
            check_time(deadline)
            removed = tuple(int(i) for i in rng.choice(np.arange(1,101),size=remove_count,replace=False))
            batch,partial = partial_batch(task,routes,removed,variant)
            logits = predict(model,batch)
            shifted = logits-logits.max(axis=1,keepdims=True)
            logprob = shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
            candidate = repair(task,partial,removed,heatmap=logprob,topk=topk,deadline=deadline,implementation=repair_implementation)
            check_time(deadline)
        except BudgetExpired:
            break
        candidate_value = task.evaluate(candidate)['net_benefit_U']
        if candidate_value > value+1e-7:
            routes,value = candidate,candidate_value
        completed += 1
        history.append(dict(iteration=step+1,net_benefit_U=value))
    return _result(task,routes,initial,history,start,completed,variant)


def vns_search(task, *, seconds, iterations, seed):
    start = time.perf_counter()
    initial = routes = task.initial()
    value = task.evaluate(routes)['net_benefit_U']
    rng = np.random.default_rng(seed)
    history, level, completed = [dict(iteration=0,net_benefit_U=value)], 0, 0
    for step in range(iterations):
        if time.perf_counter() >= start+seconds:
            break
        candidate = [list(r) for r in routes]
        if level == 0:
            src = int(rng.integers(len(candidate)))
            i = candidate[src].pop(int(rng.integers(len(candidate[src]))))
            candidate = [r for r in candidate if r]
            dest = int(rng.integers(len(candidate)+1))
            if dest == len(candidate):
                candidate.append([i])
            else:
                candidate[dest].insert(int(rng.integers(len(candidate[dest])+1)),i)
        elif level == 1:
            i,j = [int(x) for x in rng.choice(np.arange(1,101),size=2,replace=False)]
            positions = {i:(rid,pos) for rid,r in enumerate(candidate) for pos,i in enumerate(r)}
            a,b = positions[i],positions[j]
            candidate[a[0]][a[1]],candidate[b[0]][b[1]] = j,i
        elif level == 2:
            eligible = [i for i,r in enumerate(candidate) if len(r)>1]
            if eligible:
                rid = int(rng.choice(eligible))
                a,b = sorted(int(x) for x in rng.choice(len(candidate[rid]),size=2,replace=False))
                candidate[rid][a:b+1] = candidate[rid][a:b+1][::-1]
        elif len(candidate)>1:
            a,b = [int(x) for x in rng.choice(len(candidate),size=2,replace=False)]
            merged = candidate[a]+candidate[b]
            candidate = [r for rid,r in enumerate(candidate) if rid not in (a,b)]+[merged]
        candidate = tuple(tuple(r) for r in candidate)
        try:
            score = task.evaluate(candidate)['net_benefit_U']
        except ValueError:
            score = -math.inf
        if time.perf_counter() >= start+seconds:
            break
        if score > value+1e-7:
            routes,value,level = candidate,score,0
        else:
            level = (level+1)%4
        completed += 1
        history.append(dict(iteration=step+1,net_benefit_U=value))
    return _result(task,routes,initial,history,start,completed,'full_objective_VNS_dev')


def alns_search(task, *, seconds, iterations, seed, remove_count=10, initial_routes=None):
    if initial_routes is not None and task.role!='train':
        raise ValueError('stored-incumbent refinement is training-teacher only')
    # Import the pinned read-only official library, not its distance examples.
    sys.path.insert(0,str(EXTENSION_ROOT/'comparisons/static_coldchain/upstream/ALNS'))
    import alns as official_alns
    if not str(official_alns.__file__).startswith(str(EXTENSION_ROOT/'comparisons/static_coldchain/upstream/ALNS')):
        raise ValueError('ALNS was not imported from its pinned read-only source snapshot')
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel

    start = time.perf_counter()
    deadline = start+seconds
    initial = task.initial() if initial_routes is None else tuple(tuple(r) for r in initial_routes)
    task.evaluate(initial,audit=True)
    nominal = task.model.price*sum(r[3] for r in task.ins['nodes'])
    class State:
        def __init__(self,routes,pending=(),fallback=None):
            self.routes,self.pending,self.fallback = routes,pending,fallback
            self.value = math.inf if pending else nominal-task.evaluate(routes)['net_benefit_U']
        def objective(self):
            return self.value
    engine = ALNS(np.random.default_rng(seed))
    def destroy(state,rng,mode):
        if mode == 'related':
            center = int(rng.integers(1,101))
            removed = sorted(range(1,101),key=lambda i:(task.ins['distance'][center][i],i))[:remove_count]
        elif mode == 'worst':
            values = []
            for route in state.routes:
                before = task.route(route).net_benefit
                for i in route:
                    after_route = tuple(j for j in route if j != i)
                    after = task.route(after_route).net_benefit if after_route else 0
                    # Marginal loss beyond this customer's fixed nominal value.
                    values.append((task.model.price*task.ins['nodes'][i][3]-(before-after),i))
            removed = [i for _,i in sorted(values,reverse=True)[:remove_count]]
        else:
            removed = [int(i) for i in rng.choice(np.arange(1,101),remove_count,replace=False)]
        return State(remove_customers(state.routes,removed),tuple(removed),state)
    for mode in ('random','related','worst'):
        engine.add_destroy_operator(lambda state,rng,mode=mode: destroy(state,rng,mode),name=mode)
    for use_regret in (False,True):
        def restore(state,rng,use_regret=use_regret):
            try:
                return State(repair(task,state.routes,state.pending,regret=use_regret,deadline=deadline))
            except BudgetExpired:
                return state.fallback
        engine.add_repair_operator(restore,name='regret2' if use_regret else 'greedy')
    count = 0
    def stop(rng,best,current):
        nonlocal count
        if count >= iterations or time.perf_counter() >= deadline:
            return True
        count += 1
        return False
    initial_state = State(initial)
    temperature = max(1.0,0.02*initial_state.objective())
    result = engine.iterate(initial_state,RouletteWheel([25,5,1,0],0.8,3,2),
                            SimulatedAnnealing(temperature,0.01,0.98),stop)
    # Loss-cost is an affine transform of -net-benefit; no objective mismatch.
    best_values = np.maximum.accumulate([nominal-float(v) for v in result.statistics.objectives])
    history = [dict(iteration=i,net_benefit_U=float(v)) for i,v in enumerate(best_values)]
    return _result(task,result.best_state.routes,initial,history,start,count,'full_objective_ALNS_dev')
