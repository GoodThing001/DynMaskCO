"""Isolated masked insertion chain consuming per-action scores, never a model.

Scores rank first; distance is a deterministic secondary tie break. This is a
new engineering decoder, not a modification of any registered decoder.
"""
import math
import time
import numpy as np
from delivery_feasible_action_options_v1 import options


def action_chain(data, initial, mask, partial, contract, scorer, limit_s=10.):
    if not math.isfinite(limit_s) or limit_s <= 0:
        raise ValueError('Invalid cooperative component limit')
    all_orders=[i for r in initial for i in r]
    if len(set(all_orders))!=len(all_orders) or len(set(mask))!=len(mask) or not set(mask)<=set(all_orders):
        raise ValueError('Invalid masked coverage')
    routes=[[i for i in r if i not in mask] for r in initial]
    remaining=sorted(mask); trace=[]; start=time.perf_counter()
    zeros=np.zeros(np.asarray(partial).shape, dtype=float)
    while remaining:
        if time.perf_counter()-start>=limit_s:
            return dict(routes=routes,remaining=remaining,trace=trace,status='cooperative_deadline')
        customers=[]
        for i in remaining:
            enumeration=options(data,routes,i,np.asarray(partial),zeros,contract)
            rows=enumeration['rows']
            values=scorer(routes,rows)
            if len(values)!=len(rows) or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                raise ValueError('Scorer must return one finite scalar per action')
            packed=[]
            for row,value in zip(rows,values):
                packed.append([i,row['vehicle'],row['position'],row['predecessor'],row['successor'],row['retained_union_ok'],row['original_route_ok'],float(value),row['score']])
            eligible=sorted((r for r in packed if r[5] and r[6]),key=lambda r:(-r[7],r[8],r[1],r[2]))
            best=eligible[0] if eligible else None
            # Regret uses the primary action score only. Singleton gets priority.
            regret=None if len(eligible)==1 else eligible[0][7]-eligible[1][7] if eligible else 0.
            customers.append(dict(customer=i,rows=packed,best=best,regret=regret))
        feasible=[r for r in customers if r['best'] is not None]
        if not feasible:
            trace.append(dict(before=routes,remaining=remaining,customers=customers,chosen=None))
            return dict(routes=routes,remaining=remaining,trace=trace,status='no_feasible_action')
        choice=max(feasible,key=lambda r:(r['regret'] is None,0. if r['regret'] is None else r['regret'],-r['customer']))
        chosen=choice['best']
        trace.append(dict(before=[r[:] for r in routes],remaining=remaining[:],customers=customers,chosen=chosen))
        routes[chosen[1]].insert(chosen[2],chosen[0]);remaining.remove(chosen[0])
    return dict(routes=routes,remaining=[],trace=trace,status='complete')
