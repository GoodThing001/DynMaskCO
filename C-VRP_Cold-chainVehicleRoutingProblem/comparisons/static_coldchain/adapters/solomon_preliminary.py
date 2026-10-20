"""Original Solomon routing-only pilot, with independent double-precision audit.

No product labels, physical units, price, profit or quality are invented.
The large vehicle coefficient encodes the SINTEF hierarchical objective;
it is an algorithmic scalarisation, not a physical vehicle fee.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

SCRIPTS = next(p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT

ROOT = EXTENSION_ROOT / 'comparisons/static_coldchain'
DATA = EXTENSION_ROOT / 'data/classic_static_sources_20261020/solomon_100'
SCALE = 1_000_000
CASES = ('c101','c201','r101','r201','rc101','rc201')

def load(path):
    lines = path.read_text().splitlines()
    nv, capacity = next(tuple(map(int, line.split())) for line in lines if len(line.split()) == 2 and all(x.isdigit() for x in line.split()))
    nodes=[]
    for line in lines:
        parts=line.split()
        if len(parts)==7:
            try: vals=list(map(float,parts))
            except ValueError: continue
            nodes.append(vals)
    if len(nodes)!=101 or [r[0] for r in nodes]!=list(range(101)):
        raise ValueError('not an intact original 100 customer instance')
    d=[[math.hypot(a[1]-b[1],a[2]-b[2]) for b in nodes] for a in nodes]
    fixed=(len(nodes)-1+nv)*max(math.ceil(v*SCALE) for row in d for v in row)+1
    return {'name':path.stem.lower(),'nodes':nodes,'vehicles':nv,'capacity':capacity,'d':d,'vehicle_scalar':fixed,
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def audit(instance,routes):
    nodes,d=instance['nodes'],instance['d']
    seen=[]; errors=[]; tours=[]; total=0.
    if len(routes)>instance['vehicles']: errors.append('fleet_limit')
    for route in routes:
        if not route or any(type(i) is not int or not 1<=i<len(nodes) for i in route):
            errors.append('invalid_customer'); continue
        seen.extend(route)
        load=sum(nodes[i][3] for i in route)
        if load>instance['capacity']: errors.append('capacity')
        t=nodes[0][4]; last=0; distance=0.; trace=[]
        for i in route:
            arrival=t+d[last][i]; start=max(arrival,nodes[i][4])
            if start>nodes[i][5]+1e-9: errors.append('time_window')
            trace.append({'customer':i,'arrival':arrival,'service_start':start,'departure':start+nodes[i][6]})
            distance+=d[last][i]; t=start+nodes[i][6]; last=i
        t+=d[last][0]; distance+=d[last][0]
        if t>nodes[0][5]+1e-9: errors.append('depot_return')
        total+=distance
        tours.append({'route':route,'initial_load':load,'return_load':0.,'distance':distance,'return_time':t,'trace':trace})
    if sorted(seen)!=list(range(1,len(nodes))): errors.append('coverage_or_duplicate')
    return {'feasible':not errors,'errors':errors,'vehicles_used':len(routes),'distance':total,'tours':tours,
            'distance_unit':'original_benchmark_coordinate_unit_not_km','time_unit':'original_benchmark_time_unit_not_minutes',
            'quality_or_profit_evaluated':False}

def ortools_solve(ins,seconds,seed):
    from ortools.constraint_solver import pywrapcp,routing_enums_pb2
    nodes,d=ins['nodes'],ins['d']; n=len(nodes)
    manager=pywrapcp.RoutingIndexManager(n,ins['vehicles'],0)
    model=pywrapcp.RoutingModel(manager)
    distances=[[round(v*SCALE) for v in row] for row in d]
    durations=[[math.ceil(v*SCALE)+round(nodes[i][6]*SCALE) for v in row] for i,row in enumerate(d)]
    cost=model.RegisterTransitCallback(lambda a,b:distances[manager.IndexToNode(a)][manager.IndexToNode(b)])
    model.SetArcCostEvaluatorOfAllVehicles(cost)
    model.SetFixedCostOfAllVehicles(ins['vehicle_scalar'])
    dem=model.RegisterUnaryTransitCallback(lambda a:int(nodes[manager.IndexToNode(a)][3]))
    model.AddDimensionWithVehicleCapacity(dem,0,[ins['capacity']]*ins['vehicles'],True,'Capacity')
    transit=model.RegisterTransitCallback(lambda a,b:durations[manager.IndexToNode(a)][manager.IndexToNode(b)])
    horizon=round(nodes[0][5]*SCALE)
    model.AddDimension(transit,horizon,horizon,False,'Time')
    dim=model.GetDimensionOrDie('Time')
    for i in range(1,n): dim.CumulVar(manager.NodeToIndex(i)).SetRange(round(nodes[i][4]*SCALE),round(nodes[i][5]*SCALE))
    for v in range(ins['vehicles']):
        dim.CumulVar(model.Start(v)).SetRange(round(nodes[0][4]*SCALE),horizon)
        dim.CumulVar(model.End(v)).SetRange(round(nodes[0][4]*SCALE),horizon)
    p=pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(round(seconds*1000)); p.log_search=False
    solution=model.SolveWithParameters(p)
    if solution is None: return [],{'status':model.status(),'requested_seed':seed,'seed_control':'routing solver deterministic setup; no matching random seed parameter applied'}
    routes=[]
    for v in range(ins['vehicles']):
        i=model.Start(v); route=[]
        while not model.IsEnd(i):
            node=manager.IndexToNode(i)
            if node: route.append(node)
            i=solution.Value(model.NextVar(i))
        if route: routes.append(route)
    return routes,{'status':model.status(),'requested_seed':seed,'seed_control':'not_applied_to_ortools_routing'}

def pyvrp_solve(ins,seconds,seed):
    from pyvrp import Model
    from pyvrp import SolveParams, PenaltyParams
    from pyvrp.stop import MaxRuntime
    model=Model(); nodes=ins['nodes']
    loc=[model.add_location(x=int(r[1]),y=int(r[2])) for r in nodes]
    model.add_depot(loc[0],tw_early=round(nodes[0][4]*SCALE),tw_late=round(nodes[0][5]*SCALE))
    for i,r in enumerate(nodes[1:],1):
        model.add_client(loc[i],delivery=int(r[3]),service_duration=round(r[6]*SCALE),tw_early=round(r[4]*SCALE),tw_late=round(r[5]*SCALE),name=str(i))
    model.add_vehicle_type(num_available=ins['vehicles'],capacity=ins['capacity'],fixed_cost=ins['vehicle_scalar'],tw_early=round(nodes[0][4]*SCALE),tw_late=round(nodes[0][5]*SCALE))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            model.add_edge(loc[i],loc[j],distance=round(ins['d'][i][j]*SCALE),duration=math.ceil(ins['d'][i][j]*SCALE))
    # The explicit lexicographic coefficient is much larger than default
    # distance costs. Raise the penalty cap for this numeric scaling only.
    # 1e7 * conservative (n*max_arc_time + all_service) * fleet is < 2**63.
    cap=10_000_000.
    upper=cap*(len(nodes)*max(max(row) for row in ins['d'])+sum(r[6] for r in nodes))*SCALE*ins['vehicles']
    if upper>=2**63: raise ValueError('penalty_scaling_overflow_bound')
    params=SolveParams(penalty=PenaltyParams(max_penalty=cap))
    result=model.solve(MaxRuntime(seconds),seed=seed,display=False,collect_stats=False,params=params)
    # v0.14 routes expose scheduled activities with zero-based client IDs.
    routes=[[int(a.idx)+1 for a in r if a.is_client()] for r in result.best.routes()]
    return routes,{'solver_feasible':result.best.is_feasible(),'seed':seed,'iterations':result.num_iterations,'penalty_cap':cap,'overflow_bound':upper}

def bks():
    from bs4 import BeautifulSoup
    soup=BeautifulSoup((DATA/'source_page.html').read_text(encoding='utf-8'),'html.parser')
    out={}
    for tr in soup.select('tr'):
        s=tr.get_text(' ',strip=True)
        m=re.match(r'^(c\d+|r\d+|rc\d+)(b?)\s+(\d+)\s+([\d.]+)\b',s,re.I)
        if m: out[m[1].lower()]={'vehicles':int(m[3]),'distance':float(m[4]),'source_label':m[1]+m[2], 'bks_is_reference_not_optimality_proof':True}
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--seconds',type=float,default=10); p.add_argument('--cases',nargs='+',default=CASES); args=p.parse_args()
    if args.out.exists(): raise FileExistsError('new output directory required')
    args.out.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'.runtime/packages'))
    refs=bks(); rows=[]
    for case in args.cases:
        files=[f for f in (DATA/'instances').rglob('*.txt') if f.stem.lower()==case.lower()]
        if len(files)!=1: raise ValueError(f'ambiguous source {case}')
        ins=load(files[0])
        for name,solver in [('OR-Tools-GLS',ortools_solve),('PyVRP-HGS',pyvrp_solve)]:
            start=time.perf_counter()
            try:
                routes,meta=solver(ins,args.seconds,20261020)
                report=audit(ins,routes)
                row={'case':case,'method':name,'source_sha256':ins['source_sha256'],'budget_search_s':args.seconds,
                     'total_build_and_search_s':time.perf_counter()-start,'routes':routes,'report':report,'solver':meta,
                     'reference':refs.get(case),'distance_gap_pct':None}
                if report['feasible'] and case in refs and report['vehicles_used']==refs[case]['vehicles']:
                    row['distance_gap_pct']=100*(report['distance']/refs[case]['distance']-1)
            except Exception as e:
                row={'case':case,'method':name,'error_type':type(e).__name__,'error':str(e),'report':{'feasible':False}}
            rows.append(row)
            (args.out/f'{case}_{name}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k not in ('routes','solver','report')}|{'feasible':row['report']['feasible'],'vehicles':row['report'].get('vehicles_used'),'distance':row['report'].get('distance')},ensure_ascii=True),flush=True)
    versions={name:importlib.metadata.version(name) for name in ('ortools','pyvrp','numpy','vrplib')}
    result={'scope':'PRELIMINARY_ROUTING_ONLY_NOT_COLD_CHAIN_NET_PROFIT_NOT_MASKCO_COMPARISON','cases_preselected':'first original instance in each C1/C2/R1/R2/RC1/RC2 family; one pilot run per solver',
            'objective':'lexicographic_min_vehicles_then_distance','physical_prices_or_products_added':False,
            'integer_scale':SCALE,'distance_rounding':'nearest; travel durations ceil; raw double independent evaluation',
            'scalarisation':'M=(n_customers+available_vehicles)*max_scaled_arc+1; algorithmic coefficient not vehicle fee',
            'source_page_sha256':hashlib.sha256((DATA/'source_page.html').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'versions':versions,'python':platform.python_version(),'rows':rows}
    (args.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()
