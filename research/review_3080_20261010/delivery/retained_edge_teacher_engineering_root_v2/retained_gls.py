"""Undirected partial-adjacency constrained GLS, with full-C0 label selection.

Engineering prototype only. No optimality, MaskCO gain or dynamic-online claim.
"""
import copy,math,time
import numpy as np
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
from delivery_baselines import matrices
from delivery_evaluator import evaluate

def adjacency(routes,n):
    out=np.zeros((n,n),np.int8)
    for route in routes:
        chain=[0]+list(route)+[0]
        for a,b in zip(chain,chain[1:]):
            if a!=b:out[a,b]=out[b,a]=1
    return out

def constraints_input(initial,partial,n):
    a=np.asarray(partial)
    if a.shape!=(n,n) or not np.all(np.isfinite(a)) or not np.all((a==0)|(a==1)) or not np.array_equal(a,a.T) or np.any(np.diag(a)):
        raise ValueError('Expected finite binary symmetric zero-diagonal adjacency')
    if np.any(a>adjacency(initial,n)):raise ValueError('Retained edge absent from initial route')
    return a.astype(np.int8)

def solve(data,initial,partial,contract,deadline,search_s=3.):
    started=time.perf_counter();n=len(data['demands']);a=constraints_input(initial,partial,n)
    initial_report=evaluate(data,initial,contract,deadline=deadline)
    if not initial_report['hard']:raise ValueError('Unqualified own incumbent')
    if any(max(data['ready_time'][i],data['reveal_time'][i])>1e-9 for i in range(1,n)):
        raise ValueError('Only currently ready pre-dispatch engineering snapshots supported')
    d=matrices(data);v=contract.vehicles;scale=1000000
    manager=pywrapcp.RoutingIndexManager(n,v,0);routing=pywrapcp.RoutingModel(manager);solver=routing.solver()
    distance=routing.RegisterTransitCallback(lambda i,j:int(round(d[manager.IndexToNode(i),manager.IndexToNode(j)]*scale)))
    routing.SetArcCostEvaluatorOfAllVehicles(distance)
    transit=routing.RegisterTransitCallback(lambda i,j:int(math.ceil((d[manager.IndexToNode(i),manager.IndexToNode(j)]/contract.speed_kmph+data['service_time'][manager.IndexToNode(i)])*scale)))
    horizon=int(math.floor(contract.horizon_h*scale));routing.AddDimension(transit,horizon,horizon,False,'Time');td=routing.GetDimensionOrDie('Time')
    for i in range(1,n):td.CumulVar(manager.NodeToIndex(i)).SetRange(int(math.ceil(data['tw_start'][i]*scale)),int(math.floor(data['tw_end'][i]*scale)))
    for k in range(v):td.CumulVar(routing.Start(k)).SetValue(int(math.ceil(contract.loading_service_h*scale)))
    demand=routing.RegisterTransitCallback(lambda i,j:int(math.ceil(data['demands'][manager.IndexToNode(j)]*1000)))
    routing.AddDimensionWithVehicleCapacity(demand,0,[int(math.floor(contract.capacity*1000))]*v,True,'Capacity')
    retained=[]
    for i in range(n):
        for j in range(i+1,n):
            if not a[i,j]:continue
            retained.append([i,j]);jj=manager.NodeToIndex(j)
            if i:
                ii=manager.NodeToIndex(i)
                solver.Add(solver.IsEqualCstVar(routing.NextVar(ii),jj)+solver.IsEqualCstVar(routing.NextVar(jj),ii)==1)
            else:
                terms=[solver.IsEqualCstVar(routing.NextVar(routing.Start(k)),jj) for k in range(v)]
                terms +=[solver.IsEqualCstVar(routing.NextVar(jj),routing.End(k)) for k in range(v)]
                solver.Add(solver.Sum(terms)>=1)
    # Endpoints are any-vehicle undirected depot edges, matching the model's
    # symmetric partial adjacency. We do not silently pin vehicle/orientation.
    best=copy.deepcopy(initial);best_report=initial_report;records=[];callback_errors=[]
    def extract(next_value):
        routes=[]
        for k in range(v):
            idx=routing.Start(k);route=[]
            while not routing.IsEnd(idx):
                node=manager.IndexToNode(idx)
                if node:route.append(node)
                idx=next_value(routing.NextVar(idx))
            if route:routes.append(route)
        return routes
    def consider(candidate,origin):
        nonlocal best,best_report
        report=evaluate(data,candidate,contract,deadline=deadline)
        preserved=bool(np.all(adjacency(candidate,n)[a>0]>0))
        accept=preserved and report['hard'] and report['cost']<best_report['cost']-1e-12
        records.append({'routes':candidate,'report':report,'retained_edges_preserved':preserved,'accepted':bool(accept),'origin':origin})
        if accept:best,best_report=copy.deepcopy(candidate),report
    def callback():
        try:
            if time.perf_counter()>=deadline-.2:routing.solver().FinishCurrentSearch();return
            consider(extract(lambda var:var.Value()),'at_solution')
        except Exception as exc:
            callback_errors.append(type(exc).__name__+':'+str(exc));routing.solver().FinishCurrentSearch()
    routing.AddAtSolutionCallback(callback)
    params=pywrapcp.DefaultRoutingSearchParameters();params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    remaining=min(search_s,deadline-time.perf_counter()-.35)
    if remaining<=0:raise TimeoutError('No certification/transfer reserve')
    params.time_limit.FromMilliseconds(max(1,int(remaining*1000)));routing.CloseModelWithParameters(params)
    seed=copy.deepcopy(initial)+[[] for _ in range(v-len(initial))]
    assignment=routing.ReadAssignmentFromRoutes(seed,True)
    if assignment is None:raise ValueError('Initial route rejected by conservative integer model; no filtering')
    solution=routing.SolveFromAssignmentWithParameters(assignment,params)
    if callback_errors:raise RuntimeError('C0 callback errors: '+str(callback_errors))
    if solution is not None and time.perf_counter()<deadline-.1:consider(extract(lambda var:solution.Value(var)),'returned_assignment')
    if time.perf_counter()>=deadline:raise TimeoutError('Certified teacher too late')
    return {'routes':best,'report':best_report,'initial_report':initial_report,'partial_adjacency':a.tolist(),'retained_edges':retained,
      'all_candidates':records,'callback_errors':callback_errors,'method':'undirected_retained_edge_GLS','proposal_objective':'distance',
      'selection_objective':'full_delivery_C0_cost','optimality_proven':False,'wall_s':time.perf_counter()-started,'research_training_qualified':False,'stable_MaskCO_gain':False}
