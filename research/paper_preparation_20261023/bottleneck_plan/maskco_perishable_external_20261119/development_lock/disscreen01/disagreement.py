"""Recover independent route components and focus a mask on disagreement between feasible parents."""
import numpy as np

def neighbours(routes):
    return {i:tuple(sorted((a,b))) for r in routes for a,i,b in zip((0,)+r,r,r[1:]+(0,))}

def dispute_group(a,b,rng):
    left,right=neighbours(a),neighbours(b)
    disputed=np.array([i for i in range(1,101) if left[i]!=right[i]],dtype=np.int32)
    if len(disputed)>=40:
        return rng.choice(disputed,40,replace=False),len(disputed)
    rest=np.array([i for i in range(1,101) if left[i]==right[i]],dtype=np.int32)
    group=np.concatenate([disputed,rng.choice(rest,40-len(disputed),replace=False)])
    return rng.permutation(group),len(disputed)

def cross(a,b,task):
    labels=[{} for _ in range(2)]
    for side,routes in enumerate((a,b)):
        for n,route in enumerate(routes):
            for i in route:labels[side][i]=(side,n)
    graph={}
    for i in range(1,101):
        left,right=labels[0][i],labels[1][i]
        graph.setdefault(left,set()).add(right)
        graph.setdefault(right,set()).add(left)
    solved=set();out=[]
    for start in graph:
        if start in solved:continue
        pending=[start]
        sides=[[],[]]
        while pending:
            node=pending.pop()
            if node in solved:continue
            solved.add(node)
            k,n=node
            sides[k].append((a,b)[k][n])
            pending.extend(graph[node]-solved)
        out.extend(max(sides,key=lambda rs:sum(task.route(r).net_benefit for r in rs)))
    return tuple(out)
