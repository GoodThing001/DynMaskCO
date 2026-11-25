"""Economically evaluate edits that put two currently separated masked customers together."""
from functools import lru_cache

def adjacency_options(routes, customer_route, visit, partner):
    a,b=customer_route[visit],customer_route[partner]
    first,second=routes[a],routes[b]
    u,v=first.index(visit),second.index(partner)
    if a==b:
        if abs(u-v)<2:return
        for moving,anchor in ((visit,partner),(partner,visit)):
            without=tuple(n for n in first if n!=moving)
            k=without.index(anchor)
            for pos in (k,k+1):
                yield (a,), (without[:pos]+(moving,)+without[pos:],)
        lo,hi=sorted((u,v))
        for l,r in ((lo,hi),(lo+1,hi+1)):
            yield (a,), (first[:l]+first[l:r][::-1]+first[r:],)
        return
    for insertion in (v,v+1):
        yield (a,b),(first[:u]+first[u+1:],second[:insertion]+(visit,)+second[insertion:])
    for insertion in (u,u+1):
        yield (a,b),(first[:insertion]+(partner,)+first[insertion:],second[:v]+second[v+1:])
    for neighbour in (v-1,v+1):
        if 0<=neighbour<len(second):
            yield (a,b),(first[:u]+(second[neighbour],)+first[u+1:],second[:neighbour]+(visit,)+second[neighbour+1:])
    for neighbour in (u-1,u+1):
        if 0<=neighbour<len(first):
            yield (a,b),(first[:neighbour]+(partner,)+first[neighbour+1:],second[:v]+(first[neighbour],)+second[v+1:])
    yield (a,b),(first[:u+1]+second[v:],second[:v]+first[u+1:])
    yield (a,b),(second[:v+1]+first[u:],first[:u]+second[v+1:])

def disjoint_gain(moves):
    occupied=set()
    gain=0.
    count=0
    for value,ids in sorted(moves,reverse=True):
        if value>1e-7 and not occupied.intersection(ids):
            gain+=value
            occupied.update(ids)
            count+=1
    return gain,count

def evaluate_pairs(task,routes):
    routes=tuple(tuple(r) for r in routes)
    location={c:n for n,r in enumerate(routes) for c in r}
    old=[task.route(r).net_benefit for r in routes]
    @lru_cache(maxsize=None)
    def gain(i,j):
        best=0.
        ids=tuple(sorted({location[i],location[j]}))
        for before,after in adjacency_options(routes,location,i,j):
            score=0.
            for route in after:
                if route:
                    result=task.route(route)
                    if not result.feasible:break
                    score+=result.net_benefit
            else:
                best=max(best,score-sum(old[a] for a in before))
        return best,ids
    return gain
