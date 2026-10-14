"""Isolated structural insertion guard; no production or physical qualification."""
def edges(routes):
    return {tuple(sorted((a,b))) for r in routes for a,b in zip([0]+r,r+[0]) if a!=b}
def permits(routes,customer,vehicle,position,required):
    if type(customer) is not int or customer<=0 or any(customer in r for r in routes):raise ValueError('Customer must be new to skeleton')
    if type(vehicle) is not int or not 0<=vehicle<len(routes):raise ValueError('Vehicle slot')
    if type(position) is not int or not 0<=position<=len(routes[vehicle]):raise ValueError('Insertion position')
    if any(type(n) is not int or n<=0 for r in routes for n in r) or len({n for r in routes for n in r})!=sum(map(len,routes)):raise ValueError('Skeleton coverage')
    if any(len(e)!=2 or e[0]==e[1] for e in required):raise ValueError('Required edges')
    trial=[r[:] for r in routes];trial[vehicle].insert(position,customer)
    return set(required)<=edges(trial)
