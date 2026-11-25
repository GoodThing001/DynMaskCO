"""Check customer conservation and proposed edge formation, including singletons."""
from edge_moves import adjacency_options,disjoint_gain
def check():
    routes=((1,2,3,4),(5,6,7),(8,),(9,))
    loc={i:n for n,r in enumerate(routes) for i in r}
    total=0
    for i in loc:
        for j in loc:
            if i>=j:continue
            for before,after in adjacency_options(routes,loc,i,j):
                expected=sorted(k for r in before for k in routes[r])
                actual=sorted(k for r in after for k in r)
                if actual!=expected or not any({u,v}=={i,j} for r in after for u,v in zip(r,r[1:])):
                    raise ValueError((i,j,before,after))
                total+=1
    if disjoint_gain([(2.,(0,1)),(3.,(1,)),(1.,(2,3)),(0.,(4,))])!=(4.,2):raise ValueError("Overlapping move")
    print(total,"customer-conserving forced-edge edits")
if __name__=="__main__":check()
