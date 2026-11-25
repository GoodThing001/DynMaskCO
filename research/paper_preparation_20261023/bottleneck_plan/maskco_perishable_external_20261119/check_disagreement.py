"""A known mixed optimum checks the two-parent component rule and all disagreement mask lengths."""
import numpy as np
from types import SimpleNamespace
from disagreement import cross,dispute_group
def main():
    extra=tuple((i,) for i in range(7,101))
    a=((1,2),(3,),(4,5),(6,))+extra
    b=((1,),(2,3),(4,),(5,6))+extra
    def score(r):return { (1,2):4.,(2,3):1.,(4,5):1.,(5,6):4.}.get(r,0.)
    task=SimpleNamespace(route=lambda r:SimpleNamespace(net_benefit=score(r)))
    c=cross(a,b,task)
    if sum(map(score,c))!=8 or sorted(x for r in c for x in r)!=list(range(1,101)):raise ValueError("Incorrect mixed cover")
    rng=np.random.default_rng(1)
    for left,right,count in ((a,b,6),(a,a,0),((tuple(range(1,101)),),tuple((i,) for i in range(1,101)),100)):
        group,n=dispute_group(left,right,rng)
        if len(set(group))!=40 or n!=count or n==6 and not set(range(1,7))<=set(group):raise ValueError("Incorrect disputed customers")
    print("Component choice and masks passed")
if __name__=="__main__":main()
