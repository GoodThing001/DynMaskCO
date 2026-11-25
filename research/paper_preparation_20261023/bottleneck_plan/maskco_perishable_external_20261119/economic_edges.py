"""Directional savings of combining two singleton deliveries and conditional masked-edge ranks."""
import numpy as np
ALPHAS=(0.,.125,.25,.5,1.)
def economic_prior(task):
    single=[0.]+[task.route((i,)).net_benefit for i in range(1,101)]
    s=np.full((101,101),np.nan)
    for i in range(1,101):
        for j in range(1,101):
            if i==j:continue
            route=task.route((i,j))
            if route.feasible:s[i,j]=route.net_benefit-single[i]-single[j]
    good=s[np.isfinite(s)]
    if not len(good):return np.zeros((101,101)),np.zeros((101,101),dtype=bool)
    allowed=np.isfinite(s)
    q1,q3=np.quantile(good,[.25,.75]);scale=max(1.,q3-q1)
    s=np.where(allowed,s,min(good)-4*(np.ptp(good)+1))
    s[0,:]=0.;s[:,0]=0.
    return s/scale,allowed

def mask_residual(prior,allowed,heat):
    def ranks(x):
        x=x.copy();x[~allowed]=-np.inf
        return np.argsort(np.argsort(-x,axis=1,kind="stable"),axis=1,kind="stable")+1
    p,h=ranks(prior),ranks(heat)
    return np.where(allowed,np.clip(np.log(p/h),-3,3),0.)
