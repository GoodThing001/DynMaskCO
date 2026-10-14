"""Small, deterministic JAX arithmetic diagnostic; never opens original 34 inputs."""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8'))['fixed_shapes']


def values(shape, offset):
    n=int(np.prod(shape))
    return (np.sin(np.arange(offset+1,offset+n+1,dtype=np.float64))*0.07).astype(np.float32).reshape(shape)


def stages(h,a,route,prefix,suffix,scalars,k1,b1,k2,b2,barrier):
    b = jax.lax.optimization_barrier if barrier else lambda x:x
    p0=b(h[a[:,0]]);p1=b(h[a[:,3]]);p2=b(h[a[:,4]])
    r=b(route@h);p=b(prefix@h);s=b(suffix@h)
    x=b(jnp.concatenate([p0,p1,p2,r,p,s,scalars],axis=-1))
    z=b(b(x@k1)+b1)
    t=b(jnp.tanh(z))
    y=b(b(t@k2)+b2)[:,0]
    return r,p,s,x,z,t,y


def main():
    n,d,m,q,z=[SHAPE[k] for k in ('nodes','embedding','actions','scalars','hidden')]
    h=values((n,d),0)
    a=np.zeros((m,5),dtype=np.int32)
    for i in range(m):
        a[i]=[i%n,(i+1)%n,(i+2)%n,(i+3)%n,(i+4)%n]
    mask=np.zeros((m,n),dtype=np.float32)
    for i in range(m):mask[i,i%n]=1.0
    p=np.zeros((m,n),dtype=np.float32);s=np.zeros((m,n),dtype=np.float32)
    for i in range(m):p[i,(i+1)%n]=1.0;s[i,(i+2)%n]=1.0
    sc=values((m,q),500)
    k1=values((d*6+q,z),1000);b1=values((z,),5000)
    k2=values((z,1),6000);b2=values((1,),7000)
    args=tuple(jnp.asarray(v) for v in (h,a,mask,p,s,sc,k1,b1,k2,b2))
    eager=tuple(np.asarray(x) for x in stages(*args,False))
    compiled=tuple(np.asarray(x) for x in jax.jit(lambda *xs:stages(*xs,True))(*args))
    names=('route','prefix','suffix','concat','linear1_bias','tanh','linear2_bias')
    rows=[]
    for name,e,c in zip(names,eager,compiled):
        rows.append({'name':name,'shape':list(e.shape),'bitwise_equal':bool(np.array_equal(e.view(np.uint32),c.view(np.uint32))),
                     'different_float32_elements':int(np.count_nonzero(e.view(np.uint32)!=c.view(np.uint32))),
                     'max_abs_difference':float(np.max(np.abs(e.astype(np.float64)-c.astype(np.float64))))})
    out={'stage':'small_synthetic_head_reduction_diagnostic','jax_version':jax.__version__,
         'numpy_version':np.__version__,'backend':jax.default_backend(),
         'rows':rows,'first_differing_stage':next((r['name'] for r in rows if not r['bitwise_equal']),None),
         'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('REGISTRATION.json','run.py')},
         'original_34_cases_rerun':False,'original_compiled_head_strict_gate_pass':False,
         'full_backbone_loaded':False,'full_request_qualified':False,'stable_online_gain':False}
    with (HERE/'RESULT.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'first_differing_stage':out['first_differing_stage'],'final_differing_elements':rows[-1]['different_float32_elements']}))


if __name__=='__main__':main()
