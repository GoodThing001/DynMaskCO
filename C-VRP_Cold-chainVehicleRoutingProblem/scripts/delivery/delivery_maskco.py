"""Delivery route masked reconstruction; independent from I3 order generation.

This module is an engineering candidate. A numerical experiment must first seal
its contract, mask rule, adapter score, prewarm policy and common initial routes.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import time
import numpy as np
from delivery_baselines import matrices, route_ok
from delivery_evaluator import evaluate


def partial_adjacency(routes, removed, nodes):
    """Retain original unmasked edges only, never newly created shortcuts."""
    adj=np.zeros((nodes,nodes),np.float32)
    for route in routes:
        chain=[0]+list(route)+[0]
        for a,b in zip(chain,chain[1:]):
            if a!=b and a not in removed and b not in removed:
                adj[a,b]=adj[b,a]=1.
    return adj


class MaskCORouteProvider:
    """Load complete, read-only CVRP encoder AND decoder, or random same architecture."""
    def __init__(self, checkpoint, upstream_root, random_seed=None):
        upstream_root=Path(upstream_root).resolve()
        if not (upstream_root/'models/CVRPModel.py').is_file():
            raise ValueError('Explicit upstream model root is required')
        sys.path.insert(0,str(upstream_root))
        # Loading an upstream checkpoint imports its unused training logger.
        # Keep that optional logging dependency out of inference-only execution.
        logger_stub=False
        try:
            import tensorboardX
        except ImportError:
            import types
            tb=types.ModuleType('tensorboardX')
            class UnusedWriter:
                def __init__(self,*a,**k):
                    raise RuntimeError('Training logger is not part of the route inference adapter')
            tb.SummaryWriter=UnusedWriter
            sys.modules['tensorboardX']=tb
            logger_stub=True
        from training import load_ckpt
        from flax import nnx
        params,_,_,cfg,_,step=load_ckpt(str(checkpoint))
        cfg=deepcopy(cfg)
        cfg.dtype='float32'
        if cfg.encoder_input_dim!=3:
            raise ValueError('This engineering adapter requires the original 3D CVRP weights')
        if random_seed is not None:cfg.rngs=int(random_seed)
        model=cfg.construct_model()
        self.model=model if random_seed is not None else nnx.merge(nnx.graphdef(model),params)
        self.encode=nnx.jit(lambda m,x:m.encode(x))
        self.decode=nnx.jit(lambda m,h,t,a:m.decode(h,t,a,target='logit'))
        self.max_decode_s=0.
        self.identity={'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'checkpoint_step':int(step),'random_same_architecture':random_seed is not None,
            'random_seed':random_seed,'upstream_root':str(upstream_root),
            'both_encoder_decoder_used':True,'training_steps':0,'unused_logger_import_stub':logger_stub}

    def features(self, instance, capacity):
        import jax.numpy as jnp
        coords=np.asarray(instance['coords'],np.float32)
        centered=coords-coords.mean(axis=0,keepdims=True)
        scale=max(1e-8,float(np.linalg.norm(centered,axis=1).mean()))
        raw=np.concatenate((centered/scale,np.asarray(instance['demands'],np.float32)[:,None]/capacity),axis=1)
        # All input nodes are revealed and loaded at this pre-dispatch snapshot.
        return self.encode(self.model,jnp.asarray(raw[None]))

    def logits(self, features, adjacency):
        started=time.perf_counter()
        import jax.numpy as jnp
        n=len(adjacency)
        timestep=min(1.,float(adjacency[1:].sum())/(2*max(1,n-1)))
        out=self.decode(self.model,features,jnp.asarray([timestep],dtype=jnp.float32),
            jnp.asarray(adjacency[None],dtype=jnp.float32))
        result=np.asarray(out)[0]  # Blocks until encoder + decoder actually complete.
        valid=~np.eye(n,dtype=bool)
        if result.shape!=(n,n) or not np.all(np.isfinite(result[valid])):
            raise ValueError('Invalid actual MaskCO off-diagonal route logits')
        self.max_decode_s=max(self.max_decode_s,time.perf_counter()-started)
        return result

    def prewarm(self, nodes, capacity):
        """Fixed synthetic shape, no development instance or observed score used."""
        data={'coords':np.stack([np.arange(nodes),np.zeros(nodes)],axis=1),
            'demands':np.r_[0.,np.ones(nodes-1)]}
        h=self.features(data,capacity)
        adjacency=np.zeros((nodes,nodes),np.float32)
        self.logits(h,adjacency)
        compilation_inclusive_decode_s=self.max_decode_s
        # Compilation is excluded common setup by the fixed protocol. Its first
        # call must never become a per-instance runtime reserve. Measure the
        # compiled graph on the same fixed synthetic input, before dev data.
        self.max_decode_s=0.
        for _ in range(3):
            self.logits(h,adjacency)
        self.identity['prewarm_decode_including_compilation_s']=compilation_inclusive_decode_s
        self.identity['postcompile_fixed_synthetic_max_decode_s']=self.max_decode_s
        self.identity['runtime_reserve_excludes_compilation']=True


def improve_masked(instance, initial_routes, contract, provider, time_limit_s=10.,
                   seed=41, wave_h=0., iterative=False, mask_fraction=0.3,
                   guide_weight=1.0, clock=None):
    clock=time.perf_counter if clock is None else clock
    start=clock()
    if not math.isfinite(time_limit_s) or time_limit_s<=0:
        raise ValueError('Invalid total masked reconstruction budget')
    if not 0<mask_fraction<=1 or not math.isfinite(guide_weight) or guide_weight<0:
        raise ValueError('Invalid registered mask/guide parameters')
    deadline=start+time_limit_s
    initial=evaluate(instance,initial_routes,contract,wave_h,deadline,clock)
    if not initial['hard']:raise ValueError('Common initial route must be independently certified')
    best,report=deepcopy(initial_routes),initial
    dist=matrices(instance)
    n=len(dist)
    if n==1:
        return {'routes':[],'report':initial,'wall_s':clock()-start,'timeout':False,
                'cycles':0,'same_initial_assignment':True,'online_gain_established':False}
    features=provider.features(instance,contract.capacity)
    rng=np.random.default_rng(seed)
    distance_scale=max(1e-9,float(dist[~np.eye(n,dtype=bool)].mean()))
    reserve=max(0.05,2*(clock()-start),2*getattr(provider,'max_decode_s',0.))
    cycles=0
    decode_calls=0
    completed_candidates=0
    hard_feasible_candidates=0
    preserved_all=True
    while clock()<deadline-reserve and (iterative or cycles==0):
        removed=set(map(int,rng.choice(np.arange(1,n),size=max(1,int(math.ceil(mask_fraction*(n-1)))),replace=False)))
        adj=partial_adjacency(best,removed,n)
        logits=provider.logits(features,adj)
        decode_calls+=1
        reserve=max(reserve,2*getattr(provider,'max_decode_s',0.))
        if clock()>=deadline-reserve:break
        valid=~np.eye(n,dtype=bool)
        logit_scale=max(1e-8,float(np.std(logits[valid])))
        candidate=[[i for i in route if i not in removed] for route in best]
        candidate += [[] for _ in range(contract.vehicles-len(candidate))]
        remaining=sorted(removed)
        cycles+=1
        while remaining and clock()<deadline-reserve:
            choices=[]
            for i in remaining:
                options=[]
                for k,route in enumerate(candidate):
                    for at in range(len(route)+1):
                        if clock()>=deadline-reserve:break
                        a,b=(route[at-1] if at else 0),(route[at] if at<len(route) else 0)
                        if adj[a,b]>0:continue  # Original retained edges are immutable in this cycle.
                        proposed=route[:at]+[i]+route[at:]
                        if route_ok(proposed,instance,contract,dist,wave_h):
                            delta=dist[a,i]+dist[i,b]-dist[a,b]
                            old=0. if a==b else logits[a,b]
                            gain=(logits[a,i]+logits[i,b]-old)/logit_scale
                            score=delta/distance_scale-guide_weight*gain
                            options.append((score,k,at))
                options.sort()
                if options:
                    regret=options[1][0]-options[0][0] if len(options)>1 else float('inf')
                    choices.append((regret,i,options[0]))
            if not choices or clock()>=deadline-reserve:break
            _,i,(_,k,at)=max(choices,key=lambda row:(row[0],-row[1]))
            candidate[k].insert(at,i)
            remaining.remove(i)
        if remaining:continue
        completed_candidates+=1
        checked=evaluate(instance,candidate,contract,wave_h,deadline,clock)
        restored=partial_adjacency(candidate,set(),n)
        kept=bool(np.all(restored[adj>0]>0))
        preserved_all &= kept
        if not kept:raise RuntimeError('Reconstruction changed an unmasked edge')
        hard_feasible_candidates+=int(checked['hard'])
        if checked['hard'] and checked['cost']<report['cost']:
            best,report=[r for r in candidate if r],checked
    late=clock()>=deadline
    if late:best,report=deepcopy(initial_routes),initial
    return {'routes':best,'report':report,'wall_s':clock()-start,'timeout':late,
        'cycles':cycles,'same_initial_assignment':True,'retained_edges_preserved':preserved_all,
        'decode_calls':decode_calls,'completed_candidates':completed_candidates,
        'hard_feasible_candidates':hard_feasible_candidates,
        'runtime_decode_reserve_s':reserve,
        'method':'maskco_iterative' if iterative else 'maskco_single',
        'mask_fraction':mask_fraction,'guide_weight':guide_weight,
        'selection_objective':'delivery_contract_cost','online_gain_established':False}
