"""Warmed full-model worker; public packets only; killed at shared deadline.

Setup is declared explicitly before the day. No dynamically observed graph is
warmed inside this exclusion: missing registered shape falls back to incumbent.
"""
import argparse
import base64
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading
import time
import masked_fleet as mf
import public_fleet as pf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wire(value):return base64.b64encode(pickle.dumps(value,protocol=4)).decode('ascii')
def unwire(value):return pickle.loads(base64.b64decode(value))

def actual_sources(scripts,upstream):
    roots=[Path(scripts).resolve(),Path(upstream).resolve(),Path(__file__).resolve().parents[1]]
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
    return {str(p):sha(p) for p in sorted(paths) if any(p.is_relative_to(r) for r in roots)}

def serve(a):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['JAX_PLATFORM_NAME']=a.platform
    mf.bootstrap(a.scripts)
    from delivery_maskco import MaskCORouteProvider
    from flax import nnx
    import numpy as np
    shapes=sorted(set(map(int,a.shapes.split(','))))
    if any(n<3 for n in shapes):raise ValueError('registered graph shapes >=3')
    provider=MaskCORouteProvider(a.checkpoint,a.upstream,random_seed=42 if a.mode=='random' else None)
    for n in shapes:provider.prewarm(n,a.capacity)
    h=hashlib.sha256();leaves=0
    for key,value in nnx.state(provider.model,nnx.Param).flat_state():
        arr=np.asarray(value.value);h.update(str(key).encode());h.update(arr.tobytes());leaves+=1
    provider.identity.update(actual_param_hash=h.hexdigest(),actual_param_leaves=leaves)
    # Import the real certificate now so its dependency closure is sealed too.
    from coldchain_evaluator_a1 import evaluate_trace_a1
    from run_exp_energy_c0 import certify_plan
    manifest=actual_sources(a.scripts,a.upstream);checkpoint=sha(a.checkpoint)
    ready=dict(kind='ready',provider_identity=provider.identity,source_start=manifest,
               checkpoint_sha256=checkpoint,registered_shapes=shapes)
    print(json.dumps(ready),flush=True)
    for line in sys.stdin:
        request=unwire(json.loads(line)['request'])
        packet=request['packet'];deadline=request['deadline']
        if isinstance(deadline,bool) or not isinstance(deadline,Real) or not math.isfinite(deadline):raise ValueError('finite deadline')
        try:
            n=len(mf.topology(packet)[2])+1
            if n not in shapes:raise ValueError('unwarmed graph shape; no dynamic setup exclusion')
            if any(sha(p)!=v for p,v in manifest.items()) or sha(a.checkpoint)!=checkpoint:raise ValueError('actual source/weight changed')
            if time.perf_counter()>deadline:raise TimeoutError('identity validation exceeded budget')
            selected,info=mf.propose(provider,packet,a.scripts,mode=a.mode,deadline=deadline)
            cert,_=pf.certificate(packet,selected)
            end={p:sha(p) for p in manifest}
            checkpoint_end=sha(a.checkpoint)
            qualified=info.get('complete') is True and cert['qualified'] is True and end==manifest and checkpoint_end==checkpoint and time.perf_counter()<=deadline
            result=dict(qualified=qualified,selected_plan=selected,packet_sha=pf.digest(packet),vehicle_ids=mf.topology(packet)[0],
                        deadline_monotonic=deadline,proposal=info,selected_certificate=cert,source_end=end,
                        checkpoint_sha256=checkpoint,checkpoint_end_sha256=checkpoint_end,provider_identity=provider.identity,fallback=not qualified)
        except (ValueError,TimeoutError) as exc:
            result=dict(qualified=False,fallback=True,selected_plan=packet['plan'],reason=str(exc),deadline_monotonic=deadline)
        print(json.dumps(dict(kind='response',result=wire(result))),flush=True)

class ResidentActor:
    def __init__(self,scripts,upstream,checkpoint,mode,shapes,capacity=50.,platform='cpu',setup_timeout=120.):
        if mode not in mf.MODES:raise ValueError('registered mode')
        if not shapes or any(type(n) is not int or n<3 for n in shapes):raise ValueError('explicit registered shapes')
        self.checkpoint=Path(checkpoint);self.lines=queue.Queue();self.stopped=False
        env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--scripts',str(scripts),'--upstream',str(upstream),
                 '--checkpoint',str(checkpoint),'--mode',mode,'--shapes',','.join(map(str,shapes)),'--capacity',str(capacity),'--platform',platform]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=env)
        self.stderr=[]
        def reader():
            for line in self.process.stdout:
                try:self.lines.put(json.loads(line))
                except json.JSONDecodeError:self.stderr.append(line.strip())
            self.lines.put(dict(kind='closed'))
        def errors():
            for line in self.process.stderr:self.stderr.append(line.strip())
        threading.Thread(target=reader,daemon=True).start();threading.Thread(target=errors,daemon=True).start()
        try:self.ready=self.lines.get(timeout=setup_timeout)
        except queue.Empty:self.close();raise TimeoutError('fixed model setup failed')
        if self.ready.get('kind')!='ready':self.close();raise RuntimeError('worker not ready: '+str(self.stderr[-3:]))
        if self.ready['checkpoint_sha256']!=sha(self.checkpoint):self.close();raise ValueError('loaded checkpoint differs')

    def select(self,packet,started,limit=10.):
        if isinstance(started,bool) or not isinstance(started,Real) or not math.isfinite(started) or started>time.perf_counter():raise ValueError('finite elapsed monotonic start')
        if isinstance(limit,bool) or not isinstance(limit,Real) or not math.isfinite(limit) or not 0<limit<=10:raise ValueError('finite cap <=10')
        deadline=started+limit
        fallback=dict(qualified=False,fallback=True,selected_plan=packet['plan'],deadline_monotonic=deadline,timeout=False)
        if self.stopped or time.perf_counter()>=deadline:return fallback
        self.process.stdin.write(json.dumps(dict(request=wire(dict(packet=packet,deadline=deadline))))+'\n');self.process.stdin.flush()
        try:response=self.lines.get(timeout=max(0.,deadline-time.perf_counter()))
        except queue.Empty:
            self.close();fallback.update(timeout=True,reason='worker killed at shared deadline');return fallback
        if response.get('kind')!='response':self.close();fallback['reason']='worker ended';return fallback
        result=unwire(response['result'])
        identity=(result.get('source_end')==self.ready['source_start'] and result.get('checkpoint_sha256')==self.ready['checkpoint_sha256']
                  and result.get('checkpoint_end_sha256')==self.ready['checkpoint_sha256']
                  and result.get('provider_identity')==self.ready['provider_identity'] and result.get('packet_sha')==pf.digest(packet))
        if result.get('qualified') is not True or not identity or time.perf_counter()>deadline:return fallback
        return result

    def close(self):
        self.stopped=True
        if hasattr(self,'process') and self.process.poll() is None:
            self.process.kill();self.process.wait(timeout=5)
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()

def live_hook(actor,log,max_interventions=1):
    """Actual env hook; one declared archived engineering intervention by default.

    Formal full-day deployment must explicitly choose max_interventions=None and
    register the cohort. A worker timeout disables the arm, retaining incumbent.
    """
    count=[0]
    def hook(env,i,clock,event,reveal,fleet,traces,served,universe):
        if clock<=0 or (max_interventions is not None and count[0]>=max_interventions):return
        if not any(v.status in ('idle','ready') and v.needs_replan for v in fleet):return
        rp=env.replanner
        if getattr(rp,'_uniform_timing',False) is not True:raise ValueError('uniform admission wrapper required')
        started=(rp._uniform_admission_started if getattr(rp,'_uniform_admission_clock',None)==float(clock) else time.perf_counter())
        try:packet=pf.export_public(env,i,clock,event,reveal,fleet,traces,served,universe)
        except ValueError:return
        if len(pf.family(packet)[1])<2:return
        count[0]+=1;selection=actor.select(packet,started)
        fresh=pf.export_public(env,i,clock,event,reveal,fleet,traces,served,universe)
        if pf.digest(fresh)!=pf.digest(packet):selection['qualified']=False
        installed=pf.install_in_live(rp,fresh,selection)
        log.append(dict(event_id=event,clock=clock,public_packet=packet,selection=selection,installed=installed))
    return hook

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--scripts',required=True)
    p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--mode',choices=mf.MODES,required=True)
    p.add_argument('--shapes',required=True);p.add_argument('--capacity',type=float,default=50.);p.add_argument('--platform',default='cpu')
    serve(p.parse_args())
