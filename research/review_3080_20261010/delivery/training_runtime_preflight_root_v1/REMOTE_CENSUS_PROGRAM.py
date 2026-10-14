
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib, importlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import psutil
p=psutil.Process()
if 15 not in p.cpu_affinity(): raise RuntimeError('Registered former training CPU15 unavailable')
p.cpu_affinity([15])
available=psutil.virtual_memory().available
if available<8*1024**3: raise RuntimeError('8GiB CPU preflight resource barrier')
owned=[]
for q in psutil.process_iter(['pid','ppid','cmdline']):
    try:
        cmd=q.info['cmdline'] or []
        if q.pid!=p.pid and any('review_3080_20261010' in s and not s.startswith('exec(') for s in cmd):
            if any(Path(s).suffix=='.py' and ('train_' in Path(s).name or 'supervise' in Path(s).name) for s in cmd):
                owned.append({'pid':q.pid,'ppid':q.ppid(),'script_names':[Path(s).name for s in cmd if Path(s).suffix=='.py']})
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
if owned: raise RuntimeError('Owned training/supervisor still present: '+repr(owned))
import numpy as np, jax, jax.numpy as jnp, flax, optax, chex, scipy, ml_dtypes, etils
from flax import nnx
from ortools.constraint_solver import pywrapcp
import google.protobuf, ortools
bundle=Path(r'C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\remote_finite_research_root_v2')
upstream=bundle/'upstream'
if not upstream.is_dir(): raise RuntimeError('Original isolated upstream closure not found')
sys.path.insert(0,str(upstream))
# Reuse the existing trusted inspection adapter's explicit unused logging shim;
# this is import-only, never read/unpickle a checkpoint or construct a full model.
sys.path.insert(0,str(bundle/'actual_checkpoint_engineering_v1'))
from inspect_checkpoint import load_trusted
# load_trusted is not called. The shim is instead explicitly recorded here.
import types
if 'tensorboardX' not in sys.modules:
    try: import tensorboardX
    except ModuleNotFoundError:
        tb=types.ModuleType('tensorboardX')
        class UnusedWriter:
            def __init__(self,*args,**kwargs): raise RuntimeError('Logging unavailable in census')
        tb.SummaryWriter=UnusedWriter;sys.modules['tensorboardX']=tb
from helpers import sol2adj
from modules.functional import coord_normalize
from models.CVRPModel import CVRPModel
pywrapcp.DefaultRoutingSearchParameters()
packages=('numpy','jax','jaxlib','flax','optax','chex','scipy','ml_dtypes','etils','orbax-checkpoint','ortools','protobuf','psutil')
versions={n:importlib.metadata.version(n) for n in packages}
machine=platform.platform()
def synthetic():
    # Only two scalars; a runtime primitive check, not MaskCO learning evidence.
    value, gradient=jax.jit(jax.value_and_grad(lambda x:jnp.sum(x*x)))(jnp.asarray([1.,2.],dtype=jnp.float32))
    np.asarray(gradient);float(value)
    if float(value)!=5. or np.asarray(gradient).tolist()!=[2.,4.]: raise RuntimeError('JAX algebra check')
    adjacency=np.asarray(sol2adj(jnp.asarray([[0,1,2,0]],dtype=jnp.int32),dtype=jnp.float32,num_nodes=3,is_cvrp=True))
    coordinates=np.asarray(coord_normalize(jnp.asarray([[[0.,0.],[1.,0.],[0.,1.]]],dtype=jnp.float32)))
    if adjacency.shape!=(1,3,3) or not np.isfinite(coordinates).all(): raise RuntimeError('Actual helper interface')
    layer=nnx.Linear(2,1,rngs=nnx.Rngs(0));graph,state=nnx.split(layer)
    output=np.asarray(nnx.merge(graph,state)(jnp.ones((1,2),dtype=jnp.float32)))
    if output.shape!=(1,1) or not np.isfinite(output).all():raise RuntimeError('NNX split/merge runtime')
    return {'scalar_value':float(value),'scalar_gradient':np.asarray(gradient).tolist(),'helper_adjacency':adjacency.tolist(),
            'nnx_tiny_output':output.tolist(),'backend_devices':[str(d) for d in jax.devices()]}
first=synthetic()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def closure():
    files={Path(sys.executable).resolve()}
    for m in list(sys.modules.values()):
        for attr in ('__file__','__cached__'):
            name=getattr(m,attr,None)
            if name and Path(name).is_file():files.add(Path(name).resolve())
    modules=('numpy','jax','jaxlib','flax','optax','chex','scipy','ml_dtypes','etils','orbax.checkpoint','ortools','google.protobuf','psutil')
    for name in modules:
        m=importlib.import_module(name)
        roots=[Path(m.__file__).parent] if getattr(m,'__file__',None) else list(map(Path,m.__path__))
        for root in roots:
            files.update(x.resolve() for x in root.rglob('*') if x.is_file())
    for root in (Path(np.__file__).parent.parent/'numpy.libs',Path(np.__file__).parent.parent/'scipy.libs',
                 Path(np.__file__).parent.parent/'ortools.libs',Path(sys.base_prefix)/'DLLs'):
        if root.exists():files.update(x.resolve() for x in root.rglob('*') if x.is_file())
    files.update(x.resolve() for x in Path(sys.base_prefix).glob('python*.dll'))
    for entry in p.memory_maps():
        x=Path(entry.path)
        if x.is_file():files.add(x.resolve())
    return {str(x):sha(x) for x in sorted(files)}
# Prime every enumerated namespace before the first identity capture.
importlib.import_module('orbax.checkpoint')
start=closure();second=synthetic();end=closure()
changed=[x for x,h in start.items() if end.get(x)!=h]
new=sorted(set(end)-set(start))
print(json.dumps({'server_time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'python':sys.version,
 'executable':sys.executable,'platform':machine,'packages':versions,'actual_cpu_affinity':p.cpu_affinity(),
 'available_ram_start':available,'owned_training_processes':owned,'source_start':start,'source_end':end,
 'changed_or_missing_files':changed,'new_loaded_files':new,'synthetic_before':first,'synthetic_after':second,
 'pure_import_cpu_runtime_qualified':not changed and not new,
 'full_model_gradient_qualified':False,'scientific_training_start_end_qualified':False,
 'new_scientific_data_generated':False,'model_training_performed':False,'stable_online_gain':False,
 'scope':'Imported training runtime, package trees and loaded DLL census plus tiny CPU primitives only; no full-model or future-run qualification'},allow_nan=False))
