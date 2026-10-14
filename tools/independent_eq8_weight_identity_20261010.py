"""Independently rebuild actual P/R from upstream checkpoint, no pilot adapter."""
import base64,json
from pathlib import Path
from _srv import run
code=r'''import os,sys,json,hashlib,copy
from pathlib import Path
os.sched_setaffinity(0,{55})
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
root=Path('/home/hzeng/project/MASKCO-Main');cp=root/'MASKCO_code/ckpts/cvrp100.ckpt'
sys.path.insert(0,str(root/'MASKCO_code'))
from training import load_ckpt
from flax import nnx
import jax,numpy as np
params,_,_,cfg,_,step=load_ckpt(str(cp));cfg=copy.deepcopy(cfg);cfg.dtype='float32'
base=cfg.construct_model();P=nnx.merge(nnx.graphdef(base),params)
cfg.rngs=42;R=cfg.construct_model()
actual={}
for label,model in [('P',P),('R',R)]:
    state=nnx.state(model,nnx.Param);flat=jax.tree_util.tree_leaves(state);h=hashlib.sha256()
    for leaf in flat:
        value=np.ascontiguousarray(leaf);h.update(str(value.shape).encode());h.update(str(value.dtype).encode());h.update(value.tobytes())
    actual[label]={'sha256':h.hexdigest(),'parameter_leaves':len(flat)}
expected=json.loads((root/'C-VRP_Cold-chainVehicleRoutingProblem/results/research_1010_d_eq8_v1/pilot/identity_end.json').read_text())
print('INDEPENDENT_WEIGHT_JSON='+json.dumps({'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'actual_parameters':actual,'parameter_match':actual==expected['model_parameters'],'checkpoint_match':hashlib.sha256(cp.read_bytes()).hexdigest()==expected['checkpoint_sha256'],'method':'direct upstream load_ckpt/construct/merge, no pilot provider or runner import','scientific_data_or_inference_calls':0}))
'''
payload=base64.b64encode(code.encode()).decode()
raw=run("/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c \"import base64;exec(base64.b64decode('"+payload+"'))\"",180)
lines=[line.split('=',1)[1] for line in raw.splitlines() if line.startswith('INDEPENDENT_WEIGHT_JSON=')]
if len(lines)!=1:raise RuntimeError(raw[-3000:])
record=json.loads(lines[0]);out=Path('C-VRP_Cold-chainVehicleRoutingProblem/results/research_20261010/INDEPENDENT_ACTUAL_P_R_WEIGHTS.json')
with out.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record));assert record['parameter_match'] and record['checkpoint_match']
