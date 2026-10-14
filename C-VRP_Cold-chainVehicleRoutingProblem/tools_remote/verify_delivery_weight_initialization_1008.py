"""Read-only checkpoint vs its original-config initialization, no development scores."""
from pathlib import Path
from copy import deepcopy
import datetime
import hashlib
import json
import os
import sys
import traceback

def main():
    project=Path(__file__).resolve().parents[1]
    out=project/'results/delivery_weight_audit_1008_v1';out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,{55})
    os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    root=project/'results/delivery_provider_1008_v1/src'
    sys.path.insert(0,str(root/'C-VRP_Cold-chainVehicleRoutingProblem/scripts/delivery'))
    try:
        from delivery_maskco import MaskCORouteProvider
        p=MaskCORouteProvider(project.parent/'MASKCO_code/ckpts/cvrp100.ckpt',root/'MASKCO_code')
        from training import load_ckpt
        from flax import nnx
        import jax
        import numpy as np
        _,_,_,config,_,_=load_ckpt(str(project.parent/'MASKCO_code/ckpts/cvrp100.ckpt'))
        config=deepcopy(config);config.dtype='float32';fresh=config.construct_model()
        loaded=nnx.state(p.model,nnx.Param);initial=nnx.state(fresh,nnx.Param)
        assert jax.tree_util.tree_structure(loaded)==jax.tree_util.tree_structure(initial)
        la=jax.tree_util.tree_leaves(loaded);ia=jax.tree_util.tree_leaves(initial)
        changed=sum(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(la,ia))
        result={'checkpoint_sha256':p.identity['checkpoint_sha256'],'checkpoint_step_metadata':p.identity['checkpoint_step'],
            'initialization_seed_from_original_config':int(config.rngs),'parameter_arrays':len(la),
            'parameter_arrays_not_equal_original_initialization':changed,'checkpoint_is_not_original_initialization':changed>0,
            'source_root':str(root),'cpu_affinity':[55],'read_only':True,
            'online_gain_established':False,'time':str(datetime.datetime.now())}
        (out/'report.json').write_text(json.dumps(result,indent=2))
        (out/'run.exit').write_text('0')
        print(json.dumps(result),flush=True)
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2))
        (out/'run.exit').write_text('1');raise

if __name__=='__main__':main()
