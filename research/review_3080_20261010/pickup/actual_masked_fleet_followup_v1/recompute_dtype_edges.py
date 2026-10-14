"""Diagnostic recomputation: float64 norms and usable off-diagonal logits.

Preserves original model outputs/result. Float32 norm of diagonal hard-mask
sentinels can overflow and spuriously give relative_l2=0; never call equivalence.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def run(path):
    path=Path(path);z=np.load(path/'same_input_outputs.npz');a=z['logits_adapter'].astype(np.float64);b=z['logits_original'].astype(np.float64)
    mask=~np.eye(a.shape[0],dtype=bool)
    aa=a[mask];bb=b[mask];delta=aa-bb
    r=dict(raw_outputs_sha256=hashlib.sha256((path/'same_input_outputs.npz').read_bytes()).hexdigest(),
        actual_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='all nonself-edge logits; float64 arithmetic',
        edge_count=int(mask.sum()),max_abs=float(np.max(np.abs(delta))),relative_l2=float(np.linalg.norm(delta)/(np.linalg.norm(bb)+1e-300)),
        adapter_min=float(aa.min()),adapter_max=float(aa.max()),original_min=float(bb.min()),original_max=float(bb.max()),
        original_allmatrix_float32_norm_relative_zero_not_equivalence=True,numerical_equivalence_claim=False)
    (path/'VALID_EDGE_RECOMPUTATION.json').write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8');return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();print(json.dumps(run(a.directory)))
