
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import hashlib,importlib.metadata,json,sys
from pathlib import Path
import psutil
psutil.Process().cpu_affinity([15])
root=Path(__file__).parent
input_path=root/'AUDIT_INPUT.json'
data=json.loads(input_path.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
actual={};missing=[]
for s in data['source_start']:
    p=Path(s)
    if not p.is_file():missing.append(s)
    else:actual[s]=sha(p)
versions={n:importlib.metadata.version(n) for n in data['packages']}
import jax,jax.numpy as jnp,numpy as np
v,g=jax.value_and_grad(lambda x:jnp.dot(x,x))(jnp.asarray([1.,2.],dtype=jnp.float32))
print(json.dumps({'actual_hashes':actual,'missing':missing,'packages':versions,'executable':sys.executable,
    'python':sys.version,'cpu':psutil.Process().cpu_affinity(),'devices':[str(d) for d in jax.devices()],
    'independent_scalar_value':float(v),'independent_scalar_gradient':np.asarray(g).tolist(),
    'input_sha256':sha(input_path),'input_end_sha256':sha(input_path),'audit_program_sha256':sha(Path(__file__))},allow_nan=False))
