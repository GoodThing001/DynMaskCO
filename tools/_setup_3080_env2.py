"""Finish the 3080 `MASKCO_env` install where the default resolver cannot.

Two facts discovered by probing, both of which constrain what this box can do:

1. `pip install flax==0.10.4` fails on this host with
   OSError [Errno 2] ... orbax/checkpoint/experimental/v1/_src/testing/compatibility/.../
   ocdbt.process_0/d/0653e15f... , and pip's own hint names the cause: Windows Long Path support
   is disabled (LongPathsEnabled = 0). The absolute path of orbax's bundled test fixtures exceeds
   MAX_PATH under this environment's site-packages. Enabling long paths is a machine-wide registry
   change on a SHARED host, so this script does not do it; instead it installs `flax` with
   --no-deps and supplies only the dependencies `flax.nnx` actually needs. Checkpointing via
   orbax is therefore NOT available here, which is fine for training/optimisation work but must be
   recorded.

2. `pip install jax-cuda12-plugin` has NO Windows distribution at all ("from versions: none"),
   so JAX on this host is CPU-only regardless of the RTX 3080 Ti. The GPU is real (12 GB,
   driver 595.79) but not reachable from JAX; any claim that this box accelerates JAX would be
   false.

Usage:
    python tools/_setup_3080_env2.py [--config .vscode/sftp.3080.json]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402

PY = r'D:\anaconda\envs\MASKCO_env\python.exe'
NNX_DEPS = ['msgpack', 'rich', 'treescope', 'typing_extensions', 'PyYAML']


def main():
    args = sys.argv[1:]
    cfg = '.vscode/sftp.3080.json'
    if '--config' in args:
        cfg = args[args.index('--config') + 1]
    out = {}

    def ps(cmd, timeout=1800):
        return run(cmd, timeout=timeout, config=cfg)

    out['install_deps'] = ps('& "%s" -m pip install --no-input %s'
                             % (PY, ' '.join(NNX_DEPS)))
    out['install_optax'] = ps('& "%s" -m pip install --no-input optax==0.2.4' % PY)
    out['install_flax_nodeps'] = ps('& "%s" -m pip install --no-input --no-deps flax==0.10.4'
                                    % PY)
    out['installed_versions'] = ps(
        'foreach($m in @("jax","jaxlib","flax","optax","numpy","scipy","chex","ml_dtypes",'
        '"msgpack","orbax-checkpoint")){'
        ' $v = & "%s" -c "import importlib,sys;'
        ' m=importlib.import_module(sys.argv[1].replace(chr(45),chr(95)));'
        ' print(getattr(m,chr(95)*2+chr(118)+chr(101)+chr(114)+chr(115)+chr(105)+chr(111)'
        '+chr(110)+chr(95)*2,chr(63)))" $m 2>$null;'
        ' if($LASTEXITCODE -ne 0){ $v = "MISSING" };'
        ' Write-Output ("{0,-18} {1}" -f $m, $v) }' % PY)
    out['verify'] = ps(
        '& "%s" -c "'
        'import jax,flax,optax,numpy;'
        'from flax import nnx;'
        'import jax.numpy as jnp;'
        'print(\'jax\',jax.__version__,\'flax\',flax.__version__,\'optax\',optax.__version__,'
        '\'numpy\',numpy.__version__);'
        'print(\'backend\',jax.default_backend(),\'devices\',jax.devices());'
        'm=nnx.Linear(3,2,rngs=nnx.Rngs(0));'
        'f=lambda mod: ((mod(jnp.ones((1,3))))**2).sum();'
        'g=nnx.grad(f)(m);'
        'print(\'nnx_grad_finite\', all(bool(jnp.isfinite(jnp.asarray(x)).all()) '
        'for x in jax.tree_util.tree_leaves(g)))"' % PY)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'server_setup_logs'), exist_ok=True)
    freeze = ps('& "%s" -m pip freeze' % PY)
    with open(os.path.join(here, 'server_setup_logs',
                           'maskco_env_freeze_%s.txt' % stamp), 'w', encoding='utf-8') as f:
        f.write(freeze)
    out['freeze_lines'] = len([l for l in freeze.splitlines() if l.strip()])
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
