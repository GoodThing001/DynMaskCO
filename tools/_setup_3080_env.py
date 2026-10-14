"""Install the MASKCO Python stack into the Windows 3080 host's `MASKCO_env`.

Mirrors the 5090 environment: Python 3.10 (already 3.10.20 on the 3080), jax, flax==0.10.4
(0.10.7 breaks the frozen encoder's SwiGLU), optax. The CPU stack is installed first and
separately from any CUDA attempt, so a GPU-plugin failure cannot take the CPU stack down with it.

Safety: an existing shared host means changes must be auditable. The pre-install `pip freeze` and
the post-install `pip freeze` are both written next to this tool, and only `MASKCO_env` is
touched - the other conda environments (nandribo, paddle, yolov8, pt310, ...) are never modified.

Usage:
    python tools/_setup_3080_env.py [--config .vscode/sftp.3080.json] [--cuda] [--yes]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402

PY = r'D:\anaconda\envs\MASKCO_env\python.exe'
CPU_STACK = ['jax==0.6.2', 'flax==0.10.4', 'optax==0.2.4']
CUDA_STACK = ['jax-cuda12-plugin==0.6.2', 'jax-cuda12-pjrt==0.6.2']


def main():
    args = sys.argv[1:]
    cfg = '.vscode/sftp.3080.json'
    if '--config' in args:
        cfg = args[args.index('--config') + 1]
    want_cuda = '--cuda' in args
    out = {'config': cfg, 'env_python': PY}
    stamp = time.strftime('%Y%m%d-%H%M%S')

    def ps(cmd, timeout=1800):
        return run(cmd, timeout=timeout, config=cfg)

    out['python_version'] = ps('& "%s" -V' % PY)
    out['before_freeze'] = ps('& "%s" -m pip freeze' % PY)
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'server_setup_logs'), exist_ok=True)
    with open(os.path.join(here, 'server_setup_logs',
                           'maskco_env_freeze_before_%s.txt' % stamp), 'w',
              encoding='utf-8') as f:
        f.write(out['before_freeze'])

    out['install_cpu'] = ps('& "%s" -m pip install --no-input %s'
                            % (PY, ' '.join(CPU_STACK)), timeout=3600)
    out['verify_cpu'] = ps(
        '& "%s" -c "import jax,flax,optax,numpy;'
        ' print(\'jax\',jax.__version__);print(\'flax\',flax.__version__);'
        ' print(\'optax\',optax.__version__);print(\'numpy\',numpy.__version__);'
        ' print(\'backend\',jax.default_backend());print(\'devices\',jax.devices())"' % PY)

    if want_cuda:
        out['install_cuda'] = ps('& "%s" -m pip install --no-input %s'
                                 % (PY, ' '.join(CUDA_STACK)), timeout=3600)
        out['verify_cuda'] = ps(
            '& "%s" -c "import jax;print(\'backend\',jax.default_backend());'
            'print(\'devices\',jax.devices())"' % PY)

    out['after_freeze'] = ps('& "%s" -m pip freeze' % PY)
    with open(os.path.join(here, 'server_setup_logs',
                           'maskco_env_freeze_after_%s.txt' % stamp), 'w',
              encoding='utf-8') as f:
        f.write(out['after_freeze'])
    out['freeze_unchanged'] = out['before_freeze'] == out['after_freeze']
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('before_freeze', 'after_freeze')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
