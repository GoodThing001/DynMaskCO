"""R0.5 Stage B 服务器预检（B0）：记录环境证据，输出 SERVER_ENVIRONMENT.json。

在服务器（DEV-GATE 已收口后）运行：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/rrnco/bin/python CC_Compare/RRNCO/dcc_rh_v4/server_preflight.py

记录：项目/RRNCO 上游 commit、checkpoint SHA-256、LICENSE hash、依赖版本、
CUDA/驱动/GPU、工作区状态。已有环境能加载 checkpoint 就不要升级依赖。
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

_DCC = os.path.dirname(os.path.abspath(__file__))
_RRNCO_ROOT = os.path.dirname(_DCC)          # CC_Compare/RRNCO/
_REPO_ROOT = os.path.normpath(os.path.join(_DCC, '..', '..', '..'))  # MASKCO-Main/
_CKPT = os.path.join(_RRNCO_ROOT, 'checkpoints', 'rcvrptw', 'epoch_199.ckpt')
_LICENSE = os.path.join(_RRNCO_ROOT, 'LICENSE')
_OUT = os.path.join(_DCC, 'SERVER_ENVIRONMENT.json')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(cwd, *args):
    try:
        out = subprocess.run(['git', *args], cwd=cwd, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f'<err: {exc}>'


def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, '__version__', '?')
    except Exception as exc:  # noqa: BLE001
        return f'<missing: {exc}>'


def main():
    env = {
        'schema_version': 'cc-compare-server-env-v1',
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'host': platform.node(),
        'platform': platform.platform(),
        'python': sys.version,
        'project_commit': _git(_REPO_ROOT, 'rev-parse', 'HEAD'),
        'rrnco_upstream_commit': _git(_RRNCO_ROOT, 'rev-parse', 'HEAD'),
        'rrnco_remote': _git(_RRNCO_ROOT, 'remote', 'get-url', 'origin'),
        'checkpoint': {
            'path': os.path.relpath(_CKPT, _REPO_ROOT),
            'sha256': sha256_file(_CKPT),
            'size_bytes': os.path.getsize(_CKPT),
        },
        'license_sha256': sha256_file(_LICENSE),
        'dependencies': {
            'torch': _version('torch'),
            'rl4co': _version('rl4co'),
            'torchrl': _version('torchrl'),
            'tensordict': _version('tensordict'),
            'lightning': _version('lightning'),
            'numpy': _version('numpy'),
        },
    }
    # CUDA / GPU
    try:
        import torch
        env['cuda'] = {
            'available': bool(torch.cuda.is_available()),
            'device_count': torch.cuda.device_count(),
            'device_name': (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            'torch_cuda_version': torch.version.cuda,
        }
    except Exception as exc:  # noqa: BLE001
        env['cuda'] = {'error': repr(exc)}
    try:
        env['nvidia_smi'] = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'], capture_output=True, text=True,
            timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        env['nvidia_smi'] = f'<err: {exc}>'

    with open(_OUT, 'w', encoding='utf-8') as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('SERVER_ENVIRONMENT.json written:', _OUT)
    print('  checkpoint sha256 =', env['checkpoint']['sha256'])
    print('  torch =', env['dependencies']['torch'],
          ' rl4co =', env['dependencies']['rl4co'])


if __name__ == '__main__':
    main()
