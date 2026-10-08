"""RRNCO-Ordering-RH-D R1 冻结：环境身份（identity.py）。

compute 层身份 = 上游 rrnco commit + checkpoint sha256 + 依赖版本 + LICENSE sha256。
只读检查：不加载 checkpoint、不导入 torch（纯 stdlib + 文件哈希），可在任意 python 运行；
完整依赖版本以服务器 `server_preflight.py` 生成的 SERVER_ENVIRONMENT.json 为权威。

用法（仓库根）：
    python CC_Compare/RRNCO/dcc_rh_v4/identity.py
    python CC_Compare/RRNCO/dcc_rh_v4/identity.py --checkpoint CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

_DCC = os.path.dirname(os.path.abspath(__file__))
_RRNCO_ROOT = os.path.dirname(_DCC)
_REPO_ROOT = os.path.normpath(os.path.join(_DCC, '..', '..', '..'))
_CKPT_DEFAULT = os.path.join(_RRNCO_ROOT, 'checkpoints', 'rcvrptw', 'epoch_199.ckpt')
_LICENSE = os.path.join(_RRNCO_ROOT, 'LICENSE')

# 冻结值（R0.5 证据，2026-09-26）
FROZEN_CHECKPOINT_SHA256 = 'b1ff3191ee2f28c19e4807748b3c1b83522e1e96c6fb7bf9238442cf859464b3'
FROZEN_LICENSE_SHA256 = '820243863426b8f6ce5385a8e4bbe14f2da8675beec6d463e481c74bf251a491'
FROZEN_RRNCO_UPSTREAM_COMMIT = '823d510dadf4dd711730ec4fbf337c356a0de6ae'


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
        return '<err: %s>' % exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=_CKPT_DEFAULT)
    args = ap.parse_args()

    problems = []
    def require(cond, msg, got=None):
        status = 'PASS' if cond else 'FAIL'
        print('  [%s] %s%s' % (status, msg, '' if got is None else ' (got=%s)' % got))
        if not cond:
            problems.append(msg)

    # checkpoint + license + upstream commit
    ck = sha256_file(args.checkpoint) if os.path.exists(args.checkpoint) else '<missing>'
    require(ck == FROZEN_CHECKPOINT_SHA256, 'checkpoint sha256 == 冻结值', ck)
    lic = sha256_file(_LICENSE) if os.path.exists(_LICENSE) else '<missing>'
    require(lic == FROZEN_LICENSE_SHA256, 'LICENSE sha256 == 冻结值', lic)
    # 本地 rrnco checkout 允许漂移（上游只读）；compute 身份以服务器 SERVER_ENVIRONMENT
    # 记录的 commit 为准（真实运行环境）。
    up = _git(_RRNCO_ROOT, 'rev-parse', 'HEAD')
    print('  [INFO] local rrnco checkout HEAD:', up,
          '(冻结值 %s；以服务器 SERVER_ENVIRONMENT 为准)' % FROZEN_RRNCO_UPSTREAM_COMMIT)

    # SERVER_ENVIRONMENT.json（依赖版本 + commit 权威）
    senv_p = os.path.join(_DCC, 'results', 'SERVER_ENVIRONMENT.json')
    if os.path.exists(senv_p):
        senv = json.load(open(senv_p, encoding='utf-8'))
        deps = senv.get('dependencies', {})
        require(senv.get('checkpoint', {}).get('sha256') == FROZEN_CHECKPOINT_SHA256,
                'SERVER_ENVIRONMENT checkpoint sha256 一致',
                senv.get('checkpoint', {}).get('sha256'))
        require(senv.get('rrnco_upstream_commit') == FROZEN_RRNCO_UPSTREAM_COMMIT,
                'SERVER_ENVIRONMENT rrnco_upstream_commit == 冻结值',
                senv.get('rrnco_upstream_commit'))
        print('  [INFO] deps:', json.dumps(deps, sort_keys=True))
    else:
        require(False, 'results/SERVER_ENVIRONMENT.json 存在')

    if problems:
        print('\nidentity: %d 项失败' % len(problems))
        for p in problems:
            print('  -', p)
        sys.exit(1)
    print('\nidentity: ALL PASS')


if __name__ == '__main__':
    main()
