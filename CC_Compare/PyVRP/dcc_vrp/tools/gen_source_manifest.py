"""生成 PyVRP-RH-D 的 SOURCE_MANIFEST.json（identity manifest rev2）。

三个命名空间的逻辑相对路径 + 文件 SHA-256 + compute/control/analysis 分类：
  common/<file>    公共合同模块（compute：核心 7 文件）
  adapter/<file>   PyVRP dcc_vrp（compute：adapter/problem_builder/route_mapper；
                   control：identity/run_dcc/run_devproto_9x2/run_iter_sweep；
                   analysis：tests/、results 驱动、tools）
  project/<file>   项目计算模块（compute：9 文件）

另记录：pyvrp wheel sha256 / native extension sha256 / objective profile hash /
DEV manifest sha256 / 冻结配置 hash / 当前 code_hash（用与 runner 相同的
逻辑路径组合算法计算）。
"""
import hashlib
import json
import os
import sys

_DCC_VRP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from strict_online_runner import (COMMON_CORE_FILES, PROJECT_COMPUTE_FILES,
                                  code_hash)
from pyvrp_adapter import PyVRPRHDAdapter
import identity as pyvrp_identity

PROFILE = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
           r'\results\o0cc\scale_v2\objective_profile.json')
DEV_MANIFEST = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
                r'\data\baseline\50_node\dev_proto\DEV_MANIFEST.json')
WHEEL = os.path.join(_DCC_VRP, 'assets',
                     'pyvrp-0.14.0-cp312-cp312-win_amd64.whl')
FROZEN_CONFIG = os.path.join(_DCC_VRP, 'FROZEN_CONFIG.json')

ADAPTER_COMPUTE = ('pyvrp_adapter.py', 'problem_builder.py', 'route_mapper.py')
ADAPTER_CONTROL = ('identity.py', 'run_dcc.py', 'run_devproto_9x2.py',
                   'run_iter_sweep.py')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = []

    def add(logical, path, cls):
        files.append({'logical_path': logical,
                      'sha256': sha256_file(path),
                      'classification': cls})

    for f in COMMON_CORE_FILES:
        add(f'common/{f}', os.path.join(_COMMON, f), 'compute')
    for f in ADAPTER_COMPUTE:
        add(f'adapter/{f}', os.path.join(_DCC_VRP, f), 'compute')
    for f in ADAPTER_CONTROL:
        add(f'adapter/{f}', os.path.join(_DCC_VRP, f), 'control')
    for f in PROJECT_COMPUTE_FILES:
        add(f'project/{f}',
            os.path.join(_bootstrap.PROJECT_SCRIPTS_ROOT, f), 'compute')
    for f in sorted(os.listdir(os.path.join(_DCC_VRP, 'tests'))):
        if f.endswith('.py'):
            add(f'adapter/tests/{f}', os.path.join(_DCC_VRP, 'tests', f),
                'analysis')
    for f in ('tools/gen_source_manifest.py', 'verify_freeze.py'):
        p = os.path.join(_DCC_VRP, f)
        if os.path.exists(p):
            add(f'adapter/{f}', p, 'control')
    # 注意：
    #   - README.md 不进入本清单——FROZEN_CONFIG.json 才是权威方法定义，
    #     README 是可持续更新的说明文档（冻结方案 A，2026-09-09）；
    #   - FROZEN_CONFIG.json 与 FREEZE_SEAL.json 也不进入——FROZEN_CONFIG
    #     单向绑定本清单 hash，若反向包含会形成循环哈希。

    env_id = pyvrp_identity.environment_identity()
    profile_hash = json.load(open(PROFILE, encoding='utf-8'))['profile_hash']
    dev_manifest_sha256 = sha256_file(DEV_MANIFEST)

    manifest = {
        'schema_version': 'cc-compare-source-manifest-v1',
        'identity_manifest_revision': 4,
        'generated_at': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'PyVRP-RH-D',
        'expected_code_hash': code_hash(
            os.path.join(_DCC_VRP, 'pyvrp_adapter.py'),
            PyVRPRHDAdapter.compute_files()),
        'pyvrp_wheel_sha256': sha256_file(WHEEL),
        'native_extension_sha256': env_id['_pyvrp_sha256'],
        'objective_profile_hash': profile_hash,
        'dev_manifest_sha256': dev_manifest_sha256,
        'files': sorted(files, key=lambda x: x['logical_path']),
    }
    out = os.path.join(_DCC_VRP, 'SOURCE_MANIFEST.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'SOURCE_MANIFEST.json written ({len(files)} files)')
    print(f"  expected_code_hash: {manifest['expected_code_hash']}")
    print(f"  source_manifest_sha256: {sha256_file(out)}")


if __name__ == '__main__':
    main()
