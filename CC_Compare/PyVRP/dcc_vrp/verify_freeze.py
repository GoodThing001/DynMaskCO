"""PyVRP-RH-D 冻结只读验证器（与生成器分离：本文件绝不写入/重生成任何文件）。

检查项（P1 加固版）：
  1. FREEZE_SEAL 绑定的两个文件 hash + 键集合精确校验；
  2. SOURCE_MANIFEST 成员逻辑路径集合 == 独立 EXPECTED_PATHS（精确相等，
     长度 33——防止生成时遗漏文件仍自洽）；
  3. 33 个成员逐文件 hash（逻辑路径 → 实际文件，按 __file__ 相对解析，
     无 Windows 绝对路径——Linux 服务器可用同一验证器）；
  4. 重新计算 code_hash 与 manifest.expected_code_hash 一致；
  5. objective profile / DEV manifest / pyvrp wheel / native extension hash
     与 manifest 记录一致（路径经 _bootstrap 项目根解析）；
  6. FROZEN_CONFIG.identity 与 SOURCE_MANIFEST 的 profile/dev-manifest/
     wheel/native/code hash 交叉一致。

退出码：0 = 全部一致；非 0 = 冻结被破坏（输出具体差异）。
"""
import hashlib
import json
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from strict_online_runner import code_hash
from pyvrp_adapter import PyVRPRHDAdapter
import identity as pyvrp_identity

_PROJECT = _bootstrap.PROJECT_EXTENSION_ROOT
PROFILE = os.path.join(_PROJECT, 'results', 'o0cc', 'scale_v2',
                       'objective_profile.json')
DEV_MANIFEST = os.path.join(_PROJECT, 'data', 'baseline', '50_node',
                            'dev_proto', 'DEV_MANIFEST.json')
WHEEL = os.path.join(_DCC_VRP, 'assets',
                     'pyvrp-0.14.0-cp312-cp312-win_amd64.whl')

# ---- 独立的预期逻辑路径集合（与生成器约定一致；生成器遗漏即在此失配）----
EXPECTED_COMMON = ['strict_online_runner.py', 'method_adapter.py',
                   'baseline_contract.py', 'record_validation.py',
                   'trace_export.py', 'ownership_audit.py', '_bootstrap.py']
EXPECTED_ADAPTER_COMPUTE = ['pyvrp_adapter.py', 'problem_builder.py',
                            'route_mapper.py']
EXPECTED_ADAPTER_CONTROL = ['identity.py', 'run_dcc.py', 'run_devproto_9x2.py',
                            'run_iter_sweep.py', 'tools/gen_source_manifest.py',
                            'verify_freeze.py']
EXPECTED_PROJECT = ['simulation/strict_online_env.py',
                    'simulation/action_contract.py',
                    'simulation/recourse_snapshot.py',
                    'simulation/counterfactual_teacher.py',
                    'evaluation/authoritative_evaluator.py',
                    'evaluation/hard_gate.py',
                    'evaluation/coldchain_evaluator.py',
                    'coldchain/coldchain_contract.py',
                    'coldchain/coldchain_state.py']
EXPECTED_TESTS = ['test_adapter_integration.py', 'test_environment_identity.py',
                  'test_event4_regression.py', 'test_integer_boundary.py',
                  'test_penalty_protocol.py', 'test_problem_builder.py',
                  'test_empty_pool.py', 'testutil.py']

EXPECTED_PATHS = set(
    [f'common/{f}' for f in EXPECTED_COMMON]
    + [f'adapter/{f}' for f in EXPECTED_ADAPTER_COMPUTE]
    + [f'adapter/{f}' for f in EXPECTED_ADAPTER_CONTROL]
    + [f'project/{f}' for f in EXPECTED_PROJECT]
    + [f'adapter/tests/{f}' for f in EXPECTED_TESTS])

EXPECTED_SEAL_KEYS = {'source_manifest_sha256', 'frozen_config_sha256'}

NAMESPACE_ROOTS = {
    'common': _COMMON,
    'adapter': _DCC_VRP,
    'project': _bootstrap.PROJECT_SCRIPTS_ROOT,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(logical_path):
    for ns, root in NAMESPACE_ROOTS.items():
        if logical_path.startswith(ns + '/'):
            return os.path.join(root, logical_path[len(ns) + 1:])
    raise ValueError(f'未知命名空间: {logical_path}')


def main():
    problems = []

    # ---- 1. seal 绑定 + 键集合精确校验 ----
    seal = json.load(open(os.path.join(_DCC_VRP, 'FREEZE_SEAL.json'),
                          encoding='utf-8'))
    if set(seal['binds'].keys()) != EXPECTED_SEAL_KEYS:
        problems.append(f'seal 键集合不精确: {set(seal["binds"])} '
                        f'期望 {EXPECTED_SEAL_KEYS}')
    for name, bound in seal['binds'].items():
        fname = ('SOURCE_MANIFEST.json' if 'source_manifest' in name
                 else 'FROZEN_CONFIG.json')
        actual = sha256_file(os.path.join(_DCC_VRP, fname))
        if actual != bound:
            problems.append(f'seal 绑定 {fname} 不一致: 记录={bound} 实际={actual}')

    # ---- 2. manifest 逻辑路径集合精确校验 ----
    sm = json.load(open(os.path.join(_DCC_VRP, 'SOURCE_MANIFEST.json'),
                        encoding='utf-8'))
    manifest_paths = {f['logical_path'] for f in sm['files']}
    if manifest_paths != EXPECTED_PATHS:
        missing = EXPECTED_PATHS - manifest_paths
        extra = manifest_paths - EXPECTED_PATHS
        problems.append(f'逻辑路径集合不精确: missing={sorted(missing)} '
                        f'extra={sorted(extra)}')
    if len(manifest_paths) != len(EXPECTED_PATHS):
        problems.append(f'成员数量 {len(manifest_paths)} != '
                        f'{len(EXPECTED_PATHS)}')

    # ---- 3. 逐文件 hash ----
    for entry in sm['files']:
        p = resolve(entry['logical_path'])
        if not os.path.exists(p):
            problems.append(f"文件不存在: {entry['logical_path']} -> {p}")
            continue
        actual = sha256_file(p)
        if actual != entry['sha256']:
            problems.append(f"文件 hash 不一致: {entry['logical_path']} "
                            f"记录={entry['sha256'][:16]} 实际={actual[:16]}")

    # ---- 4. code_hash ----
    current_code = code_hash(os.path.join(_DCC_VRP, 'pyvrp_adapter.py'),
                             PyVRPRHDAdapter.compute_files())
    if current_code != sm['expected_code_hash']:
        problems.append(f'code_hash 不一致: 记录={sm["expected_code_hash"]} '
                        f'实际={current_code}')

    # ---- 5. 资产 hash ----
    profile_hash = json.load(open(PROFILE, encoding='utf-8'))['profile_hash']
    if profile_hash != sm['objective_profile_hash']:
        problems.append('objective profile hash 不一致')
    if sha256_file(DEV_MANIFEST) != sm['dev_manifest_sha256']:
        problems.append('DEV manifest hash 不一致')
    if sha256_file(WHEEL) != sm['pyvrp_wheel_sha256']:
        problems.append('pyvrp wheel hash 不一致')
    env_id = pyvrp_identity.check_environment()
    if env_id['_pyvrp_sha256'] != sm['native_extension_sha256']:
        problems.append('native extension hash 不一致')

    # ---- 6. FROZEN_CONFIG.identity 交叉核验 ----
    frozen = json.load(open(os.path.join(_DCC_VRP, 'FROZEN_CONFIG.json'),
                            encoding='utf-8'))
    ident = frozen.get('identity', {})
    actual_sm_hash = sha256_file(os.path.join(_DCC_VRP, 'SOURCE_MANIFEST.json'))
    cross = [
        ('source_manifest_sha256', actual_sm_hash),
        ('expected_code_hash', current_code),
        ('objective_profile_hash', profile_hash),
        ('dev_manifest_sha256', sha256_file(DEV_MANIFEST)),
        ('pyvrp_wheel_sha256', sha256_file(WHEEL)),
        ('native_extension_sha256', env_id['_pyvrp_sha256']),
    ]
    for key, expected in cross:
        if ident.get(key) != expected:
            problems.append(f'FROZEN_CONFIG.identity.{key} 不一致: '
                            f'记录={ident.get(key)} 实际={expected}')

    if problems:
        print('FREEZE VERIFICATION FAILED:')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
    print('FREEZE VERIFICATION PASS')
    print(f'  code_hash = {current_code}')
    print(f'  members verified = {len(sm["files"])} '
          f'(exact set match, seal keys exact)')
    print(f'  source_manifest_sha256 = {actual_sm_hash}')
    print(f'  frozen_config_sha256 = '
          f'{sha256_file(os.path.join(_DCC_VRP, "FROZEN_CONFIG.json"))}')


if __name__ == '__main__':
    main()
