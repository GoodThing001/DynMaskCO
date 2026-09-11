"""OR-Tools-RH-D 冻结只读验证器（与生成器分离：绝不写入/重生成任何文件）。

检查项：
  1. FREEZE_SEAL 绑定的 SOURCE_MANIFEST / FROZEN_CONFIG hash + 键集合精确；
  2. SOURCE_MANIFEST 成员逻辑路径集合 == 独立 EXPECTED_PATHS（26 项精确相等）；
  3. 26 个成员逐文件 hash；
  4. 三层身份（compute/control/analysis）重算与 manifest 一致；
  5. objective profile / DEV manifest / native / wrapper / wheel hash 与 manifest 一致；
  6. FROZEN_CONFIG.identity 与 SOURCE_MANIFEST 交叉一致；
  7. FROZEN_CONFIG 冻结配置 solution_limit=30 + 证据 hash（parity/canonical/
     selection/verifier）逐项重算一致。

退出码：0 = 全部一致；非 0 = 冻结被破坏（输出具体差异）。
"""
import hashlib
import json
import os
import sys

_DCC = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC, '..', '..', 'common'))
for p in (_DCC, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from strict_online_runner import (COMMON_CORE_FILES, PROJECT_COMPUTE_FILES,
                                  code_hash)
from ortools_adapter import ORToolsRHDAdapter
import identity as ortools_identity
from protocol_identity import CONTROL_FILES, ANALYSIS_FILES, layer_hash

_PROJECT = _bootstrap.PROJECT_EXTENSION_ROOT
PROFILE = os.path.join(_PROJECT, 'results', 'o0cc', 'scale_v2',
                       'objective_profile.json')
DEV_MANIFEST = os.path.join(_PROJECT, 'data', 'baseline', '50_node',
                            'dev_proto', 'DEV_MANIFEST.json')
RESULTS = os.path.join(_DCC, 'results')

EXPECTED_COMMON = COMMON_CORE_FILES
EXPECTED_ADAPTER_COMPUTE = ORToolsRHDAdapter.compute_files()
EXPECTED_PROJECT = PROJECT_COMPUTE_FILES
EXPECTED_CONTROL = CONTROL_FILES
EXPECTED_ANALYSIS = ANALYSIS_FILES

EXPECTED_PATHS = set(
    [f'common/{f}' for f in EXPECTED_COMMON]
    + [f'adapter/{f}' for f in EXPECTED_ADAPTER_COMPUTE]
    + [f'project/{f}' for f in EXPECTED_PROJECT]
    + [f'control/{f}' for f in EXPECTED_CONTROL]
    + [f'analysis/{f}' for f in EXPECTED_ANALYSIS])

EXPECTED_SEAL_KEYS = {'source_manifest_sha256', 'frozen_config_sha256'}

NAMESPACE_ROOTS = {
    'common': _COMMON,
    'adapter': _DCC,
    'project': _bootstrap.PROJECT_SCRIPTS_ROOT,
    'control': _DCC,
    'analysis': _DCC,
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

    # ---- 1. seal ----
    seal = json.load(open(os.path.join(_DCC, 'FREEZE_SEAL.json'), encoding='utf-8'))
    if set(seal['binds'].keys()) != EXPECTED_SEAL_KEYS:
        problems.append(f'seal 键集合不精确: {set(seal["binds"])}')
    for name, bound in seal['binds'].items():
        fname = ('SOURCE_MANIFEST.json' if 'source_manifest' in name
                 else 'FROZEN_CONFIG.json')
        actual = sha256_file(os.path.join(_DCC, fname))
        if actual != bound:
            problems.append(f'seal 绑定 {fname} 不一致: {bound} vs {actual}')

    # ---- 2. manifest 路径集合 ----
    sm = json.load(open(os.path.join(_DCC, 'SOURCE_MANIFEST.json'), encoding='utf-8'))
    manifest_paths = {f['logical_path'] for f in sm['files']}
    if manifest_paths != EXPECTED_PATHS:
        problems.append(f'逻辑路径集合不精确: missing={sorted(EXPECTED_PATHS - manifest_paths)} '
                        f'extra={sorted(manifest_paths - EXPECTED_PATHS)}')

    # ---- 3. 逐文件 hash ----
    for entry in sm['files']:
        p = resolve(entry['logical_path'])
        actual = sha256_file(p)
        if actual != entry['sha256']:
            problems.append(f'文件 hash 不一致: {entry["logical_path"]} '
                            f'{entry["sha256"][:16]} vs {actual[:16]}')

    # ---- 4. 三层身份 ----
    compute = code_hash(os.path.join(_DCC, 'ortools_adapter.py'),
                        ORToolsRHDAdapter.compute_files())
    control = layer_hash('control', CONTROL_FILES, _DCC)
    analysis = layer_hash('analysis', ANALYSIS_FILES, _DCC)
    if compute != sm['compute_hash']:
        problems.append('compute_hash 不一致')
    if control != sm['control_hash']:
        problems.append('control_hash 不一致')
    if analysis != sm['analysis_hash']:
        problems.append('analysis_hash 不一致')

    # ---- 5. 资产 ----
    profile_hash = json.load(open(PROFILE, encoding='utf-8'))['profile_hash']
    if profile_hash != sm['objective_profile_hash']:
        problems.append('objective profile hash 不一致')
    if sha256_file(DEV_MANIFEST) != sm['dev_manifest_sha256']:
        problems.append('DEV manifest hash 不一致')
    env = ortools_identity.check_environment()
    for k in ('native_extension_sha256', 'wrapper_sha256', 'wheel_sha256'):
        if env[k] != sm[k]:
            problems.append(f'{k} 不一致')

    # ---- 6. FROZEN_CONFIG 交叉核验 ----
    frozen = json.load(open(os.path.join(_DCC, 'FROZEN_CONFIG.json'), encoding='utf-8'))
    ident = frozen.get('identity', {})
    actual_sm_hash = sha256_file(os.path.join(_DCC, 'SOURCE_MANIFEST.json'))
    cross = [('source_manifest_sha256', actual_sm_hash),
             ('compute_hash', compute), ('control_hash', control),
             ('analysis_hash', analysis), ('objective_profile_hash', profile_hash),
             ('dev_manifest_sha256', sha256_file(DEV_MANIFEST)),
             ('native_extension_sha256', env['native_extension_sha256']),
             ('wrapper_sha256', env['wrapper_sha256']),
             ('wheel_sha256', env['wheel_sha256'])]
    for key, expected in cross:
        if ident.get(key) != expected:
            problems.append(f'FROZEN_CONFIG.identity.{key} 不一致')

    # ---- 7. 冻结配置 + 证据 ----
    cfg = frozen.get('config', {})
    expected_cfg = {
        'solution_limit': 30,
        'time_limit_s': 30.0,
        'solver_objective': 'distance',
        'evaluation_objective': 'coldchain_v2',
        'solver_seed': 0,
        'threads': 1,
        'first_solution_strategy': 'PATH_CHEAPEST_ARC',
        'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH',
        'capacity': 50,
        'num_vehicles': 25,
        'int_scale': 1000,
        'fallback_allowed': False,
    }
    for k, v in expected_cfg.items():
        if cfg.get(k) != v:
            problems.append(f'FROZEN_CONFIG.config.{k} != {v!r}（实际 {cfg.get(k)!r}）')
    if frozen.get('budget_selection', {}).get('verdict') != 'SELECTED_30':
        problems.append('FROZEN_CONFIG 冻结 verdict != SELECTED_30')
    ev = frozen.get('evidence', {})
    for lim in (10, 30, 100):
        pp = os.path.join(RESULTS, f'or7a_parity_l{lim}.json')
        if ev.get('or7a_parity', {}).get(f'l{lim}') != sha256_file(pp):
            problems.append(f'or7a_parity l{lim} hash 不一致')
        oc = os.path.join(RESULTS, f'or7b_9x32_l{lim}', 'canonical.COMPLETE')
        if ev.get('or7b_canonical', {}).get(f'l{lim}') != sha256_file(oc):
            problems.append(f'or7b_canonical l{lim} hash 不一致')
    sel = os.path.join(RESULTS, 'or7_budget_selection.json')
    if ev.get('budget_selection_sha256') != sha256_file(sel):
        problems.append('budget_selection hash 不一致')
    vf = os.path.join(_DCC, 'verify_or7_freeze.py')
    if ev.get('verifier_source_sha256') != sha256_file(vf):
        problems.append('verifier_source hash 不一致')

    # ---- 8. 证据包 ----
    ep_path = os.path.join(_DCC, 'EVIDENCE_PACKAGE.json')
    ep = json.load(open(ep_path, encoding='utf-8'))
    pkg = os.path.join(_DCC, ep.get('package', ''))
    if not os.path.exists(pkg):
        problems.append(f'证据包缺失: {pkg}')
    elif sha256_file(pkg) != ep.get('sha256'):
        problems.append('证据包 SHA-256 与 EVIDENCE_PACKAGE.json 不一致')
    if ev.get('evidence_package_sha256') != ep.get('sha256'):
        problems.append('FROZEN_CONFIG 证据包 hash 与 EVIDENCE_PACKAGE.json 不一致')
    if seal.get('evidence_package_sha256') != ep.get('sha256'):
        problems.append('FREEZE_SEAL 证据包 hash 与 EVIDENCE_PACKAGE.json 不一致')

    if problems:
        print('FREEZE VERIFICATION FAILED:')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
    print('FREEZE VERIFICATION PASS')
    print(f'  compute_hash = {compute}')
    print(f'  control_hash = {control}')
    print(f'  analysis_hash = {analysis}')
    print(f'  members = {len(sm["files"])} (exact set match)')
    print(f'  solution_limit = 30 (SELECTED_30)')


if __name__ == '__main__':
    main()
