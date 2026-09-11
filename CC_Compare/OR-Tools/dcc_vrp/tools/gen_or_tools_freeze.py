"""生成 OR-Tools-RH-D 正式封存三件套（SOURCE_MANIFEST / FROZEN_CONFIG / FREEZE_SEAL）。

冻结配置：solution_limit=30（OR7 预注册选择器 SELECTED_30）。绑定三层身份
（compute/control/analysis）、逐文件 hash、资产（native/wheel/profile/dev manifest）、
OR7-A parity 报告、OR7-B 顶层封条、预算选择结果、最终核验器 hash。

身份链单向绑定（无循环）：
  SOURCE_MANIFEST（源码/资产 + 三层 hash）
  ← FROZEN_CONFIG（绑定 SOURCE_MANIFEST + 冻结配置 + 证据）
  ← FREEZE_SEAL（绑定两者）

用法：
    python tools/gen_or_tools_freeze.py
"""
import hashlib
import json
import os
import sys
import time

_DCC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
ADAPTER_COMPUTE = ORToolsRHDAdapter.compute_files()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _layer_files():
    files = []
    proj = _bootstrap.PROJECT_SCRIPTS_ROOT
    for f in COMMON_CORE_FILES:
        files.append((f'common/{f}', os.path.join(_COMMON, f), 'compute'))
    for f in ADAPTER_COMPUTE:
        files.append((f'adapter/{f}', os.path.join(_DCC, f), 'compute'))
    for f in PROJECT_COMPUTE_FILES:
        files.append((f'project/{f}', os.path.join(proj, f), 'compute'))
    for f in CONTROL_FILES:
        files.append((f'control/{f}', os.path.join(_DCC, f), 'control'))
    for f in ANALYSIS_FILES:
        files.append((f'analysis/{f}', os.path.join(_DCC, f), 'analysis'))
    return files


def main():
    files = _layer_files()
    compute = code_hash(os.path.join(_DCC, 'ortools_adapter.py'), ADAPTER_COMPUTE)
    control = layer_hash('control', CONTROL_FILES, _DCC)
    analysis = layer_hash('analysis', ANALYSIS_FILES, _DCC)
    env = ortools_identity.check_environment()
    profile_hash = json.load(open(PROFILE, encoding='utf-8'))['profile_hash']
    dev_manifest_sha256 = sha256_file(DEV_MANIFEST)

    file_entries = [{'logical_path': lp, 'sha256': sha256_file(p),
                     'classification': cls}
                    for lp, p, cls in sorted(files)]

    source_manifest = {
        'schema_version': 'cc-compare-source-manifest-v1',
        'identity_manifest_revision': 1,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'OR-Tools-RH-D',
        'compute_hash': compute,
        'control_hash': control,
        'analysis_hash': analysis,
        'objective_profile_hash': profile_hash,
        'dev_manifest_sha256': dev_manifest_sha256,
        'native_extension_sha256': env['native_extension_sha256'],
        'wrapper_sha256': env['wrapper_sha256'],
        'wheel_sha256': env['wheel_sha256'],
        'files': file_entries,
    }
    sm_path = os.path.join(_DCC, 'SOURCE_MANIFEST.json')
    with open(sm_path, 'w', encoding='utf-8') as f:
        json.dump(source_manifest, f, indent=2, ensure_ascii=False)
        f.write('\n')
    sm_sha = sha256_file(sm_path)

    # 证据 hash
    parity = {f'l{lim}': sha256_file(os.path.join(RESULTS,
                                                   f'or7a_parity_l{lim}.json'))
              for lim in (10, 30, 100)}
    or7b = {f'l{lim}': sha256_file(os.path.join(RESULTS, f'or7b_9x32_l{lim}',
                                                 'canonical.COMPLETE'))
            for lim in (10, 30, 100)}
    selection_sha = sha256_file(os.path.join(RESULTS, 'or7_budget_selection.json'))
    verifier_sha = sha256_file(os.path.join(_DCC, 'verify_or7_freeze.py'))

    frozen_config = {
        'protocol_revision': 1,
        'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': {
            'method_name': 'OR-Tools-RH-D',
            'definition': 'strict-online rolling-horizon、distance-optimized、'
                          '非冷链感知，最终统一 C0 重放 D/Q/E/J_CC',
            'method_revision': '1',
            'adapter_revision': '1',
        },
        'config': {
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
        },
        'budget_selection': {
            'verdict': 'SELECTED_30',
            'selection_source': 'OR7-B-9x32',
            'reference': 30,
            'nine_cell_eq_distance': {'10': 18.048602, '30': 17.395637,
                                      '100': 17.401449},
        },
        'identity': {
            'compute_hash': compute,
            'control_hash': control,
            'analysis_hash': analysis,
            'source_manifest_sha256': sm_sha,
            'objective_profile_hash': profile_hash,
            'dev_manifest_sha256': dev_manifest_sha256,
            'native_extension_sha256': env['native_extension_sha256'],
            'wrapper_sha256': env['wrapper_sha256'],
            'wheel_sha256': env['wheel_sha256'],
        },
        'evidence': {
            'or7a_parity': parity,
            'or7b_canonical': or7b,
            'budget_selection_sha256': selection_sha,
            'verifier_source_sha256': verifier_sha,
            'determinism': '三档各 18/18 decision parity ALL_MATCH',
        },
        'revision_policy': '任何配置变化必须升级 protocol/adapter revision，'
                          '不得覆盖当前冻结版本。',
    }
    fc_path = os.path.join(_DCC, 'FROZEN_CONFIG.json')
    with open(fc_path, 'w', encoding='utf-8') as f:
        json.dump(frozen_config, f, indent=2, ensure_ascii=False)
        f.write('\n')
    fc_sha = sha256_file(fc_path)

    seal = {
        'schema_version': 'cc-compare-freeze-seal-v1',
        'method': 'OR-Tools-RH-D',
        'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'binds': {'source_manifest_sha256': sm_sha, 'frozen_config_sha256': fc_sha},
        'note': '单向绑定：SOURCE_MANIFEST 绑源码/资产；FROZEN_CONFIG 绑 '
                'SOURCE_MANIFEST + solution_limit=30 + 证据；本 seal 自身不入集合。',
    }
    seal_path = os.path.join(_DCC, 'FREEZE_SEAL.json')
    with open(seal_path, 'w', encoding='utf-8') as f:
        json.dump(seal, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print('freeze written:')
    print(f'  SOURCE_MANIFEST.json sha = {sm_sha}')
    print(f'  FROZEN_CONFIG.json sha = {fc_sha}')
    print(f'  compute_hash = {compute}')
    print(f'  control_hash = {control}')
    print(f'  analysis_hash = {analysis}')
    print(f'  files = {len(file_entries)}')


if __name__ == '__main__':
    main()
