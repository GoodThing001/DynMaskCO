"""OR5：九 cell 跨场景协议汇总（只判协议，不做方法优劣判断）。

  - 精确 9 cell、18 实例；
  - code/data/profile/config/environment 一致性由各 cell manifest 保证；
  - seed 与 scene ID 由实例记录校验（instance_validation）；
  - 每 cell 真实 solver call > 0；
  - 任何 fallback/timeout/build/map/certificate failure → Gate FAIL；
  - 输出只能是 PROTOCOL_PASS / PROTOCOL_FAIL（禁 GO/NO_HEADROOM）。

P0-3 发布单元完整绑定：aggregate 验证 summary/manifest/实例文件/cell marker 全链，
顶层 marker 绑定九个 cell 的 canonical.COMPLETE 文件 hash。
P0-2 顶层复验：verify_aggregate(out_dir) 自己读实际 pre-run，重算并比较
pre_run_manifest_sha256，从 pre-run + DEV_MANIFEST 派生精确九 cell / 实例数 /
code/profile/data/checkpoint/protocol_id/run_id，禁止调用方传入任意 cell 子集。
"""
import json
import os
import sys
import time

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from ortools_cell_summary import atomic_write_json, sha256_file
from ortools_instance_validation import CHECKPOINT_PREFIX
from protocol_identity import (CONTROL_FILES, ANALYSIS_FILES, layer_hash,
                               build_protocol_id)

CELLS = [f'{t}_{ed}' for t in ('r1', 'c1', 'rc1') for ed in ('02', '05', '08')]
DATA_DIR = os.path.join(_bootstrap.PROJECT_EXTENSION_ROOT, 'data', 'baseline',
                        '50_node', 'dev_proto')
DEV_MANIFEST = os.path.join(DATA_DIR, 'DEV_MANIFEST.json')


def _verify_cells(args_out, cell_names, instances_per_cell,
                  expected_code_hash=None, expected_profile_hash=None,
                  expected_data_sha256=None, expected_checkpoint_hash=None,
                  expected_control_hash=None, run_id=None):
    """逐 cell 验证落盘产物完整性（P0-3）。返回 (cell_summaries,
    cell_canonical_hashes)。任何缺失/不一致抛 RuntimeError。"""
    cell_summaries = {}
    cell_canonical_hashes = {}
    for name in cell_names:
        cell_dir = os.path.join(args_out, name)
        summary_path = os.path.join(cell_dir, 'summary.json')
        manifest_path = os.path.join(cell_dir, 'manifest.json')
        marker_path = os.path.join(cell_dir, 'canonical.COMPLETE')
        building_path = os.path.join(cell_dir, 'BUILDING')
        if os.path.exists(building_path):
            raise RuntimeError(f'{name} 存在 BUILDING 标记（cell 发布未完成）')
        if not os.path.exists(summary_path):
            raise RuntimeError(f'cell summary 缺失: {name}')
        if not os.path.exists(manifest_path):
            raise RuntimeError(f'cell manifest 缺失: {name}')
        if not os.path.exists(marker_path):
            raise RuntimeError(f'{name} 缺 canonical.COMPLETE 标记')
        canonical = json.load(open(marker_path, encoding='utf-8'))
        summary = json.load(open(summary_path, encoding='utf-8'))
        manifest = json.load(open(manifest_path, encoding='utf-8'))

        if canonical.get('summary_sha256') != sha256_file(summary_path):
            raise RuntimeError(f'{name} canonical.summary_sha256 与落盘 summary '
                               '不一致（标记/汇总陈旧或篡改）')
        if canonical.get('manifest_sha256') != sha256_file(manifest_path):
            raise RuntimeError(f'{name} canonical.manifest_sha256 与落盘 manifest '
                               '不一致（manifest 被替换或篡改）')
        if canonical.get('protocol_errors') != 0:
            raise RuntimeError(f'{name} canonical.protocol_errors != 0')
        if canonical.get('n_instances') != instances_per_cell:
            raise RuntimeError(f'{name} canonical.n_instances '
                               f'{canonical.get("n_instances")} != '
                               f'{instances_per_cell}')
        if sorted(canonical.get('instance_artifact_hashes', [])) != sorted(
                h['artifact_hash'] for h in summary.get('instance_hashes', [])):
            raise RuntimeError(f'{name} canonical.instance_artifact_hashes 与 '
                               'summary 不一致')
        if summary.get('n_instances') != instances_per_cell:
            raise RuntimeError(f'{name} summary.n_instances != {instances_per_cell}')

        # 实读 manifest 核对 code/data/profile/checkpoint（与 pre-run / DEV_MANIFEST 对账）
        if expected_code_hash is not None and \
                manifest.get('code_hash') != expected_code_hash:
            raise RuntimeError(f'{name} manifest.code_hash 与 pre-run 不一致')
        if expected_control_hash is not None and \
                manifest.get('control_hash') != expected_control_hash:
            raise RuntimeError(f'{name} manifest.control_hash 与 pre-run 不一致')
        if expected_profile_hash is not None and \
                manifest.get('profile_hash') != expected_profile_hash:
            raise RuntimeError(f'{name} manifest.profile_hash 与 pre-run 不一致')
        if expected_data_sha256 is not None and \
                manifest.get('data_sha256') != expected_data_sha256.get(name):
            raise RuntimeError(f'{name} manifest.data_sha256 与 DEV_MANIFEST 不一致')
        if expected_checkpoint_hash is not None and \
                manifest.get('checkpoint_hash') != expected_checkpoint_hash:
            raise RuntimeError(f'{name} manifest.checkpoint_hash 与 pre-run '
                               'native identity 不一致')
        if run_id is not None:
            if manifest.get('run_id') != run_id:
                raise RuntimeError(f'{name} manifest.run_id 与 pre-run 不一致')
            if canonical.get('run_id') != run_id:
                raise RuntimeError(f'{name} canonical.run_id 与 pre-run 不一致')

        # 实例文件 SHA-256 绑定（字节级：篡改实例 JSON 必被检出）
        inst_dir = os.path.join(cell_dir, 'instances')
        file_hashes = {}
        for i in range(instances_per_cell):
            p = os.path.join(inst_dir, f'inst_{i}.json')
            if not os.path.exists(p):
                raise RuntimeError(f'{name} 缺实例文件 inst_{i}.json')
            file_hashes[str(i)] = sha256_file(p)
        if canonical.get('instance_file_sha256') != file_hashes:
            raise RuntimeError(f'{name} canonical.instance_file_sha256 与落盘实例 '
                               '文件不一致（实例文件被替换/篡改）')

        cell_summaries[name] = summary
        cell_canonical_hashes[name] = sha256_file(marker_path)
    return cell_summaries, cell_canonical_hashes


def aggregate(args_out, cell_names, instances_per_cell, pre_run_manifest,
              expected_code_hash=None, expected_profile_hash=None,
              expected_data_sha256=None, expected_checkpoint_hash=None,
              control_hash=None, analysis_hash=None, protocol_id=None,
              run_id=None):
    """从落盘 cell summary + canonical.COMPLETE 重新汇总。

    P0-3：每个 cell 必须存在 canonical.COMPLETE 且与落盘 summary/manifest/
    实例文件全部一致。顶层 marker 绑定九个 cell 的 canonical.COMPLETE 文件
    hash + protocol_id/run_id + compute/control/analysis 三层身份。
    返回 (aggregate, canonical_path)。"""
    cell_summaries, cell_canonical_hashes = _verify_cells(
        args_out, cell_names, instances_per_cell,
        expected_code_hash=expected_code_hash,
        expected_profile_hash=expected_profile_hash,
        expected_data_sha256=expected_data_sha256,
        expected_checkpoint_hash=expected_checkpoint_hash,
        expected_control_hash=control_hash, run_id=run_id)

    n_cells = len(cell_names)
    n_expected = n_cells * instances_per_cell
    aggregate = {
        'role': 'dev_proto',
        'protocol_id': protocol_id,
        'run_id': run_id,
        'n_cells': n_cells,
        'n_instances': n_expected,
        'n_complete': sum(s['n_complete'] for s in cell_summaries.values()),
        'protocol_errors': sum(s['protocol_errors']
                               for s in cell_summaries.values()),
        'n_solver_calls': sum(s['n_solver_calls']
                              for s in cell_summaries.values()),
        'n_fallback_events': sum(s['n_fallback_events']
                                 for s in cell_summaries.values()),
        'n_fast_path': sum(s['n_fast_path'] for s in cell_summaries.values()),
        'n_waits': sum(s['n_waits'] for s in cell_summaries.values()),
    }
    all_cells_have_solves = all(s['n_solver_calls'] > 0
                                for s in cell_summaries.values())
    verdict = 'PROTOCOL_PASS' if (
        aggregate['protocol_errors'] == 0
        and aggregate['n_complete'] == n_expected
        and aggregate['n_fallback_events'] == 0
        and all_cells_have_solves) else 'PROTOCOL_FAIL'
    aggregate['verdict'] = verdict
    atomic_write_json(os.path.join(args_out, 'aggregate_summary.json'), aggregate)

    canonical = {
        'protocol_id': protocol_id,
        'run_id': run_id,
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_instances': n_expected,
        'verdict': verdict,
        'code_hash': expected_code_hash,
        'control_hash': control_hash,
        'analysis_hash': analysis_hash,
        'profile_hash': expected_profile_hash,
        'pre_run_manifest_sha256': sha256_file(pre_run_manifest),
        'aggregate_summary_sha256': sha256_file(
            os.path.join(args_out, 'aggregate_summary.json')),
        'cell_summary_sha256': {
            name: sha256_file(os.path.join(args_out, name, 'summary.json'))
            for name in cell_names},
        'cell_canonical_sha256': cell_canonical_hashes,
    }
    canonical_path = os.path.join(args_out, 'canonical.COMPLETE')
    atomic_write_json(canonical_path, canonical)
    return aggregate, canonical_path


def verify_aggregate(args_out):
    """复验顶层 marker 绑定（P0-2 + P0-3）。

    自包含：读实际 pre_run_manifest.json，重算并比较 pre_run_manifest_sha256，
    从 pre-run + DEV_MANIFEST 派生精确九 cell / 实例数 / code/profile/data/
    checkpoint/protocol_id/run_id，禁止调用方传入任意 cell 子集。任何篡改
    （pre-run / cell marker / summary / manifest / 实例文件）抛 RuntimeError。
    """
    top_path = os.path.join(args_out, 'canonical.COMPLETE')
    if not os.path.exists(top_path):
        raise RuntimeError('顶层 canonical.COMPLETE 缺失')
    top = json.load(open(top_path, encoding='utf-8'))

    pre_run_path = os.path.join(args_out, 'pre_run_manifest.json')
    if not os.path.exists(pre_run_path):
        raise RuntimeError('pre_run_manifest.json 缺失')
    if top.get('pre_run_manifest_sha256') != sha256_file(pre_run_path):
        raise RuntimeError('顶层 marker 的 pre_run_manifest_sha256 与当前 '
                           'pre-run 不一致（pre-run 被替换/篡改）')
    pre_run = json.load(open(pre_run_path, encoding='utf-8'))

    # 从 pre-run 派生身份（不得由调用方传入）
    config = pre_run.get('config') or {}
    instances_per_cell = int(config.get('instances_per_cell'))
    expected_code_hash = pre_run.get('compute_hash')
    expected_profile_hash = pre_run.get('objective_profile_hash')
    run_id = pre_run.get('run_id')
    protocol_id = pre_run.get('protocol_id')
    env = pre_run.get('ortools_environment') or {}
    native = env.get('native_extension_sha256')
    expected_checkpoint_hash = (CHECKPOINT_PREFIX + native) if native else None

    # OR7 启动项 2：实读 DEV manifest，核验其 SHA 与 pre-run 一致 + 结构合法
    if sha256_file(DEV_MANIFEST) != pre_run.get('dev_manifest_sha256'):
        raise RuntimeError('DEV manifest SHA 与 pre-run 不一致（manifest 漂移）')
    dev_manifest = json.load(open(DEV_MANIFEST, encoding='utf-8'))
    if dev_manifest.get('split_role') != 'dev_proto':
        raise RuntimeError('DEV manifest 角色非 dev_proto')
    if dev_manifest.get('problem_size') != 50 or dev_manifest.get('capacity') != 50:
        raise RuntimeError('DEV manifest problem_size/capacity 非 50')
    if len(dev_manifest.get('cells', [])) != 9:
        raise RuntimeError('DEV manifest cell 数非 9')
    expected_data_sha256 = {
        f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}": c['sha256']
        for c in dev_manifest['cells']}

    # OR7 启动项 1/2：重算三层身份 + protocol_id，与 pre-run / top canonical 对账
    control_hash = layer_hash('control', CONTROL_FILES, _DCC_VRP)
    analysis_hash = layer_hash('analysis', ANALYSIS_FILES, _DCC_VRP)
    recomputed_protocol_id = build_protocol_id(
        expected_code_hash, control_hash, analysis_hash,
        pre_run.get('dev_manifest_sha256'), expected_profile_hash, config)

    # 精确九 cell（禁止子集）
    cell_names = CELLS
    cell_summaries, cell_canonical_hashes = _verify_cells(
        args_out, cell_names, instances_per_cell,
        expected_code_hash=expected_code_hash,
        expected_profile_hash=expected_profile_hash,
        expected_data_sha256=expected_data_sha256,
        expected_checkpoint_hash=expected_checkpoint_hash,
        expected_control_hash=control_hash, run_id=run_id)

    # 顶层 marker 身份交叉核验
    if top.get('code_hash') != expected_code_hash:
        raise RuntimeError('顶层 marker code_hash 与 pre-run 不一致')
    if top.get('profile_hash') != expected_profile_hash:
        raise RuntimeError('顶层 marker profile_hash 与 pre-run 不一致')
    if top.get('control_hash') != control_hash or \
            pre_run.get('control_hash') != control_hash:
        raise RuntimeError('control_hash 与 pre-run / top canonical 不一致')
    if top.get('analysis_hash') != analysis_hash or \
            pre_run.get('analysis_hash') != analysis_hash:
        raise RuntimeError('analysis_hash 与 pre-run / top canonical 不一致')
    if top.get('run_id') != run_id:
        raise RuntimeError('顶层 marker run_id 与 pre-run 不一致')
    if protocol_id != recomputed_protocol_id or \
            top.get('protocol_id') != recomputed_protocol_id:
        raise RuntimeError('protocol_id 与重算值 / top canonical 不一致')
    if top.get('cell_canonical_sha256') != cell_canonical_hashes:
        raise RuntimeError('顶层 marker 的 cell_canonical_sha256 与当前 cell '
                           'marker 不一致（cell marker 被替换）')
    if top.get('aggregate_summary_sha256') != sha256_file(
            os.path.join(args_out, 'aggregate_summary.json')):
        raise RuntimeError('aggregate_summary.json 被替换')
    if top.get('cell_summary_sha256') != {
            name: sha256_file(os.path.join(args_out, name, 'summary.json'))
            for name in cell_names}:
        raise RuntimeError('cell summary 与顶层 marker 绑定不一致')
    return top
