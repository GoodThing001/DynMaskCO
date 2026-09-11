"""OR5：DEV-PROTO 正式控制入口（protocol Gate 专用，不做方法优劣判断）。

要求：
  - 只允许 split_role=dev_proto，精确 9 cell；
  - 每个 NPZ 实读 SHA-256 与 DEV_MANIFEST 比对；
  - 从 manifest 读真实 instance_seed / scene_instance_id；
  - 强制 v2 profile；
  - 新任务要求空目录；续跑只补真正缺失实例；
  - 每 cell 前后重验代码/数据/profile/环境身份；
  - 实例原子写；cell summary/manifest/log 原子发布；
  - canonical.COMPLETE 绑定所有产物 hash；
  - role 固定 dev_proto，输出只能是 PROTOCOL_PASS/FAIL（禁 GO/NO_HEADROOM）。

用法：
    python run_devproto.py --instances-per-cell 2 --out results/devproto_9x2
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

import numpy as np

from strict_online_runner import run_instance, load_objective_profile, \
    dataset_hash
from ortools_adapter import ORToolsRHDAdapter
import identity as ortools_identity
from ortools_cell_summary import atomic_write_json, publish_cell, sha256_file
from ortools_instance_validation import (validate_instance_record,
                                         CHECKPOINT_PREFIX)
from protocol_identity import (CONTROL_FILES, ANALYSIS_FILES, layer_hash,
                               build_protocol_id)

_PROJECT = _bootstrap.PROJECT_EXTENSION_ROOT
DATA_DIR = os.path.join(_PROJECT, 'data', 'baseline', '50_node', 'dev_proto')
PROFILE = os.path.join(_PROJECT, 'results', 'o0cc', 'scale_v2',
                       'objective_profile.json')
EXPECTED_CELLS = {(t.lower(), str(ed).replace('.', ''))
                  for t in ('R1', 'C1', 'RC1')
                  for ed in (0.2, 0.5, 0.8)}

FROZEN_CONFIG = {
    'method': 'OR-Tools-RH-D',
    'solution_limit': 30,          # 工程候选预算（OR7 才冻结）
    'time_limit_s': 30.0,          # 仅安全上限
    'ortools_version': '9.11.4210',
    'solver_objective': 'distance',          # OR-Tools 内部目标（非冷链感知）
    'evaluation_objective': 'coldchain_v2',  # 统一 C0 D/Q/E/J_CC 重放
    'capacity': 50,
    'num_vehicles': 25,
    'solver_seed': 0,
    'threads': 1,
    'first_solution_strategy': 'PATH_CHEAPEST_ARC',
    'fallback_allowed': False,
    'int_scale': 1000,
    'role': 'dev_proto',           # 禁 GO/NO_HEADROOM
}


def build_effective_config(solution_limit, instances_per_cell):
    """effective_config 绑定实际运行预算（不返回静态 FROZEN_CONFIG）。
    含 capacity/num_vehicles/solver_seed/threads/first-solution strategy +
    solver/evaluation objective 拆分（OR6.2）。"""
    cfg = dict(FROZEN_CONFIG)
    cfg['solution_limit'] = int(solution_limit)
    cfg['instances_per_cell'] = int(instances_per_cell)
    cfg['instance_set'] = [int(i) for i in range(int(instances_per_cell))]
    return cfg


def guard_output_dir(out, resume, pre_run):
    """输出目录守卫：新任务要求空目录；续跑必须与既有 pre-run manifest
    完全一致（config / code / data / profile / environment 全部逐项比对）。"""
    pre_run_path = os.path.join(out, 'pre_run_manifest.json')
    if os.path.exists(out) and os.listdir(out):
        if not resume or not os.path.exists(pre_run_path):
            raise RuntimeError('输出目录非空且无 pre-run manifest，拒绝接管')
        old = json.load(open(pre_run_path, encoding='utf-8'))
        for key in ('config', 'compute_hash', 'control_hash', 'analysis_hash',
                    'protocol_id', 'dev_manifest_sha256',
                    'objective_profile_hash', 'ortools_environment'):
            if old.get(key) != pre_run.get(key):
                raise RuntimeError(f'续跑 manifest 与当前身份不一致（{key} 漂移）')
    else:
        os.makedirs(out, exist_ok=True)
        atomic_write_json(pre_run_path, pre_run)


def resume_decision(out_path, resume, expected):
    """P0-1：续跑只补真正缺失实例；已存在但无效/混版本/损坏 → 立即拒绝覆盖
    （保留现场），而非重跑覆盖。

    返回 'reuse' 或 'compute'。文件不存在 → 'compute'；存在且完整合法 →
    'reuse'；存在但校验失败 → RuntimeError。"""
    if not os.path.exists(out_path):
        return 'compute'
    if not resume:
        raise RuntimeError(f'{out_path} 已存在且非续跑模式，拒绝覆盖')
    old = json.load(open(out_path, encoding='utf-8'))
    problems = validate_instance_record(old, expected)
    if problems:
        raise RuntimeError(f'{out_path} 已有记录校验失败，拒绝覆盖（现场保留）: '
                           f'{problems}')
    return 'reuse'


def verify_identity_still_intact(cells, profile, env_identity, frozen_compute,
                                 stage, frozen_control=None,
                                 frozen_analysis=None):
    """P0-4 + OR7 启动项 1：每 cell 前后复核 compute/control；聚合前复核
    compute/control/analysis（任何漂移立即停止）。

    重算项：compute hash、control hash（可选）、analysis hash（可选）、各 NPZ
    hash、objective profile hash、environment 身份（native extension 等）。"""
    from strict_online_runner import code_hash
    adapter_compute = ORToolsRHDAdapter.compute_files()
    compute = code_hash(os.path.join(_DCC_VRP, 'ortools_adapter.py'),
                        adapter_compute)
    profile2 = load_objective_profile(PROFILE)
    env2 = ortools_identity.check_environment()
    issues = []
    if compute != frozen_compute:
        issues.append('compute_hash 漂移')
    if frozen_control is not None:
        control = layer_hash('control', CONTROL_FILES, _DCC_VRP)
        if control != frozen_control:
            issues.append('control_hash 漂移')
    if frozen_analysis is not None:
        analysis = layer_hash('analysis', ANALYSIS_FILES, _DCC_VRP)
        if analysis != frozen_analysis:
            issues.append('analysis_hash 漂移')
    if profile2.profile_hash != profile.profile_hash:
        issues.append('profile 漂移')
    if env2 != env_identity:
        issues.append('environment 漂移')
    for c in cells:
        if sha256_file(os.path.join(DATA_DIR, c['file'])) != c['sha256']:
            issues.append(f'NPZ 漂移: {c["file"]}')
    if issues:
        raise RuntimeError(f'[{stage}] 运行中身份漂移: {"; ".join(issues)}')


def main():
    ap = argparse.ArgumentParser(description='OR-Tools-RH-D DEV-PROTO Gate')
    ap.add_argument('--instances-per-cell', type=int, default=2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--solution-limit', type=int, default=FROZEN_CONFIG['solution_limit'])
    ap.add_argument('--resume', action='store_true',
                    help='续跑：只补缺失实例（必须与既有 manifest 完全一致）')
    args = ap.parse_args()

    # ---- 参数拒绝（在创建输出目录/读数据前，OR6.2）----
    if args.solution_limit <= 0:
        raise RuntimeError(f'solution_limit 必须为正: {args.solution_limit}')
    if args.instances_per_cell <= 0:
        raise RuntimeError(f'instances_per_cell 必须为正: '
                           f'{args.instances_per_cell}')

    # ---- 身份装载 ----
    manifest_path = os.path.join(DATA_DIR, 'DEV_MANIFEST.json')
    dev_manifest = json.load(open(manifest_path, encoding='utf-8'))
    if dev_manifest['split_role'] != 'dev_proto':
        raise RuntimeError('只允许 dev_proto 角色')
    # P1-a：DEV manifest 结构校验（角色/规模/容量/精确 cell/实例数/NPZ 维度）
    if dev_manifest.get('problem_size') != 50:
        raise RuntimeError(f'problem_size 非 50: {dev_manifest.get("problem_size")}')
    if dev_manifest.get('capacity') != 50:
        raise RuntimeError(f'capacity 非 50: {dev_manifest.get("capacity")}')
    cells = dev_manifest['cells']
    if len(cells) != 9:
        raise RuntimeError(f'cell 数非 9: {len(cells)}')
    if {(c['type'].lower(), str(c['edod']).replace('.', ''))
            for c in cells} != EXPECTED_CELLS:
        raise RuntimeError('cell 集合不精确')
    for c in cells:
        p = os.path.join(DATA_DIR, c['file'])
        if sha256_file(p) != c['sha256']:
            raise RuntimeError(f'NPZ hash 与 manifest 不一致: {c["file"]}')
        n = int(c['num_instances'])
        if len(c['scene_instance_ids']) != n or len(c['instance_seeds']) != n:
            raise RuntimeError(f'{c["file"]} manifest 实例数不一致: '
                               f'scene={len(c["scene_instance_ids"])} '
                               f'seed={len(c["instance_seeds"])} n={n}')
        if args.instances_per_cell > n:
            raise RuntimeError(f'instances_per_cell={args.instances_per_cell} '
                               f'超过 {c["file"]} 实例数 {n}')
        with np.load(p, mmap_mode='r') as z:
            shape = z['coords'].shape
        if shape[0] != n:
            raise RuntimeError(f'{c["file"]} coords 第一维 {shape[0]} != {n}')
        if shape[1] != dev_manifest['problem_size'] + 1:
            raise RuntimeError(f'{c["file"]} coords 第二维 {shape[1]} != '
                               f'{dev_manifest["problem_size"] + 1}')

    profile = load_objective_profile(PROFILE)
    env_identity = ortools_identity.check_environment()
    from strict_online_runner import code_hash
    adapter_compute = ORToolsRHDAdapter.compute_files()
    compute_hash = code_hash(os.path.join(_DCC_VRP, 'ortools_adapter.py'),
                             adapter_compute)
    # OR7 启动项 1：三层身份（compute/control/analysis）分离冻结
    control_hash = layer_hash('control', CONTROL_FILES, _DCC_VRP)
    analysis_hash = layer_hash('analysis', ANALYSIS_FILES, _DCC_VRP)

    # P0-1：effective_config 绑定实际运行预算（不是静态 FROZEN_CONFIG）
    effective_config = build_effective_config(args.solution_limit,
                                              args.instances_per_cell)

    # P1-3：完整期望 checkpoint hash（native extension 完整 64 位 SHA-256），
    # 逐实例精确相等。
    expected_checkpoint_hash = (
        CHECKPOINT_PREFIX + env_identity['native_extension_sha256'])

    # P0-1：protocol_id 确定性（两次独立运行必须相同）；run_id 每次唯一，
    # 续跑沿用旧 pre-run 的 run_id。
    dev_manifest_sha256 = sha256_file(manifest_path)
    protocol_id = build_protocol_id(compute_hash, control_hash, analysis_hash,
                                    dev_manifest_sha256, profile.profile_hash,
                                    effective_config)
    pre_run_path = os.path.join(args.out, 'pre_run_manifest.json')
    if args.resume and os.path.exists(pre_run_path):
        old_run_id = json.load(open(pre_run_path, encoding='utf-8')).get('run_id')
        if not old_run_id:
            raise RuntimeError('续跑 pre-run 缺少 run_id（旧格式），拒绝接管')
        run_id = old_run_id
    else:
        run_id = uuid.uuid4().hex

    pre_run = {
        'method': FROZEN_CONFIG['method'],
        'role': 'dev_proto',
        'protocol_id': protocol_id,
        'run_id': run_id,
        'config': effective_config,
        'compute_hash': compute_hash,
        'control_hash': control_hash,
        'analysis_hash': analysis_hash,
        'instances_per_cell': args.instances_per_cell,
        'dev_manifest_sha256': dev_manifest_sha256,
        'objective_profile_hash': profile.profile_hash,
        'ortools_environment': env_identity,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # ---- 新任务：空目录；续跑：manifest 完全一致 ----
    guard_output_dir(args.out, args.resume, pre_run)

    instances = tuple(range(args.instances_per_cell))
    t0 = time.time()
    cell_summaries = {}
    cell_names = [f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}"
                  for c in cells]
    expected_data_by_cell = {
        f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}": c['sha256']
        for c in cells}
    for c in cells:
        cell_name = f"{c['type'].lower()}_{str(c['edod']).replace('.', '')}"
        cell_dir = os.path.join(args.out, cell_name)
        os.makedirs(os.path.join(cell_dir, 'instances'), exist_ok=True)
        ds = dict(np.load(os.path.join(DATA_DIR, c['file'])))
        # seed/scene 期望值由 DEV_MANIFEST 提供（P0-3：不从记录自证）
        expected_by_instance_id = {
            i: {'instance_seed': int(c['instance_seeds'][i]),
                'scene_instance_id': str(c['scene_instance_ids'][i])}
            for i in instances}
        # P0-4：每 cell 开始前复验身份（compute/control/NPZ/profile/environment）
        verify_identity_still_intact(cells, profile, env_identity, compute_hash,
                                     f'cell {cell_name} 前',
                                     frozen_control=control_hash)
        for inst in instances:
            out_path = os.path.join(cell_dir, 'instances', f'inst_{inst}.json')
            # P0-1/P0-2：续跑只补缺失；已存在但校验失败 → 立即退出不覆盖
            decision = resume_decision(out_path, args.resume, {
                'instance_id': inst,
                'scene_instance_id': expected_by_instance_id[inst]['scene_instance_id'],
                'instance_seed': expected_by_instance_id[inst]['instance_seed'],
                'data_sha256': c['sha256'],
                'profile_hash': profile.profile_hash,
                'code_hash': compute_hash,
                'checkpoint_hash': expected_checkpoint_hash,
            })
            if decision == 'reuse':
                print(f'  [{cell_name} inst {inst}] 续跑跳过（身份一致）',
                      flush=True)
                continue
            rec = run_instance(
                ds, effective_config['capacity'], effective_config['num_vehicles'],
                adapter_factory=lambda: ORToolsRHDAdapter(
                    solution_limit=effective_config['solution_limit'],
                    time_limit_s=effective_config['time_limit_s'], check_env=True),
                inst_idx=inst, objective='coldchain', profile=profile,
                seed=0, data_sha256=c['sha256'],
                adapter_module_path=os.path.join(_DCC_VRP,
                                                 'ortools_adapter.py'),
                instance_seed=c['instance_seeds'][inst],
                scene_instance_id=c['scene_instance_ids'][inst])
            atomic_write_json(out_path, rec)
            print(f'  [{cell_name} inst {inst}] complete='
                  f'{rec["outcome"]["complete"]} '
                  f'fallback_ev={rec["stats"]["fallback_triggered_events"]} '
                  f'solve={rec["stats"]["n_solver_calls"]}',
                  flush=True)
        cell_summaries[cell_name] = publish_cell(
            cell_dir, cell_name, instances, expected_by_instance_id,
            c['sha256'], profile.profile_hash, compute_hash,
            expected_checkpoint_hash, run_id, control_hash)
        # P0-4：每 cell 发布后复验身份
        verify_identity_still_intact(cells, profile, env_identity, compute_hash,
                                     f'cell {cell_name} 后',
                                     frozen_control=control_hash)
        print(f'  [{cell_name}] cell 汇总发布完成', flush=True)

    # P0-4：九 cell 聚合前统一复验身份（compute/control/analysis）
    verify_identity_still_intact(cells, profile, env_identity, compute_hash,
                                 '聚合前', frozen_control=control_hash,
                                 frozen_analysis=analysis_hash)

    # ---- 汇总（只做协议判定，独立模块 aggregate_devproto）----
    from aggregate_devproto import aggregate, verify_aggregate
    aggregate_doc, _ = aggregate(args.out, cell_names, args.instances_per_cell,
                                 pre_run_path,
                                 expected_code_hash=compute_hash,
                                 expected_profile_hash=profile.profile_hash,
                                 expected_data_sha256=expected_data_by_cell,
                                 expected_checkpoint_hash=expected_checkpoint_hash,
                                 control_hash=control_hash,
                                 analysis_hash=analysis_hash,
                                 protocol_id=protocol_id, run_id=run_id)
    # 顶层 marker 绑定自检（pre-run / cell marker 被替换必须拒绝）
    verify_aggregate(args.out)
    print(f"\n[verdict] {aggregate_doc['verdict']} "
          f"(n={aggregate_doc['n_instances']}, "
          f"complete={aggregate_doc['n_complete']}, "
          f"protocol_errors={aggregate_doc['protocol_errors']}, "
          f"fallback_events={aggregate_doc['n_fallback_events']})", flush=True)
    if aggregate_doc['verdict'] != 'PROTOCOL_PASS':
        sys.exit(1)


if __name__ == '__main__':
    main()
