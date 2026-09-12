"""共享正式门控契约（DEV-GATE / O0-CC VAL 共用）。

本模块是无副作用的纯契约层，只定义角色、manifest 校验、九 cell、seed、hash、
断点恢复与跨 split 独立性校验；不 import jax/numpy/oracle，不执行任何计算或写盘。

`run_formal_gate.py`（通用驱动器）、`run_dev_gate.py` / `run_o0cc_val.py`（薄入口）、
`aggregate_cross_cell.py`（聚合器）、`freeze_source_archive.py`（封存）与测试共用同一
份口径，避免 DEV 一套、VAL 一套导致控制链漂移。

关键规则：
  - `--role` 只允许 `dev_gate` 或 `val`；`regression`/`dev_proto` 是内部单 cell/shard
    执行角色，不是正式入口（不在此模块登记为正式角色）。
  - 九 cell 必须恰好等于 R1/C1/RC1 × EDoD 0.2/0.5/0.8；每 cell 恰好 128 实例；九 cell
    共用同一组内部 instance seeds 且每 cell 内 seed 唯一。
  - 计算/控制/分析三套代码 hash 分离：计算版本决定 oracle/评估/环境/合同行为，续跑
    必须一致；控制/分析可独立演进。
"""
import hashlib
import json
import os

# --------------------------------------------------------------------------- #
# 角色
# --------------------------------------------------------------------------- #
FORMAL_ROLES = ("dev_gate", "val")
INTERNAL_ROLES = ("regression", "dev_proto")

TYPES = ("R1", "C1", "RC1")
EDODS = (0.2, 0.5, 0.8)
EXPECTED_CELLS = [f"{t.lower()}_{int(e*10):02d}" for t in TYPES for e in EDODS]
INSTANCES_PER_CELL = 128
MANIFEST_SCHEMA = 'o0cc-split-manifest-v1'
SPLIT_REGISTRY_SCHEMA = 'o0cc-split-registry-v1'
STATISTICAL_PLAN_SCHEMA = 'o0cc-statistical-plan-v1'
# 正式 VAL 的 split registry 至少覆盖这些已登记 split（新 DEV-CAL/DEV-GATE 冻结后可扩充）。
VAL_REQUIRED_SPLIT_ROLES = ('train', 'dev_cal', 'dev_proto', 'dev_gate')

# --------------------------------------------------------------------------- #
# 版本分离文件清单
# --------------------------------------------------------------------------- #
# 计算版本（决定 oracle/评估/环境/合同行为；续跑必须一致）——与 DEV-GATE v4 冻结一致。
COMPUTE_FILES = [
    'scripts/evaluation/run_action_oracle.py',
    'scripts/project_paths.py',
    'scripts/evaluation/hard_gate.py',
    'scripts/evaluation/coldchain_evaluator.py',
    'scripts/evaluation/service_first.py',
    'scripts/evaluation/authoritative_evaluator.py',
    'scripts/simulation/sequential_oracle.py',
    'scripts/simulation/counterfactual_teacher.py',
    'scripts/simulation/jf1h_repair.py',
    'scripts/simulation/recourse_snapshot.py',
    'scripts/simulation/action_contract.py',
    'scripts/simulation/strict_online_env.py',
    'scripts/simulation/joint_fleet.py',
    'scripts/coldchain/coldchain_state.py',
    'scripts/coldchain/coldchain_contract.py',
]

# 控制版本（驱动/复用验收/断点恢复/规范汇总）——可独立演进。
CONTROL_FILES = [
    'scripts/evaluation/formal_gate_contract.py',
    'scripts/evaluation/run_formal_gate.py',
    'scripts/evaluation/run_dev_gate.py',
    'scripts/evaluation/run_o0cc_val.py',
    'scripts/evaluation/instance_validation.py',
    'scripts/evaluation/cell_summary.py',
]

# 分析版本（聚合器）——可独立演进。
ANALYSIS_FILES = [
    'scripts/evaluation/aggregate_cross_cell.py',
]

# 冻结的计算口径（DEV-GATE 与 VAL 共用同一 frozen method/config）。
FROZEN_CONFIG = {
    'capacity': 50.0,
    'num_vehicles': 25,
    'slack_vehicles': 1,
    'objective': 'coldchain',
    'local': True,
    'n_boot': 10000,
    'boot_seed': 42,
    'baseline': 'JF1-H-F',
}

# 工作区根（scripts/../.. 的 C-VRP 根，与 run_dev_gate 原逻辑一致）。
_BASE = os.path.dirname(os.path.abspath(__file__))            # scripts/evaluation
_SCRIPTS = os.path.dirname(_BASE)                             # scripts
_CVRPTW = os.path.dirname(_SCRIPTS) if os.path.basename(_SCRIPTS) == 'scripts' else _SCRIPTS


class ContractError(ValueError):
    """正式门控契约违反（角色 / manifest / seed / hash / 独立性）。"""


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def combined_hash(pairs):
    """与 _version 同规则的多文件/多键组合 hash（sorted + compact JSON）。"""
    payload = json.dumps(sorted(pairs), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _version(files):
    """多文件组合 hash + 逐文件（basename → sha256）hash。"""
    hashes = {os.path.basename(f): sha256_file(os.path.join(_CVRPTW, f)) for f in files}
    return combined_hash(hashes.items()), hashes


def cell_tag(c):
    return f"{c['type'].lower()}_{int(c['edod'] * 10):02d}"


def recompute_profile_hash(profile_path):
    """从 profile 内容重算 hash（与 ObjectiveProfile.profile_hash 同公式，不引入重依赖）。"""
    data = load_json(profile_path)
    payload = json.dumps({
        'name': data['name'],
        'distance_scale': data['distance_scale'],
        'quality_scale': data['quality_scale'],
        'energy_scale': data['energy_scale'],
        'lambda_quality': data['lambda_quality'],
        'lambda_energy': data['lambda_energy'],
        'scale_source': data.get('scale_source', 'pilot'),
        'dev_statistics': data.get('dev_statistics'),
    }, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- #
# manifest 校验
# --------------------------------------------------------------------------- #
def validate_manifest(manifest, role):
    """校验正式门控 manifest。返回 (cells, n_per_cell)。

    兼容旧 schema（`seed` / `num_instances_per_cell`）与新 schema（`root_seed` /
    `instances_per_cell`）。任何违反都抛 ContractError。
    """
    if role not in FORMAL_ROLES:
        raise ContractError(f"非法正式角色：{role!r}（仅 {FORMAL_ROLES}）")
    split_role = manifest.get('split_role')
    if split_role != role:
        raise ContractError(f"role={role} 但 manifest.split_role={split_role!r}（不一致）")

    # VAL 强制统一 schema、显式 root_seed 与完整 identity；禁止回退旧 schema / seed。
    if role == 'val':
        if manifest.get('schema') != MANIFEST_SCHEMA:
            raise ContractError(f"VAL manifest 必须 schema={MANIFEST_SCHEMA}，"
                                f"实际 {manifest.get('schema')!r}")
        root_seed = manifest.get('root_seed')
        if not isinstance(root_seed, int) or isinstance(root_seed, bool):
            raise ContractError(f"VAL manifest 必须显式整数 root_seed（禁止回退旧 seed），"
                                f"实际 {root_seed!r}")
        for k in ('generator_identity', 'contract_identity', 'profile_identity'):
            if not isinstance(manifest.get(k), dict):
                raise ContractError(f"VAL manifest 缺 identity 字段 {k}")

    cells = manifest.get('cells')
    if not isinstance(cells, list) or len(cells) != 9:
        raise ContractError(
            f"cells 必须恰好 9 个，实际 "
            f"{len(cells) if isinstance(cells, list) else type(cells).__name__}")
    tags = [cell_tag(c) for c in cells]
    if len(set(tags)) != 9:
        raise ContractError(f"cell 集合存在重复：{sorted(tags)}")
    if set(tags) != set(EXPECTED_CELLS):
        raise ContractError(f"cell 集合不符：{sorted(set(tags))} vs 预期 {EXPECTED_CELLS}")

    n_per_cell = manifest.get('instances_per_cell')
    if n_per_cell is None:
        n_per_cell = manifest.get('num_instances_per_cell')
    if not isinstance(n_per_cell, int) or isinstance(n_per_cell, bool) or n_per_cell <= 0:
        raise ContractError(f"instances_per_cell 缺失或非正：{n_per_cell!r}")
    if n_per_cell != INSTANCES_PER_CELL:
        raise ContractError(
            f"正式门控要求每 cell 恰好 {INSTANCES_PER_CELL} 实例，实际 {n_per_cell}")

    for c in cells:
        for k in ('type', 'edod', 'file', 'sha256', 'instance_seeds', 'scene_instance_ids'):
            if c.get(k) is None:
                raise ContractError(f"cell {cell_tag(c)} 缺字段 {k}")

    first_seeds = [int(s) for s in cells[0].get('instance_seeds', [])]
    if len(first_seeds) < n_per_cell:
        raise ContractError(f"cell {cell_tag(cells[0])} instance_seeds 不足 "
                            f"{n_per_cell}：{len(first_seeds)}")
    if len(set(first_seeds)) != n_per_cell:
        raise ContractError(f"cell {cell_tag(cells[0])} instance_seeds 非唯一")
    for c in cells[1:]:
        if [int(s) for s in c.get('instance_seeds', [])] != first_seeds:
            raise ContractError(f"cell {cell_tag(c)} instance_seeds 与 cell "
                                f"{cell_tag(cells[0])} 不一致（九 cell 必须共享同一组 seed）")

    for c in cells:
        sids = list(c.get('scene_instance_ids') or [])
        if len(sids) != n_per_cell:
            raise ContractError(f"cell {cell_tag(c)} scene_instance_ids 数量 "
                                f"{len(sids)} != {n_per_cell}")
        if len(set(sids)) != n_per_cell:
            raise ContractError(f"cell {cell_tag(c)} scene_instance_ids 有重复")
    return cells, n_per_cell


def manifest_root_seed(manifest):
    return manifest.get('root_seed', manifest.get('seed'))


def _cell_seed_set(manifest):
    seeds = set()
    for c in manifest.get('cells', []):
        seeds.update(int(s) for s in c.get('instance_seeds', []))
    return seeds


def _cell_scene_id_set(manifest):
    sids = set()
    for c in manifest.get('cells', []):
        sids.update(str(s) for s in (c.get('scene_instance_ids') or []))
    return sids


def check_disjoint(manifest, disjoint_paths):
    """检查 manifest 与每个 disjoint manifest 的 instance_seeds 与 scene_instance_ids 无重叠。

    返回 report（list of dict）。重叠不在此抛异常，交由调用方决定硬失败；这样驱动器能把
    完整报告写进 pre_run_manifest，而不是只输出一句「检查通过」。
    """
    this_seeds = _cell_seed_set(manifest)
    this_sids = _cell_scene_id_set(manifest)
    report = []
    for dp in (disjoint_paths or []):
        other = load_json(dp)
        seed_overlap = sorted(this_seeds & _cell_seed_set(other))
        sid_overlap = sorted(this_sids & _cell_scene_id_set(other))
        report.append({
            'manifest': dp,
            'sha256': sha256_file(dp),
            'split_role': other.get('split_role'),
            'root_seed': manifest_root_seed(other),
            'seed_overlap': seed_overlap,
            'scene_id_overlap': sid_overlap,
        })
    return report


# --------------------------------------------------------------------------- #
# VAL 冻结身份 / split registry
# --------------------------------------------------------------------------- #
def build_val_identity(manifest_obj, manifest_sha256, disjoint_sha256s, registry_sha256=None):
    """VAL 冻结身份：manifest hash / root_seed / 全体 scene ids / contract / profile /
    disjoint registry hashes / registry SHA。随计算协议一起冻结，防止续跑换 manifest 或身份。"""
    scene_ids = sorted(f"{cell_tag(c)}:{sid}"
                       for c in manifest_obj['cells'] for sid in c.get('scene_instance_ids', []))
    scene_ids_sha256 = hashlib.sha256(
        json.dumps(scene_ids, separators=(',', ':'), ensure_ascii=True).encode('utf-8')).hexdigest()
    return {
        'manifest_sha256': manifest_sha256,
        'root_seed': manifest_root_seed(manifest_obj),
        'scene_ids_sha256': scene_ids_sha256,
        'contract_hash': (manifest_obj.get('contract_identity') or {}).get('contract_hash'),
        'profile_hash': (manifest_obj.get('profile_identity') or {}).get('profile_hash'),
        'disjoint_manifest_sha256s': sorted(disjoint_sha256s),
        'registry_sha256': registry_sha256,
    }


def _is_sha256_hex(s):
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)


def load_split_registry(path):
    """加载并严格校验 split registry：schema、逐项 split_id/split_role/manifest/
    manifest_sha256（64 位 hex）、无重复 id/路径/hash。"""
    reg = load_json(path)
    if reg.get('schema') != SPLIT_REGISTRY_SCHEMA:
        raise ContractError(f"split registry 必须 schema={SPLIT_REGISTRY_SCHEMA}，"
                            f"实际 {reg.get('schema')!r}")
    splits = reg.get('splits')
    if not isinstance(splits, list) or not splits:
        raise ContractError("split registry 缺非空 splits 列表")
    seen_id, seen_path, seen_hash = set(), set(), set()
    for s in splits:
        sid = s.get('split_id')
        role = s.get('split_role')
        mp = s.get('manifest')
        mh = s.get('manifest_sha256')
        if not isinstance(sid, str) or not sid:
            raise ContractError(f"registry split 缺 split_id：{s!r}")
        if not isinstance(role, str) or not role:
            raise ContractError(f"registry split {sid} 缺 split_role")
        if not isinstance(mp, str) or not mp:
            raise ContractError(f"registry split {sid} 缺 manifest 路径")
        if not _is_sha256_hex(mh):
            raise ContractError(f"registry split {sid} manifest_sha256 非 64 位 hex：{mh!r}")
        if sid in seen_id:
            raise ContractError(f"registry split_id 重复：{sid}")
        if mp in seen_path:
            raise ContractError(f"registry manifest 路径重复：{mp}")
        if mh in seen_hash:
            raise ContractError(f"registry manifest hash 重复：{mh}")
        seen_id.add(sid); seen_path.add(mp); seen_hash.add(mh)
    return reg


def validate_val_registry(reg, required_roles=VAL_REQUIRED_SPLIT_ROLES):
    """VAL registry 覆盖性校验：必须覆盖所有预注册 split；不得含 val/test。"""
    roles = {s['split_role'] for s in reg['splits']}
    missing = [r for r in required_roles if r not in roles]
    if missing:
        raise ContractError(f"VAL split registry 未覆盖预注册 split：{missing}")
    if 'val' in roles or 'test' in roles:
        raise ContractError("VAL split registry 不得包含 val/test split")


def check_registry_disjoint(manifest_obj, registry_path, required_roles=VAL_REQUIRED_SPLIT_ROLES):
    """校验 manifest 与 registry 中所有 split 无重叠，并核验每个注册 manifest 的 hash + 覆盖性。

    返回 (report, registry)。相对路径按 registry 文件所在目录解析。
    """
    reg = load_split_registry(registry_path)
    validate_val_registry(reg, required_roles)
    base = os.path.dirname(os.path.abspath(registry_path))
    paths = []
    for s in reg['splits']:
        mp = s['manifest']
        if not os.path.isabs(mp):
            mp = os.path.join(base, mp)
        if not os.path.exists(mp):
            raise ContractError(f"registry split {s['split_id']} manifest 不存在：{mp}")
        if sha256_file(mp) != s['manifest_sha256']:
            raise ContractError(f"registry split {s['split_id']} manifest hash 不符")
        paths.append(mp)
    return check_disjoint(manifest_obj, paths), reg


# --------------------------------------------------------------------------- #
# 统计方案（预注册，与实际聚合器配置一致）
# --------------------------------------------------------------------------- #
def load_statistical_plan(path):
    """加载并做结构校验 statistical plan（正式 VAL 必须预注册，不能只作包内附件）。"""
    plan = load_json(path)
    if plan.get('schema') != STATISTICAL_PLAN_SCHEMA:
        raise ContractError(f"statistical plan 必须 schema={STATISTICAL_PLAN_SCHEMA}，"
                            f"实际 {plan.get('schema')!r}")
    for k in ('primary_endpoint', 'n_boot', 'boot_seed', 'cell_weight', 'grouping'):
        if k not in plan:
            raise ContractError(f"statistical plan 缺字段 {k}")
    return plan


def validate_statistical_plan(plan, config):
    """统计方案的主终点 / bootstrap / 等权 / 成组 / CI / GO 判据必须与聚合器冻结口径一致。"""
    if plan.get('primary_endpoint') != 'nine_cell_equal_weight_paired_delta_jcc':
        raise ContractError("statistical plan primary_endpoint 必须为 "
                            "nine_cell_equal_weight_paired_delta_jcc")
    if plan.get('n_boot') != config['n_boot']:
        raise ContractError(f"statistical plan n_boot={plan.get('n_boot')} "
                            f"!= config {config['n_boot']}")
    if plan.get('boot_seed') != config['boot_seed']:
        raise ContractError(f"statistical plan boot_seed={plan.get('boot_seed')} "
                            f"!= config {config['boot_seed']}")
    if plan.get('cell_weight') != 'equal':
        raise ContractError("statistical plan cell_weight 必须 equal（九 cell 等权）")
    if plan.get('grouping') != 'internal_seed_group':
        raise ContractError("statistical plan grouping 必须 internal_seed_group（按内部 seed 成组）")
    if plan.get('confidence_level') != 0.95:
        raise ContractError("statistical plan confidence_level 必须 0.95")
    if plan.get('ci_method') != 'percentile':
        raise ContractError("statistical plan ci_method 必须 percentile（成组 bootstrap 分位）")
    if plan.get('alternative') != 'less_than_zero':
        raise ContractError("statistical plan alternative 必须 less_than_zero")
    if plan.get('go_rule') != 'mean_lt_zero_and_upper_ci_lt_zero':
        raise ContractError("statistical plan go_rule 必须 mean_lt_zero_and_upper_ci_lt_zero")
    if plan.get('local_endpoint_role') != 'auxiliary':
        raise ContractError("statistical plan local_endpoint_role 必须 auxiliary（辅助判据）")
    if plan.get('report_dqe_components') is not True:
        raise ContractError("statistical plan report_dqe_components 必须 True（D/Q/E 分项强制报告）")
    if plan.get('service_gate_required') is not True:
        raise ContractError("statistical plan service_gate_required 必须 True（baseline/oracle 100% service）")
    if plan.get('fail_closed_on_error') is not True:
        raise ContractError("statistical plan fail_closed_on_error 必须 True（protocol/missing/non-finite fail-closed）")
    if plan.get('nominal_is_primary') is not True:
        raise ContractError("statistical plan nominal_is_primary 必须 True（nominal 唯一主分析）")
    if plan.get('sensitivity_non_gating') is not True:
        raise ContractError("statistical plan sensitivity_non_gating 必须 True（low/high 不参与 GO）")
    if plan.get('single_cell_non_gating') is not True:
        raise ContractError("statistical plan single_cell_non_gating 必须 True（单 cell 不触发整体 GO）")
    return None


# --------------------------------------------------------------------------- #
# sealed archive 校验（VAL 启动锁）
# --------------------------------------------------------------------------- #
_ARCHIVE_SEAL_REL = 'manifests/归档封存清单.json'


def verify_archive(archive, expected_bundle):
    """核验 sealed archive：文件集合精确一致 + 逐文件 hash + bundle 重算 + 预期 hash +
    seal 顶层身份与 bundle 内源码冻结清单交叉一致 + ARCHIVE_SEALED 封存标记。

    seal（归档封存清单）本身不在 bundle 内，因此其顶层身份字段不受 bundle_sha256 保护；
    这里强制它们与 bundle 内、受 bundle_sha256 保护的「源码冻结清单」逐项一致，防止只改
    seal 顶层 hash 绕过 active-code 绑定。expected_bundle 必填且必须为 64 位 hex。
    """
    if not _is_sha256_hex(expected_bundle):
        raise ContractError(f'--expected-bundle-sha256 必须是 64 位 hex：{expected_bundle!r}')
    seal_path = os.path.join(archive, _ARCHIVE_SEAL_REL)
    if not os.path.exists(seal_path):
        raise ContractError(f'archive 缺封存清单：{seal_path}')
    seal = load_json(seal_path)
    if seal.get('bundle_sha256') != expected_bundle:
        raise ContractError(f'archive bundle_sha256 与本地批准不符：'
                            f'{seal.get("bundle_sha256")} vs {expected_bundle}')
    declared = seal.get('bundle_files') or {}
    actual = {}
    for root, dirs, fnames in os.walk(archive):
        for f in fnames:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, archive).replace('\\', '/')
            if rel in ('ARCHIVE_SEALED', _ARCHIVE_SEAL_REL):
                continue
            if f.endswith('.pyc'):
                raise ContractError(f'archive 发现 .pyc：{rel}')
            actual[rel] = sha256_file(p)
    if set(actual) != set(declared):
        raise ContractError('archive 文件集合与封存清单不符')
    for rel, h in declared.items():
        if actual.get(rel) != h:
            raise ContractError(f'archive 文件 hash 不符：{rel}')
    if combined_hash(sorted(declared.items())) != seal.get('bundle_sha256'):
        raise ContractError('archive bundle_sha256 重算不符')

    # seal 顶层身份必须与 bundle 内（受 bundle_sha256 保护的）源码冻结清单逐项一致。
    frozen_path = os.path.join(archive, 'manifests', '源码冻结清单.json')
    if not os.path.exists(frozen_path):
        raise ContractError('archive 缺 manifests/源码冻结清单.json')
    frozen = load_json(frozen_path)
    for key in ('compute_sha256', 'control_sha256', 'analysis_sha256', 'runner_sha256',
                'profile_hash', 'contract_hash', 'contract_file_sha256',
                'statistical_plan_sha256', 'manifest_sha256'):
        if seal.get(key) != frozen.get(key):
            raise ContractError(f'seal.{key} 与源码冻结清单不一致')

    # ARCHIVE_SEALED 封存标记必须存在、sealed=true、bundle/archive hash 与 seal 一致。
    sealed_path = os.path.join(archive, 'ARCHIVE_SEALED')
    if not os.path.exists(sealed_path):
        raise ContractError('archive 缺 ARCHIVE_SEALED')
    sealed = load_json(sealed_path)
    if not sealed.get('sealed'):
        raise ContractError('ARCHIVE_SEALED 未 sealed=true')
    if sealed.get('bundle_sha256') != seal.get('bundle_sha256'):
        raise ContractError('ARCHIVE_SEALED.bundle_sha256 与 seal 不一致')
    if sealed.get('archive_content_sha256') != seal.get('archive_content_sha256'):
        raise ContractError('ARCHIVE_SEALED.archive_content_sha256 与 seal 不一致')
    return seal


def verify_active_code(seal):
    """运行前核验 active compute/control/analysis hash 与封存 seal 完全一致。

    防止工作区源码在封存后发生变化仍被用来处理冻结输入（gap 3）。
    """
    for key, files in (('compute_sha256', COMPUTE_FILES),
                       ('control_sha256', CONTROL_FILES),
                       ('analysis_sha256', ANALYSIS_FILES)):
        expected = seal.get(key)
        if not expected:
            raise ContractError(f'archive seal 缺 {key}')
        actual = _version(files)[0]
        if actual != expected:
            raise ContractError(f'active {key} 与封存不符：'
                                f'{actual[:12]} vs {expected[:12]}')


def _frozen_compute(cells, n_per_cell, compute_sha256, data, profile_hash, role='dev_gate',
                    val_identity=None, contract_identity=None):
    """冻结计划中「计算相关」字段（续跑必须一致；控制/分析 hash 不在此列）。

    val_identity（仅 VAL）：manifest hash / root_seed / scene ids / contract / profile /
    disjoint registry 的冻结身份。
    contract_identity（任何显式 contract 的正式角色）：raw/effective contract hash +
    contract 文件 hash + profile 文件 hash，防止续跑混入不同物理合同。
    """
    result = {
        'role': role,
        'compute_sha256': compute_sha256,
        'data_sha256': data,
        'profile_hash': profile_hash,
        'cells': [{'type': c['type'], 'edod': c['edod'], 'file': c['file'],
                   'sha256': c['sha256'], 'instance_seeds': c['instance_seeds']}
                  for c in cells],
        'instance_set': list(range(n_per_cell)),
        'config': dict(FROZEN_CONFIG),
    }
    if val_identity is not None:
        result['val_identity'] = val_identity
    if contract_identity is not None:
        result['contract_identity'] = contract_identity
    return result


def _validate_frozen_compute(old, new):
    for key in ('role', 'compute_sha256', 'data_sha256', 'profile_hash', 'cells',
                'instance_set', 'config', 'val_identity', 'contract_identity'):
        if key == 'role' and key not in old:
            # 旧 dev_gate 冻结计划无 role 字段：只允许隐含 dev_gate 续跑；VAL 必须显式 role=val。
            if new.get('role') == 'dev_gate':
                continue
            raise ContractError('VAL 冻结计划必须显式 role=val（旧 dev_gate 计划无 role，'
                                '不得用作 VAL）')
        if key in ('val_identity', 'contract_identity') and key not in old and key not in new:
            continue
        if old.get(key) != new.get(key):
            raise ContractError(
                f'冻结计划字段 {key} 不一致（拒绝续跑）：'
                f'\n  old={old.get(key)}\n  new={new.get(key)}')
