"""九场景跨 cell 按内部 seed 成组汇总（DEV-GATE 正式入口）。

九 cell（R1/C1/RC1 × EDoD 0.2/0.5/0.8）共用同一组内部 instance seeds，因此 1152 条
instance-delta 不是独立样本。正确 CI = 按内部 seed 成组 bootstrap：每次抽 N 个 seed 组
（with replacement），保留每组九 cell 的 delta，算九 cell 等权均值，再取 bootstrap CI。

主终点：九 cell 等权的 sequential 平均配对 Δ（paired delta = oracle_cost − baseline_cost）。

本脚本是正式入口，做完整防错：
  1. 身份：依据运行 manifest 的 data_sha256 与 DEV_MANIFEST cell 的 sha256 识别真实 cell，
     拒绝重复目录 / 缺失 / 非预定 cell；输入顺序只改读取顺序，不改标签。
  2. 身份字段缺失（code_sha256 / profile_hash / data_sha256 任一缺失或 null）直接拒绝。
  3. 冻结计划校验（--frozen-plan）：实际 code/profile/data/seed 映射必须等于预先冻结的
     运行计划，而非仅九份互相一致。
  4. 完整样本：formal 强制每 cell 128 实例（1152 记录）；smoke 允许小样本（如 2）。
  5. 逐实例 Gate 复算（不信任旧 summary）：protocol（strict repair）/ service / repair /
     非退化 / 有限 / inst_idx 严格整数 / local 对账，任一失败非零退出。
  6. seed 对应：九 cell 的 instance_seeds 必须一致。
  7. 角色：regression（诊断）不产正式 GO；dev_gate 输出整体判定。
  8. local 辅助：九 cell 等权（先各 cell 的 eligible 均值，再等权）；区分「完成且无
     eligible」与「未采集 local」；不可估计的 cell / 重采样直接报告，不静默删 cell 或填零。

主终点公式（不变）：
    g_s = (1/9) Σ_c (J^oracle_{c,s} − J^baseline_{c,s})
    Δ̄ = (1/N) Σ_s g_s；对 N 个 g_s 重采样得 CI。

用法：
    python scripts/evaluation/aggregate_cross_cell.py \
        --results <9 个 result 目录> --manifest <DEV_MANIFEST.json> \
        --mode dev_gate --frozen-plan <pre_run_manifest.json> --profile <objective_profile.json> \
        --out results/o0cc/dev_gate_summary.json
"""
import argparse, os, sys, json, math, hashlib
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from instance_validation import validate_instance_record, _cost

TYPES = ('R1', 'C1', 'RC1')
EDODS = (0.2, 0.5, 0.8)
EXPECTED_CELLS = [f'{t.lower()}_{int(e*10):02d}' for t in TYPES for e in EDODS]


def _fail(msg):
    print(f'ERROR: {msg}')
    sys.exit(1)


def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _verify_canonical_marker(result_dir, cell, m, frozen_compute_sha256=None, frozen_run_id=None):
    """读取 cell 前核验 canonical.COMPLETE 完成标记 + 四个产物 hash（发布完整性）。"""
    marker_path = os.path.join(result_dir, 'canonical.COMPLETE')
    if not os.path.exists(marker_path):
        _fail(f'{cell} 缺 canonical.COMPLETE 完成标记')
    marker = _read_json(marker_path)
    if marker.get('status') != 'COMPLETE':
        _fail(f'{cell} canonical 未完成（status={marker.get("status")}）')
    if marker.get('n') != m or marker.get('instance_ids') != list(range(m)):
        _fail(f'{cell} canonical instance_ids 与实例集合不符')
    if frozen_compute_sha256 is not None and marker.get('compute_sha256') != frozen_compute_sha256:
        _fail(f'{cell} canonical compute_sha256 与冻结计划不符')
    if frozen_run_id is not None and marker.get('run_id') != frozen_run_id:
        _fail(f'{cell} canonical run_id 与冻结计划不符（跨 run 混入）')
    hashes = marker.get('artifact_hashes') or {}
    for f in ('summary.json', 'manifest.json', 'per_instance.csv', 'oracle_log.jsonl'):
        p = os.path.join(result_dir, f)
        if not os.path.exists(p):
            _fail(f'{cell} 缺产物 {f}')
        if hashes.get(f) != _sha256_file(p):
            _fail(f'{cell} 产物 {f} hash 与完成标记不符（被篡改或未原子发布）')


def _validate_formal_protocol(frozen_plan, args):
    """正式模式：冻结的 instance_set + config + 统计口径必须一致，CLI 参数必须匹配冻结口径。

    防止用不同 bootstrap（n_boot/seed）或 baseline/slack/capacity 配置产出正式 verdict。
    """
    inst_set = frozen_plan.get('instance_set')
    if inst_set != list(range(128)):
        _fail(f'frozen instance_set 非 0..127：{inst_set if isinstance(inst_set, list) else inst_set!r}')
    cfg = frozen_plan.get('config') or {}
    expect = {
        'capacity': 50.0, 'num_vehicles': 25, 'slack_vehicles': 1,
        'objective': 'coldchain', 'local': True, 'baseline': 'JF1-H-F',
        'n_boot': 10000, 'boot_seed': 42,
    }
    for k, v in expect.items():
        if cfg.get(k) != v:
            _fail(f'frozen config.{k} 不符：{cfg.get(k)!r} != {v!r}')
    # CLI 参数必须与冻结口径一致
    if args.objective != cfg.get('objective'):
        _fail(f'--objective 与冻结不符：{args.objective!r} != {cfg.get("objective")!r}')
    if args.local != cfg.get('local'):
        _fail(f'--local 与冻结不符：{args.local!r} != {cfg.get("local")!r}')
    if args.n_boot != cfg.get('n_boot'):
        _fail(f'--n-boot 与冻结不符：{args.n_boot!r} != {cfg.get("n_boot")!r}')
    if args.seed != cfg.get('boot_seed'):
        _fail(f'--seed 与冻结不符：{args.seed!r} != {cfg.get("boot_seed")!r}')


def _recompute_profile_hash(profile_path):
    """从 profile 内容重算 hash（与 ObjectiveProfile.profile_hash 同公式，不引入重依赖）。"""
    data = _read_json(profile_path)
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


def _combined_hash(file_hashes):
    """与 run_dev_gate._version 同规则的多文件组合 hash（sorted + compact JSON）。"""
    payload = json.dumps(sorted(file_hashes.items()), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _resolve_identity(result_dir, manifest):
    """用运行 manifest 的 data_sha256 与 DEV_MANIFEST cell 的 sha256 识别真实 cell。"""
    rm = _read_json(os.path.join(result_dir, 'manifest.json'))
    data_hash = rm.get('data_sha256')
    if not data_hash:
        _fail(f'{result_dir}/manifest.json 缺 data_sha256')
    for c in manifest['cells']:
        if c['sha256'] == data_hash:
            return f"{c['type'].lower()}_{int(c['edod']*10):02d}", c
    _fail(f'{result_dir} 的 data_sha256 不匹配任何 DEV_MANIFEST cell')


def _validate_result_manifests(result_dirs, frozen_plan=None, profile_hash=None):
    """核对 9 个 result 的运行身份：必需字段非空 + 九份一致 + 冻结计划校验。"""
    ref = None
    for rd in result_dirs:
        rm = _read_json(os.path.join(rd, 'manifest.json'))
        code = rm.get('code_sha256')
        prof = rm.get('profile_hash')
        if not code:
            _fail(f'{rd} manifest 缺 code_sha256')
        if not prof:
            _fail(f'{rd} manifest 缺 profile_hash')
        key = (rm.get('objective'), code, prof, rm.get('n_failed'))
        if ref is None:
            ref = key
        elif key != ref:
            _fail(f'{rd} 运行身份不一致：{key} vs {ref}')
    if ref[0] != 'coldchain':
        _fail(f'objective 非 coldchain：{ref[0]}')
    if ref[3] != 0:
        _fail(f'n_failed 非 0：{ref[3]}')
    if frozen_plan is not None:
        fp_files = frozen_plan.get('compute_files')
        fp_compute = frozen_plan.get('compute_sha256')
        fp_profile = frozen_plan.get('profile_hash')
        if not fp_files:
            _fail('frozen plan 缺 compute_files')
        if not fp_compute:
            _fail('frozen plan 缺 compute_sha256')
        fp_runner = fp_files.get('run_action_oracle.py')
        if not fp_runner:
            _fail('frozen plan compute_files 缺 run_action_oracle.py')
        # 1. manifest.code_sha256（runner 单文件 hash）== 冻结计划里的 runner hash
        if ref[1] != fp_runner:
            _fail(f'runner hash 与冻结计划不符：{ref[1][:12]} vs {fp_runner[:12]}')
        # 2. 冻结计划自身一致：从 compute_files 重算组合 hash == compute_sha256
        if _combined_hash(fp_files) != fp_compute:
            _fail('frozen plan compute_sha256 与 compute_files 不一致')
        # 3. profile
        if fp_profile is not None and ref[2] != fp_profile:
            _fail(f'profile_hash 与冻结计划不符：{ref[2][:12]} vs {fp_profile[:12]}')
    if profile_hash is not None and ref[2] != profile_hash:
        _fail(f'profile 内容 hash 与 manifest 不符：{profile_hash[:12]} vs {ref[2][:12]}')
    return ref


def _validate_frozen_cells(cells, frozen_plan):
    """DEV_MANIFEST 的 cell 身份（type/edod/file/sha256/seed 映射）必须等于冻结计划。"""
    fc = frozen_plan.get('cells')
    if not fc:
        _fail('frozen plan 缺 cells（冻结 cell/seed 信息）')
    if len(fc) != len(cells):
        _fail(f'冻结计划 cells 数量 {len(fc)} 与 DEV_MANIFEST {len(cells)} 不符')
    for i, c in enumerate(cells):
        f = fc[i]
        if (c.get('type'), c.get('edod'), c.get('file'), c.get('sha256'),
                c.get('instance_seeds')) != (f.get('type'), f.get('edod'), f.get('file'),
                                             f.get('sha256'), f.get('instance_seeds')):
            _fail(f'DEV_MANIFEST cell {i} 与冻结计划不符')


def _load_instance_deltas(result_dir, cell, n, objective, require_local,
                          expected_code_sha256=None):
    """读逐实例 paired delta + local 三态，附带逐实例 Gate 复算 + 内部 ID + 计算版本身份校验。"""
    deltas = []
    local_states = []  # ('eligible', val) | ('no_eligible', None) | ('missing', None)
    for i in range(n):
        p = os.path.join(result_dir, 'instances', f'inst_{i}.json')
        if not os.path.exists(p):
            _fail(f'{cell} 缺 {p}')
        rec = _read_json(p)
        state, err = validate_instance_record(rec, i, objective, require_local=require_local,
                                              expected_code_sha256=expected_code_sha256)
        if state == 'missing':
            _fail(f'{cell} inst_{i} 缺失')
        if state in ('error', 'protocol_fail', 'service_fail', 'missing_local'):
            _fail(f'{cell} inst_{i} {state}: {err}')
        b, o = rec['baseline'], rec['oracle']
        deltas.append(_cost(o, objective) - _cost(b, objective))
        loc = rec.get('local')
        if isinstance(loc, dict):
            ed = loc.get('event_deltas') or []
            if ed:
                local_states.append(('eligible', float(np.mean(ed))))
            else:
                local_states.append(('no_eligible', None))
        else:
            local_states.append(('missing', None))
    return deltas, local_states


def _local_nine_cell_equal_weight(local_matrix, seeds, local_cells, n_boot, seed):
    """local 九 cell 等权统计量。

    local_matrix: seed -> {cell: ('eligible'|'no_eligible'|'missing', val)}。
    先各 cell 的 eligible 均值，再九 cell 等权；任一 cell 无 eligible 样本即「不可估计」。
    bootstrap 每次抽完整 seed 行后重算同一统计量；重采样后某 cell 无 eligible 的重采样
    记为不可估计（不静默删 cell / 填零 / 丢弃不利重采样）。

    返回 dict（mean/ci/可靠性与计数）。mean 为 None 表示整体不可估计。
    """
    out = {'mean': None, 'ci': None, 'non_estimable_cells': [],
           'n_missing': 0, 'n_no_eligible': 0,
           'n_boot_non_estimable': 0, 'ci_reliable': False}
    for row in local_matrix.values():
        for st in row.values():
            if st[0] == 'missing':
                out['n_missing'] += 1
            elif st[0] == 'no_eligible':
                out['n_no_eligible'] += 1

    def _cell_eligible_mean(seed_list, cell):
        vals = [local_matrix[s][cell][1] for s in seed_list
                if local_matrix[s][cell][0] == 'eligible']
        return float(np.mean(vals)) if vals else None

    cell_means = {c: _cell_eligible_mean(seeds, c) for c in local_cells}
    out['non_estimable_cells'] = [c for c, m in cell_means.items() if m is None]
    if out['non_estimable_cells']:
        return out
    out['mean'] = float(np.mean(list(cell_means.values())))

    rng = np.random.default_rng(seed + 1)
    boot_local = []
    for _ in range(n_boot):
        s_sample = rng.choice(seeds, size=len(seeds), replace=True)
        cm = {}
        ok = True
        for c in local_cells:
            v = _cell_eligible_mean(list(s_sample), c)
            if v is None:
                ok = False
                break
            cm[c] = v
        if not ok:
            out['n_boot_non_estimable'] += 1
            continue
        boot_local.append(float(np.mean(list(cm.values()))))
    if boot_local:
        lo, hi = np.percentile(boot_local, [2.5, 97.5])
        ci = [float(lo), float(hi)]
    else:
        ci = None
    if out['n_boot_non_estimable'] == 0:
        out['ci'] = ci
        out['ci_reliable'] = True
    else:
        # 出现不可估计重采样时，正式 local CI 不可用；条件化区间只作诊断，不得触发 lookahead
        out['ci'] = None
        out['ci_diagnostic'] = ci
        out['ci_reliable'] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--mode', choices=['regression', 'dev_gate'], default='regression')
    ap.add_argument('--objective', default='coldchain')
    ap.add_argument('--frozen-plan', default=None,
                    help='pre_run_manifest.json；提供时校验实际身份 == 冻结计划')
    ap.add_argument('--profile', default=None,
                    help='objective_profile.json；从内容重算 hash 并校验')
    ap.add_argument('--local', dest='local', action='store_true', default=True)
    ap.add_argument('--no-local', dest='local', action='store_false')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    manifest = _read_json(args.manifest)
    cells = manifest.get('cells')
    if not cells or len(cells) != 9:
        _fail('DEV_MANIFEST cells 缺失或非 9')

    frozen_plan = _read_json(args.frozen_plan) if args.frozen_plan else None
    expected_code_sha256 = None
    if frozen_plan is not None and frozen_plan.get('compute_files'):
        expected_code_sha256 = frozen_plan['compute_files'].get('run_action_oracle.py')
    profile_hash = _recompute_profile_hash(args.profile) if args.profile else None
    if args.profile:
        file_hash = _read_json(args.profile).get('profile_hash')
        if profile_hash != file_hash:
            _fail(f'profile 文件自洽性失败：recomputed={profile_hash[:12]} file={file_hash}')
        if frozen_plan is not None and file_hash != frozen_plan.get('profile_hash'):
            _fail(f'profile 与冻结计划 profile_hash 不符：{file_hash[:12]} '
                  f'vs {frozen_plan.get("profile_hash")}')

    # 0. 严格九个输入目录 + 正式数据角色
    if len(args.results) != 9:
        _fail(f'需要恰好 9 个目录，得到 {len(args.results)}')
    if args.mode == 'dev_gate' and manifest.get('split_role') != 'dev_gate':
        _fail(f'--mode dev_gate 要求 split_role=dev_gate，实际 {manifest.get("split_role")}')
    if args.mode == 'dev_gate':
        if frozen_plan is None:
            _fail('--mode dev_gate 要求 --frozen-plan')
        if args.profile is None:
            _fail('--mode dev_gate 要求 --profile')
        if not frozen_plan.get('cells'):
            _fail('--mode dev_gate 要求 frozen plan 含 cells（冻结 cell/seed 信息）')
        _validate_formal_protocol(frozen_plan, args)

    # 0.5 冻结 cell 身份校验（seed 映射 / 数据 hash 对不上会被发现）
    if frozen_plan is not None:
        _validate_frozen_cells(cells, frozen_plan)

    # 1. 身份识别 + 拒绝重复 + 运行身份一致
    resolved = []
    for rd in args.results:
        cell, c = _resolve_identity(rd, manifest)
        resolved.append((rd, cell, c))
    cell_ids = [cell for _, cell, _ in resolved]
    if len(set(cell_ids)) != 9:
        _fail(f'目录未覆盖 9 个唯一 cell：{cell_ids}')
    if set(cell_ids) != set(EXPECTED_CELLS):
        _fail(f'cell 集合不符：{sorted(set(cell_ids))} vs 预期 {EXPECTED_CELLS}')
    run_identity = _validate_result_manifests(args.results, frozen_plan, profile_hash)

    # 2. seed 对应验证：九 cell 的 instance_seeds 必须一致
    first_seeds = [int(s) for s in cells[0]['instance_seeds']]
    for c in cells[1:]:
        if [int(s) for s in c['instance_seeds']] != first_seeds:
            _fail('九 cell 的 instance_seeds 不一致')

    # 3. 完整样本：每个 cell 的实例文件 ID 必须一致（连续前缀 0..m-1），formal 要求 m==128
    file_id_sets = []
    for rd, cell, _ in resolved:
        ids = sorted(int(f[len('inst_'):-len('.json')]) for f in
                     os.listdir(os.path.join(rd, 'instances'))
                     if f.startswith('inst_') and f.endswith('.json'))
        file_id_sets.append(ids)
    m = len(file_id_sets[0])
    if file_id_sets[0] != list(range(m)):
        _fail(f'实例文件 ID 不是连续前缀 0..{m-1}：{file_id_sets[0][:10]}...')
    for ids in file_id_sets[1:]:
        if ids != file_id_sets[0]:
            _fail('各 cell 实例文件 ID 不一致')
    if args.mode == 'dev_gate' and m != 128:
        _fail(f'--mode dev_gate 要求每 cell 128 实例，实际 {m}')
    if m > len(first_seeds):
        _fail(f'实例数 {m} 超过 manifest instance_seeds {len(first_seeds)}')
    seeds = first_seeds[:m]
    if len(set(seeds)) != m:
        _fail(f'instance_seeds 前 {m} 个非唯一')

    # 4. 逐实例 Gate 复算 + 读 delta/local 三态
    per_cell = []
    for rd, cell, c in resolved:
        if args.mode == 'dev_gate':
            _verify_canonical_marker(rd, cell, m,
                                     frozen_compute_sha256=frozen_plan.get('compute_sha256'),
                                     frozen_run_id=(frozen_plan.get('run') or {}).get('run_id'))
        deltas, local_states = _load_instance_deltas(rd, cell, m, args.objective, args.local,
                                                     expected_code_sha256)
        per_cell.append({'cell': cell, 'deltas': deltas, 'local_states': local_states})

    # 5. 按 seed 成组（sequential 主终点：每组必须恰好 9 cell）
    seed_groups = {s: [] for s in seeds}
    for pc in per_cell:
        for i, d in enumerate(pc['deltas']):
            seed_groups[seeds[i]].append(d)
    if any(len(v) != 9 for v in seed_groups.values()):
        bad = {s: len(v) for s, v in seed_groups.items() if len(v) != 9}
        _fail(f'seed 组不完整（非 9 cell）：{list(bad.items())[:5]}')

    seq_means = np.array([float(np.mean(v)) for v in seed_groups.values()])
    seq_mean = float(seq_means.mean())
    rng = np.random.default_rng(args.seed)
    boot = np.array([rng.choice(seq_means, size=len(seq_means), replace=True).mean()
                     for _ in range(args.n_boot)])
    seq_lo, seq_hi = np.percentile(boot, [2.5, 97.5])

    # 5.5 local 九 cell 等权：先各 cell 的 eligible 均值，再等权（不按「每 seed 有值的 cell」）。
    local_cells = [pc['cell'] for pc in per_cell]
    local_matrix = {}  # seed -> {cell: ('eligible'|'no_eligible'|'missing', val)}
    for pc in per_cell:
        for i, st in enumerate(pc['local_states']):
            local_matrix.setdefault(seeds[i], {})[pc['cell']] = st

    if args.local:
        local_stat = _local_nine_cell_equal_weight(
            local_matrix, seeds, local_cells, args.n_boot, args.seed)
    else:
        local_stat = {'mean': None, 'ci': None, 'non_estimable_cells': [],
                      'n_missing': 0, 'n_no_eligible': 0,
                      'n_boot_non_estimable': 0, 'ci_reliable': False}
    local_mean = local_stat['mean']
    local_lo, local_hi = local_stat['ci'] if local_stat['ci'] is not None else (None, None)
    non_estimable_cells = local_stat['non_estimable_cells']
    n_local_missing = local_stat['n_missing']
    n_local_no_eligible = local_stat['n_no_eligible']
    n_boot_non_estimable = local_stat['n_boot_non_estimable']
    local_ci_reliable = local_stat['ci_reliable']
    local_matrix_out = {str(s): {c: (st[0], st[1]) for c, st in row.items()}
                        for s, row in local_matrix.items()}

    # 6. verdict
    if args.mode == 'regression':
        verdict = 'PROTOCOL_PASS'  # 诊断：只报告，不产正式 GO
    else:
        if seq_mean < 0 and seq_hi < 0:
            verdict = 'GO'
        elif seq_mean < 0:
            verdict = 'EVIDENCE_INSUFFICIENT'
        else:
            verdict = 'NO_HEADROOM'

    result = {
        'mode': args.mode,
        'verdict': verdict,
        'objective': args.objective,
        'per_cell': [{'cell': pc['cell'], 'mean_delta': float(np.mean(pc['deltas'])),
                      'n_instances': m} for pc in per_cell],
        'nine_cell_equal_weight_mean_delta': seq_mean,
        'grouped_bootstrap_ci': [float(seq_lo), float(seq_hi)],
        'n_seed_groups': len(seq_means),
        'n_instances_per_cell': m,
        'total_instances': m * 9,
        'local_nine_cell_mean_delta': local_mean,
        'local_grouped_bootstrap_ci': [float(local_lo), float(local_hi)]
                                      if local_lo is not None else None,
        'local_non_estimable_cells': non_estimable_cells,
        'local_missing_count': n_local_missing,
        'local_no_eligible_count': n_local_no_eligible,
        'local_bootstrap_non_estimable_count': n_boot_non_estimable,
        'local_ci_reliable': local_ci_reliable,
        'local_ci_diagnostic': local_stat.get('ci_diagnostic'),
        'local_matrix': local_matrix_out,
        'run_identity': {'objective': run_identity[0], 'code_sha256': run_identity[1],
                         'profile_hash': run_identity[2]},
        'bootstrap': {'n_boot': args.n_boot, 'seed': args.seed},
        'manifest': args.manifest,
        'frozen_plan': args.frozen_plan,
        'result_dirs': args.results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'=== 九场景跨 cell 成组汇总（mode={args.mode}）===')
    for c in result['per_cell']:
        print(f"  {c['cell']}: mean_delta={c['mean_delta']:+.4f} (n={c['n_instances']})")
    print(f'  九 cell 等权平均 Δ = {seq_mean:+.4f}')
    print(f'  成组 bootstrap 95% CI = [{seq_lo:+.4f}, {seq_hi:+.4f}]')
    if local_mean is not None:
        print(f'  local 九 cell 平均 Δ = {local_mean:+.4f}  CI = [{local_lo:+.4f}, {local_hi:+.4f}]')
    elif args.local:
        print(f'  local 九 cell 平均 Δ = 不可估计（无 eligible 的 cell：{non_estimable_cells}）')
    print(f'  verdict = {verdict}')
    print(f'  saved: {args.out}')


if __name__ == '__main__':
    main()
