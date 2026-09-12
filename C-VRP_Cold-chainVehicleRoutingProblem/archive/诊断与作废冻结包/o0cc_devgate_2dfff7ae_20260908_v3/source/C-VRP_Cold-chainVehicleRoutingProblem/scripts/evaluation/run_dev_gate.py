"""DEV-GATE128 启动驱动（控制层）。

流程：
  1. 记录 pre-run manifest（命令 / 计算·控制·分析三套代码 hash / 数据 hash / profile hash /
     实例集合 / cell 身份与 seed 映射 / 环境）；
  2. 逐 cell 跑 runner（--role regression，仅补缺失实例 = 断点续跑），总并发 8 workers（逐 cell 串行）；
  3. 复用已有实例前做严格验收（共享 instance_validation），只补「缺失」，error / protocol /
     service fail 一律停止；每次 attempt 独立落日志与记录，不覆盖历史；
  4. cell 全部实例落盘后，从全部原始实例重建 canonical summary / manifest / CSV / log
     （cell_summary.rebuild_cell_summary），保证汇总代表完整 cell 而非「最后一次补算」；
  5. 9 cell 全部 128 实例齐全后，调冻结聚合器（--mode dev_gate + --frozen-plan）输出正式判定。

版本分离：续跑完整比较「冻结计算协议」（compute_sha256 + 数据 + profile + cells/seed 映射 +
instance_set + config：capacity/slack/baseline/n_boot/boot_seed）；控制/分析 hash 可独立演进，
不因此要求重算已完成的 1152 实例。若计算版本或任一计算配置改变，续跑会正确拒绝。

用法：
    python scripts/evaluation/run_dev_gate.py \
        --data-dir data/baseline/50_node/dev_gate \
        --profile results/o0cc/scale_v2/objective_profile.json \
        --workers 8 --out results/o0cc/dev_gate
"""
import argparse, os, sys, json, subprocess, hashlib, datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS) if os.path.basename(_SCRIPTS) == 'scripts' else _SCRIPTS
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from instance_validation import validate_instance_record
from cell_summary import rebuild_cell_summary

RUNNER = os.path.join(_SCRIPTS, 'evaluation', 'run_action_oracle.py')
AGGREGATOR = os.path.join(_SCRIPTS, 'evaluation', 'aggregate_cross_cell.py')

# 计算版本：决定 oracle/评估/环境/合同行为。续跑时不得改变。
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

# 控制版本：驱动 / 复用验收 / 断点恢复 / 规范汇总。可独立演进。
CONTROL_FILES = [
    'scripts/evaluation/run_dev_gate.py',
    'scripts/evaluation/instance_validation.py',
    'scripts/evaluation/cell_summary.py',
]

# 分析版本：聚合器。可独立演进。
ANALYSIS_FILES = [
    'scripts/evaluation/aggregate_cross_cell.py',
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _version(files):
    hashes = {os.path.basename(f): _sha256_file(os.path.join(_CVRPTW, f)) for f in files}
    payload = json.dumps(sorted(hashes.items()), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest(), hashes


def _env_info():
    info = {}
    for cmd, key in [('uname -a', 'uname'), ('python --version', 'python'),
                     ('which python', 'python_path')]:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        info[key] = (r.stdout or r.stderr).strip()
    r = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',
                       shell=True, capture_output=True, text=True)
    info['gpu'] = (r.stdout or r.stderr).strip()
    return info


def _cell_tag(c):
    return f"{c['type'].lower()}_{int(c['edod']*10):02d}"


def _append_attempt(attempt_path, rec):
    with open(attempt_path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


# 冻结的计算口径（正式 DEV-GATE）
CAPACITY = 50.0
NUM_VEHICLES = 25
SLACK_VEHICLES = 1
OBJECTIVE = 'coldchain'
N_BOOT = 10000
BOOT_SEED = 42
BASELINE = 'JF1-H-F'
EXPECTED_CELLS = [f'{t.lower()}_{int(e*10):02d}' for t in ('R1', 'C1', 'RC1') for e in (0.2, 0.5, 0.8)]


def _recompute_profile_hash(profile_path):
    data = json.load(open(profile_path))
    payload = json.dumps({
        'name': data['name'], 'distance_scale': data['distance_scale'],
        'quality_scale': data['quality_scale'], 'energy_scale': data['energy_scale'],
        'lambda_quality': data['lambda_quality'], 'lambda_energy': data['lambda_energy'],
        'scale_source': data.get('scale_source', 'pilot'),
        'dev_statistics': data.get('dev_statistics'),
    }, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _frozen_compute(cells, n_per_cell, compute_sha256, data, profile_hash):
    """冻结计划中「计算相关」字段（续跑必须一致；控制/分析 hash 不在此列）。"""
    return {
        'compute_sha256': compute_sha256,
        'data_sha256': data,
        'profile_hash': profile_hash,
        'cells': [{'type': c['type'], 'edod': c['edod'], 'file': c['file'],
                   'sha256': c['sha256'], 'instance_seeds': c['instance_seeds']}
                  for c in cells],
        'instance_set': list(range(n_per_cell)),
        'config': {
            'capacity': CAPACITY, 'num_vehicles': NUM_VEHICLES,
            'slack_vehicles': SLACK_VEHICLES, 'objective': OBJECTIVE,
            'local': True, 'n_boot': N_BOOT, 'boot_seed': BOOT_SEED,
            'baseline': BASELINE,
        },
    }


def _validate_frozen_compute(old, new):
    for key in ('compute_sha256', 'data_sha256', 'profile_hash', 'cells',
                'instance_set', 'config'):
        if old.get(key) != new.get(key):
            raise SystemExit(f'冻结计划字段 {key} 不一致（拒绝续跑）：'
                             f'\n  old={old.get(key)}\n  new={new.get(key)}')


def _verify_cell_after(data_path, cell, profile_path, profile_hash, tag):
    """cell 完成后 re-verify NPZ + profile（防止运行期间被替换形成混合数据）。"""
    if _sha256_file(data_path) != cell['sha256']:
        raise SystemExit(f'[{tag}] NPZ 在运行期间被替换')
    if _recompute_profile_hash(profile_path) != profile_hash:
        raise SystemExit(f'[{tag}] profile 在运行期间被替换')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data_dir, 'DEV_MANIFEST.json')))
    cells = manifest['cells']
    if not cells or len(cells) != 9:
        raise SystemExit(f'DEV_MANIFEST cells 非 9：{len(cells) if cells else 0}')
    tags = [_cell_tag(c) for c in cells]
    if set(tags) != set(EXPECTED_CELLS):
        raise SystemExit(f'cell 集合不符：{sorted(tags)} vs 预期 {EXPECTED_CELLS}')
    if manifest.get('split_role') != 'dev_gate':
        raise SystemExit(f'要求 split_role=dev_gate，实际 {manifest.get("split_role")}')
    n_per_cell = int(manifest.get('num_instances_per_cell', 0))
    if n_per_cell != 128:
        raise SystemExit(f'num_instances_per_cell 非 128：{n_per_cell}')
    first_seeds = [int(s) for s in cells[0].get('instance_seeds', [])]
    if len(first_seeds) < n_per_cell:
        raise SystemExit(f'instance_seeds 不足 {n_per_cell}：{len(first_seeds)}')
    if len(set(first_seeds)) != n_per_cell:
        raise SystemExit(f'instance_seeds 前 {n_per_cell} 非唯一')
    for c in cells[1:]:
        if [int(s) for s in c.get('instance_seeds', [])] != first_seeds:
            raise SystemExit('九 cell instance_seeds 不一致')
    profile = json.load(open(args.profile))
    profile_hash = profile.get('profile_hash')
    if not profile_hash:
        raise SystemExit('profile 缺 profile_hash')
    # 写 pre-run manifest / 启动 runner 前先校验 profile 内容自洽（避免先建冻结记录、进 cell 才失败）
    if _recompute_profile_hash(args.profile) != profile_hash:
        raise SystemExit(f'profile 内容 hash 与自报不符：'
                         f'{_recompute_profile_hash(args.profile)[:12]} vs {profile_hash[:12]}')

    os.makedirs(args.out, exist_ok=True)

    # 1. 计算运行身份（计算/控制/分析三套 hash + 数据 NPZ 实读 hash + profile hash）
    compute_sha256, compute_files = _version(COMPUTE_FILES)
    control_sha256, control_files = _version(CONTROL_FILES)
    analysis_sha256, analysis_files = _version(ANALYSIS_FILES)
    data = {}
    for c in cells:
        npz = os.path.join(args.data_dir, c['file'])
        actual = _sha256_file(npz)
        if actual != c['sha256']:
            raise SystemExit(f"{c['file']} 数据 hash 不符：manifest={c['sha256'][:12]} "
                             f"actual={actual[:12]}")
        data[c['file']] = actual

    pre_path = os.path.join(args.out, 'pre_run_manifest.json')
    run_id = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    frozen = _frozen_compute(cells, n_per_cell, compute_sha256, data, profile_hash)

    if os.path.exists(pre_path):
        # 续跑：完整比较冻结计算协议（compute/data/profile/cells/instance_set/config）
        old = json.load(open(pre_path))
        _validate_frozen_compute(old, frozen)
        run_id = old.get('run', {}).get('run_id', run_id)
        print(f'续跑：沿用 run_id={run_id}，冻结计算协议一致（控制/分析可独立演进）')
    else:
        # 首次启动：输出目录必须为空，防止接管没有冻结记录的旧实例
        existing = os.listdir(args.out) if os.path.isdir(args.out) else []
        if existing:
            raise SystemExit(f'输出目录非空但无 pre_run_manifest.json，拒绝接管旧实例：'
                             f'{existing[:10]}')
        pre = {
            'run': {
                'run_id': run_id,
                'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
                'command': ' '.join(sys.argv),
                'data_dir': args.data_dir, 'profile': args.profile,
                'workers': args.workers, 'out': args.out,
                'instances_per_cell': n_per_cell, 'total_instances': n_per_cell * 9,
            },
            **frozen,
            'compute_files': compute_files,
            'control_sha256': control_sha256,
            'control_files': control_files,
            'analysis_sha256': analysis_sha256,
            'analysis_files': analysis_files,
            'env': _env_info(),
        }
        with open(pre_path, 'w') as f:
            json.dump(pre, f, indent=2)
        print(f'pre-run manifest: {pre_path}（run_id={run_id}）')

    attempt_path = os.path.join(args.out, 'attempts.jsonl')

    # 2. 逐 cell 跑（断点续跑：只补「缺失」实例；error / protocol / service fail 停止，不覆盖）
    for c in cells:
        tag = _cell_tag(c)
        cell_dir = os.path.join(args.out, tag)
        data_path = os.path.join(args.data_dir, c['file'])
        inst_dir = os.path.join(cell_dir, 'instances')
        os.makedirs(inst_dir, exist_ok=True)

        # 每次 cell 前核验：代码集合 + 当前 cell NPZ + profile 内容 hash（防止运行中数据/profile 被替换）
        cur_compute, _ = _version(COMPUTE_FILES)
        if cur_compute != compute_sha256:
            raise SystemExit(f'[{tag}] 计算文件集合已漂移：{cur_compute} != {compute_sha256}')
        npz_hash = _sha256_file(data_path)
        if npz_hash != c['sha256']:
            raise SystemExit(f'[{tag}] NPZ hash 漂移：{npz_hash[:12]} != {c["sha256"][:12]}')
        if _recompute_profile_hash(args.profile) != profile_hash:
            raise SystemExit(f'[{tag}] profile 内容 hash 漂移')

        missing = []
        for i in range(n_per_cell):
            p = os.path.join(inst_dir, f'inst_{i}.json')
            if not os.path.exists(p):
                missing.append(i)  # 文件不存在 → 可补算
                continue
            try:
                with open(p) as f:
                    rec = json.load(f)
            except Exception as e:
                raise SystemExit(f'[{tag}] inst_{i}.json 无法解析（保留现场，不覆盖）：{e}')
            if not isinstance(rec, dict):
                raise SystemExit(f'[{tag}] inst_{i}.json 内容非对象（保留现场，不覆盖）：'
                                 f'{type(rec).__name__}')
            state, err = validate_instance_record(rec, i, 'coldchain', require_local=True,
                                                  expected_code_sha256=compute_files['run_action_oracle.py'])
            if state == 'valid':
                continue
            # error / protocol_fail / service_fail / missing_local → 停止，不覆盖
            raise SystemExit(f'[{tag}] inst_{i} 状态 {state}（不覆盖）：{err}')

        if not missing:
            print(f'[{tag}] {n_per_cell}/{n_per_cell} 已完整，跳过补算；重建 canonical 汇总')
            rebuild_cell_summary(cell_dir, objective='coldchain', role='regression',
                                 profile_manifest=profile, data_sha256=c['sha256'],
                                 expected_code_sha256=compute_files['run_action_oracle.py'], compute_sha256=compute_sha256, run_id=run_id, workers=args.workers,
                                 launch_mode='resume-canonical', require_local=True, expected_instance_ids=list(range(n_per_cell)))
            _verify_cell_after(data_path, c, args.profile, profile_hash, tag)
            continue

        ids = ','.join(str(i) for i in missing)
        ts = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
        log = os.path.join(args.out, f'{tag}.{ts}.log')
        cmd = [sys.executable, RUNNER, '--data', data_path, '--capacity', '50', '--num_vehicles', '25',
               '--objective', 'coldchain', '--objective-profile', args.profile,
               '--local', '--role', 'regression', '--instance-ids', ids,
               '--workers', str(args.workers), '--out', cell_dir]
        print(f'[{tag}] 跑 {len(missing)}/{n_per_cell} 实例（instance-ids={ids[:60]}...）')
        attempt_id = f'{tag}.{ts}'
        start = datetime.datetime.utcnow().isoformat() + 'Z'
        # 启动前先持久化 START（即使中途断掉也有记录）
        _append_attempt(attempt_path, {
            'attempt_id': attempt_id, 'run_id': run_id, 'tag': tag,
            'status': 'START', 'timestamp_utc': start,
            'command': ' '.join(cmd), 'n_missing': len(missing),
            'instance_ids': missing, 'log': log,
            'compute_sha256': compute_sha256,
            'runner_sha256': compute_files['run_action_oracle.py'],
            'data_sha256': c['sha256'],
            'profile_hash': profile_hash,
            'control_sha256': control_sha256,
        })
        with open(log, 'w') as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        end = datetime.datetime.utcnow().isoformat() + 'Z'
        _append_attempt(attempt_path, {
            'attempt_id': attempt_id, 'run_id': run_id, 'tag': tag,
            'status': 'SUCCESS' if r.returncode == 0 else 'FAIL',
            'timestamp_utc': end, 'start_utc': start, 'end_utc': end,
            'exit_code': r.returncode, 'log': log,
            'compute_sha256': compute_sha256,
            'runner_sha256': compute_files['run_action_oracle.py'],
            'data_sha256': c['sha256'],
            'profile_hash': profile_hash,
            'control_sha256': control_sha256,
        })
        if r.returncode != 0:
            print(f'[{tag}] FAIL（exit={r.returncode}），详见 {log}。停止，不换实例。')
            sys.exit(1)
        # 全部实例落盘后重建 canonical 汇总（代表完整 cell，非本次补算子集）
        rebuild_cell_summary(cell_dir, objective='coldchain', role='regression',
                             profile_manifest=profile, data_sha256=c['sha256'],
                             expected_code_sha256=compute_files['run_action_oracle.py'], compute_sha256=compute_sha256, run_id=run_id, workers=args.workers,
                             launch_mode='canonical', require_local=True, expected_instance_ids=list(range(n_per_cell)))
        _verify_cell_after(data_path, c, args.profile, profile_hash, tag)
        print(f'[{tag}] 完成')

    # 聚合前最后核验计算文件集合
    cur_compute, _ = _version(COMPUTE_FILES)
    if cur_compute != compute_sha256:
        raise SystemExit(f'聚合前计算文件集合漂移：{cur_compute} != {compute_sha256}')

    # 3. 全部齐全后聚合（带冻结计划校验）
    results = [os.path.join(args.out, _cell_tag(c)) for c in cells]
    summary = os.path.join(args.out, 'dev_gate_summary.json')
    cmd = [sys.executable, AGGREGATOR, '--results'] + results + [
        '--manifest', os.path.join(args.data_dir, 'DEV_MANIFEST.json'),
        '--mode', 'dev_gate', '--frozen-plan', pre_path, '--profile', args.profile,
        '--out', summary]
    print('9 cell 齐全，运行聚合器（--mode dev_gate + --frozen-plan）')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print('聚合 FAIL'); sys.exit(1)
    print(f'DEV-GATE 汇总: {summary}')


if __name__ == '__main__':
    main()
