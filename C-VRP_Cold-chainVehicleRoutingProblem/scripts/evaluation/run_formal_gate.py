"""通用正式门控驱动器（DEV-GATE / O0-CC VAL 共用）。

从 `run_dev_gate.py` 的参数化重构：同一个正式门控核心，由 `--role` 区分 `dev_gate` 与
`val`。`run_dev_gate.py` 与 `run_o0cc_val.py` 只是固定 role 的薄入口，不复制计算逻辑。

流程：
  1. 加载并校验 manifest / contract / profile（角色一致性、九 cell、128 实例、共享 seed、
     VAL 的跨 split 独立性、数据文件 hash）；
  2. 计算并冻结 compute/control/analysis hash、manifest/profile/数据 hash、命令参数、环境；
  3. 任何实例运行前写 `pre_run_manifest.json`；首次运行要求输出目录为空，续跑只允许
     完全相同的冻结计算身份；
  4. 逐 cell 跑 runner（`--role regression`，仅补缺失实例），复用已有实例前做严格验收，
     error/protocol/service fail 一律停止；每次 attempt 独立落记录；
  5. cell 齐全后从全部原始实例重建 canonical 汇总；
  6. 九 cell 齐全后调聚合器（`--mode {role}` + `--frozen-plan`）输出正式判定。

用法：
    python scripts/evaluation/run_formal_gate.py \
        --role dev_gate|val --manifest <exact_manifest> --profile <exact_profile> \
        --workers 8 --out <fresh_output_dir> [--disjoint-manifest <other_manifest> ...]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import formal_gate_contract as contract
from instance_validation import validate_instance_record
from cell_summary import rebuild_cell_summary

_COLDCHAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coldchain')
if _COLDCHAIN not in sys.path:
    sys.path.insert(0, _COLDCHAIN)
from coldchain_contract import (load_coldchain_contract, ObjectiveProfile,  # noqa: E402
                                apply_objective_profile)

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_action_oracle.py')
AGGREGATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aggregate_cross_cell.py')


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


def _append_attempt(attempt_path, rec):
    with open(attempt_path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def _verify_cell_after(data_path, cell, profile_path, profile_hash, tag):
    """cell 完成后 re-verify NPZ + profile（防止运行期间被替换形成混合数据）。"""
    if contract.sha256_file(data_path) != cell['sha256']:
        raise SystemExit(f'[{tag}] NPZ 在运行期间被替换')
    if contract.recompute_profile_hash(profile_path) != profile_hash:
        raise SystemExit(f'[{tag}] profile 在运行期间被替换')


def _fail(msg):
    print(f'ERROR: {msg}')
    sys.exit(1)


def run(role, manifest, profile, workers, out, disjoint_manifests=None, split_registry=None,
        coldchain_contract=None, statistical_plan=None):
    if role not in contract.FORMAL_ROLES:
        _fail(f'--role 只允许 {contract.FORMAL_ROLES}，得到 {role!r}')
    manifest = os.path.abspath(manifest)
    profile = os.path.abspath(profile)
    out = os.path.abspath(out)
    data_dir = os.path.dirname(manifest)

    manifest_obj = contract.load_json(manifest)
    cells, n_per_cell = contract.validate_manifest(manifest_obj, role)
    manifest_sha256 = contract.sha256_file(manifest)

    # VAL：必须提供完整 split registry，做跨 split 独立性硬校验；禁止只靠可选 --disjoint-manifest。
    disjoint_report = []
    registry_sha256 = None
    if role == 'val':
        if not split_registry:
            _fail('--role val 必须提供 --split-registry（完整已登记 split 清单）')
        disjoint_report, _reg = contract.check_registry_disjoint(manifest_obj, split_registry)
        registry_sha256 = contract.sha256_file(split_registry)
    elif disjoint_manifests:
        disjoint_report = contract.check_disjoint(manifest_obj, disjoint_manifests)
    bad = [e for e in disjoint_report if e['seed_overlap'] or e['scene_id_overlap']]
    if bad:
        _fail(f'{role} manifest 与已登记 split 重叠（独立性违反）：'
              f'\n{json.dumps(bad, indent=2, ensure_ascii=False)}')

    profile_obj = contract.load_json(profile)
    profile_hash = profile_obj.get('profile_hash')
    if not profile_hash:
        _fail('profile 缺 profile_hash')
    if contract.recompute_profile_hash(profile) != profile_hash:
        _fail(f'profile 内容 hash 与自报不符：'
              f'{contract.recompute_profile_hash(profile)[:12]} vs {profile_hash[:12]}')

    # VAL：manifest 的 contract/profile identity 必须与正式 profile 一致。
    if role == 'val':
        man_contract = (manifest_obj.get('contract_identity') or {}).get('contract_hash')
        prof_contract = (profile_obj.get('provenance') or {}).get('contract_hash')
        man_profile = (manifest_obj.get('profile_identity') or {}).get('profile_hash')
        if not man_contract or not prof_contract:
            _fail('VAL manifest/profile 缺 contract identity（contract_hash）')
        if man_contract != prof_contract:
            _fail(f'VAL contract identity 不一致：manifest={man_contract} '
                  f'profile={prof_contract}')
        if man_profile and man_profile != profile_hash:
            _fail(f'VAL profile identity 不一致：manifest={man_profile} profile={profile_hash}')
        if not coldchain_contract or not statistical_plan:
            _fail('--role val 必须显式提供 coldchain contract 与 statistical plan')

    # 显式 contract 必须与 profile / manifest 三方绑定；统计方案必须与聚合器配置一致。
    loaded_contract = None
    loaded_contract_hash = None
    statistical_plan_obj = None
    if coldchain_contract:
        loaded_contract = load_coldchain_contract(coldchain_contract)
        loaded_contract_hash = loaded_contract.contract_hash
        prof_contract = (profile_obj.get('provenance') or {}).get('contract_hash')
        man_contract = (manifest_obj.get('contract_identity') or {}).get('contract_hash')
        if prof_contract and loaded_contract.contract_hash != prof_contract:
            _fail(f'loaded contract.contract_hash 与 profile.provenance.contract_hash 不一致：'
                  f'{loaded_contract.contract_hash[:12]} vs {prof_contract[:12]}')
        if man_contract and loaded_contract.contract_hash != man_contract:
            _fail(f'loaded contract.contract_hash 与 manifest.contract_identity 不一致：'
                  f'{loaded_contract.contract_hash[:12]} vs {man_contract[:12]}')
    if statistical_plan:
        statistical_plan_obj = contract.load_statistical_plan(statistical_plan)
        contract.validate_statistical_plan(statistical_plan_obj, contract.FROZEN_CONFIG)

    contract_file_sha256 = (contract.sha256_file(coldchain_contract)
                            if coldchain_contract else None)
    statistical_plan_sha256 = (contract.sha256_file(statistical_plan)
                               if statistical_plan else None)

    # 显式 contract 的冻结身份（dev_gate 与 val 都冻结）：raw/effective contract hash +
    # contract 文件 hash + profile 文件 hash。profile_file_sha256 覆盖 provenance，防止
    # 续跑时同时换 contract 又改 profile provenance 而 profile_hash 不变（P1-B）。
    contract_identity = None
    if coldchain_contract:
        prof = ObjectiveProfile(
            name=profile_obj['name'],
            distance_scale=float(profile_obj['distance_scale']),
            quality_scale=float(profile_obj['quality_scale']),
            energy_scale=float(profile_obj['energy_scale']),
            lambda_quality=float(profile_obj['lambda_quality']),
            lambda_energy=float(profile_obj['lambda_energy']),
            scale_source=profile_obj.get('scale_source', 'pilot'),
            dev_statistics=profile_obj.get('dev_statistics'),
        )
        contract_identity = {
            'contract_hash': loaded_contract_hash,
            'contract_file_sha256': contract_file_sha256,
            'profile_file_sha256': contract.sha256_file(profile),
            'effective_contract_hash': apply_objective_profile(loaded_contract, prof).contract_hash,
        }
    expected_contract_hash = (contract_identity['effective_contract_hash']
                              if contract_identity else None)

    os.makedirs(out, exist_ok=True)

    # 1. 计算运行身份（计算/控制/分析三套 hash + 数据 NPZ 实读 hash + profile hash）
    compute_sha256, compute_files = contract._version(contract.COMPUTE_FILES)
    control_sha256, control_files = contract._version(contract.CONTROL_FILES)
    analysis_sha256, analysis_files = contract._version(contract.ANALYSIS_FILES)
    data = {}
    for c in cells:
        npz = os.path.join(data_dir, c['file'])
        actual = contract.sha256_file(npz)
        if actual != c['sha256']:
            _fail(f"{c['file']} 数据 hash 不符：manifest={c['sha256'][:12]} "
                  f"actual={actual[:12]}")
        data[c['file']] = actual

    pre_path = os.path.join(out, 'pre_run_manifest.json')
    run_id = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    val_identity = None
    if role == 'val':
        disjoint_sha256s = sorted(e['sha256'] for e in disjoint_report)
        val_identity = contract.build_val_identity(manifest_obj, manifest_sha256,
                                                   disjoint_sha256s,
                                                   registry_sha256=registry_sha256)
        val_identity['statistical_plan_sha256'] = statistical_plan_sha256
        val_identity['contract_file_sha256'] = contract_file_sha256
    frozen = contract._frozen_compute(cells, n_per_cell, compute_sha256, data,
                                      profile_hash, role=role, val_identity=val_identity,
                                      contract_identity=contract_identity)

    if os.path.exists(pre_path):
        old = json.load(open(pre_path))
        contract._validate_frozen_compute(old, frozen)
        run_id = old.get('run', {}).get('run_id', run_id)
        print(f'续跑：沿用 run_id={run_id}，冻结计算协议一致（控制/分析可独立演进）')
    else:
        existing = os.listdir(out) if os.path.isdir(out) else []
        if existing:
            _fail(f'输出目录非空但无 pre_run_manifest.json，拒绝接管旧实例：'
                  f'{existing[:10]}')
        pre = {
            'run': {
                'run_id': run_id,
                'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
                'command': ' '.join(sys.argv),
                'role': role,
                'manifest': manifest,
                'manifest_sha256': manifest_sha256,
                'data_dir': data_dir, 'profile': profile,
                'workers': workers, 'out': out,
                'instances_per_cell': n_per_cell, 'total_instances': n_per_cell * 9,
            },
            **frozen,
            'compute_files': compute_files,
            'control_sha256': control_sha256,
            'control_files': control_files,
            'analysis_sha256': analysis_sha256,
            'analysis_files': analysis_files,
            'disjoint_report': disjoint_report,
            'split_registry': split_registry,
            'registry_sha256': registry_sha256,
            'env': _env_info(),
        }
        with open(pre_path, 'w') as f:
            json.dump(pre, f, indent=2)
        print(f'pre-run manifest: {pre_path}（run_id={run_id}）')

    attempt_path = os.path.join(out, 'attempts.jsonl')

    # 2. 逐 cell 跑（断点续跑：只补「缺失」实例；error / protocol / service fail 停止，不覆盖）
    cap = str(int(contract.FROZEN_CONFIG['capacity']))
    nveh = str(int(contract.FROZEN_CONFIG['num_vehicles']))
    for c in cells:
        tag = contract.cell_tag(c)
        cell_dir = os.path.join(out, tag)
        data_path = os.path.join(data_dir, c['file'])
        inst_dir = os.path.join(cell_dir, 'instances')
        os.makedirs(inst_dir, exist_ok=True)

        # 每次 cell 前核验：代码集合 + 当前 cell NPZ + profile 内容 hash
        cur_compute, _ = contract._version(contract.COMPUTE_FILES)
        if cur_compute != compute_sha256:
            _fail(f'[{tag}] 计算文件集合已漂移：{cur_compute} != {compute_sha256}')
        npz_hash = contract.sha256_file(data_path)
        if npz_hash != c['sha256']:
            _fail(f'[{tag}] NPZ hash 漂移：{npz_hash[:12]} != {c["sha256"][:12]}')
        if contract.recompute_profile_hash(profile) != profile_hash:
            _fail(f'[{tag}] profile 内容 hash 漂移')
        if coldchain_contract and contract.sha256_file(coldchain_contract) != contract_file_sha256:
            _fail(f'[{tag}] contract 文件 hash 漂移')
        if statistical_plan and contract.sha256_file(statistical_plan) != statistical_plan_sha256:
            _fail(f'[{tag}] statistical plan 文件 hash 漂移')

        missing = []
        for i in range(n_per_cell):
            p = os.path.join(inst_dir, f'inst_{i}.json')
            if not os.path.exists(p):
                missing.append(i)
                continue
            try:
                with open(p) as f:
                    rec = json.load(f)
            except Exception as e:
                _fail(f'[{tag}] inst_{i}.json 无法解析（保留现场，不覆盖）：{e}')
            if not isinstance(rec, dict):
                _fail(f'[{tag}] inst_{i}.json 内容非对象（保留现场，不覆盖）：'
                      f'{type(rec).__name__}')
            state, err = validate_instance_record(
                rec, i, 'coldchain', require_local=True,
                expected_code_sha256=compute_files['run_action_oracle.py'],
                expected_contract_hash=expected_contract_hash)
            if state == 'valid':
                continue
            _fail(f'[{tag}] inst_{i} 状态 {state}（不覆盖）：{err}')

        if not missing:
            print(f'[{tag}] {n_per_cell}/{n_per_cell} 已完整，跳过补算；重建 canonical 汇总')
            rebuild_cell_summary(cell_dir, objective='coldchain', role='regression',
                                 profile_manifest=profile_obj, data_sha256=c['sha256'],
                                 expected_code_sha256=compute_files['run_action_oracle.py'],
                                 compute_sha256=compute_sha256, run_id=run_id, workers=workers,
                                 launch_mode='resume-canonical', require_local=True,
                                 expected_instance_ids=list(range(n_per_cell)),
                                 expected_contract_hash=expected_contract_hash)
            _verify_cell_after(data_path, c, profile, profile_hash, tag)
            continue

        ids = ','.join(str(i) for i in missing)
        ts = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
        log = os.path.join(out, f'{tag}.{ts}.log')
        cmd = [sys.executable, RUNNER, '--data', data_path, '--capacity', cap,
               '--num_vehicles', nveh, '--objective', 'coldchain',
               '--objective-profile', profile, '--local', '--role', 'regression',
               '--instance-ids', ids, '--workers', str(workers), '--out', cell_dir]
        if coldchain_contract:
            cmd += ['--coldchain-contract', coldchain_contract]
        print(f'[{tag}] 跑 {len(missing)}/{n_per_cell} 实例（instance-ids={ids[:60]}...）')
        attempt_id = f'{tag}.{ts}'
        start = datetime.datetime.utcnow().isoformat() + 'Z'
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
        rebuild_cell_summary(cell_dir, objective='coldchain', role='regression',
                             profile_manifest=profile_obj, data_sha256=c['sha256'],
                             expected_code_sha256=compute_files['run_action_oracle.py'],
                             compute_sha256=compute_sha256, run_id=run_id, workers=workers,
                             launch_mode='canonical', require_local=True,
                             expected_instance_ids=list(range(n_per_cell)),
                             expected_contract_hash=expected_contract_hash)
        _verify_cell_after(data_path, c, profile, profile_hash, tag)
        print(f'[{tag}] 完成')

    # 聚合前最后核验计算文件集合
    cur_compute, _ = contract._version(contract.COMPUTE_FILES)
    if cur_compute != compute_sha256:
        _fail(f'聚合前计算文件集合漂移：{cur_compute} != {compute_sha256}')
    if coldchain_contract and contract.sha256_file(coldchain_contract) != contract_file_sha256:
        _fail('聚合前 contract 文件 hash 漂移')
    if statistical_plan and contract.sha256_file(statistical_plan) != statistical_plan_sha256:
        _fail('聚合前 statistical plan 文件 hash 漂移')

    # 3. 全部齐全后聚合（带冻结计划校验）
    results = [os.path.join(out, contract.cell_tag(c)) for c in cells]
    summary = os.path.join(out, f'{role}_summary.json')
    cmd = [sys.executable, AGGREGATOR, '--results'] + results + [
        '--manifest', manifest, '--mode', role, '--frozen-plan', pre_path,
        '--profile', profile, '--out', summary]
    print(f'9 cell 齐全，运行聚合器（--mode {role} + --frozen-plan）')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print('聚合 FAIL')
        sys.exit(1)
    print(f'{role} 汇总: {summary}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True, choices=list(contract.FORMAL_ROLES))
    ap.add_argument('--manifest', required=True, help='exact manifest 路径（不猜测）')
    ap.add_argument('--profile', required=True, help='exact profile 路径')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True, help='全新输出目录')
    ap.add_argument('--split-registry', default=None,
                    help='完整已登记 split 清单（VAL 必填，含所有 train/DEV-CAL/DEV-PROTO/DEV-GATE）')
    ap.add_argument('--disjoint-manifest', action='append', default=None,
                    help='已登记 split 的 manifest（dev_gate 用；VAL 用 --split-registry）')
    args = ap.parse_args(argv)
    if args.role == 'val':
        # 正式 VAL 必须走 run_o0cc_val.py --archive（自包含封存包 + 本地批准 bundle hash）。
        _fail('--role val 不能经本入口直接运行；请使用 '
              '`python scripts/evaluation/run_o0cc_val.py --archive <sealed> '
              '--expected-bundle-sha256 <hash>`')
    run(role=args.role, manifest=args.manifest, profile=args.profile,
        workers=args.workers, out=args.out, disjoint_manifests=args.disjoint_manifest,
        split_registry=args.split_registry)


if __name__ == '__main__':
    main()
