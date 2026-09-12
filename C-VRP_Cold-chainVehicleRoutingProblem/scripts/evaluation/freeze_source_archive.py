"""独立源码归档（按显式 role/manifest/profile 冻结，DEV-GATE 与 VAL 共用）。

把 compute + control + analysis 文件按相对目录结构复制到归档根，冻结：
  - 源码冻结清单（相对路径 → sha256）+ archive_content_sha256
  - 输入（objective_profile.json + manifest.json）
  - 数据冻结清单（9 NPZ sha256）
  - 环境清单（本地归档构建环境）
  - 证据（控制链验收报告 + 测试结果 + 封存工具自身）
  - 归档封存清单（bundle_sha256：源码+输入+证据+三个清单 的全部相对路径+hash）

不再隐式读取当前 `DEV_MANIFEST.json` 或硬编码 profile 路径；`--role`、`--manifest`、
`--profile` 显式传入，`split_role` 必须与 `--role` 一致。冻结包记录 role、manifest hash、
contract identity 与（VAL 的）disjointness 结果。

不可变：目标目录不存在或为空才允许创建；已存在且非空立即退出；完成后写 ARCHIVE_SEALED，
文件与目录均设为只读。每次源码变化都应创建新版本目录。

用法：
    python scripts/evaluation/freeze_source_archive.py \
        --role val --manifest <VAL_MANIFEST.json> --profile <profile.json> \
        --out archive/正式冻结包/O0-CC_VAL/<version>
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import sys

import formal_gate_contract as contract

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS) if os.path.basename(_SCRIPTS) == 'scripts' else _SCRIPTS


def _sha256_file(p):
    return contract.sha256_file(p)


def _combined_hash(pairs):
    return contract.combined_hash(pairs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True, choices=list(contract.FORMAL_ROLES))
    ap.add_argument('--manifest', required=True, help='exact manifest 路径')
    ap.add_argument('--profile', required=True, help='exact profile 路径')
    ap.add_argument('--split-registry', default=None,
                    help='完整已登记 split 清单（VAL 必填，写入 inputs/split_registry.json）')
    ap.add_argument('--contract', default=None,
                    help='calibrated contract JSON（可选，写入 inputs/contract.json）')
    ap.add_argument('--statistical-plan', default=None,
                    help='统计方案 JSON（可选，写入 inputs/statistical_analysis_plan.json）')
    ap.add_argument('--disjoint-manifest', action='append', default=None,
                    help='已登记 split 的 manifest（dev_gate 用；VAL 用 --split-registry）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    manifest_path = os.path.abspath(args.manifest)
    profile_path = os.path.abspath(args.profile)
    manifest_obj = contract.load_json(manifest_path)
    cells, n_per_cell = contract.validate_manifest(manifest_obj, args.role)
    split_role = manifest_obj['split_role']
    profile_obj = contract.load_json(profile_path)
    if contract.recompute_profile_hash(profile_path) != profile_obj.get('profile_hash'):
        raise SystemExit('profile 内容 hash 与自报 profile_hash 不符')

    # VAL：强制 contract + 统计方案 + split registry 齐全，并绑定三方身份 + 统计方案一致性。
    contract_hash = (profile_obj.get('provenance') or {}).get('contract_hash')
    contract_file_sha256 = None
    statistical_plan_sha256 = None
    statistical_plan = None
    if args.role == 'val':
        if not args.contract or not args.statistical_plan or not args.split_registry:
            raise SystemExit('--role val 必须同时提供 --contract / --statistical-plan / '
                             '--split-registry')
        sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'coldchain'))
        from coldchain_contract import load_coldchain_contract
        loaded = load_coldchain_contract(args.contract)
        contract_hash = loaded.contract_hash
        contract_file_sha256 = contract.sha256_file(args.contract)
        man_contract = (manifest_obj.get('contract_identity') or {}).get('contract_hash')
        prof_contract = (profile_obj.get('provenance') or {}).get('contract_hash')
        if not man_contract or not prof_contract:
            raise SystemExit('VAL manifest/profile 缺 contract identity（contract_hash）')
        if not (contract_hash == man_contract == prof_contract):
            raise SystemExit(f'contract 三方身份不一致：loaded={contract_hash[:12]} '
                             f'manifest={man_contract[:12]} profile={prof_contract[:12]}')
        statistical_plan = contract.load_statistical_plan(args.statistical_plan)
        contract.validate_statistical_plan(statistical_plan, contract.FROZEN_CONFIG)
        statistical_plan_sha256 = contract.sha256_file(args.statistical_plan)

    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f'归档目录已存在且非空，拒绝覆盖（每次源码变化创建新版本目录）：{out}')

    src_root = os.path.join(out, 'source', 'C-VRP_Cold-chainVehicleRoutingProblem')
    for sub in ('source/C-VRP_Cold-chainVehicleRoutingProblem', 'inputs', 'manifests', 'evidence'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # 1. 复制源码（保持相对目录结构）
    files = (contract.COMPUTE_FILES + contract.CONTROL_FILES + contract.ANALYSIS_FILES)
    src_manifest = {}
    for rel in files:
        dst = os.path.join(src_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(_CVRPTW, rel), dst)
        src_manifest[rel] = _sha256_file(dst)
    archive_content_sha256 = _combined_hash(src_manifest.items())

    # 2. 输入：profile + exact manifest + 9 份 NPZ + 规范化 split registry + 可选 contract/统计方案
    prof_dst = os.path.join(out, 'inputs', 'objective_profile.json')
    shutil.copy2(profile_path, prof_dst)
    manifest_dst = os.path.join(out, 'inputs', 'manifest.json')
    shutil.copy2(manifest_path, manifest_dst)

    # VAL 自包含：9 份 NPZ 平铺进 inputs/，manifest 的 `file` 基名直接相对 inputs/ 解析。
    # dev_gate 归档是纯记录（NPZ 在服务器/外部，仅记录 hash），不复制数据。
    if args.role == 'val':
        src_data_dir = os.path.dirname(manifest_path)
        for c in cells:
            npz_src = os.path.join(src_data_dir, c['file'])
            if not os.path.exists(npz_src):
                raise SystemExit(f'数据文件不存在：{npz_src}')
            shutil.copy2(npz_src, os.path.join(out, 'inputs', c['file']))

    extra_inputs = {}  # 相对路径 -> 绝对源路径（contract / 统计方案）
    if args.contract:
        extra_inputs['inputs/contract.json'] = args.contract
    if args.statistical_plan:
        extra_inputs['inputs/statistical_analysis_plan.json'] = args.statistical_plan

    # 规范化 split registry：已登记 manifests 复制进 inputs/registered_splits/，路径改 bundle-relative
    normalized_registry_path = None
    if args.split_registry:
        reg = contract.load_split_registry(args.split_registry)
        reg_dir = os.path.dirname(os.path.abspath(args.split_registry))
        reg_splits = []
        for s in reg['splits']:
            mp = s['manifest']
            if not os.path.isabs(mp):
                mp = os.path.join(reg_dir, mp)
            dst = os.path.join(out, 'inputs', 'registered_splits', f"{s['split_id']}.json")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(mp, dst)
            reg_splits.append({'split_id': s['split_id'], 'split_role': s['split_role'],
                               'manifest': f"registered_splits/{s['split_id']}.json",
                               'manifest_sha256': contract.sha256_file(dst)})
        norm_reg = {'schema': contract.SPLIT_REGISTRY_SCHEMA, 'splits': reg_splits}
        normalized_registry_path = os.path.join(out, 'inputs', 'split_registry.json')
        json.dump(norm_reg, open(normalized_registry_path, 'w'), indent=2)

    for rel, src in extra_inputs.items():
        shutil.copy2(os.path.abspath(src), os.path.join(out, rel))

    # 跨 split 独立性（VAL 用规范化 registry；dev_gate 用可选 --disjoint-manifest）
    disjoint_report = []
    registry_sha256 = None
    if args.role == 'val':
        if not normalized_registry_path:
            raise SystemExit('--role val 必须提供 --split-registry')
        disjoint_report, _reg = contract.check_registry_disjoint(manifest_obj, normalized_registry_path)
        registry_sha256 = contract.sha256_file(normalized_registry_path)
    elif args.disjoint_manifest:
        disjoint_report = contract.check_disjoint(manifest_obj, args.disjoint_manifest)
    if any(e['seed_overlap'] or e['scene_id_overlap'] for e in disjoint_report):
        raise SystemExit('manifest 与已登记 split 重叠，拒绝冻结：'
                         f'\n{json.dumps(disjoint_report, indent=2, ensure_ascii=False)}')

    # 3. 证据（报告 + 测试结果 + 测试脚本 + 封存工具自身）
    for name in ('control_chain_report.md', 'control_chain_tests.json'):
        src = os.path.join(_CVRPTW, 'results', 'o0cc', name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, 'evidence', name))
    test_script_src = os.path.join(_CVRPTW, 'scripts', 'tests', 'test_control_chain.py')
    if os.path.exists(test_script_src):
        shutil.copy2(test_script_src, os.path.join(out, 'evidence', 'test_control_chain.py'))
    tool_dst = os.path.join(out, 'evidence', 'freeze_source_archive.py')
    shutil.copy2(os.path.abspath(__file__), tool_dst)
    verifier_src = os.path.join(_BASE, 'verify_source_archive.py')
    if os.path.exists(verifier_src):
        shutil.copy2(verifier_src, os.path.join(out, 'evidence', 'verify_source_archive.py'))

    # 4. 三个清单（先写，其 hash 纳入 bundle）
    data_manifest = {'cells': [{'file': c['file'], 'sha256': c['sha256']} for c in cells],
                     'split_role': split_role,
                     'instances_per_cell': n_per_cell}
    import numpy
    env_manifest = {'role': 'local_archive_build_env', 'python': sys.version.split()[0],
                    'platform': platform.platform(), 'numpy': numpy.__version__}

    frozen_manifest = {
        'role': args.role,
        'split_role': split_role,
        'archive_content_sha256': archive_content_sha256,
        'compute_sha256': contract._version(contract.COMPUTE_FILES)[0],
        'control_sha256': contract._version(contract.CONTROL_FILES)[0],
        'analysis_sha256': contract._version(contract.ANALYSIS_FILES)[0],
        'runner_sha256': contract._version(contract.COMPUTE_FILES)[1]['run_action_oracle.py'],
        'profile_hash': profile_obj.get('profile_hash'),
        'contract_hash': contract_hash,
        'contract_file_sha256': contract_file_sha256,
        'statistical_plan_sha256': statistical_plan_sha256,
        'manifest_sha256': _sha256_file(manifest_path),
        'manifest_root_seed': contract.manifest_root_seed(manifest_obj),
        'registry_sha256': registry_sha256,
        'disjoint_report': disjoint_report,
        'files': src_manifest,
        'files_count': len(files),
    }
    json.dump(frozen_manifest, open(os.path.join(out, 'manifests', '源码冻结清单.json'), 'w'),
              indent=2)
    json.dump(data_manifest, open(os.path.join(out, 'manifests', '数据冻结清单.json'), 'w'),
              indent=2)
    json.dump(env_manifest, open(os.path.join(out, 'manifests', '环境清单.json'), 'w'),
              indent=2)

    # 5. bundle = 源码(归档相对路径) + 输入 + 证据(含工具) + 三个清单 的全部相对路径+hash
    bundle_pairs = [('source/C-VRP_Cold-chainVehicleRoutingProblem/' + rel, h)
                    for rel, h in src_manifest.items()]
    for root, dirs, fnames in os.walk(os.path.join(out, 'inputs')):
        for f in fnames:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out).replace('\\', '/')
            bundle_pairs.append((rel, _sha256_file(p)))
    for name in sorted(os.listdir(os.path.join(out, 'evidence'))):
        p = os.path.join(out, 'evidence', name)
        bundle_pairs.append(('evidence/' + name, _sha256_file(p)))
    for name in sorted(os.listdir(os.path.join(out, 'manifests'))):
        p = os.path.join(out, 'manifests', name)
        bundle_pairs.append(('manifests/' + name, _sha256_file(p)))
    bundle_sha256 = _combined_hash(bundle_pairs)

    # 6. 归档封存清单（含 bundle_sha256）
    seal_manifest = {
        'archive_format_version': 2,
        'role': args.role,
        'split_role': split_role,
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'archive_content_sha256': archive_content_sha256,
        'bundle_sha256': bundle_sha256,
        'compute_sha256': frozen_manifest['compute_sha256'],
        'control_sha256': frozen_manifest['control_sha256'],
        'analysis_sha256': frozen_manifest['analysis_sha256'],
        'runner_sha256': frozen_manifest['runner_sha256'],
        'profile_hash': frozen_manifest['profile_hash'],
        'contract_hash': frozen_manifest['contract_hash'],
        'contract_file_sha256': frozen_manifest['contract_file_sha256'],
        'statistical_plan_sha256': frozen_manifest['statistical_plan_sha256'],
        'manifest_sha256': frozen_manifest['manifest_sha256'],
        'freeze_tool_sha256': _sha256_file(tool_dst),
        'bundle_files': dict(bundle_pairs),
    }
    json.dump(seal_manifest, open(os.path.join(out, 'manifests', '归档封存清单.json'), 'w'),
              indent=2)

    # 7. 封存标记 + 只读（文件 0o444，目录 0o555）
    json.dump({'sealed': True, 'bundle_sha256': bundle_sha256,
               'archive_content_sha256': archive_content_sha256},
              open(os.path.join(out, 'ARCHIVE_SEALED'), 'w'), indent=2)
    for root, dirs, fnames in os.walk(out):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)
        for f in fnames:
            os.chmod(os.path.join(root, f), 0o444)

    print(f'归档封存完成：{out}')
    print(f'  role: {args.role}')
    print(f'  files: {len(files)}')
    print(f'  archive_content_sha256: {archive_content_sha256}')
    print(f'  bundle_sha256: {bundle_sha256}')
    print(f'  compute_sha256: {frozen_manifest["compute_sha256"]}')
    print(f'  control_sha256: {frozen_manifest["control_sha256"]}')
    print(f'  analysis_sha256: {frozen_manifest["analysis_sha256"]}')
    print(f'  profile_hash: {frozen_manifest["profile_hash"]}')
    print(f'  contract_hash: {frozen_manifest["contract_hash"]}')


if __name__ == '__main__':
    main()
