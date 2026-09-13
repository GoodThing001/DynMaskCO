"""独立源码归档（冻结计算/控制/分析源码 + 输入 + 清单 + 证据 + 封存）。

把 15 compute + 3 control + 1 analysis 文件按相对目录结构复制到归档根，冻结：
  - 源码冻结清单（相对路径 → sha256）+ archive_content_sha256（相对路径+hash 的独立归档身份）
  - 输入（objective_profile.json + DEV_MANIFEST.json）
  - 数据冻结清单（9 NPZ sha256）
  - 环境清单（本地归档构建环境）
  - 证据（控制链验收报告 + 测试结果 + 测试脚本 + 封存工具自身）
  - 归档封存清单（bundle_sha256：源码+输入+证据+三个清单+工具 的全部相对路径+hash）

不可变：目标目录不存在或为空才允许创建；已存在且非空立即退出；完成后写 ARCHIVE_SEALED，
文件与目录均设为只读（阻止 Python 生成 __pycache__）。每次源码变化都应创建新版本目录。

用法：
    python scripts/evaluation/freeze_source_archive.py \
        --out C-VRP_Cold-chainVehicleRoutingProblem/archive/o0cc_devgate_2dfff7ae_20260908_v2
"""
import argparse, os, sys, json, shutil, hashlib, platform, datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS) if os.path.basename(_SCRIPTS) == 'scripts' else _SCRIPTS
for p in (_SCRIPTS, os.path.join(_CVRPTW, 'scripts', 'evaluation')):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dev_gate as dg


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def _combined_hash(pairs):
    payload = json.dumps(sorted(pairs), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f'归档目录已存在且非空，拒绝覆盖（每次源码变化创建新版本目录）：{out}')

    src_root = os.path.join(out, 'source', 'C-VRP_Cold-chainVehicleRoutingProblem')
    for sub in ('source/C-VRP_Cold-chainVehicleRoutingProblem', 'inputs', 'manifests', 'evidence'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # 1. 复制 19 文件（保持相对目录结构）
    files = (dg.COMPUTE_FILES + dg.CONTROL_FILES + dg.ANALYSIS_FILES)
    src_manifest = {}
    for rel in files:
        dst = os.path.join(src_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(_CVRPTW, rel), dst)
        src_manifest[rel] = _sha256_file(dst)
    archive_content_sha256 = _combined_hash(src_manifest.items())

    # 2. 输入
    prof_dst = os.path.join(out, 'inputs', 'objective_profile.json')
    shutil.copy2(os.path.join(_CVRPTW, 'results', 'o0cc', 'scale_v2', 'objective_profile.json'), prof_dst)
    manifest_dst = os.path.join(out, 'inputs', 'DEV_MANIFEST.json')
    shutil.copy2(os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'dev_gate', 'DEV_MANIFEST.json'), manifest_dst)

    # 3. 证据（报告 + 测试结果 + 测试脚本 + 封存工具自身）
    evidence = {}
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
    dev_manifest = json.load(open(manifest_dst))
    data_manifest = {'cells': [{'file': c['file'], 'sha256': c['sha256']} for c in dev_manifest['cells']],
                     'split_role': dev_manifest.get('split_role'),
                     'num_instances_per_cell': dev_manifest.get('num_instances_per_cell')}
    import numpy
    env_manifest = {'role': 'local_archive_build_env', 'python': sys.version.split()[0],
                    'platform': platform.platform(), 'numpy': numpy.__version__}

    frozen_manifest = {
        'archive_content_sha256': archive_content_sha256,
        'compute_sha256': dg._version(dg.COMPUTE_FILES)[0],
        'control_sha256': dg._version(dg.CONTROL_FILES)[0],
        'analysis_sha256': dg._version(dg.ANALYSIS_FILES)[0],
        'runner_sha256': dg._version(dg.COMPUTE_FILES)[1]['run_action_oracle.py'],
        'profile_hash': json.load(open(prof_dst)).get('profile_hash'),
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
    bundle_pairs.append(('inputs/objective_profile.json', _sha256_file(prof_dst)))
    bundle_pairs.append(('inputs/DEV_MANIFEST.json', _sha256_file(manifest_dst)))
    for name in sorted(os.listdir(os.path.join(out, 'evidence'))):
        p = os.path.join(out, 'evidence', name)
        bundle_pairs.append(('evidence/' + name, _sha256_file(p)))
    for name in sorted(os.listdir(os.path.join(out, 'manifests'))):
        p = os.path.join(out, 'manifests', name)
        bundle_pairs.append(('manifests/' + name, _sha256_file(p)))
    bundle_sha256 = _combined_hash(bundle_pairs)

    # 6. 归档封存清单（含 bundle_sha256）
    seal_manifest = {
        'archive_format_version': 1,
        'created_utc': datetime.datetime.now(datetime.UTC).isoformat(),
        'archive_content_sha256': archive_content_sha256,
        'bundle_sha256': bundle_sha256,
        'compute_sha256': frozen_manifest['compute_sha256'],
        'control_sha256': frozen_manifest['control_sha256'],
        'analysis_sha256': frozen_manifest['analysis_sha256'],
        'runner_sha256': frozen_manifest['runner_sha256'],
        'profile_hash': frozen_manifest['profile_hash'],
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
    print(f'  files: {len(files)}')
    print(f'  archive_content_sha256: {archive_content_sha256}')
    print(f'  bundle_sha256: {bundle_sha256}')
    print(f'  compute_sha256: {frozen_manifest["compute_sha256"]}')
    print(f'  control_sha256: {frozen_manifest["control_sha256"]}')
    print(f'  analysis_sha256: {frozen_manifest["analysis_sha256"]}')
    print(f'  profile_hash: {frozen_manifest["profile_hash"]}')


if __name__ == '__main__':
    main()
