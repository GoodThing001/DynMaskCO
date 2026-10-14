"""归档验证器（精确文件集合 + 逐文件 hash + 重算身份）。

校验归档目录是否与封存清单一致：
  - 实际文件集合 == 声明集合（多/少都失败）；
  - 禁止 .pyc / __pycache__；
  - 逐文件 SHA-256；
  - 重算 archive_content_sha256 / bundle_sha256；
  - profile 内部 profile_hash 自洽；
  - ARCHIVE_SEALED 与封存清单绑定一致。

用法（验证时务必 -B / PYTHONDONTWRITEBYTECODE=1，避免验证本身生成 .pyc）：
    python -B scripts/evaluation/verify_source_archive.py \
        --archive C-VRP_Cold-chainVehicleRoutingProblem/archive/o0cc_devgate_2dfff7ae_20260908_v2
"""
import argparse, os, sys, json, hashlib


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def _combined_hash(pairs):
    payload = json.dumps(sorted(pairs), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _recompute_profile_hash(prof):
    payload = json.dumps({
        'name': prof['name'], 'distance_scale': prof['distance_scale'],
        'quality_scale': prof['quality_scale'], 'energy_scale': prof['energy_scale'],
        'lambda_quality': prof['lambda_quality'], 'lambda_energy': prof['lambda_energy'],
        'scale_source': prof.get('scale_source', 'pilot'),
        'dev_statistics': prof.get('dev_statistics'),
    }, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--expected-bundle-sha256', default=None,
                    help='预期 bundle hash（服务器验证收到的是本地批准的那份归档，而非仅内部自洽）')
    args = ap.parse_args()
    archive = os.path.abspath(args.archive)

    seal = json.load(open(os.path.join(archive, 'manifests', '归档封存清单.json')))
    declared = seal.get('bundle_files') or {}
    errors = []

    # 1. 收集实际文件（排除 meta 文件）
    actual = {}
    for root, dirs, fnames in os.walk(archive):
        for d in dirs:
            if d == '__pycache__':
                errors.append(f'发现 __pycache__：{os.path.join(root, d)}')
        for f in fnames:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, archive).replace('\\', '/')
            if rel in ('ARCHIVE_SEALED', 'manifests/归档封存清单.json'):
                continue
            if f.endswith('.pyc'):
                errors.append(f'发现 .pyc：{rel}')
            actual[rel] = _sha256_file(p)

    # 2. 精确文件集合
    if set(actual) != set(declared):
        missing = set(declared) - set(actual)
        extra = set(actual) - set(declared)
        if missing:
            errors.append(f'缺文件：{sorted(missing)[:8]}')
        if extra:
            errors.append(f'多文件：{sorted(extra)[:8]}')

    # 3. 逐文件 hash
    for rel, h in declared.items():
        if actual.get(rel) != h:
            errors.append(f'hash 不符：{rel}')

    # 4. 重算 bundle_sha256
    bundle = _combined_hash(sorted(declared.items()))
    if bundle != seal.get('bundle_sha256'):
        errors.append(f'bundle_sha256 不符：{bundle} vs {seal.get("bundle_sha256")}')

    # 5. 重算 archive_content_sha256（从源码冻结清单）
    src_manifest = json.load(open(os.path.join(archive, 'manifests', '源码冻结清单.json')))
    ac = _combined_hash(src_manifest['files'].items())
    if ac != seal.get('archive_content_sha256'):
        errors.append(f'archive_content_sha256 不符：{ac} vs {seal.get("archive_content_sha256")}')

    # 6. profile_hash 自洽
    prof = json.load(open(os.path.join(archive, 'inputs', 'objective_profile.json')))
    if _recompute_profile_hash(prof) != prof.get('profile_hash'):
        errors.append('profile 内容 hash 与内部 profile_hash 不符')
    if prof.get('profile_hash') != seal.get('profile_hash'):
        errors.append('profile_hash 与封存清单不符')

    # 7. ARCHIVE_SEALED 绑定
    sealed = json.load(open(os.path.join(archive, 'ARCHIVE_SEALED')))
    if not sealed.get('sealed') or sealed.get('bundle_sha256') != seal.get('bundle_sha256'):
        errors.append('ARCHIVE_SEALED.bundle_sha256 与封存清单不一致')
    if sealed.get('archive_content_sha256') != seal.get('archive_content_sha256'):
        errors.append('ARCHIVE_SEALED.archive_content_sha256 与封存清单不一致')

    # 8. 预期 bundle hash（服务器侧：收到的是本地批准的那份）
    if args.expected_bundle_sha256 and args.expected_bundle_sha256 != seal.get('bundle_sha256'):
        errors.append(f'bundle_sha256 与预期不符：{seal.get("bundle_sha256")} vs {args.expected_bundle_sha256}')

    if errors:
        print('VERIFY FAIL')
        for e in errors:
            print('  ', e)
        sys.exit(1)
    print(f'VERIFY PASS: {archive}')
    print(f'  files={len(actual)}  archive_content_sha256={seal.get("archive_content_sha256")[:16]}'
          f'  bundle_sha256={seal.get("bundle_sha256")[:16]}')


if __name__ == '__main__':
    main()
