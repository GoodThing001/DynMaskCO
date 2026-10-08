"""RRNCO-Ordering-RH-D R1 冻结：seal 核验器（verify_freeze.py）。

重算身份链：SOURCE_MANIFEST.json（每文件 sha256）→ FROZEN_CONFIG.json（manifest sha256）
→ FREEZE_SEAL.json（config/manifest/证据/checkpoint sha256），逐项对账；失败非零退出。

用法（仓库根）：
    python CC_Compare/RRNCO/dcc_rh_v4/verify_freeze.py --seal CC_Compare/RRNCO/dcc_rh_v4/FREEZE_SEAL.json
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seal', required=True)
    args = ap.parse_args()
    dcc = os.path.dirname(os.path.abspath(__file__))
    # --seal 相对路径按「仓库根」解析（用法约定：从仓库根运行，传 CC_Compare/... 路径）
    seal_p = os.path.abspath(args.seal)
    seal = json.load(open(seal_p, encoding='utf-8'))
    errors = []

    def require(cond, msg, got=None):
        status = 'PASS' if cond else 'FAIL'
        extra = '' if got is None else ' (got=%r)' % (got,)
        print('  [%s] %s%s' % (status, msg, extra))
        if not cond:
            errors.append(msg)

    # 1) SOURCE_MANIFEST 重算对账
    manifest_p = os.path.join(dcc, 'SOURCE_MANIFEST.json')
    require(os.path.exists(manifest_p), 'SOURCE_MANIFEST.json 存在')
    manifest = json.load(open(manifest_p, encoding='utf-8'))
    ok = True
    for f, exp in manifest.get('files', {}).items():
        fp = os.path.join(dcc, f)
        if not os.path.exists(fp):
            ok = False
            print('    manifest 缺文件: %s' % f)
            continue
        got = sha256_file(fp)
        if got != exp:
            ok = False
            print('    manifest hash 不符: %s' % f)
    require(ok, 'SOURCE_MANIFEST 逐文件 hash 对账一致')
    manifest_hash = sha256_file(manifest_p)
    require(seal.get('source_manifest_sha256') == manifest_hash,
            'seal.source_manifest_sha256 == 当前 SOURCE_MANIFEST hash',
            (seal.get('source_manifest_sha256'), manifest_hash))

    # 2) FROZEN_CONFIG 对账
    cfg_p = os.path.join(dcc, 'FROZEN_CONFIG.json')
    require(os.path.exists(cfg_p), 'FROZEN_CONFIG.json 存在')
    cfg_hash = sha256_file(cfg_p)
    require(seal.get('frozen_config_sha256') == cfg_hash,
            'seal.frozen_config_sha256 == 当前 FROZEN_CONFIG hash',
            (seal.get('frozen_config_sha256'), cfg_hash))
    cfg = json.load(open(cfg_p, encoding='utf-8'))
    require(cfg.get('source_manifest_sha256') == manifest_hash,
            'FROZEN_CONFIG.source_manifest_sha256 == 当前 manifest hash',
            (cfg.get('source_manifest_sha256'), manifest_hash))

    # 3) 证据对账（R0.5 evidence）
    ev_p = os.path.join(dcc, 'results', 'R0_5_EVIDENCE_MANIFEST.json')
    require(os.path.exists(ev_p), 'results/R0_5_EVIDENCE_MANIFEST.json 存在')
    ev = json.load(open(ev_p, encoding='utf-8'))
    ok = True
    for f, exp in ev.get('files', {}).items():
        fp = os.path.join(dcc, 'results', f)
        if not os.path.exists(fp):
            ok = False
            print('    证据缺文件: %s' % f)
            continue
        got = sha256_file(fp)
        if got != exp:
            ok = False
            print('    证据 hash 不符: %s' % f)
    require(ok, 'R0.5 证据逐文件 hash 对账一致')
    ev_hash = sha256_file(ev_p)
    require(seal.get('evidence_manifest_sha256') == ev_hash,
            'seal.evidence_manifest_sha256 == 当前证据 manifest hash',
            (seal.get('evidence_manifest_sha256'), ev_hash))
    # run_a/run_b verdict + checkpoint
    for name in ('r0_5_run_a.json', 'r0_5_run_b.json'):
        rp = os.path.join(dcc, 'results', name)
        require(os.path.exists(rp), 'results/%s 存在' % name)
        if os.path.exists(rp):
            r = json.load(open(rp, encoding='utf-8'))
            require(r.get('verdict') == 'PASS', '%s verdict PASS' % name, r.get('verdict'))
    ck_sha = seal.get('checkpoint_sha256')
    require(bool(ck_sha) and len(ck_sha) == 64, 'seal.checkpoint_sha256 非空且 64 位', ck_sha)
    senv = json.load(open(os.path.join(dcc, 'results', 'SERVER_ENVIRONMENT.json'),
                          encoding='utf-8'))
    require(senv.get('checkpoint', {}).get('sha256') == ck_sha,
            'SERVER_ENVIRONMENT checkpoint == seal.checkpoint_sha256',
            senv.get('checkpoint', {}).get('sha256'))

    # 4) seal 自身字段
    require(seal.get('schema_version') == 'rrnco-dcc-rh-v4-seal-v1',
            'seal schema_version 正确', seal.get('schema_version'))
    require(isinstance(seal.get('revision'), int) and seal['revision'] >= 1,
            'seal revision >= 1', seal.get('revision'))

    if errors:
        print('\nverify_freeze: %d 项失败' % len(errors))
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('\nverify_freeze: ALL PASS')


if __name__ == '__main__':
    main()
