"""verify_contract_l3.py — L3 合同迁移的只读核验器（不写任何证据，失败非零退出）。

验证 contract_migration_l3 证据目录：
  - 两份 parity verdict == BEHAVIOR_UNCHANGED_9_OF_9；
  - 四批 canonical.COMPLETE gate 全 PASS；
  - L3_EVIDENCE_MANIFEST.json 逐文件 SHA 对账；
  - 运行前后证据目录文件清单 + SHA 零变化（自身只读的自证）。

用法：
    python verify_contract_l3.py --base CC_Compare/results/contract_migration_l3
"""
import argparse
import hashlib
import json
import os
import sys

BATCHES = [
    'pyvrp/old_abf_9x1', 'pyvrp/new_7d4_9x1',
    'ortools/new_7d4_9x1', 'ortools/old_abf_9x1',
]
PARITIES = ['pyvrp_parity.json', 'ortools_parity.json']


def _snapshot(base):
    files = {}
    for root, _, names in os.walk(base):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, base).replace(os.sep, '/')
            files[rel] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    args = ap.parse_args()
    B = args.base
    errors = []

    def require(cond, msg):
        if cond:
            print(f'  [PASS] {msg}')
        else:
            errors.append(msg)
            print(f'  [FAIL] {msg}')

    before = _snapshot(B)

    # 1. 两份 parity
    for p in PARITIES:
        pp = os.path.join(B, p)
        if not os.path.exists(pp):
            require(False, f'{p} 存在')
            continue
        parity = json.load(open(pp, encoding='utf-8'))
        require(parity.get('verdict') == 'BEHAVIOR_UNCHANGED_9_OF_9',
                f'{p} verdict == BEHAVIOR_UNCHANGED_9_OF_9 (={parity.get("verdict")})')
        require(parity.get('all_unchanged') is True, f'{p} all_unchanged==True')
        require(parity.get('contract_sha_differ') is True, f'{p} contract_sha_differ==True')

    # 2. 四批 canonical.COMPLETE gate
    for b in BATCHES:
        cc = os.path.join(B, b, 'canonical.COMPLETE')
        if not os.path.exists(cc):
            require(False, f'{b}/canonical.COMPLETE 存在')
            continue
        c = json.load(open(cc, encoding='utf-8'))
        g = c.get('gate', {})
        gate_ok = (g.get('n_complete', 0) == 9 and g.get('ownership_violations', 0) == 0
                   and g.get('terminal_unresolved', 0) == 0
                   and g.get('protocol_errors', 0) == 0
                   and g.get('fallback_triggered_events', 0) == 0)
        require(gate_ok, f'{b} canonical gate 全 PASS (n_complete={g.get("n_complete")})')

    # 3. manifest 逐文件 hash
    mp = os.path.join(B, 'L3_EVIDENCE_MANIFEST.json')
    if os.path.exists(mp):
        m = json.load(open(mp, encoding='utf-8'))
        files = m.get('files', {})
        require(bool(files), 'manifest 含 files 清单')
        ok = True
        for rel, exp in files.items():
            fp = os.path.join(B, rel)
            if not os.path.exists(fp):
                ok = False
                print(f'    manifest 缺文件: {rel}')
                continue
            got = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
            if got != exp:
                ok = False
                print(f'    manifest hash 不符: {rel}')
        require(ok, 'manifest 逐文件 hash 对账一致')
    else:
        require(False, 'L3_EVIDENCE_MANIFEST.json 存在')

    # 4. 零写回（运行前后快照一致）
    after = _snapshot(B)
    require(before == after, '运行前后证据目录零写回')

    if errors:
        print(f'\nverify_contract_l3: {len(errors)} 项失败')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('\nL3 CONTRACT MIGRATION: ALL PASS')


if __name__ == '__main__':
    main()
