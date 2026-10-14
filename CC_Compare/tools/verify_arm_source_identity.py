"""核对对比方法臂的源码身份 vs 当前磁盘（2026-10-13，对比方法线）。

读 a1_c1rr2_reveal_40/gate.json（身份合格的 C1 基线）与各 a1_ext_extra/*_f/gate.json
记录的 source_sha256 / source_end_sha256，逐文件与**当前磁盘**重算 SHA-256 比较，
判定「现在重跑是否与 c1rr2 / 正式臂同协议」。

用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/verify_arm_source_identity.py [--json out.json]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..',
                                        'C-VRP_Cold-chainVehicleRoutingProblem'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def resolve(rel, ext_root):
    """Map a manifest-relative path to an absolute path."""
    if rel.startswith('../'):
        p = os.path.join(ext_root, rel)
    elif rel.startswith('scripts/') or rel.startswith('results/') or rel.startswith('data/'):
        p = os.path.join(ext_root, rel)
    else:
        p = os.path.join(ext_root, rel)
    return os.path.normpath(p)


def arm_identity(gate):
    """Return (identity_block, container_name) for either schema."""
    if 'identity' in gate:
        return gate['identity'], 'identity'
    res = gate.get('results') or {}
    for yr, yb in res.items():
        if isinstance(yb, dict) and '_identity' in yb:
            return yb['_identity'], 'results.%s._identity' % yr
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    base = os.path.join(EXT_ROOT, 'results', 'a1_c1rr2_reveal_40', 'gate.json')
    c1 = json.load(open(base, encoding='utf-8'))
    c1_id = c1['identity']

    out = {'c1rr2': {'path': base, 'source_stable': c1_id.get('source_stable'),
                     'files': {}}, 'arms': {}}

    print('c1rr2 source_stable =', c1_id.get('source_stable'))
    for phase in ('source_sha256', 'source_end_sha256'):
        for rel, h in sorted(c1_id.get(phase, {}).items()):
            p = resolve(rel, EXT_ROOT)
            cur = sha256(p) if os.path.exists(p) else None
            out['c1rr2']['files'].setdefault(rel, {})[phase] = h
            out['c1rr2']['files'][rel]['disk'] = cur
            out['c1rr2']['files'][rel]['disk_matches_start'] = (cur == h)
    diff_start = [r for r, v in out['c1rr2']['files'].items()
                  if not v['disk_matches_start']]
    print('c1rr2 files differing from disk: %d / %d' % (
        len(diff_start), len(out['c1rr2']['files'])))
    for r in diff_start:
        print('   DIFF', r)

    arms = sorted(glob.glob(os.path.join(EXT_ROOT, 'results', 'a1_ext_extra',
                                         '*_f', 'gate.json')))
    for gp in arms:
        g = json.load(open(gp, encoding='utf-8'))
        ident, where = arm_identity(g)
        if ident is None:
            continue
        name = os.path.basename(os.path.dirname(gp))
        entry = {'path': gp, 'source_stable': ident.get('source_stable'),
                 'shared_vs_c1rr2': {}, 'files_disk_stable': {}}
        for rel, h in sorted(ident.get('source_sha256', {}).items()):
            if rel in c1_id.get('source_sha256', {}):
                entry['shared_vs_c1rr2'][rel] = (
                    h == c1_id['source_sha256'][rel])
        for rel, h in sorted(ident.get('source_sha256', {}).items()):
            p = resolve(rel, EXT_ROOT)
            cur = sha256(p) if os.path.exists(p) else None
            entry['files_disk_stable'][rel] = (cur == h)
        out['arms'][name] = entry

    print()
    print('%-16s %-13s %-22s %s' % ('arm', 'src_stable', 'shared==c1rr2', 'disk==arm'))
    for name, e in sorted(out['arms'].items()):
        sh = e['shared_vs_c1rr2']
        n_ok = sum(1 for v in sh.values() if v)
        fd = e['files_disk_stable']
        n_fd = sum(1 for v in fd.values() if v)
        bad = [r for r, v in fd.items() if not v]
        print('%-16s %-13s %-22s %d/%d %s' % (
            name, e['source_stable'], '%d/%d' % (n_ok, len(sh)), n_fd, len(fd),
            ('DIFF:' + ','.join(os.path.basename(b) for b in bad)) if bad else ''))

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print('\nwrote', args.json)


if __name__ == '__main__':
    main()
