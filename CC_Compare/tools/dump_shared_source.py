"""Print the `shared_source_sha256` map from a gate.json (one line per file), for cross-machine diffing.

Usage: python CC_Compare/tools/dump_shared_source.py <gate.json>
"""
import json
import sys


def find(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, dict):
                return v
            r = find(v, key)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v, key)
            if r:
                return r
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    g = json.load(open(sys.argv[1], encoding='utf-8'))
    print('--- top-level identity ---')
    for key in ('source_stable', 'contract_sha256', 'data_sha256', 'data_meta_sha256'):
        vals = []
        find(g, key) if False else None
        # direct search for scalar values
        def collect(o, k, out):
            if isinstance(o, dict):
                for kk, vv in o.items():
                    if kk == k and not isinstance(vv, (dict, list)):
                        out.append(vv)
                    elif isinstance(vv, (dict, list)):
                        collect(vv, k, out)
            elif isinstance(o, list):
                for vv in o:
                    collect(vv, k, out)
            return out
        vals = collect(g, key, [])
        print('%s = %s' % (key, vals[:2]))
    m = find(g, 'shared_source_sha256') or {}
    print('--- shared_source_sha256 (%d files) ---' % len(m))
    for k in sorted(m):
        print('%s  %s' % (m[k], k))
    return 0


if __name__ == '__main__':
    sys.exit(main())
