"""Print sha256 of every .py under the extension's scripts/ tree (sorted), for cross-machine diffing.

Usage: python CC_Compare/tools/dump_tree_hashes.py [<extension root>]
"""
import hashlib
import os
import sys


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..',
        'C-VRP_Cold-chainVehicleRoutingProblem')
    scripts = os.path.join(root, 'scripts')
    out = []
    for dirpath, dirnames, filenames in os.walk(scripts):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace(os.sep, '/')
            out.append((rel, sha256(p)))
    for rel, h in sorted(out):
        print('%s  %s' % (h, rel))
    print('# files: %d' % len(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
