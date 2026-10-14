"""Recursive SFTP push of a local directory to the configured server root.

`_sftp_put.py` handles explicit file lists; this handles directory trees, which the two-server
workflow needs because the Linux box already has the repo but the Windows box starts empty.

Skips by default the same heavy/irrelevant patterns the sftp config ignores (VCS metadata,
caches, results, archives) so a push cannot accidentally ship hundreds of MB of results. Anything
needed but skipped can be named explicitly with --include-file.

Usage:
    python tools/_sftp_push_dir.py <local_dir_rel> [--config CFG] [--include-file REL]...
                                     [--dry-run]
"""
import json
import os
import sys

import paramiko

SKIP_DIRS = {'.git', '__pycache__', '.idea', '.vscode', 'results', 'node_modules',
             'saved_models', '.ipynb_checkpoints'}
SKIP_EXT = {'.pyc', '.pt', '.pth', '.npz', '.npy', '.zip', '.tar', '.gz', '.pkl', '.ckpt',
            '.jsonl'}


def parse_args(argv):
    cfg = '.vscode/sftp.json'
    local = None
    includes = []
    dry = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--config':
            cfg = argv[i + 1]
            i += 2
        elif a == '--include-file':
            includes.append(argv[i + 1])
            i += 2
        elif a == '--dry-run':
            dry = True
            i += 1
        elif local is None:
            local = a
            i += 1
        else:
            raise SystemExit('unexpected argument %r' % a)
    if local is None:
        raise SystemExit('usage: _sftp_push_dir.py <local_dir_rel> [--config CFG] ...')
    return cfg, local, includes, dry


def main():
    cfg, local, includes, dry = parse_args(sys.argv[1:])
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    cfgp = cfg if os.path.isabs(cfg) else os.path.join(root, cfg)
    with open(cfgp) as f:
        d = json.load(f)
    base = d['remotePath']
    if not base.endswith('/'):
        base += '/'

    jobs = []
    extra = set(includes)
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, local)):
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace('\\', '/')
            if rel in extra or (os.path.splitext(fn)[1].lower() not in SKIP_EXT):
                jobs.append((full, rel))
    total = sum(os.path.getsize(j) for j, _ in jobs)
    print('  %d files, %.1f MB -> %s' % (len(jobs), total / 1e6, base))
    if dry:
        for _, rel in jobs[:50]:
            print('   ', rel)
        return

    t = paramiko.Transport((d['host'], 22))
    t.connect(username=d['username'], password=d['password'])
    sftp = paramiko.SFTPClient.from_transport(t)
    made = set()

    def ensure(remote_dir):
        if remote_dir in made or not remote_dir:
            return
        parts = remote_dir.split('/')
        cur = ''
        for p in parts:
            if p == '':
                cur = '/'
                continue
            cur = (cur.rstrip('/') + '/' + p) if cur else p
            if cur in made:
                continue
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass
            made.add(cur)

    done = 0
    for full, rel in jobs:
        remote = base + rel
        ensure(os.path.dirname(remote))
        sftp.put(full, remote)
        done += 1
        if done % 100 == 0:
            print('  ... %d/%d' % (done, len(jobs)), flush=True)
    sftp.close()
    t.close()
    print('  pushed %d files' % done)


if __name__ == '__main__':
    main()
