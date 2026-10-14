"""Generic remote script runner for either configured server.

Builds the remote command inside Python from components, so no shell quoting is ever involved -
which matters because PowerShell mangled `2>&1`, `|`, `>` and nested quotes in every hand-typed
attempt, and because Linux and Windows hosts need different command syntax.

Usage:
    python tools/_srv_exec.py --config CFG --workdir DIR --python PY --script REL [--args ...]

Long runs: start this script itself as a local background job; the remote command then runs
synchronously over SSH and the local job holds the stream. That is simpler and more portable than
detaching on the remote host, and it avoids the Windows/Linux detachment syntax split entirely.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402


def build(cfg_path, workdir, python, script, args):
    if os.path.basename(cfg_path).endswith('3080.json'):
        return 'cd "%s"; & "%s" %s %s' % (workdir, python, script, ' '.join(args))
    return 'cd %s && %s %s %s' % (workdir, python, script, ' '.join(args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='.vscode/sftp.3080.json')
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--python', required=True)
    ap.add_argument('--script', required=True)
    ap.add_argument('--timeout', type=int, default=7200)
    known, rest = ap.parse_known_args()
    if rest and rest[0] == '--':
        rest = rest[1:]
    cmd = build(known.config, known.workdir, known.python, known.script, rest)
    print('remote command: %s' % cmd, flush=True)
    t0 = time.perf_counter()
    print(run(cmd, timeout=known.timeout, config=known.config))
    print('[local wall %.1fs]' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
