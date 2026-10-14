"""Deploy the frozen trained head to the 3080 and run the D-TRAINTASK development gate there.

Idempotent and explicit: ensures the head artifact exists remotely (uploading it if not), removes
any stale remote output directory, then runs the gate into a fresh directory so a previous partial
run can never be mistaken for the current one.

Usage:
    python tools/_deploy_and_run_3080.py [--instances 32] [--tag dev_gate_3080_v2]
"""
import argparse
import json
import os
import sys
import time

import paramiko

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = '.vscode/sftp.3080.json'
WIN_BASE = r'C:\Users\zh\project\MASKCO-Main'
PY = r'D:\anaconda\envs\MASKCO_env\python.exe'
HEAD_REL = ('C-VRP_Cold-chainVehicleRoutingProblem/results/research_20261010_continuation_v2/'
            'delivery/train_run_v1/delivery_trained_head.pkl')


def cfg():
    with open(os.path.join(ROOT, CFG)) as f:
        return json.load(f)


def ensure_head():
    d = cfg()
    sftp_base = d['remotePath']
    if not sftp_base.endswith('/'):
        sftp_base += '/'
    remote = sftp_base + HEAD_REL
    local = os.path.join(ROOT, *HEAD_REL.split('/'))
    t = paramiko.Transport((d['host'], 22))
    t.connect(username=d['username'], password=d['password'])
    sftp = paramiko.SFTPClient.from_transport(t)
    parts = remote.split('/')
    cur = ''
    for p in parts[:-1]:
        if p == '':
            cur = '/'
            continue
        cur = (cur.rstrip('/') + '/' + p) if cur else p
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass
    sftp.put(local, remote)
    st = sftp.stat(remote)
    sftp.close()
    t.close()
    return remote, st.st_size, os.path.getsize(local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, default=32)
    ap.add_argument('--tag', default='dev_gate_3080_v2')
    a = ap.parse_args()
    remote_head, rsize, lsize = ensure_head()
    print('head on 3080: %s (%d bytes; local %d, match %s)'
          % (remote_head, rsize, lsize, rsize == lsize))

    out_rel = ('C-VRP_Cold-chainVehicleRoutingProblem/results/'
               'research_20261010_continuation_v2/delivery/' + a.tag)
    out_win = out_rel.replace('/', '\\')
    print(run('if (Test-Path "%s") { Remove-Item -Recurse -Force "%s"; "removed stale" } '
              'else { "no stale dir" }' % (out_win, out_win), timeout=300, config=CFG).strip())

    script = ('C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_20261010/'
              'delivery_traintask_dev_gate.py')
    args = [
        '--scripts', 'C-VRP_Cold-chainVehicleRoutingProblem/scripts',
        '--protocol', 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_20261010/'
                      'delivery_traintask_protocol.json',
        '--recipe', 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_20261010/'
                    'delivery_traintask_recipe.json',
        '--head', HEAD_REL,
        '--out', out_rel,
        '--checkpoint', 'MASKCO_code/ckpts/cvrp100.ckpt',
        '--upstream-root', 'MASKCO_code',
        '--instances', str(a.instances),
    ]
    cmd = 'cd "%s"; & "%s" %s %s' % (WIN_BASE, PY, script, ' '.join(args))
    print('running:', cmd, flush=True)
    t0 = time.perf_counter()
    out = run(cmd, timeout=7200, config=CFG)
    print(out)
    print('[remote wall %.1fs]' % (time.perf_counter() - t0))
    print('result path (remote): %s' % (out_win + r'\result.json'))


if __name__ == '__main__':
    main()
