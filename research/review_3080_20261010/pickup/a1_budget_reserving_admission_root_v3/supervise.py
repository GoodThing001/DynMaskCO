"""Exactly one bounded owned Popen for the requested local stage."""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
CAP_S = 7200


def main():
    kind = sys.argv[1]
    if kind not in ('run', 'audit'):
        raise ValueError('Unknown stage')
    active = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = p.info['cmdline'] or []
            if any(str(HERE / (name + '.py')).lower() == s.lower() for s in cmd for name in ('run', 'audit')):
                active.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if active:
        raise RuntimeError('Exact owned workload still active')
    started = time.perf_counter()
    timed_out = False
    with (HERE / (kind + '.stdout.txt')).open('x', encoding='utf-8') as out, \
            (HERE / (kind + '.stderr.txt')).open('x', encoding='utf-8') as err:
        p = subprocess.Popen([sys.executable, '-B', str(HERE / (kind + '.py'))], stdout=out, stderr=err,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            owned = psutil.Process(p.pid)
            identity = dict(pid=p.pid, create_time=owned.create_time(), cmdline=owned.cmdline(),
                            owner=owned.username())
            p.wait(timeout=CAP_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            p.kill()
            p.wait(timeout=5.)
    receipt = dict(started=started, finished=time.perf_counter(), returncode=p.returncode, timed_out=timed_out,
                   identity=identity, reaped=p.poll() is not None, cap_s=CAP_S, normal_stage_requested=kind,
                   offline_component_not_online_budget=True)
    with (HERE / (kind.upper() + '_SUPERVISION_RECEIPT.json')).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt))
    return p.returncode


if __name__ == '__main__':
    raise SystemExit(main())
