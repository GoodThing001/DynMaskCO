"""Synthetic-only native containment qualification. Formal requests fail closed."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import json
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from journal import Journal, atomic
from windows_resources import NativeTree


def worker(root, mode, amount):
    root = Path(root)
    atomic(root/'worker_started.json', dict(pid=os.getpid(), threads=os.environ['OMP_NUM_THREADS']))
    if mode == 'exception':
        raise RuntimeError('SYNTHETIC_INJECTED_EXCEPTION')
    if mode == 'memory':
        blocks = []
        try:
            while True:
                blocks.append(bytearray(4*1024**2))
        except MemoryError:
            atomic(root/'memory_denied.json', dict(memory_error=True, blocks=len(blocks)))
        return
    if mode == 'mapped':
        # File-backed pages distinguish RSS from private job commit. Pure zeros,
        # generated here, no repository or source-day inputs are involved.
        with tempfile.TemporaryFile() as f:
            f.truncate(amount)
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                total = sum(mm[i] for i in range(0, amount, 4096))
                atomic(root/'mapped_done.json', dict(bytes=amount, checksum=total))
                time.sleep(.1)
        return
    if mode == 'child':
        try:
            subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
        except OSError as exc:
            atomic(root/'child_refused.json', dict(refused=True, winerror=exc.winerror))
            return
        raise RuntimeError('unexpected third process admission')
    if mode == 'hang':
        while True:
            time.sleep(1)  # supervisor must interrupt a blocked call, not a cooperative loop
    atomic(root/'normal_done.json', dict(ok=True))


def run(root, mode, limit, wall, unit_wall, amount):
    started = time.monotonic()
    root = Path(root)
    ledger = Journal(root, [f'synthetic_{i:02}' for i in range(19)])
    try:
        tree = NativeTree(limit)
    except OSError as exc:
        ledger.append('synthetic_00', 'END', failure_code='RESOURCE_MONITOR_UNAVAILABLE', detail=str(exc))
        atomic(root/'result.json', dict(status='BLOCKED', failure_code='RESOURCE_MONITOR_UNAVAILABLE',
                                       detail=str(exc), slots=ledger.postmortem(no_live_workers_confirmed=True)))
        return 3
    import psutil
    supervisor = psutil.Process()
    supervisor.cpu_affinity([supervisor.cpu_affinity()[0]])
    ledger.append('synthetic_00', 'START')
    unit_start = time.monotonic()
    child = None
    reason = None
    rss_peak = private_peak = 0
    try:
        child = subprocess.Popen([sys.executable, '-B', __file__, '--worker', '--root', str(root),
                                  '--mode', mode, '--amount', str(amount)],
                                 creationflags=0x4,  # Windows CREATE_SUSPENDED (not exported by subprocess)
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 close_fds=True)
        ws = tree.constrain_suspended(child.pid)
        atomic(root/'containment.json', dict(worker_pid=child.pid, supervisor_pid=os.getpid(),
                                           worker_ws=ws, native=tree.readback()))
        reserve = min(.25, wall/4, unit_wall/4)  # scaled synthetic teardown reserve only
        while True:
            processes = [supervisor]
            try:
                processes += supervisor.children(recursive=True)
            except psutil.Error:
                pass
            rss = private = 0
            for proc in processes:
                try:
                    info = proc.memory_info()
                    rss += info.rss
                    private += info.private
                except psutil.NoSuchProcess:
                    pass
            rss_peak, private_peak = max(rss_peak, rss), max(private_peak, private)
            now = time.monotonic()
            if now-started >= wall-reserve:
                reason = 'PACKAGE_TIMEOUT'
            elif rss > limit or private > limit:
                reason = 'MEMORY_LIMIT_EXCEEDED'
            elif now-unit_start >= unit_wall-reserve:
                reason = 'UNIT_TIMEOUT'
            elif child.poll() is not None:
                reason = ('MEMORY_LIMIT_EXCEEDED' if (root/'memory_denied.json').exists()
                          else 'PROCESS_EXCEPTION' if child.returncode else None)
                break
            if reason:
                child.kill()
                break
            time.sleep(.005)
        child.wait(timeout=reserve)
    except Exception as exc:
        reason = 'RESOURCE_MONITOR_UNAVAILABLE'
        if child and child.poll() is None:
            child.kill()
            child.wait(timeout=.25)
        atomic(root/'supervision_error.json', dict(detail=str(exc)))
    elapsed = time.monotonic()-started
    ledger.append('synthetic_00', 'END', failure_code=reason, elapsed_s=elapsed)
    receipt = dict(status='FAILED' if reason else 'OK', failure_code=reason, elapsed_s=elapsed,
                   worker_exited=child is None or child.poll() is not None,
                   peak_rss_sum=rss_peak, peak_private_sum=private_peak, native=tree.readback(),
                   slots=ledger.postmortem(no_live_workers_confirmed=True,
                                           package_timeout=reason == 'PACKAGE_TIMEOUT'))
    atomic(root/'result.json', receipt)
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--root', type=Path)
    ap.add_argument('--mode', default='normal', choices=['normal','hang','memory','mapped','child','exception'])
    ap.add_argument('--limit', type=int, default=6*1024**3)
    ap.add_argument('--wall', type=float, default=8)
    ap.add_argument('--unit-wall', type=float, default=6)
    ap.add_argument('--amount', type=int, default=128*1024**2)
    args = ap.parse_args()
    if args.formal:
        raise SystemExit('FORMAL_EVALUATION_NOT_AUTHORISED: no input read')
    if args.worker:
        worker(args.root, args.mode, args.amount)
    else:
        raise SystemExit(run(args.root, args.mode, args.limit, args.wall, args.unit_wall, args.amount))
