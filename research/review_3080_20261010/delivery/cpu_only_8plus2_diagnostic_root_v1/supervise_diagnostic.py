"""Supervisor for the CPU-only 8+2 s diagnostic.

Runs server_diagnostic.py as a child with a hard wall-clock timeout, and preserves the
ORIGINAL timeout / exit / reap record: returncode, whether the wall timeout fired,
whether the child was signalled, the signal name, and the reap status.  The supervisor
never rewrites the child's own DIAG_STATE.json; it writes SUPERVISOR_RECORD.json and
SUPERVISOR_STDOUT.txt beside it.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ['DIAG_OUT'])
PACKAGE = Path(os.environ['DIAG_PACKAGE'])
WALL_TIMEOUT = float(os.environ.get('DIAG_WALL_TIMEOUT', '240'))

cmd = [sys.executable, str(PACKAGE / 'server_diagnostic.py')]
env = dict(os.environ)
t0 = time.time()
record = {'command': cmd, 'wall_timeout_seconds': WALL_TIMEOUT, 'pid': None,
          'timed_out': False, 'signalled': False, 'signal_sent': None, 'signal_name': None,
          'exit_code': None, 'reaped': False, 'wall_seconds': None}
proc = subprocess.Popen(cmd, cwd='/home/hzeng/project/MASKCO-Main', env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
record['pid'] = proc.pid
(OUT / 'SUPERVISOR_RECORD.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

try:
    out, _ = proc.communicate(timeout=WALL_TIMEOUT)
except subprocess.TimeoutExpired:
    record['timed_out'] = True
    record['signal_sent'] = int(signal.SIGTERM)
    record['signal_name'] = 'SIGTERM'
    proc.send_signal(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=15)
        record['signalled'] = True
    except subprocess.TimeoutExpired:
        record['signal_sent'] = int(signal.SIGKILL)
        record['signal_name'] = 'SIGKILL'
        proc.kill()
        out, _ = proc.communicate()
        record['signalled'] = True

# explicit reap: the child has been waited for, so it cannot linger
try:
    pid, status = os.waitpid(proc.pid, os.WNOHANG)
    record['reaped'] = True
    record['reap_status'] = status
    record['reap_pid'] = pid
except ChildProcessError:
    # already reaped by communicate(); confirm via poll()
    record['reaped'] = proc.poll() is not None
    record['reap_status'] = proc.returncode
    record['reap_pid'] = proc.pid

record['exit_code'] = proc.returncode
record['wall_seconds'] = time.time() - t0
(OUT / 'SUPERVISOR_STDOUT.txt').write_text(
    (out or b'').decode('utf-8', errors='replace'), encoding='utf-8', newline='\n')
(OUT / 'SUPERVISOR_RECORD.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2))
