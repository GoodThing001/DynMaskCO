"""Supervisor r2 for the CPU-only 8+2 s diagnostic -- fail-closed child selection.

r1 -> r2 is forced by run_r2: `supervise_diagnostic.py` hardcoded the child name
(`server_diagnostic.py`), so run_r2 silently re-executed the *r1 executor* instead of the
corrected one and reproduced the r1 crash.  That failure mode must be impossible, not
merely avoided.

r2 therefore:
  * requires the child script name in `DIAG_CHILD` (no default -- the child is never guessed);
  * requires `DIAG_EXPECT_MANIFEST`, a JSON manifest of package-relative and repo-relative
    paths with their expected sha256;
  * verifies every manifest entry BEFORE launching, and refuses (exit 3, record on disk)
    on any mismatch or missing entry;
  * records the resolved child path and its actual sha256 in the supervisor record.

It still preserves the ORIGINAL timeout / exit / reap record: returncode, whether the wall
timeout fired, whether the child was signalled, the signal name, and the reap status.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ['DIAG_OUT'])
PACKAGE = Path(os.environ['DIAG_PACKAGE'])
ROOT = PACKAGE.parents[3] if (PACKAGE.parents[3] / 'MASKCO_code').is_dir() else PACKAGE.parents[4]
CHILD_NAME = os.environ.get('DIAG_CHILD')
MANIFEST_PATH = os.environ.get('DIAG_EXPECT_MANIFEST')
WALL_TIMEOUT = float(os.environ.get('DIAG_WALL_TIMEOUT', '600'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(payload):
    (OUT / 'SUPERVISOR_RECORD.json').write_text(json.dumps(payload, indent=2) + '\n',
                                                encoding='utf-8', newline='\n')


base = {'attempt_id': os.environ.get('DIAG_ATTEMPT_ID'),
        'resolved_child_name': CHILD_NAME, 'wall_timeout_seconds': WALL_TIMEOUT,
        'pid': None, 'timed_out': False, 'signalled': False, 'signal_sent': None,
        'signal_name': None, 'exit_code': None, 'reaped': False, 'wall_seconds': None,
        'refused_before_launch': False, 'identity_preflight': {}}

# ---- fail-closed preflight: never launch an unverified or unnamed child ----
if not CHILD_NAME:
    base['refused_before_launch'] = True
    base['refusal_reason'] = 'DIAG_CHILD is not set; the child script is never guessed'
    record(base)
    print(json.dumps(base, indent=2))
    raise SystemExit(3)

child = PACKAGE / CHILD_NAME
if not child.is_file():
    base['refused_before_launch'] = True
    base['refusal_reason'] = 'child script not found: %s' % child
    record(base)
    print(json.dumps(base, indent=2))
    raise SystemExit(3)

if not MANIFEST_PATH or not Path(MANIFEST_PATH).is_file():
    base['refused_before_launch'] = True
    base['refusal_reason'] = 'DIAG_EXPECT_MANIFEST missing or not a file: %r' % MANIFEST_PATH
    record(base)
    print(json.dumps(base, indent=2))
    raise SystemExit(3)

manifest = json.loads(Path(MANIFEST_PATH).read_text(encoding='utf-8'))
resolved, mismatches, absent = {}, [], []
for rel, expected in manifest.get('files', {}).items():
    path = PACKAGE / rel
    if not path.is_file():
        absent.append(rel)
        continue
    actual = sha(path)
    resolved[rel] = actual
    if actual != expected:
        mismatches.append(dict(path=rel, expected=expected, actual=actual))
for rel, expected in manifest.get('root_files', {}).items():
    path = ROOT / rel
    if not path.is_file():
        absent.append(rel)
        continue
    actual = sha(path)
    resolved[rel] = actual
    if actual != expected:
        mismatches.append(dict(path=rel, expected=expected, actual=actual))

base['identity_preflight'] = dict(manifest=str(MANIFEST_PATH), verified=resolved,
                                  mismatches=mismatches, absent=absent,
                                  manifest_sha256=sha(MANIFEST_PATH))
base['resolved_child_path'] = str(child)
base['resolved_child_sha256'] = sha(child)
if mismatches or absent:
    base['refused_before_launch'] = True
    base['refusal_reason'] = 'identity preflight failed: %d mismatch(es), %d absent' % (
        len(mismatches), len(absent))
    record(base)
    print(json.dumps(base, indent=2))
    raise SystemExit(3)

# ---- launch under a hard wall timeout -------------------------------------
cmd = [sys.executable, str(child)]
env = dict(os.environ)
env['DIAG_CHILD_LAUNCHED_BY'] = 'supervise_diagnostic_r2.py'
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
base['command'] = cmd
base['pid'] = proc.pid
record(base)

try:
    out, _ = proc.communicate(timeout=WALL_TIMEOUT)
except subprocess.TimeoutExpired:
    base['timed_out'] = True
    base['signal_sent'] = int(signal.SIGTERM)
    base['signal_name'] = 'SIGTERM'
    proc.send_signal(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=15)
        base['signalled'] = True
    except subprocess.TimeoutExpired:
        base['signal_sent'] = int(signal.SIGKILL)
        base['signal_name'] = 'SIGKILL'
        proc.kill()
        out, _ = proc.communicate()
        base['signalled'] = True

try:
    pid, status = os.waitpid(proc.pid, os.WNOHANG)
    base['reaped'] = True
    base['reap_status'] = status
    base['reap_pid'] = pid
except ChildProcessError:
    base['reaped'] = proc.poll() is not None
    base['reap_status'] = proc.returncode
    base['reap_pid'] = proc.pid

base['exit_code'] = proc.returncode
base['wall_seconds'] = time.time() - t0
(OUT / 'SUPERVISOR_STDOUT.txt').write_text((out or b'').decode('utf-8', errors='replace'),
                                           encoding='utf-8', newline='\n')
record(base)
print(json.dumps(base, indent=2))
