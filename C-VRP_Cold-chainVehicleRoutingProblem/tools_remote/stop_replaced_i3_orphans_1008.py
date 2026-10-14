"""Stop only the exact, already-replaced I3 v2 process group; preserve evidence."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
OLD = ROOT / 'results/i3_registered_1008_v2'
PGID = 708782
replacement = json.loads((OLD / 'replacement_record.json').read_text())
if replacement['replacement'] != 'results/i3_registered_1008_v3':
    raise RuntimeError('Exact replacement authorization is absent')

def members():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        pid = int(path.name)
        try:
            if os.getpgid(pid) != PGID:
                continue
            cmd = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            # Empty command is allowed only for a non-running zombie.
            status = (path / 'status').read_text().splitlines()
            zombie = any(line.startswith('State:') and 'Z' in line for line in status)
            if not zombie and (str(OLD) not in cmd or not any(
                    fragment in cmd for fragment in ('/pipeline.py', '/collect_i3_teachers.py'))):
                raise RuntimeError('Process group contains an unexpected live process: ' + cmd)
            found.append({'pid': pid, 'command': cmd, 'zombie': zombie})
        except (ProcessLookupError, FileNotFoundError):
            continue
    return found

before = members()
if not before or not any('/pipeline.py' in row['command'] for row in before):
    raise RuntimeError('Expected old supervisor no longer exists; do not signal another group')
record = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'old_stage': json.loads(
    (OLD / 'pipeline_state.json').read_text()), 'process_group': PGID, 'before': before,
    'reason': 'tmux replacement left the old timeout process group alive; duplicate collection is void',
    'scientific_parameters_changed': False, 'numerical_selection': False,
    'v3_tree_or_outputs_changed': False}
with (OLD / 'orphan_cleanup_started.json').open('x') as stream:
    json.dump(record, stream, indent=2)
os.killpg(PGID, signal.SIGTERM)
time.sleep(4)
after = members()
if any(not row['zombie'] for row in after):
    os.killpg(PGID, signal.SIGKILL)
    time.sleep(1)
    after = members()
record['after'] = after
record['live_old_processes'] = sum(not row['zombie'] for row in after)
with (OLD / 'orphan_cleanup_completed.json').open('x') as stream:
    json.dump(record, stream, indent=2)
if record['live_old_processes']:
    raise RuntimeError('Old processes survived; inspection required')
print(json.dumps({'old_live_processes': 0, 'preserved_old_outputs': True,
                  'new_formal_chain_untouched': True}))
