"""Stop our own waiting supervisor only, before any formal collection starts."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
old = ROOT / 'results/i3_registered_1008_v2'
state = json.loads((old / 'pipeline_state.json').read_text())
if (state['stage'] != 'waiting_qualified_a3_failure' or
        (old / 'teacher_collection_config.json').exists() or
        list(old.glob('teacher_day_*'))):
    raise RuntimeError('Formal work may have started; do not interrupt or replace it')
note = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'previous_stage': state['stage'],
    'reason': 'Independent adjudicator fail-closed qualification revision before formal collection',
    'formal_teacher_collection_started': False, 'scientific_parameters_changed': False,
    'replacement': 'results/i3_registered_1008_v3', 'old_qa_evidence_preserved': True}
with (old / 'replacement_record.json').open('x') as stream:
    json.dump(note, stream, indent=2)
subprocess.run(['tmux', 'kill-session', '-t', 'i3_registered_1008_v2'], check=True)
subprocess.run(['bash', str(ROOT / 'tools_remote/start_i3_registered_1008_v3.sh')], check=True)
print('REPLACED_WAITING_SUPERVISOR_ONLY=1')
