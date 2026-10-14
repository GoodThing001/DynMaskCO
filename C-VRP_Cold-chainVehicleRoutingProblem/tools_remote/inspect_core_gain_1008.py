"""Read-only status of authorized A3/I3 experiments; no interim utility selection."""
import json
from pathlib import Path
import time

ROOT = Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
result = {'server_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'a3': {}, 'i3': {}}
for seed in (20260926, 20260930, 20261001):
    folder = ROOT / ('results/a3_baseline_full-cond_%s_1007_230001' % seed)
    progress = {}
    for path in folder.glob('progress_*.txt'):
        lines = path.read_text().splitlines()
        ids = [int(line.split()[0]) for line in lines]
        progress[path.name] = {'completed': len(lines), 'unique_days': len(set(ids)),
            'last': lines[-1] if lines else None,
            'all_ok': all(line.split()[-1] == 'ok' for line in lines)}
    gate = folder / 'gate.json'
    result['a3'][str(seed)] = {'gate_complete': gate.exists(), 'progress': progress,
        'exit': (folder / 'run.exit').read_text() if (folder / 'run.exit').exists() else None}
activation = ROOT / 'results/a3_repair_chain_1007_2250/A3_SAME_SOURCE_ADJUDICATION.json'
result['a3_adjudication_exists'] = activation.exists()
if activation.exists():
    answer = json.loads(activation.read_text())
    result['a3_adjudication'] = {k: answer[k] for k in ('formal_adjudicable', 'partial', 'n_batches', 'summary_by_arm_seed')}
folder = ROOT / 'results/i3_registered_1008_v3'
for name in ('pipeline_state.json', 'QA_PASS.json', 'pipeline_failure.json'):
    path = folder / name
    result['i3'][name] = json.loads(path.read_text()) if path.exists() else None
result['i3']['qa_exit'] = (folder / 'qa.exit').read_text() if (folder / 'qa.exit').exists() else None
result['i3']['formal_teacher_shards_complete'] = len(list(folder.glob('teacher_day_*/report.json')))
result['i3']['formal_training_complete'] = {
    name: (folder / name / 'train_report.json').exists() for name in ('maskco', 'reconstruction_control')}
print(json.dumps(result, indent=2))
