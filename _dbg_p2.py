import sys, tempfile, shutil, subprocess, os, importlib.util, json
from pathlib import Path
EXT = Path(r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem')
for p in ('scripts', 'scripts/simulation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))
RUNNER = EXT / 'scripts/simulation/d3_l1_v118_forward_runner.py'
SYNTH_WORKER = EXT / 'scripts/tests/d3_l1_v118_synthetic_worker.py'
import d3_l1_v118_persistence as persistence

root = Path(tempfile.mkdtemp(prefix='v120_dbg_'))
run_dir = root / 'P2_dbg'
spec = importlib.util.spec_from_file_location('dbg', str(RUNNER))
m = importlib.util.module_from_spec(spec)
sys.modules['dbg'] = m
spec.loader.exec_module(m)
m.freeze_identity(run_dir, 'd3l1-real-f-forward-20261015-v120', now_utc='2026-10-15T00:00:00Z')
cmd = [sys.executable, str(RUNNER), '--supervise', '--out-dir', str(run_dir),
       '--synthetic-command', sys.executable, str(SYNTH_WORKER), '--out-dir', str(run_dir),
       '--mode', 'pass']
proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
print('rc', proc.returncode)
print('glob matches:', [p.name for p in run_dir.glob('EXECUTION_CONTEXT_[0-9a-f]*.json')])
print('all EXECUTION files:', [p.name for p in run_dir.glob('EXECUTION_*')])
print('guard dir:', [p.name for p in (run_dir / 'guard').glob('*')] if (run_dir / 'guard').is_dir() else 'absent')
shutil.rmtree(root, ignore_errors=True)
