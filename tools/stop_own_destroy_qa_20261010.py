"""Stop only the verified own synthetic child before any research data exist."""
import base64
from _srv import run
CODE=r'''
from pathlib import Path
import json,os,signal
r=Path('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/research_1010_destroy_eq8_v1')
if (r/'PILOT_START.json').exists() or (r/'pilot').exists():raise RuntimeError('Research phase already started; this QA-only stop is disallowed')
state=json.loads((r/'pipeline_state.json').read_text());parent=state['pid']
if state['stage']!='actual_synthetic_QA':raise RuntimeError('Not synthetic QA')
if os.stat('/proc/%d'%parent).st_uid!=os.getuid():raise RuntimeError('Owner mismatch')
target=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        status=(p/'status').read_text();pp=int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if pp==parent and str(r/'src/scripts/research_20261010/delivery_destroy_eq8.py') in cmd and str(r/'qa') in cmd:
            if p.stat().st_uid!=os.getuid():raise RuntimeError('Child owner mismatch')
            target.append(int(p.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):continue
if len(target)!=1:raise RuntimeError('Expected exactly one verified own QA child')
with (r/'QA_ENGINEERING_STOP.json').open('x') as f:json.dump({'pid':target[0],'parent':parent,'reason':'Static QA review found exceptional current-arm prefixes not durable; all-case retention requires engineering journal fix. No research data generated. New source/directory required.','scientific_parameter_change':False},f)
os.kill(target[0],signal.SIGTERM)
print(json.dumps({'stopped_own_synthetic_child':target[0],'parent_retains_failure_and_exit':parent,'research_data_generated':False}))
'''
if __name__=='__main__':
    encoded=base64.b64encode(CODE.encode()).decode()
    print(run('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c "import base64;exec(base64.b64decode(\'%s\'))"'%encoded,timeout=60))
