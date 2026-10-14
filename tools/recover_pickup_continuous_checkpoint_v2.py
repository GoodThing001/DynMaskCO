"""Preserve failed first integration; explicit old boundary completion in new tree."""
import json,shutil,psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools';P=ROOT/'research/review_3080_20261010/pickup';OLD=P/'continuous_checkpoint_root_v1';DIAG=P/'event27_settlement_diagnosis_root_v1'
def main():
    assert json.loads((OLD/'RUN_EXIT.json').read_text(encoding='utf-8'))['exit_code']==1
    diag=json.loads((DIAG/'RESULT.json').read_text(encoding='utf-8'));assert diag['source_inputs_unchanged'] and [r['base_qualified'] for r in diag['rows']]==[False,True]
    assert not [p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name=='qualify_pickup_continuous_checkpoint_v1.py' for x in p.info['cmdline'] or [])]
    for n in ('qualify_pickup_continuous_checkpoint_v1.py','audit_pickup_continuous_checkpoint_v1.py','build_pickup_continuous_audit_v1.py'):
        shutil.copy2(T/n,OLD/('FROZEN_'+n))
    shutil.copy2(T/'diagnose_pickup_event27_settlement_v1.py',DIAG/'FROZEN_diagnose_pickup_event27_settlement_v1.py')
    s=(T/'qualify_pickup_continuous_checkpoint_v1.py').read_text(encoding='utf-8').replace('continuous_checkpoint_root_v1','continuous_checkpoint_root_v2').replace('audit_pickup_continuous_checkpoint_v1.py','audit_pickup_continuous_checkpoint_v2.py')
    s=s.replace("paths=[Path(__file__),","paths=[ROOT/'tools/recover_pickup_continuous_checkpoint_v2.py',old.P/'continuous_checkpoint_root_v1/REGISTRATION.json',old.P/'continuous_checkpoint_root_v1/WORKER_FAILURE.json',old.P/'event27_settlement_diagnosis_root_v1/REGISTRATION.json',old.P/'event27_settlement_diagnosis_root_v1/RESULT.json',Path(__file__),")
    s=s.replace("            before=old.pf.digest(snapshot(clock))\n            try:env._run_loop", "            before=old.pf.digest(snapshot(clock))\n            # Sealed source stops immediately after event27 physical advance.\n            # Complete that original boundary once inside first request clock.\n            if seq==1:env._plan_and_commit(0,clock,fleet,traces,served,customers)\n            try:env._run_loop")
    s=s.replace("scope='only next two actual new reveals","scope='Engineering recovery: complete original event27 plan_and_commit once inside first request window (source was after physical advance only); preserve v1 failure. Same input/method/K10/budget; only next two actual new reveals")
    with (T/'qualify_pickup_continuous_checkpoint_v2.py').open('x',encoding='utf-8') as f:f.write(s)
    s=(T/'audit_pickup_continuous_checkpoint_v1.py').read_text(encoding='utf-8').replace('continuous_checkpoint_root_v1','continuous_checkpoint_root_v2').replace('qualify_pickup_continuous_checkpoint_v1.py','qualify_pickup_continuous_checkpoint_v2.py')
    s=s.replace("        before=old.pf.digest(snapshot())\n        try:private_env._run_loop", "        before=old.pf.digest(snapshot())\n        if index==0:private_env._plan_and_commit(0,private_clock,private_fleet,private_traces,private_served,customers)\n        try:private_env._run_loop")
    with (T/'audit_pickup_continuous_checkpoint_v2.py').open('x',encoding='utf-8') as f:f.write(s)
    print('v1 failure/source preserved; new v2 sources prepared, no execution')
if __name__=='__main__':main()
