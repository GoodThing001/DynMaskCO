"""Same-input engineering recovery after a diagnosed lazy import; no science."""
import json,shutil,hashlib
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools';P=ROOT/'research/review_3080_20261010/pickup';OLD=P/'endogenous_owner_transaction_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
assert read(OLD/'RUN_EXIT.json')['exit_code']==1
active=[p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(a).name=='qualify_pickup_endogenous_owner_transaction_v1.py' for a in p.info['cmdline'] or [])];assert not active
a=read(OLD/'SOURCE_START_0.json');b=read(OLD/'SOURCE_COMPUTE_END_0.json');added=sorted(set(b['files'])-set(a['files']));assert len(added)==2 and all('rolling_optimizer.' in p for p in added);assert all(b['files'].get(p)==h for p,h in a['files'].items())
for p,h in read(OLD/'REGISTRATION.json')['sources'].items():
    assert sha(p)==h
    if Path(p).parent==T:shutil.copy2(p,OLD/('FROZEN_'+Path(p).name))
diagnosis=dict(failure='source qualification prevented guard publication; rolling_optimizer source and existing bytecode absent at start, unchanged existing files',new_loaded_files=added,previous_files_unchanged=True,owned_processes_gone=True,production_response_not_serialized_before_guard_failure=True,not_scientific_retry=True,original_exit=1)
with (OLD/'LAZY_IMPORT_DIAGNOSIS.json').open('x',encoding='utf-8') as f:json.dump(diagnosis,f,indent=2)
for n in ('qualify_pickup_endogenous_owner_transaction','audit_pickup_endogenous_owner_transaction','run_pickup_endogenous_owner_transaction_bounded'):
    text=(T/(n+'_v1.py')).read_text(encoding='utf-8').replace('endogenous_owner_transaction_root_v1','endogenous_owner_transaction_root_v2').replace('endogenous_owner_transaction_v1.py','endogenous_owner_transaction_v2.py').replace('endogenous_owner_transaction_bounded_v1.py','endogenous_owner_transaction_bounded_v2.py')
    if n.startswith('qualify'):
        text=text.replace('import hard_admission','import hard_admission\nimport rolling_optimizer')
        text=text.replace("files=[ROOT/'tools/build_pickup_endogenous_owner_v1.py',", "files=[ROOT/'tools/recover_pickup_endogenous_owner_v2.py',OLD_FAILURE,OLD_REGISTRATION,")
        text=text.replace("CONTROL=struct.Struct", "OLD_FAILURE=P/'endogenous_owner_transaction_root_v1/LAZY_IMPORT_DIAGNOSIS.json';OLD_REGISTRATION=P/'endogenous_owner_transaction_root_v1/REGISTRATION.json'\nCONTROL=struct.Struct")
        text=text.replace("installed,reason=owner.commit(token,response)", "encode,_=old.codecs();save('RESPONSE_ATTEMPT_'+str(case)+'.json',encode(response))\n                installed,reason=owner.commit(token,response)")
    with (T/(n+'_v2.py')).open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps(diagnosis))
