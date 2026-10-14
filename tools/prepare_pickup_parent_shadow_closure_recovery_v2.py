"""Preserve failed v1; predeclare only two normal closure recovery requests."""
import ast,json,shutil,hashlib
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup';V1=P/'parent_decode_guard_shadow_root_v1';V2=P/'parent_decode_guard_shadow_root_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not V2.exists()
assert json.loads((V1/'RESULT.json').read_text())['failure_names']==['0:actual_loaded_component_closure','1:actual_loaded_component_closure']
for p in psutil.process_iter(['name','cmdline']):
    if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and p.pid!=psutil.Process().pid:
        assert not any('qualify_pickup_parent_decode_guard_shadow_v1.py' in x for x in p.info['cmdline'] or []),'v1 worker still alive'
names=['qualify_pickup_parent_decode_guard_shadow_v1.py','audit_pickup_parent_decode_guard_shadow_v1.py','pickup_loaded_closure_v1.py','run_pickup_parent_shadow_bounded_v1.py']
for name in names:
    f=ROOT/'tools'/name;target=V1/('FROZEN_'+name)
    assert not target.exists();shutil.copy2(f,target)
source=(ROOT/'tools'/names[0]).read_text(encoding='utf-8')
source=source.replace('parent_decode_guard_shadow_root_v1','parent_decode_guard_shadow_root_v2').replace('audit_pickup_parent_decode_guard_shadow_v1.py','audit_pickup_parent_decode_guard_shadow_v2.py').replace('run_pickup_parent_shadow_bounded_v1.py','run_pickup_parent_shadow_bounded_v2.py')
source=source.replace("items=[Path(__file__),", "items=[Path(__file__),ROOT/'tools/prepare_pickup_parent_shadow_closure_recovery_v2.py',")
source=source.replace("return {str(p):sha(p) for p in items}","items += sorted(p for p in V1.iterdir() if p.is_file())\n    return {str(p):sha(p) for p in items}")
source=source.replace("CONTROL=struct.Struct", "V1=P/'parent_decode_guard_shadow_root_v1'\nCONTROL=struct.Struct")
source=source.replace('owner.signature();capture();start=capture(identities())','old.codecs();owner.signature();capture();start=capture(identities())')
source=source.replace("six parent-role stage faults only callback15", "v2 only two normal closure recovery; six original v1 stage faults retained without rerun")
source=source.replace("new_C0_or_SAA_execution=False,", "original_v1_registration_sha256=sha(V1/'REGISTRATION.json'),original_v1_result_sha256=sha(V1/'RESULT.json'),original_failures_preserved=True,preinitialize_actual_audit_io=True,new_C0_or_SAA_execution=False,")
source=source.replace("        records += [execute(1+fault,items[0],fault) for fault in range(1,7)]\n",'')
target=ROOT/'tools/qualify_pickup_parent_decode_guard_shadow_v2.py';assert not target.exists();ast.parse(source);target.write_text(source,encoding='utf-8')
source=(ROOT/'tools'/names[1]).read_text(encoding='utf-8').replace('parent_decode_guard_shadow_root_v1','parent_decode_guard_shadow_root_v2')
source=source.replace("def sha(p):", "V1=P/'parent_decode_guard_shadow_root_v1'\ndef sha(p):")
source=source.replace("psutil.Process().cpu_affinity([7]);reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');_,decode=old.codecs();checks=[]", "psutil.Process().cpu_affinity([7]);reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');original=read(V1/'RESULT.json');_,decode=old.codecs();checks=[]\n    records=raw['records']+original['records'][2:]")
source=source.replace("len(raw['records'])==8","len(raw['records'])==2 and len(records)==8")
source=source.replace("for r in raw['records']","for r in records")
source=source.replace("for record in raw['records']:","ck('original_failure_kept',original['failure_names']==['0:actual_loaded_component_closure','1:actual_loaded_component_closure'] and reg['original_v1_result_sha256']==sha(V1/'RESULT.json'))\n    for record in records:")
source=source.replace("n=str(record['case']);exp,prior=expected[record['callback']]", "n=str(record['case']);exp,prior=expected[record['callback']];evidence=V1 if record['fault'] else OUT")
source=source.replace("begin=read(OUT/('SOURCE_START_'", "begin=read(evidence/('SOURCE_START_'")
source=source.replace("not (OUT/('REPLICA_RESULT_'", "not (evidence/('REPLICA_RESULT_'")
source=source.replace("ck(n+':owned_worker_reaped',record['reaped'] and record['pid']==record['ready']['pid'])", "active=[p for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any('qualify_pickup_parent_decode_guard_shadow_v' in x for x in p.info['cmdline'] or [])]\n        ck(n+':owned_worker_reaped',record['reaped'] and record['pid']==record['ready']['pid'] and not active)")
source=source.replace('registered_sha256=sha', 'original_v1_sha256=sha(V1/\'RESULT.json\'),original_runtime_checks=35,original_runtime_failed=2,normal_recovery_requests=2,original_six_faults_rerun=False,registered_sha256=sha')
target=ROOT/'tools/audit_pickup_parent_decode_guard_shadow_v2.py';assert not target.exists();ast.parse(source);target.write_text(source,encoding='utf-8')
source=(ROOT/'tools'/names[3]).read_text(encoding='utf-8').replace('parent_decode_guard_shadow_root_v1','parent_decode_guard_shadow_root_v2').replace('qualify_pickup_parent_decode_guard_shadow_v1.py','qualify_pickup_parent_decode_guard_shadow_v2.py').replace('audit_pickup_parent_decode_guard_shadow_v1.py','audit_pickup_parent_decode_guard_shadow_v2.py')
target=ROOT/'tools/run_pickup_parent_shadow_bounded_v2.py';assert not target.exists();ast.parse(source);target.write_text(source,encoding='utf-8')
with (V1/'FAILURE_MECHANISM_AND_SOURCE_PRESERVATION.json').open('x',encoding='utf-8') as f:json.dump(dict(failed_checks=2,source_changed=False,added_source='scripts/correction_engineering/audit_io.py and existing pyc imported lazily by actual public digest',v2_recovery='same two sealed endpoints and guard/deadlines; preinitialize actual audit_io; no six fault rerun; original FAIL unchanged',frozen={n:sha(V1/('FROZEN_'+n)) for n in names}),f,indent=2)
print(json.dumps(dict(v1_preserved=True,new_sources=3,six_faults_rerun=False)))
