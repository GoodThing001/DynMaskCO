"""Preserve failed closure and repeat only engineering after a proven seal bug."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=D/'multistep_insertable_teacher_engineering_root_v1'
    raw=json.loads((old/'RESULT.json').read_text());audit=json.loads((old/'INDEPENDENT_MULTISTEP_REPLAY.json').read_text())
    assert raw['failed']==1 and audit['failed']==2 and raw['state_count']==8
    extra=raw['sealed']['new_loaded_files']
    assert len(extra)==4 and all('/email/' in p.replace('\\','/') for p in extra) and not raw['sealed']['changed']
    source=ROOT/'tools/qualify_delivery_multistep_teacher_root_v1.py';text=source.read_text()
    text=text.replace("multistep_insertable_teacher_engineering_root_v1", "multistep_insertable_teacher_engineering_root_v2")
    marker='    import numpy, ortools, google.protobuf\n'
    assert text.count(marker)==1
    text=text.replace(marker,marker+"    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'ortools', 'protobuf')}\n    machine = platform.platform()\n")
    text=text.replace("'platform': platform.platform()", "'platform': machine")
    text=text.replace("'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'ortools', 'protobuf')}", "'packages': versions")
    target=ROOT/'tools/qualify_delivery_multistep_teacher_root_v2.py'
    with target.open('x') as f:f.write(text)
    original_audit=ROOT/'tools/audit_delivery_multistep_teacher_root_v1.py';newaudit=ROOT/'tools/audit_delivery_multistep_teacher_root_v2.py'
    with newaudit.open('x') as f:f.write(original_audit.read_text().replace('multistep_insertable_teacher_engineering_root_v1','multistep_insertable_teacher_engineering_root_v2'))
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'failure':'Metadata version lookup lazily imported email parser after closure file enumeration but before first request. No source changes; start file list incomplete, so not environment qualified.',
        'original_raw_sha256':sha(old/'RESULT.json'),'original_audit_sha256':sha(old/'INDEPENDENT_MULTISTEP_REPLAY.json'),
        'all_old_states_preserved':True,'new_directory':'multistep_insertable_teacher_engineering_root_v2',
        'repair':'Resolve metadata and platform before enumerating actual loaded files.',
        'scientific_parameters_changed':False,'new_science_data':False,'repeat_reason':'only identical observed-fixture engineering QA after independent closure failure',
        'new_launcher_sha256':sha(target),'new_audit_sha256':sha(newaudit)}
    with (old/'CLOSURE_ENUMERATION_FAILURE.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'preserved_old_failure':True,'new_launcher':str(target)}))
if __name__=='__main__':main()
