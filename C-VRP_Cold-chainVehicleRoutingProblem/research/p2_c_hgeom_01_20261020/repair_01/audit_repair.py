"""Read-only prerequisite audit: does not import or execute the native candidate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def review():
    reg=json.loads((HERE/'repair_registration.json').read_text(encoding='utf-8'))
    old=json.loads((HERE.parent/'run_identity.json').read_text(encoding='utf-8'))
    workspace=HERE.parents[3]
    mismatches=[]
    for name,h in {**reg['preserve'],**reg['contracts'],**old['frozen_original_files']}.items():
        if sha(workspace/name)!=h:mismatches.append(name)
    # This merely verifies existing files; no installation, compiler or native
    # API is invoked. The observed HTTP catalogue records are separate evidence.
    compiler_candidates=[p for p in (shutil.which('cl.exe'),shutil.which('clang-cl.exe')) if p]
    sdk_roots=[Path('C:/Program Files (x86)/Windows Kits/10'),
               Path('C:/Program Files (x86)/Windows Kits/8.1')]
    headers=[str(p) for root in sdk_roots if root.exists() for p in root.rglob('winnt.h')]
    helper=HERE/'sdk_check.dll'
    return dict(schema='hgeom_repair01_prerequisite_audit_v1',status='BLOCKED',
                original_hash_mismatches=mismatches,
                compiler_on_current_path=compiler_candidates,checked_sdk_headers=headers,
                compiled_helper_exists=helper.exists(),
                blocker='INDEPENDENT_SDK_CHECKER_BUILD_NOT_ESTABLISHED',
                native_cases_run=0,formal_units_evaluated=0,new_T2_created=False,
                note='No claim that all possible compiler locations were exhaustively inventoried. Host installation paths and portable acquisition evidence are described in qualification_report.md.')


if __name__=='__main__':
    target=HERE/'preflight.json'
    if target.exists():raise SystemExit('Refuse overwrite prerequisite receipt')
    result=review()
    target.write_bytes((json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    print(json.dumps(result,ensure_ascii=False))
