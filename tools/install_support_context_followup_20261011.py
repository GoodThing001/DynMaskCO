"""Install only the reviewed narrow context patch after exact source checks."""
import hashlib
import json
from pathlib import Path
import shutil
import difflib

ROOT=Path(__file__).resolve().parents[1]
AGENT=Path(r'C:\Users\mmc\.codex\worktrees\ba63\MASKCO-Main\.arbor\reproduction\research\review_3080_20261010\reproduction\followup_20261011')
DEST=ROOT/'research/review_3080_20261010/reproduction/followup_20261011'
CURRENT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_claim_gate/program_support_audit.py'
BASE='98dc3791336dc6873903f6a6a05874efdeb3b7dba88e75575001eb6fe7985d41'
AFTER='cfcb1c323f95de2b2c9806b35a5fba51e4fa9b2b8df460ce92f9780c1a3d700b'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    candidate=AGENT/'program_support_current_context_candidate.py'
    if sha(CURRENT)!=BASE or sha(candidate)!=AFTER:
        raise ValueError('Current base or candidate SHA mismatch; no stale overwrite')
    old=CURRENT.read_text(encoding='utf-8').splitlines(keepends=True)
    new=candidate.read_text(encoding='utf-8').splitlines(keepends=True)
    patch=''.join(difflib.unified_diff(old,new,fromfile='a/scripts/research_claim_gate/program_support_audit.py',tofile='b/scripts/research_claim_gate/program_support_audit.py'))
    expected=(AGENT/'CURRENT_CONTEXT_NARROW.patch').read_text(encoding='utf-8')
    if patch!=expected:
        raise ValueError('Candidate differs from reviewed narrow patch')
    DEST.mkdir(exist_ok=False)
    records=[]
    for p in AGENT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        q=DEST/p.relative_to(AGENT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        records.append({'path':q.relative_to(DEST).as_posix(),'sha256':sha(q)})
    backup=DEST/'installed_source_backup';backup.mkdir()
    shutil.copy2(CURRENT,backup/CURRENT.name)
    shutil.copy2(candidate,CURRENT)
    with (DEST/'ROOT_INSTALL_IDENTITY.json').open('x',encoding='utf-8') as f:
        json.dump({'base_sha256':BASE,'after_sha256':sha(CURRENT),'patch_verified':True,
                   'old_protocol_or_results_modified':False,'files':records},f,indent=2)
    print(json.dumps({'narrow_patch_installed':True,'files_integrated':len(records),'old_results_modified':False}))

if __name__=='__main__':main()
