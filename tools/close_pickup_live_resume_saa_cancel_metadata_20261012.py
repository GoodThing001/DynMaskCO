"""Correct only the current cross-line next-action pointer, preserving prior bytes."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/live_resume_saa_cancel_root_v1';B=R/'current_entries_before_live_resume_saa_cancel_20261012'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
state=E/'results/research_20261010_continuation_v2/WORK_STATUS.json';index=B/'UPDATE_IDENTITY.json';archive=O/'ARCHIVE_IDENTITY.json'
for source,name in ((state,'PRIOR_CROSS_LINE_WORK_STATUS.json'),(index,'PRIOR_CROSS_LINE_UPDATE_IDENTITY.json'),(archive,'PRIOR_CROSS_LINE_ARCHIVE_IDENTITY.json')):
    dest=O/name;assert not dest.exists();shutil.copy2(source,dest)
obj=read(state);before=obj['delivery']['next'];assert 'Continue pickup epoch5 actual pending order7' in before
obj['delivery']['next']=before.replace('Continue pickup epoch5 actual pending order7','Continue pickup selected epoch6/callback23 after actual order7 commit; pending order58 has only been accepted in a discarded partial-scoring branch')
state.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
idx=read(index)
for x in idx['files']:
    if x['path']=='results/research_20261010_continuation_v2/WORK_STATUS.json':x['after_sha256']=sha(state)
index.write_text(json.dumps(idx,ensure_ascii=False,indent=2),encoding='utf-8');arc=read(archive);arc['entry_update_sha256']=sha(index);archive.write_text(json.dumps(arc,ensure_ascii=False,indent=2),encoding='utf-8')
with (O/'CROSS_LINE_NEXT_POINTER_CORRECTION.json').open('x',encoding='utf-8') as f:json.dump(dict(before=before,after=obj['delivery']['next'],only_current_next_action_metadata=True,no_result_or_registration_changed=True,prior_bytes_preserved=True),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(current_entries=16,cross_line_pointer='epoch6 pending order58',qualified_results_unchanged=True)))
