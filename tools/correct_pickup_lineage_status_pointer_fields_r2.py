"""Align legacy caller fields with new actual13 evidence, retaining initial bytes."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';O=R/'pickup/lineage_header_cancel_root_v1';B=R/'current_entries_before_lineage_header_cancel_20261012'
def main():
    identity=read(B/'UPDATE_IDENTITY.json');assert all(sha(E/x['path'])==x['after_sha256'] for x in identity['files'])
    f=E/'results/research_20261010_continuation_v2/WORK_STATUS.json';previous=O/'WORK_STATUS_BEFORE_LEGACY_POINTER_CORRECTION_R2.json';assert not previous.exists();shutil.copy2(f,previous);v=read(f)
    assert v['pickup']['current_authoritative_epoch']==13 and v['pickup']['caller_selected_epoch']==12
    v['updated']=datetime.now(timezone.utc).isoformat();v['pickup'].update(caller_selected_epoch=13,caller_selected_callback=30,caller_selected_order=48,caller_selected_slot=1,discarded_pending_callback=31,discarded_pending_order=39,next_resume_callback=31)
    f.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    fresh=dict(identity);fresh['supersedes_pointer_metadata_only']=sha(B/'UPDATE_IDENTITY.json');fresh['files']=[dict(x,after_sha256=sha(E/x['path'])) for x in identity['files']];create(B/'UPDATE_IDENTITY_R2.json',fresh)
    arc=read(O/'ARCHIVE_IDENTITY.json');arc['entry_update_sha256']=sha(B/'UPDATE_IDENTITY_R2.json');arc['entry_update_path']=str(B/'UPDATE_IDENTITY_R2.json');arc['initial_archive_identity_sha256']=sha(O/'ARCHIVE_IDENTITY.json');create(O/'ARCHIVE_IDENTITY_R2.json',arc)
    create(O/'LEGACY_POINTER_FIELD_CORRECTION_R2.json',dict(created_utc=v['updated'],previous_WORK_STATUS_sha256=sha(previous),current_WORK_STATUS_sha256=sha(f),old_caller_selected_epoch=12,new_caller_selected_epoch=13,next_resume_callback=31,discarded_pending_order=39,initial_manifests_preserved=True,reason='initial recording updated current_authoritative fields but omitted existing caller_selected/discarded_pending/next_resume aliases; align all seven legacy fields to actual evidence',no_production_or_audit_rerun=True))
    print('{"legacy_pointer_fields_aligned":7,"experiment_rerun":false}')
if __name__=='__main__':main()
