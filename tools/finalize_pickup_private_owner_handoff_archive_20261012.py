"""Archive identity completion; does not reexecute any qualified workload."""
import json,hashlib,psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/private_owner_handoff_root_v1';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
reg=read(OUT/'REGISTRATION.json');archive=read(OUT/'ARCHIVE_IDENTITY.json');backup=R/'current_entries_before_private_owner_handoff_20261012';entries=read(backup/'UPDATE_IDENTITY.json')['files'];s=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json')
names={'qualify_pickup_private_owner_handoff_v1.py','audit_pickup_private_owner_handoff_v1.py'}
active=[dict(pid=p.pid,cmd=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(arg).name in names for arg in p.info['cmdline'] or [])]
assert not active and all(sha(p)==h for p,h in reg['sources'].items()) and all(sha(p)==h for p,h in archive['frozen'].items())
assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in entries)
assert not s['stable_online_gain_established'] and not s['MaskCO_attribution_established'] and not s['pickup']['full_live_request_qualified'] and not s['pickup']['full_loaded_dependency_closure_qualified'] and not s['pickup']['full_live_owner_transaction_qualified']
update=read(backup/'UPDATE_IDENTITY.json');assert sha(E/update['report'])==archive['report_sha256']
receipt=R/'AUTOMATION_PRIVATE_OWNER_HANDOFF_UPDATE_20261012.json'
result=dict(registered_files_exact=len(reg['sources']),frozen_files_exact=len(archive['frozen']),current_entries_backed_and_exact=len(entries),report_exact=True,owned_workers_active=active,full_live_request_qualified=False,full_live_owner_transaction_qualified=False,stable_online_gain_established=False,automation_receipt_sha256=sha(receipt),shared_MD_uploaded=False)
with (OUT/'FINAL_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
