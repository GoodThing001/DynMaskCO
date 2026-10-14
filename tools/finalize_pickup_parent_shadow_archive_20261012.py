"""Final archive identities and exact owned process command arguments, no QA replay."""
import json,hashlib,psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/parent_decode_guard_shadow_root_v2';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
reg=read(OUT/'REGISTRATION.json');archive=read(OUT/'ARCHIVE_IDENTITY.json');backup=R/'current_entries_before_parent_shadow_20261012';entries=read(backup/'UPDATE_IDENTITY.json')['files'];s=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json')
owned_names={'qualify_pickup_parent_decode_guard_shadow_v1.py','qualify_pickup_parent_decode_guard_shadow_v2.py','audit_pickup_parent_decode_guard_shadow_v1.py','audit_pickup_parent_decode_guard_shadow_v2.py'}
active=[dict(pid=p.pid,cmd=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(arg).name in owned_names for arg in p.info['cmdline'] or [])]
changed=[p for p,h in reg['sources'].items() if sha(p)!=h];changed_frozen=[p for p,h in archive['frozen'].items() if sha(p)!=h]
changed_entries=[x['path'] for x in entries if sha(E/x['path'])!=x['after_sha256'] or sha(backup/x['path'])!=x['before_sha256']]
assert not active and not changed and not changed_frozen and not changed_entries,(active,changed,changed_frozen,changed_entries)
assert not s['stable_online_gain_established'] and not s['pickup']['full_live_request_qualified'] and not s['pickup']['full_loaded_dependency_closure_qualified'] and not s['pickup']['live_owner_handoff_qualified']
result=dict(registered_files_exact=len(reg['sources']),frozen_files_exact=len(archive['frozen']),current_entries_backed_and_exact=len(entries),owned_workers_active=active,full_live_request_qualified=False,stable_online_gain_established=False,automation_receipt_sha256=sha(R/'AUTOMATION_PARENT_SHADOW_UPDATE_20261012.json'),archive_reader_correction='UTF-8 explicit and actual script argument basename identity; prior inline reader self-matched its own -c text, no worker restart')
with (OUT/'FINAL_ARCHIVE_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
