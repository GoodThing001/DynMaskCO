"""Correct only stored service provenance after R2's temporal assumption failed."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools';OUT=ROOT/'research/review_3080_20261010/pickup/resident_two_events_root_v1'
old=T/'audit_pickup_resident_projection_recovery_r2.py';raw=json.loads((OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R2.json').read_text(encoding='utf-8'));assert raw['failed']==1 and raw['failure_names']==['original_stored_physical_service52']
shutil.copy2(old,OUT/('FROZEN_'+old.name))
text=old.read_text(encoding='utf-8').replace('INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R2.json','INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R3.json')
text=text.replace("newly=set(np.flatnonzero(events[0]['served']))-set(np.flatnonzero(initial['served']));ck('original_stored_physical_service52',newly=={52} and any(s.node==52 for t in events[0]['traces'] for s in t.services))", "previous=decode(read(SOURCE));before_previous=previous['after_commit'];newly=set(np.flatnonzero(initial['served']))-set(np.flatnonzero(before_previous['served']));ck('original_prior_physical_service52',newly=={52} and 52 in set(np.flatnonzero(events[0]['served'])) and any(s.node==52 for t in initial['traces'] for s in t.services))\n    ck('R2_only_service_provenance_failure_preserved',read(OUT/'INDEPENDENT_PRIVATE_PUBLIC_SUFFIX_RECOVERY_R2.json')['failure_names']==['original_stored_physical_service52'])")
text=text.replace("'only_just_served52_removed'","'only_already_served52_removed'")
text=text.replace("qualified_composition='original1215 successful checks plus this narrow private/public suffix evidence'", "qualified_composition='original1215 successful checks plus this narrow private/public suffix evidence; original R2 service-timing assumption failure preserved'")
with (T/'audit_pickup_resident_projection_recovery_r3.py').open('x',encoding='utf-8') as f:f.write(text)
