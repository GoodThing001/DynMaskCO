"""Seal own output only; recover QA runtime source versions by matching hashes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def old_resident_bytes():
    b=(HERE/'resident_actor.py').read_bytes()
    b=b.replace(b' or started>time.perf_counter()',b'')
    b=b.replace(b'finite elapsed monotonic start',b'finite monotonic start')
    b=b.replace(b'            checkpoint_end=sha(a.checkpoint)\n',b'')
    b=b.replace(b' and checkpoint_end==checkpoint',b'')
    b=b.replace(b'checkpoint_sha256=checkpoint,checkpoint_end_sha256=checkpoint_end,',b'checkpoint_sha256=checkpoint,')
    b=b.replace(b"                  and result.get('checkpoint_end_sha256')==self.ready['checkpoint_sha256']\n",b'')
    return b

def seal():
    # Preserve exact runtime files corresponding to each original registration.
    for directory,field in [('wiring_qa_v1','source_start'),('resident_qa_v1','source_sha256')]:
        reg=json.loads((HERE/directory/'REGISTRATION.json').read_text(encoding='utf-8'))
        snapshot=HERE/directory/'source_snapshot';snapshot.mkdir(exist_ok=True)
        rows=[]
        for original,want in reg[field].items():
            p=Path(original)
            if not p.is_relative_to(HERE):continue
            data=old_resident_bytes() if p.name=='resident_actor.py' else p.read_bytes()
            if hashlib.sha256(data).hexdigest()!=want:raise ValueError('runtime source bytes not recovered: '+p.name)
            destination=snapshot/p.name
            if destination.exists() and destination.read_bytes()!=data:raise ValueError('never overwrite sealed runtime source')
            destination.write_bytes(data);rows.append(dict(original=original,sealed=str(destination.relative_to(HERE)),sha256=want))
        (snapshot/'MAPPING.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    inspection=Path('D:/PyCharm_/MASKCO-Main/research/review_3080_20261010/delivery/remote_actual_bundle_v2/downloaded/ACTUAL_MODEL_INSPECTION.json')
    original=json.loads(inspection.read_text(encoding='utf-8'))
    evidence=dict(actual_inspection_path=str(inspection),actual_inspection_sha256=sha(inspection),checkpoint_sha256=original['checkpoint_sha256'],
                  checkpoint_step=original['checkpoint_step'],model_config=original['model_config'],parameter_leaves=original['parameter_leaves'],
                  adapter_declared_dtype='float32',softcap_nonnull_incompatibility_applies_here=False,
                  original_path_same_input_numerical_verification_completed=False)
    (HERE/'ACTUAL_CONFIGURATION_EVIDENCE.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    manifest={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='DELIVERABLE_MANIFEST.json'}
    (HERE/'DELIVERABLE_MANIFEST.json').write_text(json.dumps(dict(sealed_at=datetime.now(timezone.utc).isoformat(),files=manifest,
        engineering_only=True,formal_online_gain_established=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(manifest),all_runtime_source_snapshots_match=True)))
if __name__=='__main__':seal()
