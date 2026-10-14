"""Preserve original closure FAIL; prospectively same tensor contract with IO warmed."""
from pathlib import Path
from datetime import datetime,timezone
import json,ast,hashlib,shutil,psutil,os
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';old=D/'action_tensor_binding_root_v1';new=D/'action_tensor_binding_root_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((old/'RESULT.json').read_text());assert [c['name'] for c in r['checks'] if not c['passed']]==['loaded_identity']
assert not [p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(x).name=='qualify_delivery_action_tensor_binding_v1.py' for x in p.info['cmdline'] or [])]
b=json.loads((old/'SOURCE_START.json').read_text());e=json.loads((old/'SOURCE_END.json').read_text());assert all(e['files'].get(p)==h for p,h in b['files'].items());extra={p:h for p,h in e['files'].items() if p not in b['files']};assert len(extra)==6 and all('/zipfile/' in p.replace('\\','/') for p in extra)
for n in ('qualify_delivery_action_tensor_binding_v1.py','audit_delivery_action_tensor_binding_v1.py','delivery_action_tensor_binding_v1.py'):
    f=old/('FROZEN_'+n);assert not f.exists();shutil.copy2(ROOT/'tools'/n,f)
for kind in ('qualify','audit'):
    src=ROOT/'tools'/f'{kind}_delivery_action_tensor_binding_v1.py';p=src.read_text(encoding='utf-8').replace('action_tensor_binding_root_v1','action_tensor_binding_root_v2').replace("tools/audit_delivery_action_tensor_binding_v1.py","tools/audit_delivery_action_tensor_binding_v2.py")
    # Actual sample NPZ encode/decode before begin closes zipfile and array-header lazy dependencies.
    p=p.replace('import numpy as np,psutil','import numpy as np,psutil,io,zipfile')
    p=p.replace("list(psutil.process_iter(['name']));capture();", "list(psutil.process_iter(['name']));warm=io.BytesIO();np.savez(warm,sample=np.zeros((1,3,9),dtype=np.float32));warm.seek(0);sample=np.load(warm,allow_pickle=False);sample['sample'];sample.close();capture();")
    if kind=='qualify':
        p=p.replace("inputs=[CHAIN,OLD,", "inputs=[ROOT/'tools/recover_delivery_action_tensor_binding_v2.py',old_result,old_begin,old_end,CHAIN,OLD,")
        p=p.replace('def main():',"old_result=D/'action_tensor_binding_root_v1/RESULT.json';old_begin=D/'action_tensor_binding_root_v1/SOURCE_START.json';old_end=D/'action_tensor_binding_root_v1/SOURCE_END.json'\ndef main():")
    ast.parse(p)
    with (ROOT/'tools'/f'{kind}_delivery_action_tensor_binding_v2.py').open('x',encoding='utf-8',newline='\n') as f:f.write(p)
out=dict(created_utc=datetime.now(timezone.utc).isoformat(),original_directory=str(old),next_directory=str(new),original_result_sha256=sha(old/'RESULT.json'),original_registration_sha256=sha(old/'REGISTRATION.json'),original_start_sha256=sha(old/'SOURCE_START.json'),original_end_sha256=sha(old/'SOURCE_END.json'),closure_only_missing_start_files=extra,no_actual_existing_file_changed=True,old_tensor_sha256=sha(old/'TENSORS.npz'),same_inputs_target_dtype_anchor_mask_and_scores=True,binding_module_sha256=sha(ROOT/'tools/delivery_action_tensor_binding_v1.py'),new_producer_sha256=sha(ROOT/'tools/qualify_delivery_action_tensor_binding_v2.py'),new_audit_sha256=sha(ROOT/'tools/audit_delivery_action_tensor_binding_v2.py'),scientific_family_not_started=True)
with (old/'CLOSURE_FAILURE_RECOVERY_REGISTRATION_R2.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print('original closureFAIL preserved; new same-contract programs created before execution')
