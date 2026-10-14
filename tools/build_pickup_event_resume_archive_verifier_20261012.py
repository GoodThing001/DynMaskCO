"""New read-only identities verifier; no experiment or prior QA re-execution."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
p=(ROOT/'tools/verify_pickup_selector_event_loss_archive_20261012.py').read_text(encoding='utf-8').replace('selector_event_loss','event_loss_resume').replace('SELECTOR_EVENT_LOSS','EVENT_LOSS_RESUME')
p=p.replace("('SOURCE_COMPUTE_9.json','SOURCE_END_9.json','SOURCE_COMPUTE_10.json','SOURCE_END_10.json','SOURCE_AT_SELECTOR_PUBLICATION.json')","('DUPLICATE_SOURCE_END.json','SOURCE_COMPUTE_11.json','SOURCE_END_11.json','SOURCE_COMPUTE_12.json','SOURCE_END_12.json')")
p=p.replace("==2314 and state['pickup']['caller_selected_epoch']==10","==1807 and state['pickup']['caller_selected_epoch']==12").replace("caller_selected_order']==5","caller_selected_order']==76").replace("next_resume_callback']==28","next_resume_callback']==30").replace('actual selected slot0 epoch10/callback27','actual selected slot0 epoch12/callback29')
p=p.replace("'不要重复本22/2314' in a['prompt'] and '8.0261676秒' in a['prompt'] and 'callback27/order5' in a['prompt']","'不要重复本19/1807' in a['prompt'] and '.020752300042659044秒' in a['prompt'] and 'callback29/order76' in a['prompt']")
p=p.replace("ck('killed_worker_not_natural0',receipt['worker_was_terminated'] and receipt['worker_actual_return_code']==read(O/'RESULT.json')['actual_worker_exit'] and receipt['worker_natural_exit_0'] is False)","ck('normal_worker_actual_natural0',not receipt['worker_was_terminated'] and receipt['worker_actual_return_code']==read(O/'RESULT.json')['actual_worker_exit']==0 and receipt['worker_natural_exit_0'] is True)")
p=p.replace("out=dict(created_utc=", "prior=R/'pickup/selector_event_loss_root_v1';ck('source_actual_selected10_receipt',sha(prior/'ACTUAL_SHARED_PAYLOAD_10.bin')==read(prior/'HOST_10.json')['ticket']['payload_sha256']=='a4b0c063e7b6959654429bc969f7072fa1de466898ec60be4c194c8283449dbe')\nfor f in O.iterdir():\n    if f.is_file():ck('new_artifact_readable:'+f.name,f.stat().st_size>0)\nout=dict(created_utc=")
ast.parse(p)
with (ROOT/'tools/verify_pickup_event_loss_resume_archive_20261012.py').open('x',encoding='utf-8',newline='\n') as f:f.write(p)
print('new final verifier prepared')
