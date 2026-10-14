from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
p=(ROOT/'tools/verify_pickup_inactive_slot_cancel_archive_20261012.py').read_text(encoding='utf-8')
p=p.replace('inactive_slot_cancel','prepared_slot_cancel')
p=p.replace("'SOURCE_COMPUTE_7.json','SOURCE_END_7.json','SOURCE_COMPUTE_8.json','SOURCE_END_8.json','SOURCE_AT_PARTIAL_WRITE.json'","'SOURCE_COMPUTE_8.json','SOURCE_END_8.json','SOURCE_COMPUTE_9.json','SOURCE_END_9.json','SOURCE_AT_PREPARED_WRITE.json'")
p=p.replace("state['pickup']['caller_selected_epoch']==7","state['pickup']['caller_selected_epoch']==8").replace("state['pickup']['discarded_pending_order']==27","state['pickup']['discarded_pending_order']==36").replace('actual selected slot1 epoch7/callback24','actual selected slot1 epoch8/callback25')
p=p.replace("'不要重复本19/2309' in a['prompt'] and '8.0217174秒' in a['prompt'] and 'callback25/order27' in a['prompt']","'不要重复本19/2309完整未选组件' in a['prompt'] and '8.0248976秒' in a['prompt'] and 'callback26/order36' in a['prompt']")
needle="names={'qualify_pickup_prepared_slot_cancel_v1.py'"
pos=p.index(needle)
p=p[:pos]+"ck('automation_receipt',read(R/'AUTOMATION_PREPARED_SLOT_CANCEL_UPDATE_20261012.json')['automation_id']=='maskco')\n"+p[pos:]
ast.parse(p)
with (ROOT/'tools/verify_pickup_prepared_slot_cancel_archive_20261012.py').open('x',encoding='utf-8',newline='\n') as f:f.write(p)
print('new read-only verifier saved')
