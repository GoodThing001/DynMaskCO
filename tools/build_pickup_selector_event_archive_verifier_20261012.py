from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
p=(ROOT/'tools/verify_pickup_prepared_slot_cancel_archive_20261012.py').read_text(encoding='utf-8')
p=p.replace('prepared_slot_cancel','selector_event_loss').replace('AUTOMATION_PREPARED_SLOT_CANCEL_UPDATE','AUTOMATION_SELECTOR_EVENT_LOSS_UPDATE')
p=p.replace("'SOURCE_COMPUTE_8.json','SOURCE_END_8.json','SOURCE_COMPUTE_9.json','SOURCE_END_9.json','SOURCE_AT_PREPARED_WRITE.json'","'SOURCE_COMPUTE_9.json','SOURCE_END_9.json','SOURCE_COMPUTE_10.json','SOURCE_END_10.json','SOURCE_AT_SELECTOR_PUBLICATION.json'")
p=p.replace("state['pickup']['selector_event_loss_independent_checks']==2309","state['pickup']['selector_event_loss_independent_checks']==2314").replace("state['pickup']['caller_selected_epoch']==8","state['pickup']['caller_selected_epoch']==10").replace("state['pickup']['discarded_pending_order']==36","state['pickup']['discarded_pending_order'] is None and state['pickup']['caller_selected_order']==5 and state['pickup']['next_resume_callback']==28")
p=p.replace('actual selected slot1 epoch8/callback25','actual selected slot0 epoch10/callback27')
p=p.replace("'不要重复本19/2309完整未选组件' in a['prompt'] and '8.0248976秒' in a['prompt'] and 'callback26/order36' in a['prompt']","'不要重复本22/2314' in a['prompt'] and '8.0261676秒' in a['prompt'] and 'callback27/order5' in a['prompt']")
ast.parse(p)
with (ROOT/'tools/verify_pickup_selector_event_loss_archive_20261012.py').open('x',encoding='utf-8',newline='\n') as f:f.write(p)
print('new read-only verifier saved')
