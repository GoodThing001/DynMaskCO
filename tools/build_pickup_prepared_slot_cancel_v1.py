"""Prospective next two events: valid inactive envelope is not a selected commit."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]

def create(n,t):
    ast.parse(t)
    with (ROOT/'tools'/n).open('x',encoding='utf-8',newline='\n') as f:f.write(t)

def advance(t):
    t=t.replace('inactive_slot_cancel','prepared_slot_cancel').replace('INACTIVE_SLOT_CANCEL','PREPARED_SLOT_CANCEL')
    t=t.replace("live_resume_saa_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_6.bin","inactive_slot_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_7.bin")
    t=t.replace("timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin","live_resume_saa_cancel_root_v1/ACTUAL_SHARED_PAYLOAD_6.bin")
    for a,b in [('OWNER_STATE_8','OWNER_STATE_9'),('ACTUAL_WRITER_PAYLOAD_8','ACTUAL_WRITER_PAYLOAD_9'),('ACTUAL_WRITER_PAYLOAD_7','ACTUAL_WRITER_PAYLOAD_8'),('ACTUAL_SHARED_ENVELOPE_7','ACTUAL_SHARED_ENVELOPE_8'),('ACTUAL_SHARED_PAYLOAD_7','ACTUAL_SHARED_PAYLOAD_8')]:
        # SOURCE path is excluded from filename replacements.
        t=t.replace("'"+a,"'"+b)
    return t

def main():
    p=advance((ROOT/'tools/qualify_pickup_inactive_slot_cancel_v1.py').read_text(encoding='utf-8'))
    # Numeric changes are non-cascading explicit matches.
    p=p.replace('seq!=len(records)+7 or seq>8','seq!=len(records)+8 or seq>9')
    p=p.replace('if seq==8:break','if seq==9:break').replace('if seq==7:','if seq==8:')
    p=p.replace('epoch!=6 or not 0<size<=MAX','epoch!=7 or not 0<size<=MAX').replace('selected actual epoch6 bound','selected actual epoch7 bound')
    p=p.replace("cp['epoch']==6","cp['epoch']==7").replace('pack_into(blob.buf,0,6,len(payload)','pack_into(blob.buf,0,7,len(payload)')
    p=p.replace('for seq in (7,8):','for seq in (8,9):').replace('4 if seq==8 else 0','5 if seq==9 else 0').replace('final_reap_included=seq==8','final_reap_included=seq==9')
    start=p.index('            if fault==4:');end=p.index("            if time.perf_counter()>=deadline or struct.unpack_from('<I',ctrl.buf,44)[0]:raise ValueError('before selector deadline fence')",start)
    p=p[:start]+WORKER_PREPARED+p[end:]
    p=p.replace('stage!=4','stage!=5').replace("fixed inactive partial-write stage absent / forbidden publish","fixed complete inactive-envelope stage absent / forbidden select")
    start=p.index(";w=read(OUT/'PARTIAL_WRITE_WITNESS.json')");end=p.index('\n            f=time.perf_counter()',start)
    p=p[:start]+";w=read(OUT/'PREPARED_WRITE_WITNESS.json');off=w['inactive_slot']*SLOT;raw_save('ACTUAL_INACTIVE_ENVELOPE.bin',bytes(blob.buf[off:off+HEADER.size+w['total_payload_bytes']]));raw_save('ACTUAL_INACTIVE_PAYLOAD.bin',bytes(blob.buf[off+HEADER.size:off+HEADER.size+w['total_payload_bytes']]))"+p[end:]
    p=p.replace("w=read(OUT/'PARTIAL_WRITE_WITNESS.json')","w=read(OUT/'PREPARED_WRITE_WITNESS.json')")
    p=p.replace("timed_selected6_restore","timed_selected7_restore").replace("normal_order58_next25_same_resident","normal_order27_next26_same_resident").replace("==[24,25] and actual['records'][0]['order']==58","==[25,26] and actual['records'][0]['order']==27")
    start=p.index("        ck('actual_partial_inactive_prefix'");end=p.index("        ck('active7_byte_exact_retained'",start)
    p=p[:start]+CHECK_PREPARED+p[end:]
    p=p.replace('active7_byte_exact_retained','active8_byte_exact_retained').replace('no_late_epoch8_select_ack','no_late_epoch9_select_ack').replace("host[1]['stage']==4","host[1]['stage']==5").replace("SOURCE_AT_PARTIAL_WRITE.json","SOURCE_AT_PREPARED_WRITE.json")
    start=p.index("scope='New engineering only:");end=p.index(",K=10",start)
    p=p[:start]+"scope='New engineering only: restore actual selected epoch7/callback24 raw bytes inside first request, process callback25/order27 for first caller-authoritative commit epoch8. Same resident next callback26 fully computes and privately installs then writes complete valid raw payload and header to inactive slot0, waits12s before selector/event/ACK; cancel/reap, selected slot1 epoch8 must remain exact. A complete inactive candidate is not a caller-selected commit. No old half-write QA, no models/science/new day, no arbitrary memory ordering/selector tearing/persistence/external effects qualification.'"+p[end:]
    p=p.replace("partial_payload_fraction='floor(len/2)'","fault_boundary='complete_inactive_payload_header_before_selector'")
    create('qualify_pickup_prepared_slot_cancel_v1.py',p)
    a=advance((ROOT/'tools/audit_pickup_inactive_slot_cancel_v1.py').read_text(encoding='utf-8'))
    a=a.replace('index+7','index+8').replace('24+index','25+index').replace('selected_epoch6_actual_start','selected_epoch7_actual_start').replace("restore['source_epoch']==6","restore['source_epoch']==7").replace('epoch==7 and bh.hex()','epoch==8 and bh.hex()').replace("h['stage']==4","h['stage']==5").replace('inactive_partial_cancel_keeps_selected','complete_inactive_candidate_not_selected')
    start=a.index("    witness=read(OUT/'PARTIAL_WRITE_WITNESS.json')");end=a.index("    ck('all_actual_audit_inputs_unchanged'",start)
    a=a[:start]+AUDIT_PREPARED+a[end:]
    create('audit_pickup_prepared_slot_cancel_v1.py',a)
    print('created prospective producer/audit; no execution')

WORKER_PREPARED="""            blob.buf[offset+HEADER.size:offset+HEADER.size+len(payload)]=payload;HEADER.pack_into(blob.buf,offset,seq,len(payload),bytes.fromhex(before),hashlib.sha256(payload).digest(),bytes.fromhex(after))
            if fault==5:
                raw_save('PREPARED_INACTIVE_ENVELOPE.bin',bytes(blob.buf[offset:offset+HEADER.size+len(payload)]))
                save('PREPARED_WRITE_WITNESS.json',dict(sequence=seq,selected_slot=active,inactive_slot=inactive,total_payload_bytes=len(payload),entered=time.perf_counter(),selector_changed=False,new_header_written=True,complete_payload_written=True,checkpoint_event_sent=False))
                at=capture(sources());save('SOURCE_AT_PREPARED_WRITE.json',at);struct.pack_into('<I',ctrl.buf,40,5);time.sleep(12.)
                raise TimeoutError('complete inactive candidate cancelled before selector/event')
"""
CHECK_PREPARED="""        ck('actual_complete_inactive_candidate',(OUT/'ACTUAL_INACTIVE_ENVELOPE.bin').read_bytes()==(OUT/'PREPARED_INACTIVE_ENVELOPE.bin').read_bytes() and (OUT/'ACTUAL_INACTIVE_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_9.bin').read_bytes() and w['selected_slot']==1 and w['inactive_slot']==0)
        epoch,size,bh,ph,sh=HEADER.unpack_from((OUT/'ACTUAL_INACTIVE_ENVELOPE.bin').read_bytes())
        ck('inactive_new9_header_exact',epoch==9 and size==w['total_payload_bytes'] and ph.hex()==sha(OUT/'ACTUAL_WRITER_PAYLOAD_9.bin') and bh.hex()==actual['records'][1]['resident_before'] and sh.hex()==actual['records'][1]['resident_after'] and w['new_header_written'] and w['complete_payload_written'] and not w['selector_changed'] and not w['checkpoint_event_sent'])
"""
AUDIT_PREPARED="""    witness=read(OUT/'PREPARED_WRITE_WITNESS.json');h=raw['host'][1];head=struct.Struct('<QQ32s32s32s')
    envelope=(OUT/'ACTUAL_INACTIVE_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];candidate=pickle.loads(payload)
    ck('prepared:complete_actual_envelope',envelope==(OUT/'PREPARED_INACTIVE_ENVELOPE.bin').read_bytes() and payload==(OUT/'ACTUAL_INACTIVE_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD_9.bin').read_bytes() and len(payload)==size==witness['total_payload_bytes'])
    ck('prepared:new9_header_SHA_lineage',epoch==candidate['epoch']==9 and hashlib.sha256(payload).digest()==ph and bh.hex()==candidate['base_digest']==actual['records'][1]['resident_before'] and sh.hex()==old.pf.digest(candidate['state'])==actual['records'][1]['resident_after'])
    for k,v in candidate['state'].items():ck('prepared:private_candidate_field:'+k,old.pf.digest(v)==old.pf.digest(actual['records'][1]['private_after'][k]))
    ck('prepared:inactive_slot0_not_selection',witness['inactive_slot']==0 and witness['selected_slot']==1 and witness['new_header_written'] and witness['complete_payload_written'] and not witness['selector_changed'] and not witness['checkpoint_event_sent'])
    ck('prepared:after_private_commit_before_cutoff',actual['records'][1]['resident_finished']<=witness['entered']<h['cutoff']<=h['finished'] and h['wall_s']<10.)
    envelope=(OUT/'RETAINED_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];c=pickle.loads(payload)
    ck('prepared:authority_is_actual8_not_candidate9',epoch==c['epoch']==8 and sh.hex()==old.pf.digest(c['state'])==actual['records'][0]['resident_after'] and sh.hex()!=actual['records'][1]['resident_after'] and payload==(OUT/'RETAINED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_SHARED_PAYLOAD_8.bin').read_bytes() and hashlib.sha256(payload).digest()==ph and len(payload)==size)
    bs=read(OUT/'SOURCE_START.json');fe=read(OUT/'SOURCE_AT_PREPARED_WRITE.json');ck('prepared:start_to_actual_fault_files_maps',bs['files']==fe['files'] and bs['module_files']==fe['module_files'])
    for p,h in fe['files'].items():ck('prepared:loaded:'+p,sha(p)==h)
"""
if __name__=='__main__':main()
