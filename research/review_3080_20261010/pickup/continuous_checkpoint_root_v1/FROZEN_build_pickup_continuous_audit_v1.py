"""Create a new audit from original-library reference, no previous audit rerun."""
from pathlib import Path
T=Path(__file__).resolve().parent
s=(T/'audit_pickup_resident_two_events_v1.py').read_text(encoding='utf-8')
s=s.replace('import copy,json,pickle,hashlib','import copy,json,pickle,hashlib,ctypes,os,struct')
s=s.replace('import numpy as np,psutil','import numpy as np,psutil\nfrom pickup_loaded_closure_v1 import capture,sha\nimport hard_admission,rolling_optimizer')
s=s.replace("OUT=P/'resident_two_events_root_v1';SOURCE=P/'endogenous_owner_transaction_root_v2/OWNER_RESULT_0.json'","OUT=P/'continuous_checkpoint_root_v1';SOURCE=P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin'")
s=s.replace("decode(read(OUT/'RESIDENT_RESULT.json'));initial=decode(read(SOURCE))['after_primary_advance'];checks=[]","decode(read(OUT/'OWNER_STATE_2.json'));initial=pickle.loads(SOURCE.read_bytes())['state'];checks=[]")
s=s.replace("    ck('sealed_actual_post15_physical_start'","    ck('sealed_event27_actual_start'")
s=s.replace("row['callback']==16+index","row['callback']==18+index")
s=s.replace("old.pf.digest(private_rp.export_state())==old.pf.digest(row['before'])==old.pf.digest(prior)","old.pf.digest(private_rp.export_state())==old.pf.digest(row['before'])")
# Do not conflate old served private plan prefix with actual public suffix.
insertion="""        suffix={str(v):[int(j) for j in private_rp.export_state()['plan'].get(str(v),private_rp.export_state()['plan'].get(v,[])) if not private_served[j]] for v in range(len(private_fleet))}
        for v in private_fleet:
            if v.status=='committed' and v.committed_next not in (None,0):suffix[str(v.vehicle_id)]=[int(v.committed_next)]+[j for j in suffix[str(v.vehicle_id)] if j!=v.committed_next]
        ck(name+':direct_public_remaining_suffix',old.pf.digest(suffix)==old.pf.digest(prior['plan']))
"""
s=s.replace("        for key,value in packet['dataset'].items():",insertion+"        for key,value in packet['dataset'].items():")
oldtime="        h=raw['host'][index];ck(name+':exact_normal_elapsed'"
start=s.index(oldtime);end=s.index("\n        begin=read",start)
s=s[:start]+"""        h=raw['host'][index];ck(name+':exact_elapsed_and_ticket',h['finished']-h['started']==h['wall_s'] and h['started']<=row['resident_started']<=row['started']<=response['worker_finished']<=row['committed']<=row['resident_finished']<=h['chosen_at']<h['cutoff']<h['started']+10. and h['finished']<h['started']+10. and h['ticket']['digest']==row['resident_after'])
        envelope=(OUT/('ACTUAL_SHARED_ENVELOPE_'+str(index+1)+'.bin')).read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];actual_checkpoint=pickle.loads(payload)
        ck(name+':actual_original_shared_bytes',payload==(OUT/('ACTUAL_SHARED_PAYLOAD_'+str(index+1)+'.bin')).read_bytes()==(OUT/('ACTUAL_WRITER_PAYLOAD_'+str(index+1)+'.bin')).read_bytes() and hashlib.sha256(payload).digest()==ph and len(payload)==size==row['payload_size'])
        ck(name+':actual_checkpoint_lineage',epoch==actual_checkpoint['epoch']==index+1 and bh.hex()==actual_checkpoint['base_digest']==row['resident_before'] and sh.hex()==old.pf.digest(actual_checkpoint['state'])==old.pf.digest(snapshot()) and ph.hex()==row['payload_sha256']==h['ticket']['payload_sha256'])
"""+s[end:]
s=s.replace("names={'qualify_pickup_resident_two_events_v1.py'}","names={'qualify_pickup_continuous_checkpoint_v1.py'}")
s=s.replace("raw['host'][1]['final_reap_included'] and not raw['host'][0]['final_reap_included']","raw['host'][1]['final_reap_included'] and raw['host'][1]['reaped'] and not raw['host'][0]['final_reap_included']")
pos=s.index("    failed=[c['name']")
extra="""    rh=raw['recovery_host'];rec=decode(read(OUT/'RECOVERY.json'));ck('lost_second_ACK_after_selected_commit',raw['host'][0]['ack_present'] and not raw['host'][1]['ack_present'] and raw['host'][1]['stage']==2 and raw['host'][1]['chosen_at']<raw['host'][1]['cutoff'])
    for k,v in snapshot().items():ck('restored_private:'+k,old.pf.digest(v)==old.pf.digest(rec['state'][k]))
    ck('duplicate_committed_second_ticket_zero_execution',rec['physical_calls']==rec['admission_calls']==0 and rec['epoch']==rh['sequence']==2 and rec['digest']==rec['duplicate_digest']==rh['digest']==raw['host'][1]['ticket']['digest'])
    ck('separate_recovery_budget_and_reap',rh['finished']-rh['started']==rh['wall_s'] and rh['started']<=rec['started']<=rec['finished']<=rh['finished']<rh['started']+10. and rh['exit_code']==0 and rh['reaped'])
    rs=read(OUT/'RECOVERY_SOURCE_START.json');re=read(OUT/'RECOVERY_SOURCE_END.json');ck('actual_recovery_loaded_closure',rs['files']==re['files'] and rs['module_files']==re['module_files'] and rec['loaded_closure_qualified'])
    for p,h in re['files'].items():ck('recovery_loaded:'+p,sha(p)==h)
    ck('all_actual_audit_inputs_unchanged',all(sha(p)==h for p,h in identity.items()))
    audit_end=capture(identity)
    for p,h in audit_begin['files'].items():ck('audit_loaded:'+p,audit_end['files'].get(p)==h==sha(p))
    ck('audit_actual_loaded_closure',audit_begin['files']==audit_end['files'] and audit_begin['module_files']==audit_end['module_files'])
    with (OUT/'AUDIT_SOURCE_END.json').open('x',encoding='utf-8') as f:json.dump(audit_end,f,indent=2)
"""
s=s[:pos]+extra+s[pos:]
# Warm audit actual dependencies before its workload, then bind all actual inputs.
pos=s.index("    for index,row in enumerate(actual['records']):")
warm="""    dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']));capture()
    identity={str(p):sha(p) for p in list(OUT.iterdir())+[SOURCE,Path(__file__)] if p.is_file()}
    with (OUT/'AUDIT_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(dict(inputs=identity,scope='two new actual reveals, direct original admission/all40K10/C0/actual private events and actual raw checkpoint lineage; no new worker/engine/projector reference'),f,indent=2)
    audit_begin=capture(identity)
    with (OUT/'AUDIT_SOURCE_START.json').open('x',encoding='utf-8') as f:json.dump(audit_begin,f,indent=2)
"""
s=s[:pos]+warm+s[pos:]
s=s.replace('INDEPENDENT_RESIDENT_TWO_EVENTS.json','INDEPENDENT_CONTINUOUS_CHECKPOINT.json')
with (T/'audit_pickup_continuous_checkpoint_v1.py').open('x',encoding='utf-8') as f:f.write(s)
