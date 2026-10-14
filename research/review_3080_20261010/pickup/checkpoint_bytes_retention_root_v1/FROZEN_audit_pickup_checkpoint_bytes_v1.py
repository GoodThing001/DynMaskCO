"""Audit actual archived shared bytes, never repickle a JSON tree."""
import ctypes,hashlib,json,os,pickle,struct,sys
from pathlib import Path
from dataclasses import is_dataclass,fields
import numpy as np
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
from pickup_loaded_closure_v1 import capture,sha
OUT=old.P/'checkpoint_bytes_retention_root_v1'
INPUT=old.P/'checkpoint_ack_root_v1/PHYSICAL_ATTEMPT_0.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def encode(v):
    if is_dataclass(v):return {'__dataclass__':type(v).__module__+'.'+type(v).__name__,'fields':{f.name:encode(getattr(v,f.name)) for f in fields(v)}}
    if isinstance(v,(np.ndarray,np.generic)):return {'__array__':str(v.dtype),'shape':list(v.shape),'values':encode(v.tolist())}
    if isinstance(v,float) and not np.isfinite(v):return {'__float__':repr(v)}
    if isinstance(v,dict):return {'__mapping__':[[encode(k),encode(x)] for k,x in v.items()]}
    if isinstance(v,(tuple,list,set,frozenset)):return {'__sequence__':type(v).__name__,'values':[encode(x) for x in (sorted(v) if isinstance(v,(set,frozenset)) else v)]}
    return v
def digest(v):return hashlib.sha256(json.dumps(encode(v),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);assert np.__version__=='1.26.4'
    dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'))
    list(psutil.process_iter(['name','cmdline']));capture()
    _,decode=old.codecs();expected=decode(read(INPUT))['checkpoint'];digest(expected)
    # Warm deserializer's actual types before its closure. No state execution.
    pickle.loads((OUT/'ACTUAL_SHARED_PAYLOAD.bin').read_bytes());capture()
    identity={str(p):sha(p) for p in list(OUT.iterdir())+[INPUT,Path(__file__)] if p.is_file()}
    save('AUDIT_REGISTRATION.json',dict(inputs=identity,scope='actual raw writer/shared envelope/payload identity, exact value fields and independent header validation; no serialization reconstruction/physical/admission/solver'))
    begin=capture(identity);save('AUDIT_SOURCE_START.json',begin);checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    h=struct.Struct('<QQ32s32s32s'); envelope=(OUT/'ACTUAL_SHARED_ENVELOPE.bin').read_bytes();epoch,size,base,ph,sh=h.unpack_from(envelope);payload=envelope[h.size:]
    ck('raw_archive_exact_length',size==len(payload) and len(envelope)==h.size+size)
    ck('writer_shared_payload_actual_bytes',payload==(OUT/'ACTUAL_SHARED_PAYLOAD.bin').read_bytes()==(OUT/'ACTUAL_WRITER_PAYLOAD.bin').read_bytes())
    ck('raw_payload_SHA',hashlib.sha256(payload).digest()==ph)
    actual=pickle.loads(payload)
    def exact(n,a,b):
        if isinstance(a,(np.ndarray,np.generic)):
            ck(n+':bits',type(a) is type(b) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
        elif is_dataclass(a):
            ck(n+':class',type(a) is type(b))
            for f in fields(a):exact(n+'.'+f.name,getattr(a,f.name),getattr(b,f.name))
        elif isinstance(a,dict):
            ck(n+':keys',list(a)==list(b))
            for k,v in a.items():exact(n+'.'+str(k),v,b[k])
        elif isinstance(a,(list,tuple)):
            ck(n+':sequence',type(a) is type(b) and len(a)==len(b))
            for i,(x,y) in enumerate(zip(a,b)):exact(n+'.'+str(i),x,y)
        elif isinstance(a,float):ck(n+':hex',type(a) is type(b) and a.hex()==b.hex())
        else:ck(n+':value',type(a) is type(b) and a==b)
    exact('private_checkpoint',expected,actual)
    ck('independent_state_digest',digest(actual['state'])==sh.hex())
    ck('expected_epoch_and_base',epoch==actual['epoch']==expected['epoch'] and base.hex()==actual['base_digest']==expected['base_digest'])
    # Independent in-memory negative boundaries, not old runtime fault replays.
    def valid(raw):
        e,s,b,p,t=h.unpack_from(raw);body=raw[h.size:]
        return e==expected['epoch'] and s==len(body) and s<=2*1024*1024 and b.hex()==expected['base_digest'] and hashlib.sha256(body).digest()==p
    damaged=bytearray(envelope);damaged[-1]^=1;ck('byte_corruption_rejected',not valid(damaged))
    wrong=bytearray(envelope);struct.pack_into('<Q',wrong,0,epoch+1);ck('wrong_epoch_rejected',not valid(wrong))
    ck('normal_envelope_accepted',valid(envelope))
    for p in psutil.process_iter(['name','cmdline']):
        if (p.info['name'] or '').lower()=='python.exe' and any(Path(x).name=='qualify_pickup_checkpoint_bytes_v1.py' for x in p.info['cmdline'] or []):ck('owned_writer_gone:'+str(p.pid),False)
    end=capture(identity);save('AUDIT_SOURCE_END.json',end)
    for p,v in begin['files'].items():ck('audit_source:'+p,end['files'].get(p)==v==sha(p))
    ck('audit_loaded_closure',begin['files']==end['files'] and begin['module_files']==end['module_files'])
    s=read(OUT/'SOURCE_START.json');e=read(OUT/'SOURCE_END.json')
    for p,v in s['files'].items():ck('writer_source:'+p,e['files'].get(p)==v==sha(p))
    ck('writer_module_and_file_closure',s['files']==e['files'] and s['module_files']==e['module_files'])
    ck('all_input_bytes_unchanged',all(sha(p)==v for p,v in identity.items()))
    r=read(OUT/'RESULT.json');ck('timed_archive_and_reap',r['reaped'] and r['finished']-r['started']==r['wall_s'] and r['wall_s']<10.)
    fail=[c['name'] for c in checks if not c['pass_']]
    save('INDEPENDENT_BYTES.json',dict(checks=checks,checks_count=len(checks),failed=len(fail),failure_names=fail,new_payload_bytes_qualified=not fail,original_v1_payloads_recovered=False,original_v1_FAIL_unchanged=True,full_live_request_qualified=False,physical_ack_loss_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False))
    save('AUDIT_RUN_EXIT.json',dict(exit_code=int(bool(fail))));print(json.dumps(dict(checks=len(checks),failed=len(fail),failure_names=fail,audit_files=len(begin['files']),writer_files=len(s['files']))));return int(bool(fail))
if __name__=='__main__':raise SystemExit(main())
