import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; sys.path.insert(0,str(ROOT/'tools'))
import json,ast
from engineering_stage_closure_v1 import initialize,closure,same,sha,save

def main():
    initialize(); reg=json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8')); result=json.loads((HERE/'RESULT.json').read_text(encoding='utf-8'))
    start=closure([Path(__file__)]); save(HERE/'AUDIT_SOURCE_START.json',start); checks=[]
    def ck(n,v): checks.append({'name':n,'pass':bool(v)})
    for i,(p,h) in enumerate(reg['source_files'].items()):
        ck('original:'+str(i),sha(p)==h)
        ck('archived_original:'+str(i),sha(HERE/'original_bytes'/('input_%d.dat'%i))==h)
        ck('copied_before:'+str(i),result['before'][i]['sha256']==h)
        ck('held_byte_and_fileid:'+str(i),result['before'][i]==result['after'][i] and result['before'][i]['info']['links']==1)
        ck('private_target:'+str(i),Path(result['before'][i]['path']).resolve().is_relative_to((HERE/'owned').resolve()))
    for mode in ('leased','released','owner_held','owner_exited'):
        p=json.loads((HERE/(mode+'_RESULT.json')).read_text(encoding='utf-8'))
        ck('actual_exit:'+mode,p['actual_returncode']==0 and p['reaped'])
        ck('CPU7:'+mode,p['identity']['cpu_affinity']==[7])
        ck('child_closed:'+mode,p['source_closed'])
        if mode=='leased':
            for row in p['rows'][:6]: ck('sharing_rejection:'+row['name'],not row['success'] and row['winerror']==32)
            ck('new_file_not_protected',p['rows'][6]['success'])
        else: ck('write_boundary:'+mode,p['rows'][0]['success']==(mode!='owner_held'))
        for src,h in p['source_start']['files'].items(): ck('child_input:'+mode+':'+src,sha(src)==h)
    ck('new_file_real',(HERE/'owned/unleased_extra.py').read_bytes()==b'UNREGISTERED=1\n')
    ck('manifest_gap_explicit',result['directory_manifest_accepts'] is False and result['directory_additions']==['unleased_extra.py'])
    owner=json.loads((HERE/'OWNER_READY.json').read_text(encoding='utf-8')); exit=json.loads((HERE/'OWNER_EXIT.json').read_text(encoding='utf-8'))
    ck('exception_owner_closed_only_at_ready',same(owner['source_start'],owner['source_at_ready']))
    ck('exception_exit17',exit['returncode']==17 and exit['reaped'])
    ck('source_calls_actual_exit',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_exit' for n in ast.walk(ast.parse((HERE/'owner.py').read_text(encoding='utf-8')))))
    ck('owner_no_fake_source_end',not (HERE/'OWNER_SOURCE_END.json').exists())
    prod_start=json.loads((HERE/'SOURCE_START.json').read_text(encoding='utf-8')); prod_end=json.loads((HERE/'SOURCE_END.json').read_text(encoding='utf-8')); ck('parent_closed',same(prod_start,prod_end))
    for src,h in prod_start['files'].items(): ck('parent_input:'+src,sha(src)==h)
    end=closure([Path(__file__)]); save(HERE/'AUDIT_SOURCE_END.json',end); ck('audit_closed',same(start,end))
    ck('no_full_claim',result['full_runtime_guard_qualified'] is False and result['full_request_budget_qualified'] is False)
    save(HERE/'INDEPENDENT_AUDIT.json',{'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'checks':checks,'scope':'independent saved bytes/file-ID/actual cross-process receipts, original OS implementation shared; lease probes not rerun'})
    print(json.dumps({'checks':len(checks),'failed':sum(not x['pass'] for x in checks)})); return int(any(not x['pass'] for x in checks))

if __name__=='__main__': raise SystemExit(main())
