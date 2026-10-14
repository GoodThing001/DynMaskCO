import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; sys.path.insert(0,str(ROOT/'tools'))
import json,subprocess,time
from engineering_stage_closure_v1 import initialize,closure,same,identity,sha,save
from lease_runtime import acquire,info,release,owned

def probe(mode):
    p=subprocess.Popen([sys.executable,str(HERE/'probe.py'),mode],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    try: out,err=p.communicate(timeout=10)
    except subprocess.TimeoutExpired: p.kill(); out,err=p.communicate(timeout=3); raise TimeoutError('Owned probe expired')
    (HERE/(mode+'.stdout')).open('xb').write(out); (HERE/(mode+'.stderr')).open('xb').write(err)
    if p.returncode!=0: raise RuntimeError('Probe failed: '+err.decode('utf-8',errors='replace'))
    v=json.loads(out); v['actual_returncode']=p.returncode; v['reaped']=p.poll() is not None
    save(HERE/(mode+'_RESULT.json'),v); return v

def main():
    initialize(); reg=json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8'))
    for path,h in reg['source_files'].items(): assert sha(path)==h
    base=HERE/'owned'; base.mkdir(exist_ok=False); (HERE/'original_bytes').mkdir(exist_ok=False)
    copies=[]
    for i,(src,h) in enumerate(reg['source_files'].items()):
        raw=Path(src).read_bytes(); assert sha(src)==h
        q=owned(base/('copy_%d'%i+Path(src).suffix),base)
        q.open('xb').write(raw); (HERE/'original_bytes'/('input_%d.dat'%i)).open('xb').write(raw); copies.append(q)
    start=closure([Path(__file__)]); save(HERE/'SOURCE_START.json',start); t0=time.perf_counter()
    handles=[]; owner=None
    try:
        handles=[acquire(q,base) for q in copies]
        before=[{'path':str(q),'sha256':sha(q),'info':info(h)} for q,h in zip(copies,handles)]
        blocked=probe('leased')
        after=[{'path':str(q),'sha256':sha(q),'info':info(h)} for q,h in zip(copies,handles)]
        additions=sorted(q.name for q in base.iterdir() if q.is_file() and q not in copies)
        directory_manifest_accepts=not additions
        for h in handles: release(h)
        handles=[]
        released=probe('released')
        owner=subprocess.Popen([sys.executable,str(HERE/'owner.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
        cap=time.perf_counter()+5
        while not (HERE/'OWNER_READY.json').exists():
            if owner.poll() is not None or time.perf_counter()>cap: raise RuntimeError('Owner not ready')
            time.sleep(.02)
        held=probe('owner_held'); (HERE/'OWNER_EXIT_SIGNAL').open('x').write('exit')
        ostdout,ostderr=owner.communicate(timeout=5)
        (HERE/'owner.stdout').open('xb').write(ostdout); (HERE/'owner.stderr').open('xb').write(ostderr)
        owner_receipt={'returncode':owner.returncode,'reaped':owner.poll() is not None,'deliberate_abnormal_exit':True,'python_CloseHandle_not_called':True}
        save(HERE/'OWNER_EXIT.json',owner_receipt)
        exited=probe('owner_exited')
        checks=[('copied_exactly_before_and_after_lease',before==after),('six_mutations_blocked_by_sharing',all(not x['success'] and x['winerror']==32 for x in blocked['rows'][:6])),('directory_addition_really_allowed',blocked['rows'][6]['success'] and additions==['unleased_extra.py']),('directory_manifest_rejects',not directory_manifest_accepts),('released_write_succeeds',released['rows'][0]['success']),('separate_owner_blocks_write',not held['rows'][0]['success'] and held['rows'][0]['winerror']==32),('abnormal_owner_exit17',owner.returncode==17),('OS_teardown_releases_handle',exited['rows'][0]['success']),('original_source_unchanged',all(sha(p)==h for p,h in reg['source_files'].items()))]
        end=closure([Path(__file__)]); save(HERE/'SOURCE_END.json',end); checks.append(('parent_source_closed',same(start,end)))
        result={'checks':[{'name':n,'pass':v} for n,v in checks],'checks_count':len(checks),'failed':sum(not v for _,v in checks),'before':before,'after':after,'directory_additions':additions,'directory_manifest_accepts':directory_manifest_accepts,'owner_exit':owner_receipt,'elapsed_component':time.perf_counter()-t0,'environment':'local Python3.12.7 CPU7 NTFS file copies only','full_runtime_guard_qualified':False,'full_request_budget_qualified':False,'stable_online_gain':False,'parent_identity':identity()}
        save(HERE/'RESULT.json',result); print(json.dumps({'checks':len(checks),'failed':result['failed']}))
        return int(result['failed']>0)
    finally:
        for h in handles: release(h)
        if owner is not None and owner.poll() is None:
            owner.kill(); owner.wait(timeout=3)

if __name__=='__main__': raise SystemExit(main())
