import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; sys.path.insert(0,str(ROOT/'tools'))
import json,time
from engineering_stage_closure_v1 import initialize,closure,same,identity,sha,save
from lease_runtime import owned

def main():
    initialize(); mode=sys.argv[1]; base=HERE/'owned'; p=owned(base/'copy_0.py',base); pyc=owned(base/'copy_1.pyc',base)
    start=closure([Path(__file__)]); rows=[]
    def attempt(name,fn):
        t=time.perf_counter()
        try: fn(); row={'name':name,'success':True,'winerror':None}
        except OSError as e: row={'name':name,'success':False,'winerror':getattr(e,'winerror',None),'errno':e.errno}
        row['elapsed']=time.perf_counter()-t; rows.append(row)
    def write(target):
        owned(target,base)
        with target.open('r+b') as f: f.write(b'# changed by owned probe\n')
    if mode=='leased':
        replacement=owned(HERE/'scratch'/'replacement.dat',HERE)
        replacement.parent.mkdir(exist_ok=True); replacement.write_bytes(b'owned replacement')
        attempt('write',lambda:write(p)); attempt('truncate',lambda:os.truncate(p,0))
        attempt('replace',lambda:os.replace(replacement,p)); attempt('rename',lambda:os.rename(p,owned(base/'moved.py',base)))
        attempt('delete',lambda:os.unlink(p)); attempt('pyc_write',lambda:write(pyc))
        extra=owned(base/'unleased_extra.py',base)
        attempt('directory_add',lambda:extra.open('x',encoding='utf-8').write('UNREGISTERED=1\n'))
    elif mode in ('released','owner_held','owner_exited'): attempt('write',lambda:write(p))
    else: raise ValueError('Unknown bounded probe mode')
    end=closure([Path(__file__)]); out={'mode':mode,'identity':identity(),'rows':rows,'source_start':start,'source_end':end,'source_closed':same(start,end)}
    print(json.dumps(out,ensure_ascii=True))

if __name__=='__main__': main()
