import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
from pathlib import Path
import json,subprocess,time
import psutil

def main():
    folder=Path(sys.argv[1]).resolve(); name=sys.argv[2]
    expected=Path(__file__).resolve().parents[1]/'research/review_3080_20261010'
    assert folder.is_relative_to(expected.resolve()) and name in ('run','audit')
    psutil.Process().cpu_affinity([7])
    out=(folder/(name+'.stdout')).open('xb'); err=(folder/(name+'.stderr')).open('xb'); t0=time.perf_counter()
    p=None; timed=False; creation=None; cmd=None; owner=None
    try:
        p=subprocess.Popen([sys.executable,str(folder/(name+'.py'))],stdout=out,stderr=err,creationflags=0x08000000)
        process=psutil.Process(p.pid); creation=process.create_time(); cmd=process.cmdline(); owner=process.username()
        try: code=p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            timed=True
            # Only our actual process tree, anchored by this Popen and identity.
            for child in process.children(recursive=True):
                child.kill(); child.wait(timeout=3)
            p.kill(); code=p.wait(timeout=3)
    finally:
        if p is not None and p.poll() is None: p.kill(); p.wait(timeout=3)
        out.close(); err.close()
    receipt={'exit_code':code,'timed_out':timed,'wall_seconds':time.perf_counter()-t0,'pid':p.pid,'create_time':creation,'cmdline':cmd,'owner':owner,'reaped':p.poll() is not None,'cap_seconds':60,'science_or_full_request_budget':False}
    target=folder/('RUN_EXIT.json' if name=='run' else 'AUDIT_RUN_EXIT.json')
    with target.open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps(receipt)); return code

if __name__=='__main__': raise SystemExit(main())
