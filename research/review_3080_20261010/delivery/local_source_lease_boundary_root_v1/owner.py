import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; sys.path.insert(0,str(ROOT/'tools'))
import time
from engineering_stage_closure_v1 import initialize,closure,identity,sha,save
from lease_runtime import acquire,info

def main():
    initialize(); base=HERE/'owned'; path=base/'copy_0.py'
    start=closure([Path(__file__)]); h=acquire(path,base)
    save(HERE/'OWNER_READY.json',{'identity':identity(),'handle_file':info(h),'sha256':sha(path),'source_start':start,'source_at_ready':closure([Path(__file__)])})
    deadline=time.perf_counter()+15
    while time.perf_counter()<deadline and not (HERE/'OWNER_EXIT_SIGNAL').exists(): time.sleep(.02)
    # Deliberately bypass Python finally/CloseHandle. OS process teardown must
    # release only this process's non-inherited owned-file handle.
    os._exit(17)

if __name__=='__main__': main()
