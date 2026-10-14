"""600-second owned child supervisor; preserve all actual logs and return code."""
from pathlib import Path
import subprocess,os,sys,time,json,psutil
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'research/review_3080_20261010/pickup/resident_day_tail_root_v1';TOOLS=ROOT/'tools'
def main():
    if O.exists():raise FileExistsError('Never restart existing engineering tree')
    logs=ROOT/'research/review_3080_20261010/pickup/resident_day_tail_launch_root_v1';logs.mkdir(exist_ok=False);start=time.perf_counter();failure=None
    with (logs/'stdout.txt').open('xb') as out,(logs/'stderr.txt').open('xb') as err:
        p=subprocess.Popen([sys.executable,'-B',str(TOOLS/'qualify_pickup_resident_day_tail_v1.py')],stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),creationflags=subprocess.CREATE_NO_WINDOW)
        with (logs/'START.json').open('x') as f:json.dump(dict(parent_pid=os.getpid(),worker_pid=p.pid,registered_max_s=600,started=start),f)
        try:code=p.wait(timeout=600.)
        except subprocess.TimeoutExpired:
            children=psutil.Process(p.pid).children(recursive=True)
            for child in reversed(children):
                try:child.kill()
                except psutil.NoSuchProcess:pass
            p.kill();p.wait();gone,live=psutil.wait_procs(children,timeout=2.);code=124;failure=dict(reason='registered_timeout',owned_children_alive=[c.pid for c in live])
    with (logs/'EXIT.json').open('x') as f:json.dump(dict(exit_code=code,failure=failure,finished=time.perf_counter(),wall_s=time.perf_counter()-start,parent_pid=os.getpid(),actual_producer_returncode=p.returncode),f,indent=2)
    print(json.dumps(dict(exit_code=code,failure=failure,wall_s=time.perf_counter()-start)));return code
if __name__=='__main__':raise SystemExit(main())
