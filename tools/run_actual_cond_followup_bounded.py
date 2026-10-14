"""Complete the already observed fixture's actual conditional report arm."""
import json,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/review_3080_20261010/pickup/actual_masked_fleet_followup_v1'
OUT=P/'actual_cond_hist_root_v1'
def main():
    wrapper=P/'actual_cond_hist_root_v1_supervision';wrapper.mkdir(exist_ok=False)
    start=time.time()
    argv=[sys.executable,'-B',str(P/'cond_hist_followup.py'),'--scripts',str(ROOT/'C-VRP_Cold-chainVehicleRoutingProblem/scripts'),
          '--archive-trace',str(ROOT/'research/review_3080_20261010/pickup/actual_masked_fleet_engineering_v1/wiring_qa_v1/own_uncond_dynamic_trace.jsonl'),
          '--out',str(OUT)]
    (wrapper/'REGISTRATION.json').write_text(json.dumps({'engineering_only':True,'new_days':0,'max_seconds':900,'argv':argv,
        'start_utc_unix':start,'only_owned_descendants_may_be_killed':True}),encoding='utf-8')
    with (wrapper/'stdout.log').open('w',encoding='utf-8') as stdout,(wrapper/'stderr.log').open('w',encoding='utf-8') as stderr:
        p=subprocess.Popen(argv,stdout=stdout,stderr=stderr)
        try:code=p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            children=psutil.Process(p.pid).children(recursive=True)
            for c in reversed(children):
                try:c.kill()
                except psutil.NoSuchProcess:pass
            p.kill();p.wait(timeout=10);psutil.wait_procs(children,timeout=10);code=124
    result={'exit_code':code,'wall_seconds':time.time()-start,'engineering_only':True,'scientific_gain':False}
    (wrapper/'RUN_EXIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));return code
if __name__=='__main__':sys.exit(main())
