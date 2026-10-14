"""Run two bounded engineering replays of the same previously inspected day.

This is not fresh research evidence: it qualifies conserved shadow accounting
and the exact original 200-day prior without changing original source/results.
"""
import ast, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup'
OUT=P/'complete_history_energy_qa_root_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)

def main():
    OUT.mkdir(exist_ok=False)
    parent=P/'actual_masked_fleet_followup_v1/cond_hist_followup.py'
    history=P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl'
    correction=P/'shadow_energy_correction_root_v1/energy_complete_saa.py'
    trace=P/'actual_masked_fleet_followup_v1/full_day_qa_v1/own_uncond_dynamic_trace.jsonl'
    body=parent.read_text(encoding='utf-8')
    body=body.replace('import hard_admission as ha',
        "sys.path.insert(0,"+repr(str(parent.parent))+ ")\n"+
        "sys.path.insert(0,"+repr(str(correction.parent))+")\nimport hard_admission as ha",1)
    start=body.index('    historic=[]')
    end=body.index('        def _saa_decide',start)
    body=body[:start]+'''    import pickle
    from energy_complete_saa import EnergyCompleteSAA
    if not hasattr(conditional_base,'common_prior'):
        with Path(config['history_pickle']).open('rb') as f:
            conditional_base.common_prior=pickle.load(f)
    historic=conditional_base.common_prior['history']
    class Observed(EnergyCompleteSAA if config['ledger_corrected'] else SaaReplanner):
'''+body[end:]
    old="contract=make_c0_contract_v2();history=[(np.linspace(.5,15.5,30),np.asarray([(j+d)%3 for j in range(30)],int)) for d in range(3)]"
    assert body.count(old)==1
    body=body.replace(old,"contract=make_c0_contract_v2()\n    import pickle\n    with Path(a.history_pickle).open('rb') as f: prior=pickle.load(f)\n    history=prior['history_times_classes']")
    body=body.replace('history=history)\n    sha=',"history=history,history_pickle=a.history_pickle,ledger_corrected=a.corrected)\n    sha=",1)
    body=body.replace("'P-REAL-COND-HIST-SAA-REPORT-ENGINEERING-v1'","'P-COMMON200-SHADOW-LEDGER-ENGINEERING-root-v1'")
    body=body.replace("'explicit three synthetic fixture days with fixed known x/y/demand/TW/service; NOT actual I3 training200'",
        "'exact original I3 historical seed20260925 all200 days; independently regenerated and qualified'")
    body=body.replace("source_start=pf.source_hashes(Path(a.scripts).resolve()),",
        "history_pickle_sha256=sha(a.history_pickle),ledger_corrected=a.corrected,adapter_sha256=sha(__file__),source_start=pf.source_hashes(Path(a.scripts).resolve()),")
    body=body.replace("p.add_argument('--archive-trace');", "p.add_argument('--history-pickle');p.add_argument('--corrected',action='store_true');p.add_argument('--archive-trace');")
    ast.parse(body)
    worker=OUT/'complete_history_worker.py';worker.write_text(body,encoding='utf-8')
    write(OUT/'REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        scope='two arms one already-observed 80-order engineering fixture; no fresh evaluation day or CI',
        parent_sha256=sha(parent),worker_sha256=sha(worker),history_sha256=sha(history),
        correction_sha256=sha(correction),observed_trace_sha256=sha(trace),
        arms=['original_ledger','complete_ledger'],K=10,paired_branches=20,history_days=200,
        per_arm_outer_bound_seconds=240,request_limit_seconds=10,
        primary_online_gate=False,stable_online_gain=False,all_five_arm_scoring_qualified=False))
    results=[]
    for arm in ('original_ledger','complete_ledger'):
        argv=[sys.executable,'-B',str(worker),'--scripts',str(E/'scripts'),
            '--archive-trace',str(trace),'--out',str(OUT/arm),'--history-pickle',str(history)]
        if arm=='complete_ledger':argv+=['--corrected']
        started=time.perf_counter()
        with (OUT/(arm+'.stdout.log')).open('xb') as stdout,(OUT/(arm+'.stderr.log')).open('xb') as stderr:
            proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=dict(os.environ,
                PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            try:code=proc.wait(timeout=240)
            except subprocess.TimeoutExpired:
                children=psutil.Process(proc.pid).children(recursive=True)
                for child in reversed(children):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                proc.kill();proc.wait(timeout=10);psutil.wait_procs(children,timeout=10);code=124
        results.append(dict(arm=arm,exit=code,wall_seconds=time.perf_counter()-started))
        write(OUT/(arm+'_RUN_EXIT.json'),results[-1])
        print(json.dumps(results[-1]),flush=True)
        if code:break
    write(OUT/'RUN_EXIT.json',dict(arms=results,old_sources_preserved=sha(parent)==json.loads((OUT/'REGISTRATION.json').read_text())['parent_sha256'],
        complete=len(results)==2 and all(x['exit']==0 for x in results),formal_gain=False))
    return int(len(results)!=2 or any(x['exit']!=0 for x in results))

if __name__=='__main__':sys.exit(main())
