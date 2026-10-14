"""Preregistered paired old boundary completion diagnosis, no common scoring."""
import copy,json,pickle,time,types,ast,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
import psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_observed_day_extension_v1 as observed
import qualify_pickup_checkpoint_ack_v1 as source_api
import pickup_undecided_public_v1 as projector
import pickup_pending_transaction_v1 as engine
from pickup_loaded_closure_v1 import sha
OUT=old.P/'event27_settlement_diagnosis_root_v1';INPUT=old.P/'checkpoint_bytes_retention_root_v1/ACTUAL_SHARED_PAYLOAD.bin';FAILED=old.P/'continuous_checkpoint_root_v1'
def save(n,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def main():
    psutil.Process().cpu_affinity([7]);OUT.mkdir(exist_ok=False);paths=[Path(__file__),INPUT,FAILED/'REGISTRATION.json',FAILED/'WORKER_FAILURE.json',FAILED/'RUN_EXIT.json',old.HISTORY,observed.fixture.TRACE,old.ROOT/'tools/pickup_pending_transaction_v1.py',old.E/'scripts/simulation/strict_online_env.py'];ids={str(p):sha(p) for p in paths}
    save('REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=ids,scope='paired fixed same event27 sealed source: zero vs one original plan_and_commit completion at source boundary, then stop at first actual next reveal and original current-only admission/full C0 base certificate. Both retained. No K10, production guard, checkpoint publication, scientific data or gains. Diagnostic hypotheses only; no continuous trial re-dispatch.',cap_s=30.,source_boundary_completion=[False,True],new_scientific_data=False,scientific_family=False))
    encode,decode=old.codecs();template,dataset=observed.setup();initial=pickle.loads(INPUT.read_bytes())['state'];history=pickle.loads(old.HISTORY.read_bytes());rows=[]
    class Stop(Exception):pass
    for settle in (False,True):
        b=source_api.restore(template,dataset,initial);before=old.pf.digest(source_api.state(b));boundary={};s=time.perf_counter()
        if settle:b.env._plan_and_commit(0,b.clock,b.fleet,b.traces,b.served,b.customers)
        settled=copy.deepcopy(source_api.state(b))
        def on_reveal(self,env,i,clock,fleet,served,visible):boundary.update(clock=clock,visible=visible);raise Stop()
        b.rp.on_reveal=types.MethodType(on_reveal,b.rp)
        reveals=b.reveals
        try:b.env._run_loop(0,b.clock,sum(t<=b.clock+1e-6 for t in reveals),b.fleet,b.traces,b.served,b.customers,reveals,float(b.env.tw_end[0].max()))
        except Stop:b.clock=boundary['clock']
        else:raise ValueError('fixed first next reveal absent')
        visible=boundary['visible'];pending=sorted((int(j) for j in visible if b.env.demands[0,j]>0 and not b.served[j] and j not in b.rp._accepted|b.rp._rejected),key=lambda j:(b.env.reveal_time[0,j],j));assert len(pending)==1
        packet=projector.export_undecided(b.env,0,b.clock,b.env.event_id,sum(t<=b.clock+1e-6 for t in reveals),b.fleet,b.traces,b.served,b.customers,pending[0]);save('INPUT_'+str(int(settle))+'.json',encode(dict(source_digest=before,after_source_completion=settled,at_next_reveal=source_api.state(b),packet=packet)))
        base=engine.decide(packet,history,old.E/'scripts',s+30.);save('BASE_'+str(int(settle))+'.json',encode(base));rows.append(dict(source_completion=settle,order=pending[0],clock=float(b.clock),event_id=b.env.event_id,source_replan_before=initial['replan_count'],source_replan_after=settled['replan_count'],base_qualified=base['qualified'],hard=base['certificate']['evaluation']['hard_feasible'],failures=base['certificate']['evaluation']['failures'],wall_s=time.perf_counter()-s,source_digest=before,packet_digest=old.pf.digest(packet)))
    source=old.E/'scripts/simulation/strict_online_env.py';tree=ast.parse(source.read_text(encoding='utf-8'));loop=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_run_loop');calls=[dict(method=n.func.attr,line=n.lineno) for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('_advance_fleet','on_reveal','_plan_and_commit')];save('ORIGINAL_LOOP_AST.json',dict(source_sha256=sha(source),calls=calls))
    save('RESULT.json',dict(rows=rows,original_loop_calls=calls,source_inputs_unchanged=all(sha(p)==h for p,h in ids.items()),interpretation='Candidate omitted old boundary completion: paired original full C0 reports and states retain evidence. No conclusion of whole algorithm correctness or stable gain.',full_live_request_qualified=False,stable_online_gain=False));print(json.dumps(dict(rows=rows,source_inputs_unchanged=all(sha(p)==h for p,h in ids.items()))));return 0
if __name__=='__main__':raise SystemExit(main())
