"""New pending-input adaptation QA, reusing sealed inputs, not old QA reruns."""
import copy,json,os,pickle,queue,subprocess,sys,threading,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np
import psutil
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from joint_install import JointPolicyOwner,transactional_class,state_dict
from run_exp_energy_c0 import C0ReserveReplanner
from pickup_pending_transaction_v1 import compute
import pickup_partial_common_v1,pickup_partial_trace_v1,pickup_partial_known_v1,pickup_partial_validation_v1
import pickup_undecided_public_v1
ROOT=old.ROOT;P=old.P;E=old.E;OUT=P/'pending_admission_transaction_root_v1'
INPUT=P/'undecided_public_input_root_v2/RESULT.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,x):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def cases():
    _,decode=old.codecs();raw=read(INPUT);rows=[]
    for index in (0,9,13):
        row=next(r for r in raw['records'] if r['kind']=='derived_old_state' and r['index']==index and r['variant']==0)
        rows.append(dict(name='derived:'+str(index),packet=decode(row['packet'])))
    first=next(r for r in raw['records'] if r['kind']=='actual_first_reveal_hook' and r['variant']==0)
    rows.append(dict(name='sealed_first_reveal',packet=decode(first['packet'])))
    # Deliberately engineering-only multi-pending boundary, selected by ID only.
    packet=copy.deepcopy(rows[2]['packet']);prior=packet['snapshot']['replanner_state']
    other=min(prior['rejected']);prior['rejected']=sorted(set(prior['rejected'])-{other})
    packet['admission_input']['pending']=sorted(set(packet['admission_input']['pending'])|{other})
    packet['snapshot']['state_hash']=snapshot_state_hash(packet['snapshot'])
    pickup_undecided_public_v1.validate_undecided(packet)
    rows.append(dict(name='derived_two_pending',packet=packet,extra_pending=other))
    return rows
def sources():
    manifest=old.pf.source_hashes(E/'scripts')
    names=('qualify_pickup_pending_transaction_v1.py','audit_pickup_pending_transaction_v1.py','pickup_pending_transaction_v1.py',
           'pickup_partial_common_v1.py','pickup_partial_trace_v1.py','pickup_partial_known_v1.py','pickup_partial_validation_v1.py','pickup_undecided_public_v1.py')
    for p in [ROOT/'tools'/n for n in names]+[INPUT,old.HISTORY,Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__)]:
        manifest[str(p)]=old.sha(p)
    return manifest
def worker():
    psutil.Process().cpu_affinity([7])
    if np.__version__!='1.26.4':raise ValueError('original environment')
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    manifest=sources();print(json.dumps(dict(kind='ready',sources=manifest,cpu=[7],numpy=np.__version__)),flush=True)
    for line in sys.stdin:
        q=old.wire.unwire(json.loads(line)['request'])
        try:
            response=compute(q['packet'],history,E/'scripts',q['deadline'])
            response['source_end']={p:old.sha(p) for p in manifest};response['source_qualified']=response['source_end']==manifest
        except Exception:response=dict(kind='error',traceback=traceback.format_exc())
        print(json.dumps(dict(kind='response',result=old.wire.wire(response))),flush=True)
class Worker:
    def __init__(self):
        self.q=queue.Queue();self.errors=[]
        self.p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        def pump():
            for line in self.p.stdout:
                try:self.q.put(json.loads(line))
                except Exception:self.errors.append(line)
            self.q.put(dict(kind='closed'))
        def err():
            for line in self.p.stderr:self.errors.append(line)
        threading.Thread(target=pump,daemon=True).start();threading.Thread(target=err,daemon=True).start()
        try:self.ready=self.q.get(timeout=30)
        except queue.Empty:self.close();raise TimeoutError('setup')
        if self.ready.get('kind')!='ready':self.close();raise ValueError(repr(self.errors))
    def ask(self,packet,deadline):
        self.p.stdin.write(json.dumps(dict(request=old.wire.wire(dict(packet=packet,deadline=deadline))))+'\n');self.p.stdin.flush()
        try:r=self.q.get(timeout=max(0.,deadline-time.perf_counter()))
        except queue.Empty:self.close();raise TimeoutError('request')
        if r.get('kind')!='response':raise ValueError('worker ended '+repr(self.errors))
        return old.wire.unwire(r['result'])
    def close(self):
        if self.p.poll() is None:self.p.kill()
        self.p.wait(timeout=5)
def main():
    psutil.Process().cpu_affinity([7]);encode,_=old.codecs();rows=cases();OUT.mkdir(exist_ok=False)
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=old.sha(INPUT),sources=sources(),
             input_names=[r['name'] for r in rows],inputs_reused_from_completed_public_component=True,
             multi_pending_derivation='request13, remove minimum other rejected ID; engineering only, not actual simultaneous reveal',
             method='own_uncond_dynamic current-only; own_incumbent proposal; original common history K10; full C0 with explicit provisional pending partition; compiled JointPolicyOwner',
             evaluator_change='only observed admission coverage recognizes pending, retains all real demands/features; pending not served or penalized; original full-day evaluator unchanged',
             K=10,common_seed=202610111301,normal_cap_s=10.,cpu=[7],numpy='1.26.4',new_scientific_data=False,
             real_event_ownership_integrated=False,parent_transport_preemptible=False,parent_decode_commit_preemptible=False,
             full_live_10s_qualified=False,full_loaded_dependency_closure_qualified=False,formal_learned_controls=False,stable_online_gain=False)
    save('REGISTRATION.json',reg);checks=[];records=[];boundaries=[];w=None
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    try:
        for row in rows:
            packet=row['packet'];name=row['name'];o=packet['admission_input']['current_order']
            rp=transactional_class(C0ReserveReplanner)(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.)
            rp.restore_state(packet['snapshot']['replanner_state']);fleet,_,_=restore_recourse_snapshot(packet['snapshot'])
            owner=JointPolicyOwner(rp,fleet,packet['snapshot']['clock'],packet['snapshot']['event_id']);before=rp.export_state();ptr=rp._policy
            w=Worker();started=time.perf_counter();deadline=started+10.
            token=owner.begin(str(o),o,deadline,old.pf.digest(packet));response=w.ask(packet,deadline);received=time.perf_counter()
            if response.get('kind')!='response':raise ValueError(response)
            installed,reason=owner.commit(token,response);committed=time.perf_counter();w.close();finished=time.perf_counter();after=rp.export_state()
            current=response['scoring_packet'];remaining=set(packet['admission_input']['pending'])-{o}
            ck(name+':worker_current_qualified',response['base_response']['qualified'] and response['certified'])
            ck(name+':joint_publish',installed and reason=='published' and rp._policy is response['compiled_state'] and rp._policy is not ptr)
            ck(name+':state_exact',state_dict(response['compiled_state'])==after)
            ck(name+':current_decided_only',set(after['accepted'])-{o}==set(before['accepted']) and set(after['rejected'])-{o}==set(before['rejected']) and o in set(after['accepted'])|set(after['rejected']))
            ck(name+':other_pending_retained',set(current['admission_input']['pending'])==remaining and not remaining&(set(after['accepted'])|set(after['rejected'])))
            ck(name+':pending_not_served',not remaining&{s.node for t in response['selected_raw'] for s in t.services})
            ck(name+':pending_not_charged',response['selected_certificate']['evaluation']['reject_loss']==sum(packet['penalties'][int(packet['dataset']['temp_class'][0,j])] for j in after['rejected']))
            ck(name+':features_not_masked',old.pf.digest(packet['dataset'])==old.pf.digest(current['dataset']))
            ck(name+':common20branches',all(c['qualified'] and len(c['branch_values'])==10 for c in response['common_scores']['candidate_scores'].values()))
            ck(name+':all_C0_hard',response['selected_certificate']['qualified'])
            ck(name+':incumbent_unchanged_proposer',not response['accepted_route'] and response['proposal']==current['plan'])
            ck(name+':normal_elapsed_includes_parent_cleanup',started<=received<=committed<=finished<deadline and w.p.poll() is not None)
            ck(name+':selected_sources_same',response['source_end']==w.ready['sources'])
            records.append(dict(name=name,packet=packet,before=before,after=after,response=response,ready=w.ready,
                                started=started,received=received,committed=committed,finished=finished,wall_s=finished-started,installed=installed,reason=reason,worker_pid=w.p.pid))
            w=None
        # New semantic boundaries only; no old timeout QA repeated.
        from pickup_partial_validation_v1 import validate_partial
        base=records[-1]['response']['scoring_packet'];pending=base['admission_input']['pending'][0]
        for label in ('pending_rejected','pending_accepted','pending_in_route','pending_missing','decision_label'):
            trial=copy.deepcopy(base)
            if label=='pending_rejected':trial['snapshot']['replanner_state']['rejected'].append(pending)
            elif label=='pending_accepted':trial['snapshot']['replanner_state']['accepted'].append(pending)
            elif label=='pending_in_route':trial['plan']['0'].append(pending)
            elif label=='pending_missing':trial['admission_input']['pending'].remove(pending)
            else:trial['admission_input']['current_decision']='accepted' if trial['admission_input']['current_decision']=='rejected' else 'rejected'
            trial['snapshot']['state_hash']=snapshot_state_hash(trial['snapshot'])
            try:validate_partial(trial);error=None
            except ValueError as exc:error=str(exc)
            ck('new_boundary:'+label,error is not None);boundaries.append(dict(name=label,error=error))
        ck('registered_sources_end',all(old.sha(p)==h for p,h in reg['sources'].items()))
        result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),records=records,boundaries=boundaries,
                    full_live_10s_qualified=False,continuous_live_event_qualified=False,formal_learned_controls=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)
        save('RESULT.json',encode(result));save('RUN_EXIT.json',dict(exit_code=int(result['failed']>0)))
        print(json.dumps(dict(checks=len(checks),failed=result['failed'],records=len(records),walls=[r['wall_s'] for r in records])));return int(result['failed']>0)
    except Exception:
        if w:w.close()
        save('FAILURE.json',dict(traceback=traceback.format_exc(),checks=checks,records=encode(records)));save('RUN_EXIT.json',dict(exit_code=1));raise
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:raise SystemExit(main())
