"""P2-C-CTRL-01: fixed-root certified-greedy myopic opportunity census.

Only the admission rule changes. Shared insertion, both reference gates,
planning, physical scheduling and the OPP observer remain frozen. No labels.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_pos01c_opportunity as opp
import p2_pos01c_smoke_v2 as b

ROOT, WORKSPACE = b.ROOT, b.WORKSPACE
SAA = opp.OUT
OUT = ROOT/'research/p2_c_ctrl_01_20261019'
BASELINE = '510f4e5b70f9c85f0791f6718ca5cf26c1e66623'
MAX_OUTPUT = 64*1024**2
RULE = 'CERTIFIED_GREEDY_MYOPIC_ACCEPT_IFF_SHARED_CHECKS_PASS_WITHIN_EVENT_6S'
imports = opp.imports


def myopic_class():
    """No native C0 estimated-energy gate, no future vote or alternative checker.

    This is a certified feasible-first myopic adapter, not a reproduction claim
    for raw C0ReserveReplanner(mode='myopic'). Declaration precedes SOURCE runs.
    """
    _, t, r, _, _ = imports()
    class CertifiedMyopic(t.PublicSaaReplanner):
        def __init__(self, spec):
            super().__init__(spec)
            self.decisions = []

        def export_state(self):
            state = super().export_state()
            state['p2_ctrl01'] = {'rule': RULE, 'decisions': copy.deepcopy(self.decisions)}
            return state

        def on_reveal(self, engine, inst, clock, vehicles, served, visible_ids):
            proxy, published = r.saa_public_env(engine, inst, clock)
            self._reset_if_new(inst)
            reserved = proxy.get_reserved_customers(vehicles)
            new = sorted((c for c in published if c and not served[c] and c not in reserved
                          and c not in self._accepted and c not in self._rejected),
                         key=lambda c: (proxy.reveal_time[0,c], c))
            if not new:
                return
            start = time.perf_counter()
            self._sync_id_map(proxy,0,clock)
            for o in new:
                self.admission_count += 1
                saved = copy.deepcopy(self._plan)
                if time.perf_counter()>start+self.spec.time_limit:
                    ok,reason = False,'EVENT_TIMEOUT_BEFORE_INSERT'
                else:
                    # Exact PublicSaaReplanner implementation: same geometric
                    # insertion, C0 certificate and independent P2 reference gate.
                    ok = self._try_insert_certified(proxy,0,o,vehicles,served,clock)
                    reason = 'SHARED_CHECKS_PASS' if ok else 'SHARED_INSERT_OR_CERT_REJECT'
                    if time.perf_counter()>start+self.spec.time_limit:
                        ok,reason = False,'EVENT_TIMEOUT_AFTER_INSERT'
                if ok:
                    self._accepted.add(o)
                else:
                    self._plan=saved
                    self._rejected.add(o)
                    if reason.startswith('EVENT_TIMEOUT'):
                        self.timeouts += 1
                self.decisions.append(dict(clock=float(clock),order=int(o),accepted=bool(ok),
                    reason=reason,elapsed_event_s=time.perf_counter()-start))
    return CertifiedMyopic


def occupancy(view):
    _,_,r,_,_=imports()
    plans,allowed=r.plans_from_view(view),r.mutable_ids(view)
    raw={str(vid):len(p.suffix) for vid,p in plans.items()}
    legal={str(vid):len(p.suffix) if vid in allowed else 0 for vid,p in plans.items()}
    return dict(provisional_suffix_lengths=raw,legal_mask_suffix_lengths=legal,
                legal_vehicle_ids=sorted(allowed),total_mutable_customers=sum(legal.values()),
                accepted_unserved=len(set(view.accepted)-set(view.served)))


def historical_qualification(spec):
    """Mechanical reuse audit only: never rerun the SAA source days."""
    registration=json.loads((SAA/'registration.json').read_text())
    result=json.loads((SAA/'result.json').read_text())
    phases=json.loads((SAA/'phases.json').read_text())
    for group in ('source_sha256','preserved_sha256'):
        if any(b.sha(WORKSPACE/p)!=h for p,h in registration[group].items()):
            raise ValueError('HISTORICAL_SAA_IDENTITY_MISMATCH: '+group)
    verification=json.loads((SAA/'verification.json').read_text())
    if any(b.sha(SAA/p)!=h for p,h in verification['artifact_sha256'].items()):
        raise ValueError('HISTORICAL_SAA_RECEIPT_HASH_MISMATCH')
    if (result['status']!='COMPLETE' or result['identity']['signature']!=spec.signature or
            b.sha(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt')!=registration['checkpoint_sha256']):
        raise ValueError('HISTORICAL_SAA_CONTRACT_CONFIG_OR_COMPLETENESS_MISMATCH')
    rows={(p['day'],p['ordinal']):p for p in phases}
    if opp.summaries(rows,result['days'])!=result['summary']:
        raise ValueError('HISTORICAL_SAA_STATISTIC_DEFINITION_MISMATCH')
    return dict(status='PASS_REFERENCE_COMPARISON_ONLY',source_files=len(registration['source_sha256']),
                preserved_files=len(registration['preserved_sha256']),phases=len(phases),
                spec_signature=spec.signature,contract_hash=spec.contract.contract_hash,
                baseline_result_sha256=b.sha(SAA/'result.json'),
                baseline_phases_sha256=b.sha(SAA/'phases.json'),
                same_denominator_function='frozen p2_pos01c_opportunity.summaries',
                same_observer='frozen p2_pos01c_opportunity.observe',
                changed_factor=RULE,SAA_source_reruns=0)


def run_day(day,emit,spec,manifest,history_ids):
    np,t,r,Proposal,_=imports()
    descriptor=[d for d in manifest['days'] if d['role']=='SOURCE_TRAIN'][day]
    with np.load(ROOT/descriptor['relative_path'],allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    if t.dataset_digest(data)!=descriptor['dataset_sha256']:
        raise ValueError('SOURCE_DATA_IDENTITY_DRIFT')
    sampler=Proposal(spec.history,n_neighbors=10)
    current={'ordinal':-1,'observation':None,'previous_accepted':set(),'previous_rejected':set()}
    class ObservedPolicy(myopic_class()):
        def on_reveal(self,engine,inst,clock,vehicles,served,visible):
            started=time.monotonic()
            old_checks,old_timeouts=len(self.admission_checks),self.timeouts
            old_decisions=len(self.decisions)
            old_a,old_r=set(self._accepted),set(self._rejected)
            try:
                return super().on_reveal(engine,inst,clock,vehicles,served,visible)
            finally:
                emit('ADMISSION',dict(day=day,event_id=engine.event_id,clock=clock,
                    elapsed_s=time.monotonic()-started,timeouts=self.timeouts-old_timeouts,
                    accepted=sorted(self._accepted-old_a),rejected=sorted(self._rejected-old_r),
                    reference_checks=self.admission_checks[old_checks:],decisions=self.decisions[old_decisions:]))

        def plan(self,engine,inst,clock,vehicles,served,visible,replan_ids=None):
            current['ordinal']+=1
            ordinal=current['ordinal'];current['observation']=None
            emit('BEGIN',dict(day=day,ordinal=ordinal,event_id=engine.event_id,clock=clock))
            started=time.monotonic();physical=opp.physical_identity(vehicles)
            try:
                answer=super().plan(engine,inst,clock,vehicles,served,visible,replan_ids)
                if physical!=opp.physical_identity(vehicles):
                    raise ValueError('PLANNER_CHANGED_FROZEN_PHYSICAL_STATE')
                if current['observation'] is None:
                    raise ValueError('NATURAL_POST_ADMISSION_PHASE_NOT_OBSERVED')
                if (not current['previous_accepted']<=self._accepted or
                        not current['previous_rejected']<=self._rejected):
                    raise ValueError('IRREVOCABLE_LEDGER_REGRESSION')
                current['previous_accepted']=set(self._accepted)
                current['previous_rejected']=set(self._rejected)
                row=current['observation']
                row.update(day=day,ordinal=ordinal,event_id=engine.event_id,status='COMPLETE',
                    plan_including_observer_s=time.monotonic()-started)
                emit('PHASE',row)
                return answer
            except BaseException as exc:
                emit('PHASE_ERROR',dict(day=day,ordinal=ordinal,clock=clock,
                    partial_observation=current['observation'],error=type(exc).__name__+': '+str(exc),
                    elapsed_s=time.monotonic()-started))
                raise
    policy=ObservedPolicy(spec)
    class CommitObservedEnv(t.SaaPhaseEnv):
        def _commit(self,inst,clock,vehicle,traces,next_node):
            result=super()._commit(inst,clock,vehicle,traces,next_node)
            emit('COMMIT',dict(day=day,event_id=self.event_id,clock=float(clock),
                vehicle_id=vehicle.vehicle_id,node=int(next_node),status=vehicle.status,
                dispatch_time=vehicle.dispatch_time,ready_time=vehicle.ready_time))
            return result
    env=CommitObservedEnv(data,50.,1.5,num_vehicles=15,replanner=policy,
                         coldchain_contract=spec.contract,booking_horizon=16.)
    def hook(engine,view,context):
        started=time.monotonic()
        row=opp.observe(engine,view,context,policy,sampler,history_ids)
        row['occupancy']=occupancy(view)
        row['observer_s']=time.monotonic()-started
        current['observation']=row
        if not row['P0_reference_check']['ok']:
            raise ValueError('CURRENT_P0_REFERENCE_FAILURE: '+str(row['P0_reference_check']))
    env.p2_capture=hook
    traces,served=env.run(0)
    services=[dict(vehicle_id=tr.vehicle_id,order=int(s.node),depart=s.depart_time,
                   arrival=s.arrival_time,start=s.service_start,finish=s.service_finish)
              for tr in traces for s in tr.services]
    sequence=[s['order'] for s in services]
    dispatched=[tr for tr in traces if tr.dispatch_time is not None]
    if (set(sequence)!=policy._accepted or len(sequence)!=len(set(sequence)) or
            policy._accepted & policy._rejected or policy._accepted | policy._rejected!=set(range(1,51)) or
            any(tr.final_coldchain_state is None or not tr.final_coldchain_state.closed for tr in dispatched)):
        raise ValueError('TERMINAL_SERVICE_ADMISSION_OR_RETURN_LEDGER_INCOMPLETE')
    if policy.draw_audit:
        raise ValueError('MYOPIC_UNEXPECTED_FUTURE_SAMPLING')
    return dict(day=day,day_id=descriptor['day_id'],status='COMPLETE',
                source_sha256=descriptor['file_sha256'],root_seed=descriptor['root_seed'],
                phase_attempts=current['ordinal']+1,accepted=sorted(policy._accepted),
                rejected=sorted(policy._rejected),served=sequence,timeouts=policy.timeouts,
                draw_audit=policy.draw_audit,admission_checks=policy.admission_checks,
                decisions=policy.decisions,services=services,
                dispatched_vehicle_ids=[tr.vehicle_id for tr in dispatched],
                dispatch_times={str(tr.vehicle_id):tr.dispatch_time for tr in dispatched},
                return_arrivals={str(tr.vehicle_id):tr.return_arrival for tr in dispatched},
                terminal_utility='NOT_EVALUATED',terminal_labels=0)


def worker(conn,gate):
    gate.wait()
    sink=open(b.os.devnull,'w');sys.stdout=sys.stderr=sink
    try:
        spec,manifest,history_ids=opp.make_spec()
        qualification=historical_qualification(spec)
        b.send(conn,'READY',dict(signature=spec.signature,contract_hash=spec.contract.contract_hash,
            historical_SAA=qualification,admission_rule=RULE,
            SaaSpec={k:b.plain(v) for k,v in asdict(spec).items() if k not in ('history','contract')}))
        def emit(kind,row):
            b.send(conn,kind,row)
            ack,_,_=b.receive(conn)
            if ack!='ACK':
                raise ValueError('SUPERVISOR_REFUSED_OBSERVATION')
        for day in range(2):
            emit('DAY_BEGIN',dict(day=day))
            result=run_day(day,emit,spec,manifest,history_ids)
            emit('DAY_END',result)
        b.send(conn,'DONE',{})
    except BaseException as exc:
        b.send(conn,'ERROR',dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()[-12000:]))


def preserved():
    old=opp.preserved()
    old.update({p.relative_to(WORKSPACE).as_posix():b.sha(p) for p in SAA.rglob('*') if p.is_file()})
    return old


summaries=opp.summaries


def execute():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():
        raise RuntimeError('SINGLE_USE_ID_ALREADY_ATTEMPTED; NO_RETRY')
    store = b.Store(OUT, cap=MAX_OUTPUT)
    started, deadline = time.monotonic(), time.monotonic()+1200.
    registration = dict(run_id=OUT.name, baseline=BASELINE, qualification='REFERENCE_ONLY',
        preserved_sha256=preserved(), source_sha256=b.sources(),
        checkpoint_sha256=b.sha(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt'),
        old_configuration='same frozen contract/history; myopic uses no stochastic forecast draws',
        admission_rule=RULE, sole_factor='replace SAA forecast vote by shared-check feasible-first acceptance',
        denominator='every natural engine-triggered plan entry; no dedup/filter; unfinished attempts retained',
        mask_seed=0, future_rule='original 10-neighbor support audit only; no new future draws',
        resource=dict(CPU_workers=1, BLAS_threads=1, wall_s=1200, RAM_bytes=8*b.GIB,
            job_hard_bytes=7*b.GIB, supervisor_stop_bytes=512*1024**2, output_bytes=MAX_OUTPUT),
        prohibited=['MaskCO repair','counterfactual continuation','terminal utility labels','training','GPU','final seeds'],
        source_run_limit='exactly once per two registered SOURCE roots; no restart or replacement')
    store.write('registration.json', b.encoded(registration))
    result = dict(status='BLOCKED', days=[dict(day=i,status='NOT_ATTEMPTED') for i in range(2)],
                  qualification='REFERENCE_ONLY', stop_reason=None, phase_rows={}, preflight={})
    rows, selected, proc, conn, job = {}, [0,0], None, None, None
    peaks = dict(job_committed_bytes=0, supervisor_bytes=0)
    def journal(kind, data):
        packet = b.encoded(dict(kind=kind, payload=data))+b'\n'
        if store.used()+len(packet)>MAX_OUTPUT-8*1024**2:
            raise RuntimeError('OUTPUT_LIMIT_BEFORE_APPEND')
        with (OUT/'journal.jsonl').open('ab') as f:
            f.write(packet)
    try:
        # Expected synthetic resource faults, before any SOURCE execution.
        probe_store = b.Store(OUT/'output_probe', cap=9*1024**2)
        try:
            probe_store.write('refused.bin', b'x'*(1024**2+1))
        except RuntimeError:
            assert not (OUT/'output_probe/refused.bin').exists()
        else:
            raise RuntimeError('OUTPUT_PROBE_FAILED')
        probe = b.Session(store, deadline, memory=64*1024**2)
        try:
            assert probe.call({'op':'probe-memory'},10)['allocation_denied']
        finally:
            probe.close()
        probe = b.Session(store, deadline)
        try:
            try:
                probe.call({'op':'probe-timeout'},3)
            except TimeoutError:
                pass
            else:
                raise RuntimeError('TIMEOUT_PROBE_FAILED')
        finally:
            probe.close()
        result['preflight']['resource_guards'] = 'PASS_EXPECTED_SYNTHETIC_FAULTS_RECLAIMED'
        ctx = mp.get_context('spawn')
        conn, child = ctx.Pipe()
        gate = ctx.Event()
        job = b.Job(7*b.GIB)
        proc = ctx.Process(target=worker, args=(child,gate))
        proc.start()
        child.close()
        job.assign(proc.pid)
        gate.set()
        while time.monotonic()<deadline-5.:
            peaks['job_committed_bytes'] = max(peaks['job_committed_bytes'], job.peak())
            peaks['supervisor_bytes'] = max(peaks['supervisor_bytes'], b.supervisor_memory())
            if peaks['supervisor_bytes']>512*1024**2 or store.used()>MAX_OUTPUT-8*1024**2:
                raise RuntimeError('SUPERVISOR_RESOURCE_LIMIT')
            if not conn.poll(.05):
                if not proc.is_alive():
                    raise RuntimeError('WORKER_EXIT_WITHOUT_RECEIPT')
                continue
            kind, data, _ = b.receive(conn)
            if kind == 'ERROR':
                journal(kind,data)
                raise RuntimeError(data['error'])
            if kind == 'DONE':
                result['status']='COMPLETE'
                break
            journal(kind,data)
            if kind == 'READY':
                result['identity']=data
                continue
            if kind == 'DAY_BEGIN':
                result['days'][data['day']].update(status='RUNNING')
            elif kind == 'BEGIN':
                key=(data['day'],data['ordinal'])
                if key in rows:
                    raise ValueError('DUPLICATE_PHASE_ORDINAL')
                rows[key]=dict(data,status='UNFINISHED')
            elif kind in ('PHASE','PHASE_ERROR'):
                key=(data['day'],data['ordinal'])
                if key not in rows:
                    raise ValueError('PHASE_WITHOUT_BEGIN')
                rows[key]=data if kind=='PHASE' else dict(data,status='EXCEPTION')
                if kind=='PHASE' and data['reference_usable'] and selected[data['day']]<2:
                    store.write('selected_day%d_%d.json'%(data['day'],selected[data['day']]),b.encoded(data))
                    selected[data['day']]+=1
            elif kind == 'DAY_END':
                result['days'][data['day']]=data
            b.send(conn,'ACK')
        else:
            raise TimeoutError('GLOBAL_20_MINUTE_LIMIT')
    except BaseException as exc:
        result.update(status='BLOCKED',stop_reason=type(exc).__name__+': '+str(exc))
    finally:
        if job is not None:
            try:
                peaks['job_committed_bytes']=max(peaks['job_committed_bytes'],job.peak())
                job.close()
            except BaseException as exc:
                result.update(status='BLOCKED',cleanup_error=str(exc))
        if proc is not None:
            proc.join(1.)
            if proc.is_alive():
                proc.kill(); proc.join(.5)
            if proc.is_alive():
                result.update(status='BLOCKED',cleanup_error='WORKER_NOT_RECLAIMED')
        if conn is not None:
            conn.close()
        for day in result['days']:
            if day['status']=='RUNNING':
                day.update(status='INCOMPLETE',stop_reason=result['stop_reason'])
        for name,expected in (('source_sha256', b.sources()),('preserved_sha256',preserved())):
            equal=expected==registration[name]
            result[name+'_unchanged']=equal
            if not equal:
                result.update(status='BLOCKED',stop_reason='IDENTITY_DRIFT_AFTER_RUN')
        result.update(summary=summaries(rows,result['days']),resource_peaks=peaks,
            elapsed_s=time.monotonic()-started, output_bytes_before_final=store.used(),
            model_loads=0, mask_reconstructions=0, real_counterfactuals=0, terminal_labels=0,
            training_steps=0, GPU=False, full_certified=False, final_test_used=False)
        result.pop('phase_rows')
        store.write('phases.json', b.encoded([rows[k] for k in sorted(rows)]),final=True)
        store.write('result.json', b.encoded(result),final=True)
    print(json.dumps(dict(status=result['status'],summary=result['summary'],elapsed_s=result['elapsed_s'])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-once', action='store_true',required=True)
    parser.parse_args()
    execute()
