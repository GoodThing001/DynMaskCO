"""P2-C-OPP-01: one-shot natural SAA census; no repair/labels/model loading.

Only this isolated observer/runner is new. All admission, scheduling, commit,
WAIT/RETURN and random sampling decisions delegate to the frozen P2-B controller.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_pos01c_smoke_v2 as b  # CPU/thread setup + existing Windows job primitives

ROOT, WORKSPACE = b.ROOT, b.WORKSPACE
OUT = ROOT/'research/p2_c_opp_01_20261019'
BASELINE = '1428eb96ebec41f4988d5a96569b8dd32df06068'
MAX_OUTPUT = 64*1024**2


def imports():
    np, teacher, ref = b.imports()
    from p2_pos01c_gap import CapacityConditionedProposal, PublicPrefix
    return np, teacher, ref, CapacityConditionedProposal, PublicPrefix


def raw_mask_proposals(view, plans):
    """Read-only expansion of the frozen four rules, cross-checked with mask_pool.

    It does not replace the real mask generator, score masks or repair anything.
    """
    np, _, r, _, _ = imports()
    allowed = r.mutable_ids(view)
    pool = sorted(c for vid, p in plans.items() if vid in allowed for c in p.suffix
                  if c in view.accepted and c not in view.served)
    names = ('distance', 'resource', 'frontier', 'random')
    if len(pool) < 2 or not all(r.route_check(view, p) for p in plans.values()):
        raw = {name: () for name in names}
    else:
        e, orders = r.public_env(view), {o.oid: o for o in view.orders}
        vmap = {v.vehicle_id: v for v in view.vehicles}
        detour, pressure, first = {}, {}, []
        for vid, p in plans.items():
            route_time = p.anchor_time
            for i, c in enumerate(p.suffix):
                pred = p.anchor_node if i == 0 else p.suffix[i-1]
                arrival = route_time+float(e.dist_mat[0, pred, c])/view.tw_speed
                route_time = max(arrival, orders[c].tw_start)+orders[c].service
                if c not in pool:
                    continue
                succ = 0 if i+1 == len(p.suffix) else p.suffix[i+1]
                detour[c] = float(e.dist_mat[0, pred, c]+e.dist_mat[0, c, succ]-e.dist_mat[0, pred, succ])
                cc = vmap[vid].coldchain_state
                lots = () if cc is None else cc.cargo_manifest
                ratios = [1-l.quality_remaining/l.initial_quality for l in lots
                          if l.temp_class == orders[c].temp_class]
                slack = orders[c].tw_end-arrival
                pressure[c] = (orders[c].demand/(view.capacity-r.CAPACITY_MARGIN)+
                    1/(1+max(0, slack)/max(orders[0].tw_end, 1e-9))+max(ratios, default=0))
                if i == 0:
                    first.append(c)
        distance = sorted(pool, key=lambda c: (-detour[c], c))[:2]
        resource = sorted(pool, key=lambda c: (-pressure[c], c))[:2]
        frontier = []
        if first:
            a = min(first, key=lambda c: (-pressure[c], c))
            c = min((c for c in pool if c != a), key=lambda c: (e.dist_mat[0, a, c], c))
            frontier = [a, c]
        random = np.random.default_rng(0).choice(pool, 2, replace=False).tolist()
        raw = dict(zip(names, (tuple(sorted(x)) for x in (distance, resource, frontier, random))))
    unique = tuple(dict.fromkeys(pair for pair in raw.values() if pair))
    if unique != r.mask_pool(view, plans, seed=0):
        raise ValueError('RAW_PROPOSALS_DISAGREE_WITH_FROZEN_MASK_POOL')
    return dict(raw=raw, rules_evaluated=4, nonempty_raw=sum(bool(x) for x in raw.values()),
                masks=unique, distinct=len(unique),
                duplicates=sum(bool(x) for x in raw.values())-len(unique), mutable=pool)


def support(view, sampler):
    """Same kNN/count audit; after booking, full HIST suffixes are naturally empty.

    GAP's PublicPrefix forbids clock>booking_horizon for *drawing*. Natural SAA
    still has legal return/service decision phases after booking. This local
    diagnostic prefix validates against the public depot deadline instead; no
    sampler, booking contract, feature clock or future distribution is changed.
    """
    _, _, r, _, PublicPrefix = imports()
    from scenario_saa import ScenarioOrder
    r.validate_view(view)
    @dataclass(frozen=True)
    class NaturalPrefix(PublicPrefix):
        def free_slots(self):
            ids = [o.oid for o in self.orders]
            if (not 0 <= self.clock <= next(o.tw_end for o in view.orders if o.oid == 0)+r.EVENT_TOL
                    or len(ids) != len(set(ids)) or any(not 1 <= i <= 50 for i in ids)):
                raise ValueError('INVALID_NATURAL_PUBLIC_PREFIX')
            return 50-len(ids)
    prefix = NaturalPrefix(view.clock, view.booking_horizon, 50, tuple(
        ScenarioOrder(o.oid, o.reveal, o.x, o.y, o.demand, o.temp_class,
                      o.tw_start, o.tw_end, o.service) for o in view.orders if o.oid))
    return sampler.support(prefix)  # support ONLY; draw is never called


def physical_identity(vehicles):
    _, _, r, _, _ = imports()
    return r.identity([{k: v for k, v in asdict(x).items() if k != 'mutable_suffix'}
                       for x in vehicles])


def observe(engine, view, context, policy, sampler, history_ids):
    _, _, r, _, _ = imports()
    before = r.identity([view, policy.export_state()])
    _, clock, _, _, vehicles, _, served, _ = context
    if clock != view.clock or set(view.rejected) != policy._rejected or set(view.accepted) != policy._accepted:
        raise ValueError('POST_ADMISSION_LEDGER_MISMATCH')
    by_id = {v.vehicle_id: v for v in vehicles}
    for v in view.vehicles:
        raw = by_id[v.vehicle_id]
        for key in r.Vehicle.__dataclass_fields__:
            if key == 'ready_time':
                expected = max(clock, raw.ready_time) if raw.status in ('idle', 'ready') else raw.ready_time
            elif key == 'mutable_suffix':
                expected = tuple(c for c in policy._plan.get(v.vehicle_id, ())
                                 if not served[c] and c != raw.committed_next)
            elif key == 'served_route':
                expected = tuple(raw.served_route)
            else:
                expected = getattr(raw, key)
            if getattr(v, key) != expected:
                raise ValueError('FROZEN_PHYSICAL_PREFIX_OR_P0_MISMATCH: '+key)
    plans = r.plans_from_view(view)
    proposals = raw_mask_proposals(view, plans)
    audit = support(view, sampler)
    gate = r.reference_check(view, plans)
    if not all(r.route_check(view, p) for p in plans.values()):
        raise ValueError('CONSERVATIVE_CAPACITY_TW_OR_RETURN_DOMAIN_FAILED')
    if before != r.identity([view, policy.export_state()]):
        raise ValueError('OBSERVER_MUTATED_STATE')
    return dict(clock=view.clock, public_identity=r.identity(view), public_view=view,
                ledger={'accepted': view.accepted, 'rejected': view.rejected, 'served': view.served},
                masks=proposals, future_support=dict(audit,
                    nearest_day_ids=[history_ids[i] for i in audit['nearest']],
                    compatible_day_ids=[history_ids[i] for i in audit['eligible']]),
                multi=proposals['distinct'] >= 2, supported=len(audit['eligible']) >= 2,
                joint=proposals['distinct'] >= 2 and len(audit['eligible']) >= 2,
                reference_usable=gate['ok'] and proposals['distinct'] >= 2 and len(audit['eligible']) >= 2,
                P0_reference_check=gate, immutable_observer=True,
                frozen_commitments_and_ledger='PASS', conservative_capacity='capacity-1e-4',
                post_booking=view.clock >= view.booking_horizon-r.EVENT_TOL,
                compatible_suffixes_all_empty=bool(audit['eligible']) and
                    all(audit['suffix_counts'][i] == 0 for i in audit['eligible']),
                controller_signature=policy.spec.signature,
                timeouts_so_far=policy.timeouts, admission_checks=policy.admission_checks)


def make_spec():
    np, t, _, _, _ = imports()
    import p2_pos01c_r0_prepare as prep
    from scenario_saa import make_c0_contract_v2, build_history
    reg, manifest = prep.verify()
    descriptors = [d for d in manifest['days'] if d['role'] == 'HIST_TRAIN']
    hist = []
    for d in descriptors:
        with np.load(ROOT/d['relative_path'], allow_pickle=False) as z:
            hist.extend(build_history(z))
    spec = t.SaaSpec(contract=make_c0_contract_v2(), history=tuple(tuple(d) for d in hist),
        booking_horizon=16., budget=186.06639500089943, capacity=50., tw_speed=1.5,
        K=4, n_neighbors=10, time_limit=6., cooling_share=4.452, reject_penalty=(5.,10.,15.),
        stream_id=b.STREAM_ID, stream_seed=2026101903, shadow_mode='ls',
        energy_pricing='marginal', standby_orders=8.)
    old = json.loads((b.OUT/'registration.json').read_text())
    if (len(hist) != 16 or spec.contract.contract_hash != reg['contract_hash'] or
            b.sha(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt') != old['checkpoint_sha256']):
        raise ValueError('FROZEN_CONTRACT_HISTORY_OR_CHECKPOINT_IDENTITY_DRIFT')
    assert old['B_ref_kwh'] == spec.budget and old['cooling_share'] == spec.cooling_share
    assert old['SAA'] == dict(K=4, event_time_limit_s=6, n_neighbors=10, shadow_mode='ls',
        energy_pricing='marginal', standby_orders=8, future_policy='reveal', incr_eval=False,
        stream_id=b.STREAM_ID, stream_seed=2026101903)
    return spec, manifest, [d['day_id'] for d in descriptors]


def run_day(day, emit, spec, manifest, history_ids):
    np, t, r, Proposal, _ = imports()
    descriptor = [d for d in manifest['days'] if d['role'] == 'SOURCE_TRAIN'][day]
    with np.load(ROOT/descriptor['relative_path'], allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if t.dataset_digest(data) != descriptor['dataset_sha256']:
        raise ValueError('SOURCE_DATA_IDENTITY_DRIFT')
    sampler = Proposal(spec.history, n_neighbors=10)
    current = {'ordinal': -1, 'observation': None, 'previous_accepted': set(), 'previous_rejected': set()}
    class ObservedPolicy(t.PublicSaaReplanner):
        def on_reveal(self, engine, inst, clock, vehicles, served, visible):
            started = time.monotonic()
            old_checks, old_timeouts = len(self.admission_checks), self.timeouts
            old_a, old_r = set(self._accepted), set(self._rejected)
            try:
                return super().on_reveal(engine, inst, clock, vehicles, served, visible)
            finally:
                emit('ADMISSION', dict(day=day, event_id=engine.event_id, clock=clock,
                    elapsed_s=time.monotonic()-started, timeouts=self.timeouts-old_timeouts,
                    accepted=sorted(self._accepted-old_a), rejected=sorted(self._rejected-old_r),
                    reference_checks=self.admission_checks[old_checks:]))

        def plan(self, engine, inst, clock, vehicles, served, visible, replan_ids=None):
            current['ordinal'] += 1
            ordinal = current['ordinal']
            current['observation'] = None
            emit('BEGIN', dict(day=day, ordinal=ordinal, event_id=engine.event_id, clock=clock))
            started = time.monotonic()
            physical = physical_identity(vehicles)
            try:
                answer = super().plan(engine, inst, clock, vehicles, served, visible, replan_ids)
                if physical != physical_identity(vehicles):
                    raise ValueError('PLANNER_CHANGED_FROZEN_PHYSICAL_STATE')
                if current['observation'] is None:
                    raise ValueError('NATURAL_POST_ADMISSION_PHASE_NOT_OBSERVED')
                if (not current['previous_accepted'] <= self._accepted or
                        not current['previous_rejected'] <= self._rejected):
                    raise ValueError('IRREVOCABLE_LEDGER_REGRESSION')
                current['previous_accepted'] = set(self._accepted)
                current['previous_rejected'] = set(self._rejected)
                row = current['observation']
                row.update(day=day, ordinal=ordinal, event_id=engine.event_id, status='COMPLETE',
                    plan_including_observer_s=time.monotonic()-started)
                emit('PHASE', row)
                return answer
            except BaseException as exc:
                emit('PHASE_ERROR', dict(day=day, ordinal=ordinal, clock=clock,
                    partial_observation=current['observation'],
                    error=type(exc).__name__+': '+str(exc), elapsed_s=time.monotonic()-started))
                raise
    policy = ObservedPolicy(spec)
    env = t.SaaPhaseEnv(data, 50., 1.5, num_vehicles=15, replanner=policy,
                        coldchain_contract=spec.contract, booking_horizon=16.)
    def hook(engine, view, context):
        started = time.monotonic()
        row = observe(engine, view, context, policy, sampler, history_ids)
        row['observer_s'] = time.monotonic()-started
        current['observation'] = row
    env.p2_capture = hook
    traces, served = env.run(0)  # exactly one natural execution; no evaluation call
    sequence = [s.node for trace in traces for s in trace.services]
    if (set(sequence) != policy._accepted or len(sequence) != len(set(sequence)) or
            policy._accepted & policy._rejected or
            policy._accepted | policy._rejected != set(range(1,51))):
        raise ValueError('TERMINAL_SERVICE_OR_ADMISSION_LEDGER_INCOMPLETE')
    return dict(day=day, day_id=descriptor['day_id'], status='COMPLETE',
                source_sha256=descriptor['file_sha256'], root_seed=descriptor['root_seed'],
                phase_attempts=current['ordinal']+1, accepted=sorted(policy._accepted),
                rejected=sorted(policy._rejected), served=sequence, timeouts=policy.timeouts,
                draw_audit=policy.draw_audit, admission_checks=policy.admission_checks,
                terminal_utility='NOT_EVALUATED', terminal_labels=0)


def worker(conn, gate):
    gate.wait()
    sink = open(b.os.devnull, 'w')
    sys.stdout = sys.stderr = sink
    try:
        spec, manifest, history_ids = make_spec()
        b.send(conn, 'READY', dict(signature=spec.signature, contract_hash=spec.contract.contract_hash,
            SaaSpec={k:b.plain(v) for k,v in asdict(spec).items() if k not in ('history','contract')}))
        def emit(kind, row):
            b.send(conn, kind, row)
            ack, _, _ = b.receive(conn)
            if ack != 'ACK':
                raise ValueError('SUPERVISOR_REFUSED_OBSERVATION')
        for day in range(2):
            emit('DAY_BEGIN', dict(day=day))
            result = run_day(day, emit, spec, manifest, history_ids)
            emit('DAY_END', result)
        b.send(conn, 'DONE', {})
    except BaseException as exc:
        b.send(conn, 'ERROR', dict(error=type(exc).__name__+': '+str(exc),
                                 traceback=traceback.format_exc()[-12000:]))


def preserved():
    old = b.preserved()
    for folder in (b.OUT, ROOT/'research/p2_c_gap_01_20261019'):
        old.update({p.relative_to(WORKSPACE).as_posix(): b.sha(p)
                    for p in folder.rglob('*') if p.is_file()})
    return old


def summaries(rows, days):
    result = []
    for day in range(2):
        seen = [v for (d,_),v in rows.items() if d == day]
        valid = [v for v in seen if v.get('status') == 'COMPLETE']
        n = len(seen)
        counts = dict(no_mask=sum(v['masks']['distinct']==0 for v in valid),
            single_mask=sum(v['masks']['distinct']==1 for v in valid),
            multi_mask=sum(v['multi'] for v in valid),
            NO_SUPPORT=sum(v['future_support']['status']=='NO_SUPPORT' for v in valid),
            LOW_SUPPORT=sum(v['future_support']['status']=='LOW_SUPPORT' for v in valid),
            sufficient_support=sum(v['supported'] for v in valid), joint=sum(v['joint'] for v in valid),
            reference_usable_joint=sum(v['reference_usable'] for v in valid),
            P0_reference_rejections=sum(not v['P0_reference_check']['ok'] for v in valid),
            post_booking=sum(v.get('post_booking',False) for v in valid),
            compatible_suffixes_all_empty=sum(v.get('compatible_suffixes_all_empty',False) for v in valid),
            incomplete=n-len(valid))
        complete = days[day]['status'] == 'COMPLETE' and not counts['incomplete']
        rates = {key:counts[num]/n if n and complete else None for key,num in
                 (('p_multi','multi_mask'),('p_support','sufficient_support'),('p_joint','joint'))}
        result.append(dict(day=day, status=days[day]['status'], observed_denominator=n,
            complete_root_day=complete, counts=counts, rates=rates,
            partial_observed_rates={key:counts[num]/n if n else None for key,num in
                (('p_multi','multi_mask'),('p_support','sufficient_support'),('p_joint','joint'))},
            unobserved_future_phase_count='UNKNOWN_NOT_ZERO' if not complete else 0))
    return result


def execute():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():
        raise RuntimeError('SINGLE_USE_ID_ALREADY_ATTEMPTED; NO_RETRY')
    store = b.Store(OUT, cap=MAX_OUTPUT)
    started, deadline = time.monotonic(), time.monotonic()+1200.
    registration = dict(run_id=OUT.name, baseline=BASELINE, qualification='REFERENCE_ONLY',
        preserved_sha256=preserved(), source_sha256=b.sources(),
        checkpoint_sha256=b.sha(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt'),
        old_configuration='p2_pos01c_smoke_20261019_v2/registration.json; unchanged; same random stream',
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
