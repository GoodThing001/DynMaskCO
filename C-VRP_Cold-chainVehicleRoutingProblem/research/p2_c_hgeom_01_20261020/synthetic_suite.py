"""Constructed fixtures only. Never load official phases, views or checkpoints."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import dataclasses
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import psutil
from source_bridge import build
from recovery import enumerate_closure, structure, anchors
from geometry import residual, primary, relation, package_gate
from journal import Journal, atomic

HERE = Path(__file__).resolve().parent
DENIED_REAL_INPUTS = []


def input_guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).replace('\\', '/').lower()
        if name.endswith(('/phases.json','.npz','.ckpt')):
            DENIED_REAL_INPUTS.append(name)
            raise PermissionError('PHASE1_REAL_INPUT_FORBIDDEN')


def fixture(ref, *, shifted=0., committed=False):
    orders = tuple(ref.Order(i, x+shifted if i else 0., y, 0. if not i else 1.,
                             0., 100., 0., 0, 1., 0.)
                   for i, (x,y) in enumerate([(0.,0.),(.2,.1),(.7,.2),(.1,.5)]))
    if committed:
        vehicles = (ref.Vehicle(0, status='committed', committed_next=1,
                               committed_arrive=.2, committed_finish=.5, mutable_suffix=(2,)),
                    ref.Vehicle(1, mutable_suffix=(3,)), ref.Vehicle(2, status='closed'))
    else:
        vehicles = (ref.Vehicle(0, mutable_suffix=(1,2)), ref.Vehicle(1, mutable_suffix=(3,)),
                    ref.Vehicle(2, status='closed'))
    return ref.PublicView(orders, vehicles, (1,2,3), (), (), (0,1), (), 0., 100., 50., 1., 1000.,
                          NS(thermal=NS(supported_temp_classes=(0,1,2))))


def independent_closure(ac, ref, view, base, mask, allowed):
    """Separate exhaustive implementation: order permutations, recursive inserts.

    Does not call enumerate_closure, shared legal_actions, structure or hash-only
    dedup. This oracle checks the synthetic relation's full terminal set.
    """
    result = {}
    def key(plans):
        return tuple((vid,p.anchor_node,float(p.anchor_time).hex(),float(p.anchor_load).hex(),tuple(p.suffix))
                     for vid,p in sorted(plans.items()))
    def rec(plans, seq):
        if not seq:
            result[key(plans)] = plans
            return
        c = seq[0]
        candidates,_ = ac.enumerate_actions_from_plans(ref.public_env(view),0,plans,c,
                                                       allowed_vehicle_ids=tuple(sorted(allowed)))
        for cand in candidates:
            trial = ac.apply_action(plans,cand.action,allowed_vehicle_ids=tuple(sorted(allowed)))
            if cand.feasible and all(ref.route_check(view,p) for p in trial.values()):
                rec(trial,seq[1:])
    partial = {vid:dataclasses.replace(p,suffix=tuple(c for c in p.suffix if c not in mask))
               for vid,p in base.items()}
    for order in itertools.permutations(mask):
        rec(partial,order)
    return result


def run(output):
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('refuse overwrite synthetic run')
    output.mkdir(parents=True)
    ac,ref,shared,bridge=build()
    rows=[]
    def record(name, fn):
        start=time.monotonic()
        try:
            detail=fn()
            rows.append(dict(name=name,ok=True,detail=detail,wall_s=time.monotonic()-start))
        except Exception as exc:
            rows.append(dict(name=name,ok=False,error=type(exc).__name__+': '+str(exc),
                             wall_s=time.monotonic()-start))
        atomic(output/'progress.json',dict(checks=rows,synthetic_only=True))

    view=fixture(ref); base=shared.plans_of(view); allowed=shared.shared_allowed(view); mask=(1,2)
    oracle=independent_closure(ac,ref,view,base,mask,allowed)
    targets=sorted({ac.plan_hash(p) for p in oracle.values()})
    def checker(v,b):
        return lambda p: shared.check_plan(v,p,0,shared.anchor_of(b,0),
                                           int(b[0].anchor_node),set(v.accepted)-set(v.served))
    def closure(**kw):
        return enumerate_closure(shared,view,base,mask,allowed,targets,check=checker(view,base),**kw)
    full=closure()
    def verify(condition, detail=None):
        assert condition, detail
        return detail
    record('bruteforce_independent_synthetic_closure',lambda:verify(
        full['closure_exhausted'] and {r['structure'] for r in full['records']}==set(oracle),
        dict(structures=len(oracle),expanded=full['enumeration_cost']['expanded_states'])))
    record('empty_suffix_and_closed_vehicle',lambda:verify(
        2 not in base and residual({0:ac.VehiclePlan(0,1,0.,0.,())},ref.public_env(view).dist_mat)
        ==float(ref.public_env(view).dist_mat[0,1,0])))
    record('cross_vehicle_reassignment',lambda:verify(any(1 in r['plans'][1].suffix for r in full['records'])))
    def exact():
        v=fixture(ref,committed=True);b=shared.plans_of(v)
        result=enumerate_closure(shared,v,b,(2,),shared.shared_allowed(v),[],check=checker(v,b))
        good=all(anchors(x['plans'])==anchors(b) and all(1 not in p.suffix for p in x['plans'].values())
                 for x in result['records'])
        p=dict(base);p[0]=dataclasses.replace(p[0],suffix=(1,1,2))
        return verify(good and not checker(view,base)(p)[0])
    record('exact_once_and_all_anchors',exact)
    def state_binding():
        other=fixture(ref,shifted=2.)
        p=base
        return verify(ac.plan_hash(p)==ac.plan_hash(p) and
                      residual(p,ref.public_env(view).dist_mat)!=residual(p,ref.public_env(other).dist_mat))
    record('same_hash_different_public_state',state_binding)
    def collision():
        result=enumerate_closure(shared,view,base,mask,allowed,['same'],check=checker(view,base),
                                 hash_fn=lambda p:'same')
        return verify(len(result['matches']['same'])==len(oracle) and not result['uniqueness_proved']
                      and 'RECOVERY_NOT_UNIQUE' in result['failure_codes'])
    record('collision_retains_distinct_structures',collision)
    def found_not_unique():
        first=closure(cap=1)['records'][0]['plan_hash']
        r=enumerate_closure(shared,view,base,mask,allowed,[first],check=checker(view,base),cap=1)
        return verify(r['targets_found']==r['n_targets']==1 and not r['uniqueness_proved']
                      and not r['closure_exhausted'])
    record('all_targets_found_not_exhausted',found_not_unique)
    def cap_test():
        capped=closure(cap=1)
        exact=enumerate_closure(shared,view,base,(),allowed,[shared.signature(base)],
                               check=checker(view,base),cap=1)
        return verify(capped['closure_status']=='CAPPED' and not capped['closure_exhausted']
                      and exact['closure_status']=='EXHAUSTED')
    record('cap_exact_boundary_and_unfinished_frontier',cap_test)
    def missing():
        args=(shared,view,base,mask,allowed,['absent'])
        a=enumerate_closure(*args,check=checker(view,base))
        b=enumerate_closure(*args,check=checker(view,base),cap=1)
        return verify('RECOVERY_NOT_FOUND' in a['failure_codes'] and
                      'TARGET_NOT_OBSERVED_IN_PARTIAL_CLOSURE' in b['failure_codes'])
    record('missing_in_complete_vs_partial',missing)
    record('source_route_relation_and_action_parity',lambda:verify(
        all(ref.route_check(view,p) for r in full['records'] for p in r['plans'].values())
        and not any(x in sys.modules for x in ('jax','flax','torch','mpre_policy'))))
    def nonfinite():
        matrix=ref.public_env(view).dist_mat.copy();matrix[0,0,1]=math.nan
        caught=False
        try:residual({0:ac.VehiclePlan(0,0,0.,0.,(1,))},matrix)
        except ValueError:caught=True
        return verify(caught and primary(0.,[0.])['ratio'] is None and
                      relation(0.,0.)['relative_gap'] is None)
    record('nonfinite_distance_and_zero_denominator',nonfinite)
    record('one_percent_exact_boundary',lambda:verify(primary(99.,[100.])['improve']
        and not primary(math.nextafter(99.,math.inf),[100.])['improve']))
    record('enum_positive_zero_negative_gap',lambda:verify(
        [relation(x,1.)['distance_relation'] for x in (1.001,1.,.999)]==
        ['ENUM_STRICTLY_BETTER','ENUM_TIE','CONSISTENCY_ERROR'] and
        relation(1.001,1.)['within_one_percent']))
    def sum_test():
        p={2:ac.VehiclePlan(2,0,0.,0.,(1,2)),0:ac.VehiclePlan(0,0,0.,0.,(3,))}
        matrix=np.zeros((1,4,4));matrix[0,0,3]=1e16;matrix[0,3,0]=1.
        matrix[0,0,1]=1.;matrix[0,1,2]=1.
        return verify(residual(p,matrix)==math.fsum([1e16,1.,1.,1.,0.]) and
                      residual(p,matrix)==residual(dict(reversed(list(p.items()))),matrix))
    record('fsum_order_and_no_rounding',sum_test)
    record('primary_set_membership',lambda:verify(primary(10.,[12.,9.])['delta']==1.))
    def gate():
        rs=[dict(unit_key=str(i),n_targets=7 if i<9 else 6,targets_found=7 if i<9 else 6,
                 uniqueness_proved=True,closure_exhausted=True,legal=True,failure_codes=[]) for i in range(19)]
        # 9*7+10*6 =123, including all19 slots; deletion or any failed flag invalidates.
        assert package_gate(rs)
        rs[0]['uniqueness_proved']=False
        return verify(not package_gate(rs) and not package_gate(rs[1:]))
    record('package_gate_all_19_and_123',gate)
    def journal_fault():
        j=Journal(output/'journal_fault',[str(i) for i in range(19)])
        j.append('0','START');j.append('0','CANDIDATE',structure='synthetic')
        with j.path.open('ab') as f:f.write(b'{broken')
        atomic(j.root/'candidate.json',dict(complete=True))
        (j.root/'candidate.json.tmp').write_bytes(b'{partial')
        pm=j.postmortem(no_live_workers_confirmed=True)
        return verify(len(pm['slots'])==19 and pm['damaged_tail'] and
                      pm['slots'][0]['failure_code']=='PROCESS_INTERRUPTED' and
                      json.loads((j.root/'candidate.json').read_text())['complete'])
    record('journal_atomic_interruption',journal_fault)

    def native(name, mode, *, wall=8.,unit=6.,limit=96*1024**2):
        folder=output/name
        cmd=[sys.executable,'-B',str(HERE/'supervisor.py'),'--root',str(folder),
             '--mode',mode,'--wall',str(wall),'--unit-wall',str(unit),'--limit',str(limit)]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=wall+5)
        result=json.loads((folder/'result.json').read_text())
        result['process_exit']=p.returncode
        return result
    native_results={}
    def native_case(name,mode,expected,**kw):
        receipt=native(name,mode,**kw);native_results[name]=receipt
        assert receipt['failure_code']==expected,(receipt,expected)
        assert receipt['worker_exited'] and len(receipt['slots']['slots'])==19
        return dict(receipt=str((output/name/'result.json').relative_to(HERE)),
                    native=receipt['native'],elapsed_s=receipt['elapsed_s'])
    record('worker_exception',lambda:native_case('exception','exception','PROCESS_EXCEPTION'))
    record('unit_timeout_native_block',lambda:native_case('unit_timeout','hang','UNIT_TIMEOUT',wall=5.,unit=2.))
    record('package_timeout_not_attempted_denominator',lambda:native_case('package_timeout','hang','PACKAGE_TIMEOUT',wall=2.,unit=5.))
    record('native_memory_limit_and_failed_allocation',lambda:native_case('memory','memory','MEMORY_LIMIT_EXCEEDED'))
    def rss_scope():
        receipt=native('mapped','mapped');native_results['mapped']=receipt
        return verify(receipt['status']=='OK' and receipt['peak_rss_sum']<=96*1024**2
                      and receipt['native']['flags']&0x200 and
                      receipt['native']['supervisor_hard_ws']['flags']&4 and
                      receipt['native']['active_process_limit']==2,
                      dict(receipt=str((output/'mapped/result.json').relative_to(HERE)),
                           peak_rss_sum=receipt['peak_rss_sum'],peak_private_sum=receipt['peak_private_sum']))
    record('supervisor_plus_worker_rss_and_commit_scope',rss_scope)
    def threads():
        receipt=native('child','child')
        started=json.loads((output/'child/worker_started.json').read_text())
        return verify(receipt['status']=='OK' and (output/'child/child_refused.json').exists()
                      and started['threads']=='1',dict(refused_extra_process=True))
    record('single_worker_and_numeric_threads',threads)

    def forced_kill(supervisor_kill):
        folder=output/('kill_supervisor' if supervisor_kill else 'kill_worker')
        proc=subprocess.Popen([sys.executable,'-B',str(HERE/'supervisor.py'),'--root',str(folder),
                               '--mode','hang','--wall','8','--unit-wall','6'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.monotonic()+5
        while not (folder/'worker_started.json').exists():
            assert time.monotonic()<deadline,'worker failed to start'
            time.sleep(.01)
        pid=json.loads((folder/'worker_started.json').read_text())['pid']
        if supervisor_kill:proc.kill()
        else:psutil.Process(pid).kill()
        proc.wait(timeout=3)
        deadline=time.monotonic()+2
        while psutil.pid_exists(pid) and time.monotonic()<deadline:time.sleep(.01)
        assert not psutil.pid_exists(pid),'worker survived supervisor death'
        j=Journal(folder,[],create=False)
        receipt=j.postmortem(no_live_workers_confirmed=True)
        atomic(folder/'postmortem.json',receipt)
        assert len(receipt['slots'])==19 and not receipt['computation_resumed']
        return dict(worker_pid=pid,worker_dead=True,postmortem=str((folder/'postmortem.json').relative_to(HERE)))
    record('worker_hard_kill',lambda:forced_kill(False))
    record('supervisor_hard_kill_kills_descendants',lambda:forced_kill(True))
    def drift():
        # Alter only a copied expected digest, never source files or the T1.
        reg=json.loads((HERE/'registration.json').read_text(encoding='utf-8'))
        expected=dict(reg['seven_elements']['identity_and_delivery']['existing_source_hashes'])
        expected['C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/action_contract.py']='mismatch'
        try:build(expected_hashes=expected)
        except ValueError as exc:return verify('IDENTITY_MISMATCH' in str(exc))
        raise AssertionError('source identity drift accepted')
    record('identity_drift_before_work',drift)
    def formal_guard():
        p=subprocess.run([sys.executable,'-B',str(HERE/'supervisor.py'),'--formal'],
                          capture_output=True,text=True,timeout=5)
        return verify(p.returncode!=0 and 'FORMAL_EVALUATION_NOT_AUTHORISED' in p.stderr)
    record('formal_evaluation_refused_before_data_access',formal_guard)
    record('no_real_input_reads_during_synthetic',lambda:verify(
        not DENIED_REAL_INPUTS and not any(x in sys.modules for x in ('jax','flax','torch')),
        dict(audit_hook_installed=True,forbidden_read_attempts=len(DENIED_REAL_INPUTS))))
    record('read_only_postmortem_no_restart',lambda:verify(
        not Journal(output/'kill_supervisor',[],create=False).postmortem(
            no_live_workers_confirmed=True)['computation_resumed']))
    # Reserved audit name is answered by a separate executable, not this test file.
    result=dict(schema='hgeom_synthetic_regression_v1',synthetic_only=True,checks=rows,
                all_ok=all(r['ok'] for r in rows),n_checks=len(rows),n_ok=sum(r['ok'] for r in rows),
                bridge=bridge,formal_evaluations=0,checkpoint_loads=0,
                pending_independent_check='source_and_result_independent_recheck')
    atomic(output/'regression.json',result)
    print(json.dumps(dict(all_ok=result['all_ok'],n_checks=len(rows),n_ok=result['n_ok'],
                          failed=[r for r in rows if not r['ok']]),ensure_ascii=False))
    return result


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.addaudithook(input_guard)
    raise SystemExit(0 if run(sys.argv[1])['all_ok'] else 1)
