"""Independent fresh-instance regeneration, route/manifest replay and paired CI.

Does not import the research runner or reuse its gate decision.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parents[1]/'C-VRP_Cold-chainVehicleRoutingProblem'
for d in ('delivery','coldchain'):sys.path.insert(0,str(E/'scripts'/d))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def regenerate(seed,i):
    rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
    x={'coords':np.r_[np.array([[5.,5.]]),rng.uniform(0,10,(50,2))].tolist(),
       'demands':np.r_[0,rng.integers(1,6,50)].tolist(),'temp_class':np.r_[0,rng.integers(0,3,50)].tolist(),
       'reveal_time':[0.]*51,'ready_time':[0.]*51,'tw_start':[0.]*51,'tw_end':[8.]*51,'service_time':[0.]+[.03]*50}
    return x,[list(range(1+10*k,11+10*k)) for k in range(5)]
def edges(routes):
    ans=set()
    for r in routes:
        chain=[0]+r+[0]
        ans.update(tuple(sorted((a,b))) for a,b in zip(chain,chain[1:]) if a!=b)
    return ans

def equivalent(left,right):
    """Exact structure/discrete values, 1e-9 absolute tolerance for CPU floats.

    Cross-platform QA first failed byte equality on 54 energy values whose
    largest difference was 2.78e-16; retained diagnostic precedes this repair.
    This does not modify costs, hard feasibility, data or the numerical gate.
    """
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(equivalent(left[k],right[k]) for k in left)
    if isinstance(left,list):
        return isinstance(right,list) and len(left)==len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    if type(left) is float or type(right) is float:
        return type(left) in (int,float) and type(right) in (int,float) and math.isfinite(left) and math.isfinite(right) and math.isclose(left,right,rel_tol=0.,abs_tol=1e-9)
    return type(left) is type(right) and left==right

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    folder=Path(args.directory);result=json.loads((folder/'result.json').read_text(encoding='utf-8'));reg=json.loads((folder/'PROTOCOL.json').read_text(encoding='utf-8'))
    checks=[]
    def check(name,ok):checks.append({'name':name,'pass':bool(ok)})
    c=DeliveryContract();rows=result['rows'];check('32_unique_exact_instances',[r['instance_id'] for r in rows]==list(range(32)))
    identity=json.loads((folder/'identity_start.json').read_text())
    check('actual_contract',identity['contract_sha256']==c.identity())
    check('actual_protocol',identity['protocol_sha256']==hashlib.sha256((folder/'PROTOCOL.json').read_bytes()).hexdigest())
    check('registered_immutable_fields',reg['steps']==8 and reg['time_limit_s']==10 and reg['guide_weight']==1 and reg['mask_fraction']==.3 and reg['bootstrap_repetitions']==2000 and reg['bootstrap_seed']==2026101077)
    check('actual_checkpoint',identity['checkpoint_sha256']=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db')
    check('actual_cpu_threads',identity['cpu_affinity']==[55] and all(v=='1' for v in identity['threads'].values()))
    manifest=json.loads((folder/'SOURCE_MANIFEST.json').read_text())
    check('actual_manifest',identity['source_manifest_sha256']==digest(manifest))
    for name in ('scripts/delivery/delivery_contract.py','scripts/delivery/delivery_evaluator.py','scripts/coldchain/coldchain_state.py','scripts/coldchain/coldchain_contract.py'):
        check('local_replay_source_'+name,name in manifest and hashlib.sha256((E/name).read_bytes()).hexdigest()==manifest[name])
    check('new_data_seed',reg['dataset_seed']==2026101002)
    check('identity_start_end',json.loads((folder/'identity_start.json').read_text())==json.loads((folder/'identity_end.json').read_text()))
    check('actual_source_identity',json.loads((folder/'INDEPENDENT_ACTUAL_END_IDENTITY.json').read_text())['match'] is True)
    final=[];first=[];qualified=True
    for row in rows:
        i=row['instance_id'];data,initial=regenerate(reg['dataset_seed'],i)
        check(f'{i}_data_bytes',digest([data,initial])==row['data_initial_sha256'] and data==row['data'] and initial==row['initial_routes'])
        rng=np.random.default_rng(np.random.SeedSequence([reg['mask_seed'],i]));schedule=[sorted(map(int,rng.choice(np.arange(1,51),15,replace=False))) for _ in range(8)]
        check(f'{i}_mask_schedule',schedule==row['schedule'] and digest(schedule)==row['schedule_sha256'])
        order=list(np.random.default_rng(np.random.SeedSequence([reg['arm_order_seed'],i])).permutation(['P','R','H']))
        check(f'{i}_arm_order',order==row['arm_order'])
        init=evaluate(data,initial,c);check(f'{i}_initial_hard',init['hard'] is True)
        costs=[];early=[]
        for arm in ('P','R','H'):
            a=row['arms'][arm];last_routes=initial;last_cost=init['cost'];check(f'{i}_{arm}_8logs',[s['step'] for s in a['attempts']]==list(range(1,9)))
            expected_first=last_cost
            for s,mask in zip(a['attempts'],schedule):
                label=f'{i}_{arm}_{s["step"]}'
                check(label+'_common_mask',s['removed']==mask and s['mask_sha256']==digest(mask))
                check(label+'_before_incumbent',s['incumbent_before']==last_routes and math.isclose(s['cost_before'],last_cost,abs_tol=1e-8))
                retained={e for e in edges(last_routes) if e[0] not in mask and e[1] not in mask}
                check(label+'_retained_edges',retained=={tuple(e) for e in s['retained_edges']})
                if s['candidate_complete']:
                    candidate=s['candidate_routes'];replayed=evaluate(data,candidate,c)
                    check(label+'_retained_preserved',retained.issubset(edges(candidate)))
                    if s['candidate_hard']:
                        check(label+'_candidate_physical_hard',replayed['hard'] is True)
                        check(label+'_candidate_cost',math.isclose(replayed['cost'],s['candidate_report']['cost'],abs_tol=1e-8))
                        for key in ('distance_km','energy_kwh','quality_loss_value'):
                            check(label+'_'+key,math.isclose(replayed[key],s['candidate_report'][key],abs_tol=1e-8))
                        check(label+'_manifest',equivalent(replayed['delivered'],s['candidate_report']['delivered']))
                        check(label+'_trajectory',equivalent(replayed['traces'],s['candidate_report']['traces']))
                    if s['accepted']:
                        check(label+'_legal_accept',s['candidate_hard'] is True and replayed['hard'] is True and replayed['cost']<last_cost)
                        last_routes=candidate;last_cost=replayed['cost']
                else:check(label+'_not_accept_incomplete',s['accepted'] is False)
                check(label+'_after_incumbent',s['incumbent_after']==last_routes and math.isclose(s['cost_after'],last_cost,abs_tol=1e-8))
                if s['step']==1:expected_first=last_cost
            submitted=evaluate(data,a['routes'],c)
            check(f'{i}_{arm}_submit_hard',submitted['hard'] is True)
            check(f'{i}_{arm}_submit_components',all(math.isclose(submitted[k],a['report'][k],abs_tol=1e-8) for k in ('cost','distance_km','energy_kwh','quality_loss_value')))
            check(f'{i}_{arm}_submit_manifest',equivalent(submitted['delivered'],a['report']['delivered']))
            check(f'{i}_{arm}_submit_trajectory',equivalent(submitted['traces'],a['report']['traces']))
            check(f'{i}_{arm}_timeout_policy',a['routes']==(initial if a['timeout'] else last_routes))
            check(f'{i}_{arm}_NN_activation',a['decode_calls']==(0 if arm=='H' else a['actual_attempts']))
            check(f'{i}_{arm}_first_endpoint',math.isclose(a['first_endpoint_cost'],expected_first,abs_tol=1e-8))
            qualified &= a['fixed_schedule_qualified'] is True
            check(f'{i}_{arm}_reported_schedule',a['fixed_schedule_qualified']==(a['actual_attempts']==8 and not a['timeout']))
            check(f'{i}_{arm}_endpoint',math.isclose(a['final_endpoint_cost'],submitted['cost'],abs_tol=1e-8))
            costs.append(submitted['cost']);early.append(a['first_endpoint_cost'])
        final.append(costs);first.append(early)
    check('all_8_actual_attempts_no_deadline',qualified)
    final=np.asarray(final,float);first=np.asarray(first,float);check('finite_all_costs',np.isfinite(final).all() and np.isfinite(first).all())
    rng=np.random.default_rng(reg['bootstrap_seed']);idx=rng.integers(0,len(final),(reg['bootstrap_repetitions'],len(final)))
    comparisons={}
    for j,name in ((1,'P_over_R'),(2,'P_over_H')):
        gain=(final[:,j]-final[:,0])/final[:,j];samples=gain[idx].mean(1);ci=np.quantile(samples,[.025,.975]).tolist()
        comparisons[name]={'mean_relative_gain':float(gain.mean()),'descriptive_95_CI':ci,'exploratory_threshold_met':bool(gain.mean()>=.01 and ci[0]>0)}
    output={'checks':checks,'independently_replayed_routes':len(rows)*3,'candidate_replay_included':True,'all_qualification_checks_pass':all(x['pass'] for x in checks),
        'mean_final_costs':dict(zip(('P','R','H'),final.mean(0).tolist())),'mean_first_costs':dict(zip(('P','R','H'),first.mean(0).tolist())),
        'comparisons':comparisons,'exploratory_attribution_clue':all(x['exploratory_threshold_met'] for x in comparisons.values()) and all(x['pass'] for x in checks),
        'online_gain_established':False,'stable_gain_established':False,'scope':'n32_exploratory_fixed_attempts_snapshot_not_confirmation_or_online'}
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k!='checks'}));print('failed_checks',sum(not x['pass'] for x in checks))
if __name__=='__main__':main()
