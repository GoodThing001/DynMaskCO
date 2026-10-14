"""Independently regenerate32 instances and replay96 finals and768 candidates."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
from audit_eq8_research_20261010 import regenerate,edges,equivalent,digest,DeliveryContract,evaluate,E

ROOT=E/'results/research_20261010_continuation_v2'
RAW=ROOT/'delivery_raw_v1'

def main():
    folder=RAW/'pilot'; result=json.loads((folder/'result.json').read_text())
    protocol=json.loads((folder/'PROTOCOL.json').read_text());identity=json.loads((folder/'identity_start.json').read_text())
    rows=result['rows'];checks=[]
    def check(name,ok):checks.append({'name':name,'pass':bool(ok)})
    c=DeliveryContract();arms=('Pguided','Rguided','Pdiscard')
    check('32_fixed_unique_ids',[r['instance_id'] for r in rows]==list(range(17000,17032)))
    check('three_registered_arms',protocol['arms']==list(arms))
    expected={'dataset_seed':202610101701,'mask_seed':202610101702,'arm_order_seed':202610101704,'bootstrap_seed':202610101703,'steps':8,'mask_fraction':.5,'time_limit_s':10.,'random_model_seed':42,'bootstrap_repetitions':2000}
    check('independent_registered_values',all(protocol.get(k)==v for k,v in expected.items()))
    template=json.loads((RAW/'PRE_DATA_PILOT_TEMPLATE.json').read_text())
    check('only_actual_QA_hash_filled',{k:v for k,v in protocol.items() if k!='synthetic_qa_artifact_sha256'}=={k:v for k,v in template.items() if k!='synthetic_qa_artifact_sha256'})
    check('QA_hash',protocol['synthetic_qa_artifact_sha256']==hashlib.sha256((RAW/'qa/result.json').read_bytes()).hexdigest())
    check('actual_protocol',identity['protocol_sha256']==hashlib.sha256((folder/'PROTOCOL.json').read_bytes()).hexdigest())
    check('actual_contract',identity['contract_sha256']==c.identity())
    check('CPU54_threads1',identity['cpu_affinity']==[54] and all(v=='1' for v in identity['threads'].values()))
    check('weight_identity',identity['checkpoint_sha256']=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db' and identity['model_parameters']['P']['sha256']!=identity['model_parameters']['R']['sha256'])
    check('identity_start_end',identity==json.loads((folder/'identity_end.json').read_text()))
    manifest=json.loads((RAW/'SOURCE_MANIFEST.json').read_text())
    check('manifest_identity',identity['source_manifest_sha256']==digest(manifest))
    for rel,h in manifest.items():check('downloaded_source_'+rel,hashlib.sha256((RAW/'src'/rel).read_bytes()).hexdigest()==h)
    for rel in ['scripts/delivery/delivery_contract.py','scripts/delivery/delivery_evaluator.py','scripts/coldchain/coldchain_state.py','scripts/coldchain/coldchain_contract.py']:
        check('actual_local_replay_source_'+rel,hashlib.sha256((E/rel).read_bytes()).hexdigest()==manifest[rel])
    check('progress_result_exact',[json.loads(s) for s in (folder/'progress.jsonl').read_text().splitlines()]==rows)
    final=[];walls={a:[] for a in arms};decoded={a:0 for a in arms};timeouts={a:0 for a in arms};candidate_count=0
    for row in rows:
        iid=row['instance_id'];data,initial=regenerate(protocol['dataset_seed'],iid)
        check(f'{iid}_fresh_data',data==row['data'] and initial==row['initial_routes'] and digest([data,initial])==row['data_initial_sha256'])
        check(f'{iid}_arm_order',list(np.random.default_rng(np.random.SeedSequence([protocol['arm_order_seed'],iid])).permutation(list(arms)))==row['arm_order'])
        init=evaluate(data,initial,c);check(f'{iid}_initial',equivalent(init,row['initial_report']) and init['hard'])
        costs=[]
        for arm in arms:
            a=row['arms'][arm];routes=initial;cost=init['cost'];walls[arm].append(a['wall_s']);decoded[arm]+=a['decode_calls'];timeouts[arm]+=a['timeout']
            check(f'{iid}_{arm}_8_actual',len(a['attempts'])==8 and a['decode_calls']==8 and a['complete_candidates']==8 and a['hard_candidates']==8 and a['fixed_schedule_qualified'] and not a['timeout'] and a['wall_s']<10)
            for step,s in enumerate(a['attempts'],1):
                name=f'{iid}_{arm}_{step}';mask=s['removed'];removed=set(mask)
                check(name+'_step',s['step']==step and s['decode_calls']==1 and s['candidate_complete'] and s['candidate_hard'])
                check(name+'_mask',len(mask)==25 and len(removed)==25 and all(type(x) is int and 1<=x<=50 for x in mask) and digest(mask)==s['mask_sha256'])
                if arm=='Pdiscard':
                    expected=sorted(map(int,np.random.default_rng(np.random.SeedSequence([protocol['mask_seed'],iid,step])).choice(np.arange(1,51),25,False)))
                    check(name+'_uniform_exact_rng',mask==expected)
                check(name+'_role',s['guide_outputs_used'] is (arm!='Pdiscard') and s['logits_shape']==[51,51] and len(s['logits_sha256'])==64)
                check(name+'_before',routes==s['incumbent_before'] and math.isclose(cost,s['cost_before'],abs_tol=1e-9))
                full=np.zeros((51,51),dtype=np.float32)
                for u,v in edges(routes):full[u,v]=full[v,u]=1.
                check(name+'_decode_input',hashlib.sha256(full.tobytes()).hexdigest()==s['decode_input_adjacency_sha256'])
                retained={edge for edge in edges(routes) if not removed.intersection(edge)}
                adj=np.zeros((51,51),dtype=np.float32)
                for u,v in retained:adj[u,v]=adj[v,u]=1.
                check(name+'_retained_identity',retained=={tuple(e) for e in s['retained_edges']} and hashlib.sha256(adj.tobytes()).hexdigest()==s['adjacency_sha256'])
                candidate=s['candidate_routes'];replayed=evaluate(data,candidate,c);candidate_count+=1
                check(name+'_full_physical_replay',replayed['hard'] and equivalent(replayed,s['candidate_report']))
                check(name+'_keep_edges',retained.issubset(edges(candidate)) and s['retained_edges_preserved'])
                if s['accepted']:
                    check(name+'_legal_improvement',replayed['cost']<cost and s['elapsed_s']<10)
                    routes=candidate;cost=replayed['cost']
                check(name+'_after',routes==s['incumbent_after'] and math.isclose(cost,s['cost_after'],abs_tol=1e-9))
            submitted=evaluate(data,a['routes'],c)
            check(f'{iid}_{arm}_final',submitted['hard'] and equivalent(submitted,a['report']) and a['routes']==routes)
            costs.append(submitted['cost'])
        final.append(costs)
    final=np.asarray(final);idx=np.random.default_rng(protocol['bootstrap_seed']).integers(0,32,(2000,32));paired={}
    for j,label in [(2,'Pguided_over_Pdiscard'),(1,'Pguided_over_Rguided')]:
        d=(final[:,j]-final[:,0])/final[:,j];ci=np.percentile(d[idx].mean(1),[2.5,97.5]).tolist()
        pair={'mean_relative_gain':float(d.mean()),'descriptive_95_CI':ci}
        check(label+'_independent_mean_CI',equivalent(pair,{k:result['paired'][label][k] for k in pair}))
        paired[label]=pair
    failures=[x for x in checks if not x['pass']]
    out={'all_qualification_checks_pass':not failures,'check_count':len(checks),'failed_checks':failures,'full_checks':checks,
        'independent_instance_count':32,'independent_candidate_replays':candidate_count,'independent_final_replays':96,
        'mean_costs':dict(zip(arms,final.mean(0).tolist())),'mean_wall_seconds':{a:float(np.mean(v)) for a,v in walls.items()},
        'actual_decode_calls':decoded,'timeouts':timeouts,'paired':paired,
        'numeric_signal_established':not failures and all(p['mean_relative_gain']>=.01 and p['descriptive_95_CI'][0]>0 for p in paired.values()),
        'stable_online_gain_established':False,'MaskCO_attribution_established':False,
        'limitations':['Shared versioned C0 evaluator replay is independent of the runner, not independent physical calibration.',
        'Guided masks and logits SHA are recorded; full logits arrays absent, so their selection is not independently recomputed here.',
        'CPU float replay uses1e-9 absolute tolerance; all discrete identifiers and structure exact.',
        'Complete generator/model-load/end-identity evidence is separately audited; this is a32 snapshot destroy-role test, not trained masked generation or dynamic online confirmation.']}
    target=ROOT/'INDEPENDENT_DESTROY_EQ8_REPLAY.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(out,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k not in ('full_checks',)}))
    return 0 if not failures else 1

if __name__=='__main__':raise SystemExit(main())
