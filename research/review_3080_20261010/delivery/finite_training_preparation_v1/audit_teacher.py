"""Independent row regeneration and whole C0 replay; no model/teacher reruns."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from finite_common import sha,digest,dump,check_dependency_identity


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('root','upstream','registration','teacher-dir','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    reg=json.loads(a.registration.read_text(encoding='utf-8'))
    check_dependency_identity(reg,a.root,a.upstream)
    summary=json.loads((a.teacher_dir/'TEACHER_SUMMARY.json').read_text(encoding='utf-8'))
    archive=a.teacher_dir/'TEACHER_STATES.jsonl'
    if sha(archive)!=summary['teacher_archive_sha256']:
        raise ValueError('Archive identity mismatch')
    for folder in ('delivery','coldchain','research_20261010'):
        sys.path.insert(0,str(a.root/'scripts'/folder))
    from delivery_equal_schedule import generate,make_schedule
    from delivery_evaluator import evaluate
    from delivery_contract import DeliveryContract
    from delivery_maskco import partial_adjacency
    contract=DeliveryContract()
    rows=[json.loads(s) for s in archive.read_text(encoding='utf-8').splitlines()]
    if reg['status']=='REGISTERED_ENGINEERING_ONLY':
        ids=[900000,900001]
    else:
        ids=reg['train_instance_ids']
    expected={(i,j) for i in ids for j in range(4)}
    checks=[]
    def check(label,value): checks.append({'check':label,'pass':bool(value)})
    check('exact_unique_keys',len(rows)==len(expected) and {tuple(r['state_key']) for r in rows}==expected)
    counts={k:0 for k in ('positive','negative','small','invalid')}
    max_error=0.
    for row in rows:
        key=str(row['state_key'])
        data,incumbent=generate(summary['dataset_seed'],row['instance_id'])
        check(key+'source_integer_types',type(row['instance_id'])is int and type(row['mask_ordinal'])is int
              and all(type(v)is int for v in row['state_key']+row['mask']))
        check(key+'data',digest(data)==row['data_sha256'] and digest(data)==digest(row['data']))
        check(key+'initial',incumbent==row['incumbent_routes'])
        mask=make_schedule(51,4,summary['mask_seed'],row['instance_id'],fraction=.5)[row['mask_ordinal']]
        partial=partial_adjacency(incumbent,set(mask),51)
        check(key+'mask_and_context',row['mask']==mask and np.array_equal(partial,np.asarray(row['partial_adjacency'])))
        initial=evaluate(data,incumbent,contract)
        candidate=evaluate(data,row['candidate_routes'],contract)
        target=evaluate(data,row['CE_target_routes'],contract)
        valid=(candidate['hard']is True and candidate['complete']is True and row['status']=='complete' and row['late']is False)
        delta=initial['cost']-candidate['cost'] if valid else None
        sign=0 if delta is None or abs(delta)<=1e-8 else (1 if delta>0 else -1)
        category='invalid' if not valid else ('small' if sign==0 else ('positive' if sign>0 else 'negative'))
        preferred=row['candidate_routes'] if valid and delta>1e-8 else incumbent
        check(key+'pair_sign_category',row['preference_sign']==sign and row['category']==category and row['teacher_pair_valid']==valid)
        check(key+'lower_C0_CE_target',row['CE_target_routes']==preferred)
        check(key+'full_target_hard',target['hard']is True and target['complete']is True)
        check(key+'all_states_retained',row['all_states_retained']is True)
        for name,actual,saved in (('initial',initial,row['initial_report']),('target',target,row['CE_target_certified_report'])):
            error=max(abs(actual[k]-saved[k]) for k in ('cost','distance_km','energy_kwh','quality_loss_value'))
            max_error=max(max_error,error)
            check(key+name+'cost_decomposition',error<=1e-8)
            check(key+name+'manifest_and_events',digest(actual['delivered'])==digest(saved['delivered'])
                  and digest(actual['traces'])==digest(saved['traces']))
        check(key+'target_retained_edges',np.all(partial_adjacency(preferred,set(),51)[partial>0]>0))
        check(key+'state_sha',digest({k:v for k,v in row.items() if k!='state_sha256'})==row['state_sha256'])
        counts[category]+=1
    failures=[x for x in checks if not x['pass']]
    result={'gate_name':'teacher_engineering','observer':'isolated_agent_local',
        'independent_acceptance':not failures,'registration_sha256':sha(a.registration),
        'contract_hash':contract.identity(),'teacher_archive_sha256':sha(archive),'states':len(rows),
        'categories':counts,'checks':len(checks),'failures':failures,'max_cost_decomposition_error':max_error,
        'source_unchanged_rechecked':True,'new_scientific_training':False,'dynamic_online_gain_claim':False}
    dump(a.out,result)
    print(json.dumps({k:result[k] for k in ('independent_acceptance','states','categories','checks','failures')}))
    if failures: raise ValueError('Independent teacher replay not accepted')
