"""Recompute the preregistered paired delivery pilot gates and full qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from run_delivery_pilot import digest,generate


def main(result_path,protocol,out):
    result=json.loads(Path(result_path).read_text());reg=json.loads(Path(protocol).read_text())
    root=Path(result_path).parent;identity=result['identity'];rows=result['rows']
    expected=reg['dataset']['instance_ids'];ids=[r['instance_id'] for r in rows]
    valid=ids==expected and len(set(ids))==len(expected)==32
    start=json.loads((root/'identity_start.json').read_text());end=json.loads((root/'identity_end.json').read_text())
    valid &= identity==start==end and result.get('source_stable') is True and result.get('identity_verified') is True
    valid &= identity['protocol_sha256']==hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
    contract=DeliveryContract(**reg['contract_overrides']);valid &= identity['contract_hash']==contract.identity()
    valid &= identity['checkpoint_sha256']==reg['weights']['sha256'] and identity['timing_hash']==digest(reg['timing'])
    costs={arm:[] for arm in reg['arms']};timeouts={arm:0 for arm in reg['arms']}
    replayed_dataset={}
    for row in rows:
        data,initial=generate(reg,row['instance_id'])
        if row['data_initial_hash']!=digest([data,initial]) or row['identity']!=identity:valid=False
        replayed_dataset[str(row['instance_id'])]={'instance':data,'initial_routes':initial,'data_initial_hash':digest([data,initial])}
        if set(row['arms'])!=set(reg['arms']):valid=False;continue
        for arm in reg['arms']:
            r=row['arms'][arm];check=evaluate(data,r['routes'],contract)
            valid &= bool(check['hard'] and r['report']['hard'] and np.isfinite(check['cost']) and abs(check['cost']-r['report']['cost'])<=1e-9)
            costs[arm].append(check['cost']);timeouts[arm]+=int(r['timeout'])
            valid &= bool(np.isfinite(r['wall_s']) and r['wall_s']>=0)
            if r['wall_s']>=reg['timing']['total_per_arm_per_instance_s']:
                valid &= r['timeout'] and r['routes']==initial
    valid &= digest(replayed_dataset)==identity['data_sha256']
    rng=np.random.default_rng(reg['bootstrap']['seed']);indices=rng.integers(0,len(rows),(reg['bootstrap']['repetitions'],len(rows)))
    comparisons={}
    for control in reg['strong_controls']+[reg['pretraining_control']]:
        a=np.asarray(costs[control]);b=np.asarray(costs[reg['primary_method']])
        fractional=(a-b)/a;boot=fractional[indices].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
        threshold=reg['practical_threshold'] if control in reg['strong_controls'] else 0.
        passed=bool(valid and fractional.mean()>=threshold and ci[0]>0)
        comparisons[control]={'absolute_control_minus_single_mean':float((a-b).mean()),
            'fractional_gain_mean':float(fractional.mean()),'fractional_gain_ci95':ci,'pass':passed}
    iteration=np.asarray(costs['maskco_single'])-np.asarray(costs['maskco_iterative'])
    activity={arm:sum(int(row['arms'][arm].get('decode_calls',0)) for row in rows)
              for arm in reg['arms'] if arm.startswith('maskco')}
    method_active=all(v>0 for v in activity.values())
    result={'formal_pilot_adjudicable':bool(valid),'instances':len(rows),'mean_cost':{k:float(np.mean(v)) for k,v in costs.items()},
        'model_decode_calls':activity,'active_method_comparison_qualified':bool(valid and method_active),
        'timeout_counts':timeouts,'comparisons':comparisons,
        'same_weight_iteration_absolute_gain_mean':float(iteration.mean()),
        'same_weight_iteration_absolute_gain_ci95':np.quantile(iteration[indices].mean(1),[.025,.975]).tolist(),
        'pilot_pass':bool(valid and method_active and all(c['pass'] for c in comparisons.values())),
        'scope':'route_transfer_clue_only','online_gain_established':False,
        'next':'preregister real delivery online contract if pass; otherwise archive and diagnose'}
    Path(out).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.result,a.protocol,a.out)
