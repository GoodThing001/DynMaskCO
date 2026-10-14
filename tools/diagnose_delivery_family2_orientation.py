"""Selected observed-pair structural diagnosis; no scientific data/search/train."""
import ast, hashlib, json, sys
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from delivery_oriented_action_codec import encode, decode, insertion_targets, replay

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'research/review_3080_20261010'
D = R/'delivery'
OUT = D/'family2_orientation_action_diagnosis_root_v1'
RAW = D/'multistep_insertable_family2_root_v1/downloaded/STATES.jsonl'
AUDIT = sorted((D/'multistep_insertable_family2_data_audit_root_v2').glob('inspection_*'))[-1]/'INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json'
CONTRACT = ROOT/'C-VRP_Cold-chainVehicleRoutingProblem/scripts/delivery/delivery_contract.py'
OBJECTIVE = D/'multistep_full_gradient_root_v2/features_objective.py'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(name, x):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)

def undirected(routes):
    return {tuple(sorted((a,b))) for r in routes for a,b in zip([0]+r, r+[0]) if a != b}

def main():
    OUT.mkdir(exist_ok=False)
    inputs = {str(p):sha(p) for p in [RAW, AUDIT, CONTRACT, OBJECTIVE, Path(__file__), Path(__file__).with_name('delivery_oriented_action_codec.py')]}
    save('ENGINEERING_SCOPE.json', {'registered_utc':datetime.now(timezone.utc).isoformat(), 'inputs':inputs,
        'scope':'posthoc four observed alias pairs, lossless codec only', 'new_scientific_family':False,
        'new_solver_teacher_model_run':False, 'physical_replay':False, 'learned_decoder_reachability_qualified':False})
    audit=json.loads(AUDIT.read_text(encoding='utf-8'))
    assert audit['raw_replay_failure_names']==[] and audit['failed']==36
    keys={tuple(s['key']) for s in audit['summaries'] if not s['oracle_exact'] and s['oracle_same_undirected_adjacency']}
    assert len(keys)==4
    weights={}
    for node in ast.walk(ast.parse(CONTRACT.read_text(encoding='utf-8'))):
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ('distance_weight','energy_weight','quality_weight'):
            weights[node.target.id]=Decimal(str(ast.literal_eval(node.value)))
    assert len(weights)==3
    checks=[];pairs=[];observed=[]
    def ck(name, value): checks.append({'name':name,'pass':bool(value)})
    with RAW.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);key=(row['instance_id'],row['step'])
            if key not in keys: continue
            t,o=row['teacher']['routes'],row['oracle']['routes']
            changed=[v for v in range(len(t)) if t[v]!=o[v]]
            prefix=str(key)
            ck(prefix+':one_changed_vehicle',len(changed)==1)
            ck(prefix+':same_assignment',all(set(a)==set(b) for a,b in zip(t,o)))
            ck(prefix+':pure_reversal',all(o[v]==t[v][::-1] for v in changed))
            ck(prefix+':same_undirected',undirected(t)==undirected(o))
            ck(prefix+':distinct_directed_target',encode(t)!=encode(o))
            for name,routes in [('teacher',t),('oracle',o)]:
                ck(prefix+':'+name+':directed_roundtrip',decode(encode(routes))==routes)
                skeleton,events=insertion_targets(row['request']['initial'],routes,row['request']['mask'])
                ck(prefix+':'+name+':anchored_actions',replay(skeleton,events)==routes)
                ck(prefix+':'+name+':all_masked_once',sorted(e['customer'] for e in events)==row['request']['mask'])
            terms={k:Decimal(str(row['oracle']['report'][k]))-Decimal(str(row['teacher']['report'][k])) for k in ('cost','distance_km','energy_kwh','quality_loss_value')}
            weighted=weights['distance_weight']*terms['distance_km']+weights['energy_weight']*terms['energy_kwh']+weights['quality_weight']*terms['quality_loss_value']
            ck(prefix+':zero_distance_delta',terms['distance_km']==0)
            ck(prefix+':cost_decomposition',abs(weighted-terms['cost'])<Decimal('1e-12'))
            for v in range(len(t)):
                if v not in changed:
                    ck(prefix+':unchanged_vehicle_trace:'+str(v),row['teacher']['report']['traces'][v]==row['oracle']['report']['traces'][v])
            pairs.append({'key':list(key),'changed_vehicle':changed,'teacher_route':t[changed[0]],'oracle_route':o[changed[0]],
                          'oracle_minus_teacher':{k:float(v) for k,v in terms.items()},'weighted_cost_delta':float(weighted),
                          'same_vehicle_assignment':True,'direction_only':True,'new_target_is_codec_not_trained_decoder':True})
            observed.append({k:row[k] for k in ('instance_id','step','request','partial','initial_report')} | {
                'teacher':{'routes':t,'report':row['teacher']['report']},'oracle':row['oracle']})
    ck('all_four_observed_keys',{tuple(p['key']) for p in pairs}==keys)
    empty=[[],[2,1],[],[3],[]]
    ck('empty_slot_roundtrip',decode(encode(empty))==empty)
    ck('slot_swap_distinguishable',encode([[1],[],[2]])!=encode([[2],[],[1]]))
    skeleton,events=insertion_targets([[1,2],[3],[]],[[2,1],[3],[]],[1,2])
    ck('all_masked_vehicle_actions',replay(skeleton,events)==[[2,1],[3],[]])
    for name,callback in [
        ('duplicate_orders',lambda:encode([[1],[1]])),
        ('unmasked_order_changed',lambda:insertion_targets([[1,2,3]],[[3,2,1]],[2])),
        ('unmasked_vehicle_changed',lambda:insertion_targets([[1,2],[3]],[[2],[3,1]],[2])),
        ('disconnected_cycle',lambda:decode({'schema':'vehicle-directed-arcs-v1','vehicle_slots':1,'arcs':[[0,1,2],[0,2,1]]})),
        ('stale_anchor',lambda:replay([[1]], [{'customer':2,'vehicle':0,'position':1,'predecessor':3,'successor':0}]))]:
        rejected=False
        try:callback()
        except ValueError:rejected=True
        ck('reject:'+name,rejected)
    for p,h in inputs.items():ck('unchanged:'+p,sha(p)==h)
    save('OBSERVED_PAIRS.json',observed)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'checks_count':len(checks),
            'failed':sum(not c['pass'] for c in checks),'pairs':pairs,'all_aliases_direction_only':True,
            'vehicle_assignment_explanation_for_these_four':False,'recorded_cost_arithmetic_only':True,
            'codec_structural_roundtrip_only':True,'production_decoder_modified':False,'new_optimizer_steps':0,
            'family2_barrier_unchanged_FAIL':True,'stable_online_gain':False,'shared_MD_uploaded':False}
    save('RESULT.json',result);save('RUN_EXIT.json',{'exit_code':int(result['failed']>0)})
    print(json.dumps({k:result[k] for k in ('checks_count','failed','all_aliases_direction_only','stable_online_gain')}))
    return int(result['failed']>0)
if __name__=='__main__':sys.exit(main())
