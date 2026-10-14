"""Independent coordinate arithmetic and ordinal action checks, no decoder calls."""
import ast,hashlib,json,os,sys,traceback
from decimal import Decimal,localcontext
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'research/review_3080_20261010/delivery/family2_remaining_action_diagnosis_root_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def files():
    paths={Path(sys.executable).resolve(),Path(__file__).resolve()}
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file())
    return {str(p):sha(p) for p in paths}
def main():
    if sys.argv[1:]==['--register']:
        inputs={str(O/n):sha(O/n) for n in ('REGISTRATION.json','RESULT.json','RUN_EXIT.json','OBSERVED_ACTION_RECORDS.json')}
        inputs[str(Path(__file__))]=sha(__file__)
        save('AUDIT_REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,
            arithmetic='60 digit Decimal.from_float coordinates; population scales from ordered off diagonal; signed final-target edge indicators; absolute1e-10 diagnostic arithmetic tolerance from prior registration',
            independence='No imports of diagnostic, original decoder, physical evaluator, model or solver. Ordinal vehicle/order invariant identifies first recorded mismatch.',
            full_scientific_closure_qualified=False));print('{"audit_registered":true}');return 0
    reg=read(O/'AUDIT_REGISTRATION.json');prior=read(O/'REGISTRATION.json');res=read(O/'RESULT.json');rows=read(O/'OBSERVED_ACTION_RECORDS.json');checks=[];detail=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    for p,h in reg['inputs'].items():ck('input_start:'+p,sha(p)==h)
    start=files();save('AUDIT_SOURCE_START.json',dict(files=start,python=sys.version))
    def identity_ok(routes,locations):
        for slot,r in enumerate(routes):
            ranks=[locations[n][1] for n in r]
            if any(locations[n][0]!=slot for n in r) or any(a>=b for a,b in zip(ranks,ranks[1:])):return False
        return True
    bykey={tuple(r['key']):r for r in res['records']};tol=Decimal(str(prior['score_arithmetic_abs_tolerance']))
    with localcontext() as ctx:
        ctx.prec=60
        for x in rows:
            key=(x['instance_id'],x['step']);name=str(key);ref=bykey[key];target=x['teacher_routes'];loc={n:(v,k) for v,r in enumerate(target) for k,n in enumerate(r)}
            coords=[[Decimal.from_float(float(c)) for c in row] for row in x['request']['data']['coords']];n=len(coords)
            dm=[[(sum((a-b)**2 for a,b in zip(coords[u],coords[v]))).sqrt() for v in range(n)] for u in range(n)]
            scale=sum(dm[u][v] for u in range(n) for v in range(n) if u!=v)/Decimal(n*(n-1))
            edge=set()
            for r in target:
                p=[0]+r+[0]
                for u,v in zip(p,p[1:]):
                    if u!=v:edge.add(frozenset((u,v)))
            density=Decimal(2*len(edge))/Decimal(n*(n-1));ls=(density*(1-density)).sqrt()
            def arithmetic(r,i,at):
                u=r[at-1] if at else 0;v=r[at] if at<len(r) else 0
                g=int(frozenset((u,i)) in edge)+int(frozenset((i,v)) in edge)-(0 if u==v else int(frozenset((u,v)) in edge))
                d=dm[u][i]+dm[i][v]-dm[u][v]
                return u,v,g,d,d/scale-Decimal(g)/ls
            mask=set(x['request']['mask']);state=[[n for n in r if n not in mask] for r in x['request']['initial']];first=None
            ck(name+':initial_ordinal_invariant',identity_ok(state,loc))
            for j,e in enumerate(x['oracle']['insertions']):
                i,v,at=e['customer'],e['vehicle'],e['position'];pre=[r[:] for r in state];before=identity_ok(pre,loc);u,b,g,d,sc=arithmetic(pre[v],i,at)
                ck(name+':stored_score_decimal:'+str(j),abs(sc-Decimal.from_float(e['score']))<=tol)
                state[v].insert(at,i);after=identity_ok(state,loc)
                if before and not after and first is None:
                    tv,rank=loc[i];ta=sum(loc[k][1]<rank for k in pre[tv]);au,ab,ag,ad,asc=arithmetic(pre[tv],i,ta);gap=asc-sc
                    relation='chosen_strictly_lower_score' if gap>tol else ('target_strictly_lower_score_filter_or_missing_context' if gap< -tol else 'score_tie')
                    first=j
                    ck(name+':first_index',j==ref['insertion_index']);ck(name+':customer',i==ref['customer']);ck(name+':pre_routes',pre==ref['pre_routes'])
                    for label,values in [('chosen',(v,at,u,b,g,d,sc)),('teacher_compatible',(tv,ta,au,ab,ag,ad,asc))]:
                        z=ref[label]
                        for f,value in zip(('vehicle','position','predecessor','successor','binary_gain'),values[:5]):ck(name+':'+label+':'+f,z[f]==value)
                        ck(name+':'+label+':distance_decimal',abs(values[5]-Decimal.from_float(z['distance_delta']))<=tol)
                        ck(name+':'+label+':score_decimal',abs(values[6]-Decimal.from_float(z['score']))<=tol)
                    ck(name+':score_relation',relation==ref['score_relation']);ck(name+':gap_decimal',abs(gap-Decimal.from_float(ref['target_minus_chosen_score']))<=tol)
                    ck(name+':slot_class',ref['wrong_vehicle']==(v!=tv));ck(name+':blocked_partial_anchor',ref['target_edge_blocked']==(x['partial'][au][ab]>0))
                    detail.append(dict(key=list(key),first_insertion=j,score_relation=relation,decimal_gap=str(gap),wrong_vehicle=v!=tv,target_gain=ag))
            ck(name+':saved_final_routes',state==x['oracle']['routes']);ck(name+':first_failure_exists',first is not None)
    ck('exact30_same_sets',len(detail)==len(rows)==30 and {tuple(x['key']) for x in detail}==set(bykey))
    # Inspect code, do not execute production decoder or qualify its filters.
    tree=ast.parse(Path(next(p for p in prior['inputs'] if p.endswith('diagnose_delivery_remaining_actions_v1.py'))).read_text(encoding='utf-8'))
    forbidden={'reconstruct','route_ok','evaluate','solve','construct_model'}
    ck('no_decoder_physics_solver_calls',not any(isinstance(z,ast.Call) and isinstance(z.func,ast.Name) and z.func.id in forbidden for z in ast.walk(tree)))
    for p,h in reg['inputs'].items():ck('input_end:'+p,sha(p)==h)
    end=files();ck('audit_loaded_files_unchanged',start==end);save('AUDIT_SOURCE_END.json',dict(files=end,changed=[p for p,h in start.items() if end.get(p)!=h],new_files=sorted(set(end)-set(start))))
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),details=detail,
        independent_high_precision_coordinate_arithmetic=True,exact_float_bit_reproduction_claimed=False,actual_production_feasibility_filters_replayed=False,
        all30_recorded_actions_scoped=True,scientific_family2_still_FAIL=True,production_decoder_modified=False,full_scientific_closure_qualified=False,stable_online_gain=False)
    save('INDEPENDENT_REMAINING_ACTIONS.json',out);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(out['failed']>0),pid=os.getpid()));print(json.dumps(dict(checks_count=len(checks),failed=out['failed'],loaded_files=len(start))));return int(out['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        save('AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()));raise
