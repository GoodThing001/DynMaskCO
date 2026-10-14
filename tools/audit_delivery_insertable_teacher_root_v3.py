"""Independently replay every stored constrained teacher candidate; no solver run."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';D=ROOT/'research/review_3080_20261010/delivery';OUT=D/'insertable_teacher_engineering_root_v3'
for sub in ('delivery','coldchain'):sys.path.insert(0,str(E/'scripts'/sub))
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def edges(routes):
    result=set()
    for r in routes:
        chain=[0]+r+[0]
        result.update(tuple(sorted((a,b))) for a,b in zip(chain,chain[1:]) if a!=b)
    return result
def main():
    r=json.loads((OUT/'RESULT.json').read_text());reg=json.loads((OUT/'ENGINEERING_REGISTRATION.json').read_text());checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('reported_engineering_pass',r['failed']==0 and r['retained_teacher_engineering_qualified'])
    ck('raw_reg_bound',r['registration_sha256']==sha(OUT/'ENGINEERING_REGISTRATION.json'))
    ck('actual_module',reg['teacher_sha256']==sha(OUT/'retained_gls.py')==r['ready']['source_sha256'])
    ck('actual_launcher',reg['program_sha256']==sha(ROOT/'tools/qualify_delivery_insertable_teacher_root_v3.py'))
    fp=D/'finite_training_preparation_v1/ENGINEERING_TEACHER_QA_v1/TEACHER_STATES.jsonl';ck('old_fixture_bytes',reg['fixtures_sha256']==sha(fp))
    old=[json.loads(s) for s in fp.read_text().splitlines()];fixture={x['instance_id']:x for x in old if x['mask_ordinal']==0}
    ck('exact_charged_affinity',reg['actual_parent_worker_cpu']==r['ready']['cpu']==[7])
    ck('no_new_science',reg['research_new_days']==0 and r['new_scientific_family_registered'] is False)
    summaries=[];count=0
    rows=[x for x in r['records'] if 'answer' in x]
    ck('four_expected_requests',[(x['instance_id'],x['mask_kind']) for x in rows]==[(i,k) for i in (900000,900001) for k in ('partial','all_edges')])
    for x in rows:
        name=str((x['instance_id'],x['mask_kind']));q=x['request'];f=fixture[x['instance_id']];a=np.asarray(q['partial']);required={tuple(sorted((int(i),int(j)))) for i,j in np.argwhere(np.triu(a,1)>0)}
        ck(name+':registered_existing_data',q['data']==f['data'] and q['initial']==f['incumbent_routes'])
        ck(name+':all_original_unmasked_edges_only',required<=edges(q['initial']))
        expected=edges(q['initial']) if x['mask_kind']=='all_edges' else {tuple(sorted((int(i),int(j)))) for i,j in np.argwhere(np.triu(f['partial_adjacency'],1)>0)}
        ck(name+':exact_registered_mask',required==expected)
        ck(name+':actual_request_clock_cap',0<=x['wall_s']<10. and x['started']<=x['received']<=q['deadline'] and math.isclose(q['deadline']-x['started'],10.,abs_tol=1e-8))
        removed=set(f['mask']) if x['mask_kind']=='partial' else set()
        ck(name+':actual_mask_registered',q['masked_orders']==(f['mask'] if x['mask_kind']=='partial' else []))
        kept=[[o for o in rr if o not in removed] for rr in q['initial']]
        ck(name+':all_old_fixture_vehicles_anchored',all(kept))
        def fixed_skeleton(routes):
            return len(routes)==len(kept) and [[o for o in rr if o not in removed] for rr in routes]==kept
        out=x['answer']['result'];ck(name+':selected_fixed_insertion_skeleton',fixed_skeleton(out['routes']))
        ck(name+':actual_worker_skeleton',out['fixed_unmasked_vehicle_order']==kept and out['masked_orders']==q['masked_orders'])
        initial=evaluate(q['data'],q['initial'],DeliveryContract());best_routes=q['initial'];best_cost=initial['cost']
        ck(name+':initial_hard',initial['hard'] and initial['complete']);ck(name+':initial_cost',math.isclose(initial['cost'],out['initial_report']['cost'],abs_tol=1e-9))
        ck(name+':exact_undirected_constraint_list',set(map(tuple,out['retained_edges']))==required)
        for j,c in enumerate(out['all_candidates']):
            count+=1;ev=evaluate(q['data'],c['routes'],DeliveryContract());preserved=required<=edges(c['routes'])
            ck(name+':'+str(j)+':all_retained_edges',preserved and c['retained_edges_preserved']);ck(name+':'+str(j)+':unmasked_original_vehicle_order',fixed_skeleton(c['routes']))
            ck(name+':'+str(j)+':all_physical_flags_replayed',all(ev[k]==c['report'][k] for k in ('hard','complete','errors')))
            for k in ('cost','distance_km','energy_kwh','quality_loss_value'):ck(name+':'+str(j)+':'+k,math.isclose(ev[k],c['report'][k],rel_tol=1e-12,abs_tol=1e-9))
            accepted=preserved and ev['hard'] and ev['cost']<best_cost-1e-12;ck(name+':'+str(j)+':fullC0_selection',c['accepted']==accepted)
            if accepted:best_routes,best_cost=c['routes'],ev['cost']
        final=evaluate(q['data'],out['routes'],DeliveryContract());ck(name+':final_complete_hard',final['complete'] and final['hard']);ck(name+':selected_best_all_records',out['routes']==best_routes and math.isclose(final['cost'],best_cost,abs_tol=1e-9));ck(name+':never_worse_than_own',best_cost<=initial['cost']+1e-9)
        summaries.append({'instance':x['instance_id'],'mask':x['mask_kind'],'candidates':len(out['all_candidates']),'retained_edges':len(required),'request_s':x['wall_s'],'old_fixture_initial_cost':initial['cost'],'teacher_C0_cost':final['cost'],'engineering_numbers_not_method_evidence':True})
    fault=r['records'][-1];ck('delayed_fault_rejected',fault['reply_kind']=='timeout' and fault['late_reply_applied'] is False and fault['fallback_routes']==fixture[900000]['incumbent_routes'])
    ck('all_candidates_present',count==316);failed=sum(not c['pass'] for c in checks)
    report={'checks':checks,'checks_count':len(checks),'failed':failed,'independent_retained_teacher_replay_pass':failed==0,'raw_sha256':sha(OUT/'RESULT.json'),'summaries':summaries,
      'old_fixtures_only':True,'all_targets_pass_structural_insertion_necessity':failed==0,'specific_learned_greedy_decoder_can_realize_teacher':False,'new_scientific_family_registered':False,'teacher_optimality_proven':False,'MaskCO_attribution_established':False,'stable_online_gain':False,
      'qualification_limits':['Independent arithmetic/route replay shares fullC0 physical evaluator, not independent physical calibration','Current module/launcher/reg/fixture byte identities verified; complete interpreter and dependency start/end closure not yet registered for science','Search preserves unmasked vehicle/order and original edges; this is structural insertion necessity, not proof of learned greedy decoder reachability or an optimal target','No task training, matched deployment-state teacher generation or new research cohort yet']}
    with (OUT/'INDEPENDENT_RETAINED_TEACHER_REPLAY.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'recorded_candidates':count,'summary':summaries}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
