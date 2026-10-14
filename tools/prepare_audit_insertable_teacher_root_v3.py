"""Prepare isolated independent replay plus exact fixed-vehicle/order necessity."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'tools/audit_delivery_retained_teacher_root_v2.py').read_text()
s=s.replace("OUT=D/'retained_edge_teacher_engineering_root_v2'","OUT=D/'insertable_teacher_engineering_root_v3'")
s=s.replace('qualify_delivery_retained_teacher_root_v2.py','qualify_delivery_insertable_teacher_root_v3.py').replace('count==274','count==316')
s=s.replace("        out=x['answer']['result'];initial=",'''        removed=set(f['mask']) if x['mask_kind']=='partial' else set()
        ck(name+':actual_mask_registered',q['masked_orders']==(f['mask'] if x['mask_kind']=='partial' else []))
        kept=[[o for o in rr if o not in removed] for rr in q['initial']]
        ck(name+':all_old_fixture_vehicles_anchored',all(kept))
        def fixed_skeleton(routes):
            return len(routes)==len(kept) and [[o for o in rr if o not in removed] for rr in routes]==kept
        out=x['answer']['result'];ck(name+':selected_fixed_insertion_skeleton',fixed_skeleton(out['routes']))
        ck(name+':actual_worker_skeleton',out['fixed_unmasked_vehicle_order']==kept and out['masked_orders']==q['masked_orders'])
        initial=''')
s=s.replace("            ck(name+':'+str(j)+':all_retained_edges',preserved and c['retained_edges_preserved'])","            ck(name+':'+str(j)+':all_retained_edges',preserved and c['retained_edges_preserved']);ck(name+':'+str(j)+':unmasked_original_vehicle_order',fixed_skeleton(c['routes']))")
s=s.replace("'old_fixtures_only':True,","'old_fixtures_only':True,'all_targets_pass_structural_insertion_necessity':failed==0,'specific_learned_greedy_decoder_can_realize_teacher':False,")
s=s.replace("'Search is distance GLS plus fullC0 recorded-candidate selection, no optimal target guarantee'","'Search preserves unmasked vehicle/order and original edges; this is structural insertion necessity, not proof of learned greedy decoder reachability or an optimal target'")
with (ROOT/'tools/audit_delivery_insertable_teacher_root_v3.py').open('x',encoding='utf-8') as f:f.write(s)
print('Independent replay plus fixed-skeleton necessity audit prepared.')
