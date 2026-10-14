"""Add one untested rejection boundary; preserve original transaction QA."""
import json,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools';P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'joint_policy_rejection_root_v1'
OUT.mkdir(exist_ok=False)
shutil.copyfile(P/'joint_policy_transaction_root_v3/joint_install.py',OUT/'joint_install.py')
code=(T/'qualify_pickup_joint_policy_transaction_root_v3.py').read_text()
code=code.replace("OUT=P/'joint_policy_transaction_root_v3'", "OUT=P/'joint_policy_rejection_root_v1'")
a=code.index("    rows=raw['admissions'];indices=");b=code.index('\ndef worker():',a)
code=code[:a]+"    rows=raw['admissions'];return [(13,rows[13])]\n"+code[b:]
code=code.replace('Three already recorded admission requests;','One previously recorded rejection request;')
code=code.replace('first two recorded admissions and first pre-admission state with >=2 mutable orders; no outcome-based selection','Outcome-selected ENGINEERING boundary only: first archived rejection still rejected by canonical own_uncond_dynamic among observed rejection indices 0,6,7,13; index69 also probed. No scientific sample, no gain estimate, no historical outcome equivalence claim')
code=code.replace("        template=records[-1]", """                ck(rid+':actual_new_order_rejected',row['order'] not in before['accepted'] and row['order'] not in after['accepted'] and row['order'] in after['rejected'])
                ck(rid+':rejection_preserves_accepted_and_budget',set(after['accepted'])==set(before['accepted']) and after['budget']==before['budget'])
                ck(rid+':rejected_order_not_in_plan',all(row['order'] not in route for route in after['plan'].values()))
                ck(rid+':actual_branch_matches_archived_rejection',row['accepted'] is False and row['order'] in row['base_response']['state']['rejected'])
        template=records[-1]""")
a=code.index('        template=records[-1]');b=code.index('    except Exception as exc:',a)
code=code[:a]+code[b:]
a=code.index("    for phase in ('projection'");b=code.index("    ck('all_six_normal_requests_retained'",a)
code=code[:a]+code[b:]
code=code.replace('all_six_normal_requests_retained','both_rejection_route_modes_retained')
code=code.replace("'whole_live_10s_qualified':False", "'new_order_rejection_boundary_qualified':failed==0,'whole_live_10s_qualified':False")
with (T/'qualify_pickup_joint_rejection_root_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(code)
compile(code,'new_rejection.py','exec')
audit=(T/'audit_pickup_joint_policy_transaction_root_v3.py').read_text().replace('joint_policy_transaction_root_v3','joint_policy_rejection_root_v1').replace('qualify_pickup_joint_policy_rejection_root_v1.py','qualify_pickup_joint_rejection_root_v1.py')
audit=audit.replace('all_six_unique_expected_requests','both_unique_expected_rejections').replace('len(expected)==6','len(expected)==2')
a=audit.index('    expected_faults=');b=audit.index('    remaining=[]',a)
audit=audit[:a]+"    ck('no_repeat_of_prior14_completed_faults',raw['faults']==[])\n"+audit[b:]
audit=audit.replace("        summaries.append(", """        ck(label+':actual_rejected_new_order',r['order'] not in r['before']['accepted'] and r['order'] in r['after']['rejected'] and r['order'] not in r['after']['accepted'])
        ck(label+':no_acceptance_or_budget_debit',set(r['before']['accepted'])==set(r['after']['accepted']) and r['before']['budget']==r['after']['budget'])
        ck(label+':rejected_order_not_routed',all(r['order'] not in route for route in r['after']['plan'].values()))
        ck(label+':observed_original_rejection',row['accepted'] is False and r['order'] in row['base_response']['state']['rejected'])
        summaries.append(""")
audit=audit.replace('INDEPENDENT_JOINT_TRANSACTION_REPLAY.json','INDEPENDENT_REJECTION_TRANSACTION_REPLAY.json')
audit=audit.replace("'joint_observed_policy_transaction_qualified':failed==0", "'joint_observed_policy_transaction_qualified':failed==0,'new_order_rejection_boundary_qualified':failed==0")
with (T/'audit_pickup_joint_rejection_root_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(audit)
compile(audit,'rejection_audit.py','exec')
receipt={'purpose':'Previously uncovered rejection engineering boundary; not scientific outcome selection','probed_indices':[0,6,7,13,69],'archived_accepted':[False]*5,'canonical_direct_accepted':[True,True,True,False,True],'selected_index':13,'history_parameter': 'original qualified COMMON_HISTORY.pkl history_times_classes','historical_admission_equivalence_qualified':False,'source_original_unmodified':True}
with (OUT/'BOUNDARY_SELECTION_PREFLIGHT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print('New rejection engineering programs prepared, existing component unchanged.')
