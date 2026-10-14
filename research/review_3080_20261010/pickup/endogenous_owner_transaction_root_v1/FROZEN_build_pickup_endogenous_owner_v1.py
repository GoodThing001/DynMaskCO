"""Create a new engineering tree's programs; never edit qualified predecessors."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tools'
q=(T/'qualify_pickup_private_owner_handoff_v1.py').read_text(encoding='utf-8')
q=q.replace('Prepared-only ownership handoff: private projection through actual fleet commit.','Endogenous current admission/K10/C0 in the actual private environment owner.')
q=q.replace('This is not an endogenous admission/proposal/SAA transaction or a scientific day.','One observed engineering request plus one new post-selection cancellation; no learned proposal or scientific day.')
q=q.replace('private_owner_handoff','endogenous_owner_transaction')
q=q.replace('import pickup_undecided_public_v1 as projector','import pickup_undecided_public_v1 as projector\nimport pickup_pending_transaction_v1 as engine\nimport pickup_partial_common_v1,pickup_partial_trace_v1,pickup_partial_known_v1,pickup_partial_validation_v1\nimport hard_admission\nfrom run_exp_rule_control import UncondDynamicReplanner\nfrom energy_complete_saa import EnergyCompleteSAA\nfrom scenario_saa import CondHistoricalSampler')
q=q.replace("files=[Path(__file__),", "files=[ROOT/'tools/build_pickup_endogenous_owner_v1.py',old.HISTORY,ROOT/'tools/pickup_pending_transaction_v1.py',Path(__file__),")
q=q.replace("response_blob=pickle.dumps(sealed_response['response'],protocol=5);baseline", "history=pickle.loads(old.HISTORY.read_bytes());baseline")
q=q.replace('if fault==n:time.sleep(2.)','if fault==n:time.sleep(12.)')
old="response=pickle.loads(response_blob);token=owner.begin(str(b.current),b.current,deadline,public_hash);response['deadline']=deadline"
new="token=owner.begin(str(b.current),b.current,deadline,public_hash);response=engine.compute(decoded,history,old.E/'scripts',deadline)\n                end_compute=capture(sources());response['source_qualified']=begin['files']==end_compute['files'] and begin['module_files']==end_compute['module_files'];save('SOURCE_COMPUTE_END_'+str(case)+'.json',end_compute)\n                response['source_start']=begin['files'];response['source_end']=end_compute['files']"
assert old in q;q=q.replace(old,new)
q=q.replace('prepared_response_reused=True,new_admission_or_SAA=False','prepared_response_reused=False,new_admission_or_SAA=True,response=response')
q=q.replace("REPLY.pack_into(reply.buf,0,seq,1,2,bytes.fromhex(response['compiled_state_sha256']))", "encode,_=old.codecs();save('CANDIDATE_RESULT_'+str(case)+'.json',encode(record))\n                REPLY.pack_into(reply.buf,0,seq,1,2,bytes.fromhex(response['compiled_state_sha256']))")
q=q.replace('before=old.pf.digest(state(b));b.clock=next_physical_clock(b);','mark(3,fault);before=old.pf.digest(state(b));b.clock=next_physical_clock(b);')
q=q.replace("expected=sealed()[1]['response']['compiled_state_sha256'];cap=.4 if fault else 10.;reserve=.15 if fault else 2.","cap=10.;reserve=2.")
q=q.replace("if published!=expected:raise ValueError('prepared identity')","if len(published)!=64:raise ValueError('bounded compiled identity')")
q=q.replace('CONTROL.pack_into(ctrl.buf,0,2,compute,2,0)','CONTROL.pack_into(ctrl.buf,0,2,compute,2,fault)')
q=q.replace('records=[execute(0,0,baseline),execute(1,1,baseline),execute(2,2,baseline)]','records=[execute(0,0,baseline),execute(1,3,baseline)]')
q=q.replace("r['stage_reached']==r['fault'] and r['error']=='cancelled' and r['prepared'] is None and not r['actual_process_owner_selected']","r['stage_reached']==3 and r['error']=='advance_cancelled' and r['prepared'] is not None and r['promotion'] is not None and not r['actual_process_owner_selected']")
q=q.replace("'actual_projection_encoding_cancel'","'post_selection_cancel_rollback'")
q=q.replace("prepared_private_owner_component_qualified=not failed","endogenous_observed_owner_component_qualified=not failed")
q=q.replace("selection='only original first callback15; one normal prepared-only ownership promotion plus public projection and encoding two new delay faults',prepared_response_reused=True,new_admission_or_SAA=False", "selection='only original first callback15; one endogenous current admission/K10/C0/commit and natural physical progress, plus one delay after host owner selection before physical advance',prepared_response_reused=False,new_admission_or_SAA=True")
# The registration line has no response variable; remove replacement introduced there.
q=q.replace('prepared_response_reused=False,new_admission_or_SAA=True,response=response,new_scientific_data=False','prepared_response_reused=False,new_admission_or_SAA=True,new_scientific_data=False')
q=q.replace('normal_cap_s=10.,normal_reserve_s=2.,fault_cap_s=.4,fault_reserve_s=.15,delay_s=2.', 'normal_cap_s=10.,normal_reserve_s=2.,fault_cap_s=10.,fault_reserve_s=2.,delay_s=12.,common_seed=202610111301,K=10')
q=q.replace('one normal prepared-only ownership promotion plus public projection and encoding two new delay faults','one endogenous normal and one post-selection/pre-physical-entry cancellation')
q=q.replace("actual_owner_scope='restored original private next-reveal environment; prepared policy through original plan_and_commit and one natural physical advance, then stop'", "actual_owner_scope='restored original private next-reveal environment; endogenous current-only own_uncond_dynamic, own_incumbent proposal, original K10/full C0/compile, real plan_and_commit and one physical advance; host rollback discards delayed private branch'")
with (T/'qualify_pickup_endogenous_owner_transaction_v1.py').open('x',encoding='utf-8') as f:f.write(q)
s=(T/'run_pickup_private_owner_handoff_bounded_v1.py').read_text(encoding='utf-8').replace('private_owner_handoff','endogenous_owner_transaction')
with (T/'run_pickup_endogenous_owner_transaction_bounded_v1.py').open('x',encoding='utf-8') as f:f.write(s)
a=(T/'audit_pickup_private_owner_handoff_v1.py').read_text(encoding='utf-8').replace('private_owner_handoff','endogenous_owner_transaction')
a=a.replace("response=decode(json.loads(next(f)))['response']", "sealed_response=decode(json.loads(next(f)))['response']\n    response=actual['response']")
a=a.replace('compiled=response[\'compiled_state\'];state=',"\n    from run_exp_rule_control import UncondDynamicReplanner\n    from energy_complete_saa import EnergyCompleteSAA\n    from scenario_saa import CondHistoricalSampler\n    from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash\n    import pickle\n    with old.HISTORY.open('rb') as f:history=pickle.load(f)\n    prior=packet['snapshot']['replanner_state'];o=request['order'];name='endogenous';a=set(prior['accepted']);r=set(prior['rejected']);pending=set(packet['admission_input']['pending']);current=response['scoring_packet']\n"+"compiled=response['compiled_state'];state=")
# Insert independently assembled original admission and K10 branches from existing reference,
# without calling the new producer/adaptation as reference or replaying previous QA cases.
ref=(T/'audit_pickup_pending_transaction_v1.py').read_text(encoding='utf-8')
block=ref[ref.index('        rp=UncondDynamicReplanner'):ref.index("        ck(name+':joint_compiled_export'")]
block='\n'.join(line[8:] for line in block.splitlines()).replace('copy.deepcopy(state)','copy.deepcopy(prior)',1)
block='\n'.join('    '+line for line in block.splitlines())
a=a.replace("    rp.restore_state(copy.deepcopy(state));env._plan_and_commit", block+"\n    ck('independent_compiled_matches_original_admission',state==replay)\n    # Restore the original private physical reference, because admission replay used public copies.\n    rp=Policy(packet['budget'],packet['capacity'],packet['booking_horizon'],packet['contract'],0.);rp.restore_state(copy.deepcopy(state))\n    env=StrictOnlineEnv(copy.deepcopy(dataset),packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy()\n    # Recover the pre-admission actual physical baseline directly from the frozen predecessor's original replay state.\n    _,dec=old.codecs();sealed_owner=dec(read(P/'private_owner_handoff_root_v1/OWNER_RESULT_0.json'))\n    physical=sealed_owner['public_packet']['snapshot'];fleet,traces,served=restore_recourse_snapshot(physical);env.event_id=packet['snapshot']['event_id'];env.replan_count=packet['snapshot']['replan_count'];clock=packet['snapshot']['clock']\n    env._plan_and_commit")
a=a.replace("reg['prepared_response_reused'] and not reg['new_admission_or_SAA'] and not reg['new_scientific_data'] and not actual['new_admission_or_SAA']","not reg['prepared_response_reused'] and reg['new_admission_or_SAA'] and not reg['new_scientific_data'] and actual['new_admission_or_SAA']")
a=a.replace("r['cap_s']==(.4 if r['fault'] else 10.) and r['reserve_s']==(.15 if r['fault'] else 2.)","r['cap_s']==10. and r['reserve_s']==2.")
a=a.replace("r['stage_reached']==r['fault'] and r['error']=='cancelled' and not r['actual_process_owner_selected'] and r['prepared'] is None", "r['stage_reached']==3 and r['error']=='advance_cancelled' and not r['actual_process_owner_selected'] and r['prepared'] is not None and r['promotion'] is not None")
a=a.replace('new_SAA_or_admission_reexecution=False','new_SAA_or_admission_reexecution=True')
with (T/'audit_pickup_endogenous_owner_transaction_v1.py').open('x',encoding='utf-8') as f:f.write(a)
