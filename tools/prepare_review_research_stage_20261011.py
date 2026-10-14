"""Append-only review acceptance and preregistration; no scientific generation."""
import hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,r):
    with p.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    b=R/'reproduction/formal_run_requirements_root_v2'
    test=(b/'ROOT_FINAL_TEST_RUN.log').read_text(encoding='utf-8')
    if 'Ran 26 tests' not in test or '\nOK' not in test:raise ValueError('Final26 regression not qualified')
    acceptance={'stage':'root_v2_final_necessary_subset_acceptance','adapter_sha256':sha(b/'new_run_gate_adapter.py'),
        'tests_sha256':sha(b/'test_new_run_gate.py'),'actual_test_run_sha256':sha(b/'ROOT_FINAL_TEST_RUN.log'),
        'tests':26,'failures':0,'program_sha256':'bc8cfcb51dba940af539e23194fad7c7e7caf681a1e98e68a5b296c7286632d5',
        'scope_amendment_sha256':'2184df0054789f1866b959fa462e4cf660ba1b372553b04ac7292f64df1b6035',
        'supersedes_preliminary_file':'ROOT_PROTOCOL_CORRECTION.json','preliminary_adapter_hash_is_current':False,
        'old_package_or_archive_modified':False,'formal_online_qualified':False,'stable_core_online_gain':False}
    write(b/'FINAL_ROOT_ACCEPTANCE.json',acceptance)
    (b/'ROOT_V2_SCOPE.md').write_text("""# 当前根审查版本

此副本的最终字节以 FINAL_ROOT_ACCEPTANCE.json 为准。早期 ROOT_PROTOCOL_CORRECTION.json 的 new_adapter_sha256 是修订中间状态，保留供追溯，不能当作最终字节。

已固定当前 PROGRAM bc8cfcb5…及追加范围注册2184df00…；历史PROGRAM8edbc3cd…只验证父谱系。取货主确认角色为 qualified_strong_nonlearned_rolling，阈值20；纯重建为归因对照，cond_hist必须作为报告臂保留。已测两个运行点不证明强滚动在所有日支配cond_hist。本门仅校验必要元数据子集，尚不覆盖所有 own-base/random 控制或实际因果、完整物理、进程终止语义。

最终26项回归通过。正式资格、实际激活独立重放和稳定核心在线增益仍为false；presence、文件哈希、设计检查通过均不能自授科学资格。原v1包、现行旧183文件和既有结果字节未改。
""",encoding='utf-8')
    source=R/'delivery/finite_training_preparation_v1';dest=R/'delivery/finite_training_research_root_v2'
    dest.mkdir(exist_ok=False)
    for name in ('finite_common.py','teacher_pipeline.py','train_full_masked.py'):
        shutil.copy2(source/name,dest/name)
    # Require independent whole fresh teacher acceptance before the training stage.
    p=dest/'train_full_masked.py';text=p.read_text(encoding='utf-8')
    text=text.replace("archive = a.teacher_dir / 'TEACHER_STATES.jsonl'","""archive = a.teacher_dir / 'TEACHER_STATES.jsonl'
    if a.mode == 'research':
        if a.accepted_teacher is None:
            raise ValueError('Fresh256 states require independent complete teacher acceptance')
        accepted = json.loads(a.accepted_teacher.read_text(encoding='utf-8'))
        if (accepted.get('independent_acceptance') is not True or
            accepted.get('registration_sha256') != sha(a.registration) or
            accepted.get('teacher_archive_sha256') != sha(archive) or
            accepted.get('states') != 256 or accepted.get('failed') != 0):
            raise ValueError('Fresh teacher independent stage not qualified')""")
    text=text.replace("ap.add_argument('--cpu',type=int,required=True)","ap.add_argument('--accepted-teacher',type=Path)\n    ap.add_argument('--cpu',type=int,required=True)")
    text=text.replace("finite = all(np.isfinite(np.asarray(g.value)).all() for _,g in grads.flat_state())","""gradient_inventory = inventory(grads)
                actual_before = inventory(params)
                finite = len(gradient_inventory)==234 and all(np.isfinite(np.asarray(g.value)).all() for _,g in grads.flat_state())""")
    text=text.replace("'all234_parameter_leaves_in_gradient':True,'gradient_finite':bool(finite)","""'all234_parameter_leaves_in_gradient':len(gradient_inventory)==234,'gradient_finite':bool(finite),
                    'actual_gradient_leaf_count':len(gradient_inventory),'actual_gradient_tensor_digest':digest(gradient_inventory),
                    'actual_before_parameter_digest':digest(actual_before),'actual_after_parameter_digest':digest(inventory(params))""")
    p.write_text(text,encoding='utf-8')
    reg=load(source/'FULL_MASKED_FINITE_PROTOCOL_DRAFT.json')
    reg.update(status='REGISTERED_RESEARCH',registration_id='D-FULLMASK-SNAPSHOT-TRAIN-root-v2',
        registered_utc=datetime.now(timezone.utc).isoformat(),
        parent_draft_sha256=sha(source/'FULL_MASKED_FINITE_PROTOCOL_DRAFT.json'),
        current_program_sha256=acceptance['program_sha256'],scope_amendment_sha256=acceptance['scope_amendment_sha256'],
        train_instance_ids=list(range(64)),
        seeds=dict(zip(['train','teacher_mask','training_schedule','development','arm_order','bootstrap','confirmation1','confirmation2','confirmation3'],range(202610111101,202610111110))),
        deployment_arms=['paired_objective','reconstruction_control','original_warmstart','random10D_untrained','nonlearning_fixed8','GLS'],
        deployment_scope='eight fixed mask opportunities under common10s ceiling; not ANY50/anytime and not equal consumed wall',
        confirmation_primary='snapshot-only: relative complete cost reduction vs random/nonlearning_fixed8/GLS >=.01, vs CE-only >0; all adjusted intervals positive in all three cohorts',
        attribution='same-init paired-objective versus CE-only; does not establish pretrained versus random-trained attribution',
        family_index=1,
        family_error_budget=.05/(1*2)/2,
        confirmation_contrasts=4,
        bootstrap_two_sided_alpha=.05/(1*2)/2/(4*3),
        bootstrap_method='paired whole instance bootstrap100000 draws percohort; adjusted two-sided percentiles; each tail MC>=25; no pooling/selecting seeds',
        bootstrap_repetitions=100000,
        minimum_expected_tail_MC=25,
        development_scope='16 fresh snapshots diagnostic only; fixed final weights; no checkpoint or hyperparameter selection',
        confirmation_execution_enabled=False,
        confirmation_start_barriers=['independent256 teacher acceptance','independent128 steps and actual final weight reload','new final weights all-arm hard10s activation and timeout regression','balanced registered arm order with full raw route preservation','explicit snapshot versus dynamic claim separation'],
        online_stable_gain_claim_enabled=False,
        finite_training_source_changes='new copied trainer requires independent256 teacher acceptance and records actual234 gradient and parameter digests perstep')
    evidence={'teacher_engineering':source/'ROOT_ENGINEERING_TEACHER_AUDIT.json',
        'actual_10D_objective':R/'delivery/remote_features_bundle_v1/downloaded/INDEPENDENT_ROOT_ACTUAL_FEATURES_OBJECTIVE_AUDIT_v2.json',
        'deployment_10s_budget':R/'delivery/remote_model_budget_bundle_root_v1/downloaded/INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json'}
    gates=dest/'gates';gates.mkdir();reg['accepted_gate_sha256']={}
    for name,artifact in evidence.items():
        raw=load(artifact)
        if name=='actual_10D_objective':
            assert raw['n_checks']==330 and raw['n_failures']==0
            gate={'gate_name':name,'contract_hash':reg['contract_hash'],'independent_acceptance':True,
                'evidence_file':str(artifact),'evidence_sha256':sha(artifact),'checks':330,'failures':0,
                'limits':raw['limitations'],'dynamic_online_qualified':False}
            write(gates/(name+'.json'),gate)
        else:
            assert raw['independent_acceptance'] is True
            shutil.copy2(artifact,gates/(name+'.json'))
        reg['accepted_gate_sha256'][name]=sha(gates/(name+'.json'))
    reg['programs']={name:sha(dest/name) for name in reg['programs']}
    write(dest/'RESEARCH_REGISTRATION.json',reg)
    # Validate prospective gate now; generation takes place only after remote seal.
    sys.path.insert(0,str(dest))
    from finite_common import verify_research_gate,check_dependency_identity
    verify_research_gate(reg,{name:gates/(name+'.json') for name in evidence})
    check_dependency_identity(reg,E,ROOT/'MASKCO_code')
    write(dest/'LOCAL_PREGENERATION_ACCEPTANCE.json',{'registration_sha256':sha(dest/'RESEARCH_REGISTRATION.json'),
        'gate_hashes':reg['accepted_gate_sha256'],'dependency_and_program_bytes_match':True,
        'new_data_generated':False,'fresh_teacher_accepted':False,'training_started':False,
        'formal_confirmation_or_dynamic_online_authorized_here':False,
        'planned_next':'generate all64x4 retained states, independently replay; only then fixed128 each'})
    print(json.dumps({'B_final26_pass':True,'D_registration_sha256':sha(dest/'RESEARCH_REGISTRATION.json'),'new_data_generated':False}))
if __name__=='__main__':main()
