"""Archive saved-only synchronized score diagnostics and update current pointers."""
import datetime
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT/'research/review_3080_20261010'
STAGE=TREE/'delivery/score_split_timed_execution_root_v1'
RAW=STAGE/'downloaded'
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def create(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def seal():
    files=[STAGE/'EXECUTION_REGISTRATION.json',STAGE/'DOWNLOAD_MANIFEST.json']
    files+=sorted(p for p in STAGE.glob('DOWNLOAD_PREFLIGHT_*.json'))
    files+=sorted(p for p in STAGE.glob('RESOURCE_PREFLIGHT_*.json'))
    files+=sorted(p for p in RAW.rglob('*') if p.is_file())
    create(STAGE/'SAVED_RECORDING_REGISTRATION.json',dict(
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tool_sha256=sha(Path(__file__)),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        method='Saved independent exact rational completed spans only; preserve failures/open records; no model/parser/C0/QA replay, no timing-equivalence or scientific benefit claim.'))


def record():
    reg=read(STAGE/'SAVED_RECORDING_REGISTRATION.json')
    assert sha(Path(__file__))==reg['tool_sha256']
    assert all(sha(ROOT/p)==h for p,h in reg['inputs'].items())
    manifest=read(STAGE/'DOWNLOAD_MANIFEST.json')
    for item in manifest:
        p=RAW/item['name'];assert p.stat().st_size==item['size'] and sha(p)==item['sha256']
    observations=[json.loads(read(p)['stdout']) for p in STAGE.glob('DOWNLOAD_PREFLIGHT_*.json')]
    assert observations and all(x['complete'] and not x['owned_processes'] for x in observations)
    r=read(RAW/'RESULT.json');a=read(RAW/'INDEPENDENT_TIMED_REQUEST_AUDIT.json')
    assert len(r['receipts'])==2
    workers=[x for x in a['phase_summaries'] if x['role']=='WORKER']
    assert len(workers)==2
    rows=[]
    for x,receipt in zip(workers,r['receipts']):
        totals={k:Fraction(*v['exact_completed_seconds']) for k,v in x['phase_totals'].items()}
        rows.append(dict(arm=receipt['arm'],elapsed_seconds=receipt['elapsed'],selected=receipt['selected'],
            actual_worker_returncode=receipt['actual_worker_returncode'],worker_reaped=receipt['worker_reaped'],
            natural_exit=receipt['natural_exit'],completed_insertions=x['phase_totals'].get('9',{}).get('completed',0),
            phase_counts={k:{t:v[t] for t in ('completed','open','exception')} for k,v in x['phase_totals'].items()},
            completed_phase_exact_seconds={k:[v.numerator,v.denominator] for k,v in totals.items()},
            completed_phase_seconds={k:float(v) for k,v in totals.items()},
            closed=x['closed'],nested_phase_totals_not_added=True,
            dependency_gate_reached='14' in totals,parameter_gate_reached='15' in totals))
    source_start=read(RAW/'TIMED_AUDIT_SOURCE_START.json');source_end=read(RAW/'TIMED_AUDIT_SOURCE_END.json')
    exact_source=(source_start['files']==source_end['files'] and source_start['module_files']==source_end['module_files'])
    scope=dict(registration_sha256=sha(STAGE/'EXECUTION_REGISTRATION.json'),
        requests=2,rows=rows,independent_checks=a['checks_count'],independent_failures=a['failed'],
        independent_failure_names=a['failure_names'],saved_new_forward_replays=a['new_timed_evidence_forward_replays'],
        numpy_head_max_abs_error=a['max_abs_numpy_head_error'],numpy_head_only_bound=2e-5,
        raw_replay_qualified=a['raw_replay_qualified'],
        missing_completed_source_end=a['missing_completed_source_end'],
        independent_loaded_files=len(source_start['files']),independent_module_files=len(source_start['module_files']),
        independent_source_start_end_equal=exact_source,
        run_exit=read(RAW/'RUN_EXIT.json'),audit_exit=read(RAW/'AUDIT_RUN_EXIT.json'),chain_exit=read(RAW/'CHAIN_EXIT.json'),
        download_files=len(manifest),download_bytes=sum(x['size'] for x in manifest),download_all_hashes_match=True,
        actual_owned_tree_exited_before_download=True,
        additional_sync_barriers=True,old_phase7_timing_not_counterfactual_equivalent=True,
        model_expressions_and_parameters_unchanged=True,optimizer_steps=0,
        numeric_contract_unchanged=True,old_QA_repeated=False,
        complete_chain_qualified=False,full_request_budget_qualified=False,formal_controls_qualified=False,
        full_scientific_loaded_closure_qualified=False,learned_output_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,
        GPU_used=False,shared_MD_uploaded=False,final_pickup_seeds_used=False)
    create(STAGE/'STAGE_ADJUDICATION.json',scope)
    report=EXT/'02_配送线/文档/结果与进度/配送_解码动作头与结果转换同步细分计时验收_2026-10-13.md'
    lines=['# 配送解码、动作头与结果转换同步细分计时验收','',
        '事前注册SHA：`'+scope['registration_sha256']+'`。新独占两请求仍固定旧key[12,3]/25masked，原pretrained initial/fresh888同headseed20261012、0更新，不按教师标签/成本选样，不预热同样本，不重复旧2/34/277或其它已完成QA。', '',
        '新engine仅在原phase7内部增加phase20原decode与同步、21原final_norm与同步、22原动作头与同步、23原NumPy转换。原表达式顺序/模型/参数/动作/物理门保留；新增同步会改变调度，下面是本新测量路径的实际段耗时，不能替代旧路径反事实时间或宣称加速。19旧阶段与4新阶段仍在8秒执行+2秒回收内，最终依赖/参数门仍在发布前计时内。', '',
        '| 臂 | 含回收请求秒 | 完成插入 | decode秒 | final_norm秒 | action head秒 | host转换秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        t=row['completed_phase_seconds']
        lines.append('| %s | %.9f | %d | %.9f | %.9f | %.9f | %.9f |'%(row['arm'],row['elapsed_seconds'],row['completed_insertions'],t.get('20',0),t.get('21',0),t.get('22',0),t.get('23',0)))
    lines += ['',
        '表中只累加完成记录；被kill的open段不算完成，phase7包含四子段，不能把父子段求和冒充整个请求。完整候选/C0/放行未到达不意味着不可达或成本为零，生产被kill没有自然末端source闭包。超时保留原计划/实际返回码和全部部分链。', '',
        '独立核查%d项，失败%d；新保存完整前向重放%d。独立原checkpoint234叶76263424参数/fresh888/同seed四叶动作头/10D归一/当前锚点/原JAX完整评分、局部过滤与客户竞争/已完成转移/原incumbent C0/原trace字节及四子段嵌套关系。NumPy头最大差%.12g只适用原登记新头跨库2e-5，参数/特征/latent/JAX仍精确；共享原JAX/Transformer/C0，不是另一模型或物理实现。'%(a['checks_count'],a['failed'],a['new_timed_evidence_forward_replays'],a['max_abs_numpy_head_error']), '',
        '生产退出与独立/监督退出原记录分别为%d/%d/%d；模型worker实际退出码单独保留，driver0不冒充worker自然0。独立加载%d文件/%dmodule，起止身份一致=%s；%d原件%d字节在实际本人树退出后有界只读下载逐SHA一致。'%(scope['run_exit']['exit_code'],scope['audit_exit']['exit_code'],scope['chain_exit']['exit_code'],scope['independent_loaded_files'],scope['independent_module_files'],str(exact_source),scope['download_files'],scope['download_bytes']), '',
        '下一优先根据动作头与decode的实际分段决定保持模型/动作语义的调度实现，尤其检查每步动作数量变化时的重复原语调度；这里不唯一证明编译/JIT原因。下一新事前工程必须包含独立原始表达式一致性、固定预算、全参数/实际依赖计时内门，不靠放宽8秒或预热旧样本。正式已训同初始化CE/同任务已训随机、各控制/等耗时MaskCO迭代/独立强动态接拒和科学完整依赖仍待；第2族FAIL封闭、第3族未启动。取货已权威终局64/callback80/pending空，不跑callback81或旧无模型QA。稳定、可归因MaskCO核心在线增益尚未建立。','']
    create(report,'\n'.join(lines))
    next_action='New score_split_timed_execution_root_v1 two fixed initial requests: see STAGE_ADJUDICATION exact completed phases20decode/21norm/22head/23host and original7 nesting. Additional synchronization explicitly registered; not old timing counterfactual or speed gain. Independent raw failures='+str(a['failed'])+'; preserve all original outputs. Do NOT rerun two split requests/new saved forward replays/audit/old QA. Next implement bounded semantics-preserving scheduling based on measured head/decode segments and independently verify original expressions, parameters/actions under fixed8+2 and final gates in window; no label/cost model selection/sample warmup. Formal trained CE/task random/strongdynamic/equal-time iteration/scientific closure false, family2closed/family3notstarted, pickupterminal64/pendingempty/no81.'
    note='\n> 2026-10-13 配送评分四子段实际诊断：新score_split_timed_execution_root_v1两固定initial请求，phase20decode/21norm/22head/23host同步计时；新增同步非旧流程反事实时间，原表达式/参数未变。独立%d项%d失败，新保存前向%d；详细耗时见新报告与STAGE_ADJUDICATION。预算8+2保持，部分链/原计划回退/实际退出原件保留，科学完整预算/正式控制/稳定增益不升级。下一优先语义保持调度实现，不重复本两请求/前向/审计或旧QA；第2族封闭、第3族未启，取货终局64/pending空不重跑。\n'%(a['checks_count'],a['failed'],a['new_timed_evidence_forward_replays'])
    entries=[EXT/'项目当前状态.md',EXT/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',EXT/'02_配送线/README.md',EXT/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup=TREE/'current_entries_before_score_split_20261013';backup.mkdir(exist_ok=False);changes=[]
    for p in entries:
        raw=p.read_bytes();q=backup/p.relative_to(EXT);q.parent.mkdir(parents=True,exist_ok=True);q.open('xb').write(raw)
        if p.suffix=='.json':
            w=json.loads(raw.decode('utf-8-sig'));w['updated']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
            w['delivery'].update(score_split_component=scope,next=next_action)
            w['delivery']['active_score_split_stage']['status']='completed_actual_tree_exited_no_scientific_gate_upgrade'
            w['latest_delivery_stage_receipts_root']=STAGE.relative_to(ROOT).as_posix()
            new=json.dumps(w,ensure_ascii=False,indent=2)+'\n'
        else:
            new=note+raw.decode('utf-8-sig')
            if p==EXT/'02_配送线/README.md':new+='\n- [评分四子段计时验收](文档/结果与进度/'+report.name+')\n'
        assert p.read_bytes()==raw
        p.write_text(new,encoding='utf-8',newline='\n')
        changes.append(dict(path=p.relative_to(EXT).as_posix(),before_sha256=sha(q),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=report.relative_to(EXT).as_posix(),shared_MD_uploaded=False))
    create(TREE/'reproduction/SCORE_SPLIT_TIMING_COMPONENT_EVIDENCE_20261013.json',scope)
    create(STAGE/'LOCAL_RECORDING_RECEIPT.json',dict(entries=len(changes),report_sha256=sha(report),next=next_action,raw_inputs_unchanged=all(sha(ROOT/p)==h for p,h in reg['inputs'].items())))
    print(json.dumps(dict(recorded=True,rows=rows,independent_failures=a['failed'],stable_online_gain=False)))


if __name__=='__main__':{'seal':seal,'record':record}[sys.argv[1]]()
