"""Close the authorized three-line stage locally, preserving scientific assets."""
from pathlib import Path
import datetime, hashlib, json, os, re, zipfile
EXT=Path(__file__).resolve().parents[1]
OUT=EXT/'results/i3_online_recovery_1009_v1'
REPORT=EXT/'01_取货线/文档/结果与进度/I3_首开发40日正式裁决与失败诊断_2026-10-09.md'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
final=read(OUT/'LOCAL_ONLINE_FINAL_ADJUDICATION_20261009.json')
identity=read(OUT/'INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009_v2.json')
assert final['formal_adjudicable'] and not final['registered_primary_passed'] and identity['qualified']
delivery=read(EXT/'02_配送线/文献工件/2026-10-09/LOCAL_EVIDENCE_ACCEPTANCE.json')
negative=read(EXT/'03_负结果线/核查工件/论文封存与收口检查_2026-10-09.json')
candidate=read(EXT/'results/three_line_parallel_20261009/PARENT_STATIC_CANDIDATE_ACCEPTANCE.json')
assert not delivery['protected_file_mismatches'] and candidate['all_checks_pass']
package=EXT/'negative_result_paper'
original_mismatch=[r['path'] for r in negative['package_preservation_baseline']['files']
                   if sha(package/r['path'])!=r['sha256']]
assert not original_mismatch
diagnosis={'scope':'Arithmetic and descriptive analysis of the original 160 day summaries and prior risk diagnostic; no policy/model runs',
 'registered_primary_failed':True,'decision_objective_gain_not_established':True,
 'utility_gap_decomposition':final['paired_comparisons']['cond_hist']['decomposition'],
 'observed':'I3 revenue lower by96.75; reject loss higher by48.375; fuel lower by0.445433. Exact net -144.679567.',
 'timeouts':'I3 has fewer timeout events than cond/reconstruction; this does not exclude timing-related effects at different states.',
 'unidentified_hypotheses':['teacher unlimited-time vs deployment10s','fixed teacher-policy state coverage',
     'generator lacks fleet/resource inputs','surrogate-to-utility and slot-dependence mismatch'],
 'mechanism_causally_identified':False,'new_mechanism_experiments':False,
 'raw_online_trajectory_replay_available':False,'no_numeric_retry':True,'next_science_requires_separate_authorization':True}
save(OUT/'LOCAL_FAILURE_DIAGNOSIS_20261009.json',diagnosis)
paths=[
 '项目当前状态.md','00_项目导航.md',
 'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md',
 'docs/当前规划/结果与进度/执行进度表.md',
 'docs/当前规划/决策与审计/三线并行推进_执行范围与验收_2026-10-09.md',
 'docs/当前规划/决策与审计/取货与配送双线独立实验_执行与方向决策_2026-10-08.md',
 'docs/当前规划/决策与审计/I3定时跟进与自动裁决_2026-10-08.md',
 '01_取货线/README.md','01_取货线/文档索引.md','01_取货线/实验结果索引.md',
 '01_取货线/文档/协议与审计/I3固定教师决策训练_执行预声明_2026-10-08.md',
 '01_取货线/文档/协议与审计/方向A_I3采集位置修订提案_2026-10-08.md',
 '01_取货线/文档/结果与进度/I3_固定训练验收与在线入口恢复_2026-10-09.md',
 '01_取货线/文档/结果与进度/I3_128状态独立验收与固定训练启动_2026-10-09.md',
 '01_取货线/文档/结果与进度/I3_教师目标留出风险独立裁决_2026-10-09.md',
 'results/README.md'
]
updated=[]
for rel in paths:
    p=EXT/rel;old=p.read_text(encoding='utf-8')
    link=os.path.relpath(REPORT,p.parent).replace('\\','/')
    notice=('> **2026-10-09 取货首开发40日正式结案**：128状态、两模型各192步及四臂160日资格通过；'
       'I3/暖启动/纯重建/cond平均效用456.317/446.969/473.361/600.997，全部hard=1。'
       'I3−cond为−144.680 [−191.109,−100.125]，主门FAIL；I3−纯重建−17.044 [−47.283,14.581]，归因未建立。'
       '348项实际结束封存、数据/合同/池再生身份及2000次配对CI独立复核通过，全部超时保留。'
       '原失败分叉诊断与本地归档完成，不复验双种子、不重跑；最终27/28未触碰。'
       '配送待审设计和负结果本轮有限交付已验收，按约定收束定时任务；待审科学/同步/发布不自动执行。'
       '仍无MaskCO单步核心在线增益、等预算迭代或完整强动态对照证据。'
       '[正式裁决与诊断]('+link+')。以下较早时间状态为历史快照；文档仍仅本地。\n\n')
    assert '**2026-10-09 取货首开发40日正式结案**' not in old
    pos=old.find('\n')
    new=old[:pos+1]+'\n'+notice+old[pos+1:].lstrip('\n')
    if rel=='00_项目导航.md':
        new=new.replace('128状态及两模型192步已验收；入口工程恢复后执行四臂40日首门',
                        '128/192/40日资格通过；首主门失败，诊断归档完成，当前有限链收束')
    if rel=='01_取货线/README.md':
        new=new.replace('128状态及独立验收 → I3/纯重建各192步 → 四臂40日首门 → 过门才双种子。',
                        '128状态及两模型192步已合格完成；四臂40日主门失败并归档，按原分叉未启动双种子。')
    p.write_text(new,encoding='utf-8');updated.append(rel)
state_path=EXT/'results/three_line_parallel_20261009/WORK_STATUS.json'
state=read(state_path)
state['pickup'].update(status='closed_first_primary_gate_failed',last_actual_server_check=identity['checked_at_server'],
 owned_spawn_workers=0,first_formal_gate_present=True,
 last_actual_check_archive='results/i3_online_recovery_1009_v1/INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009_v2.json',
 three_stage_qualifications=final['three_stage_qualifications'],first_primary_passed=False,
 objective_attribution_passed=False,diagnosis_and_archive_completed=True,
 formal_report=str(REPORT.relative_to(EXT)).replace('\\','/'),
 action='Closed registered numerical chain after failed first primary; no two-seed repetition, parameter changes or final tests.')
state['authorized_remaining_stage']='None in current bounded three-line authorization. New science, remote synchronization and publication require separately authorized scope.'
state['all_authorized_bounded_stages_completed']=True
state['automation']['pause_reason']='First pickup primary failed; diagnosis/archive completed; bounded delivery design and negative-paper closeout completed.'
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Archive original downloads, independent audits and previous qualification reports.
zip_path=OUT/'I3_FINAL_READONLY_EVIDENCE_20261009.zip'
assert not zip_path.exists()
files=[p for p in OUT.rglob('*') if p.is_file() and p!=zip_path and p.suffix in ('.json','.sha256','.exit','.log','.py','.sh')]
files += [REPORT,EXT/'tools/audit_i3_online_final_local_1009.py',EXT.parent/'tools/audit_i3_final_download_1009.py',
 EXT/'results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json',
 EXT/'results/i3_registered_1008_v5/LOCAL_FIXED_SCHEDULE_AUDIT_20261009.json']
with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,str(p.relative_to(EXT.parent)).replace('\\','/'))
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
links=[]
for p in [REPORT]+[EXT/r for r in paths]:
    content=p.read_text(encoding='utf-8')
    targets=re.findall(r'\]\(([^)]+)\)',content) if p==REPORT else [
        os.path.relpath(REPORT,p.parent).replace('\\','/')]
    for t in targets:
        if t.startswith(('http:','https:')):continue
        target=p.parent/t.split('#')[0]
        assert target.exists(),(p,t)
        links.append({'from':str(p.relative_to(EXT)),'to':t})
completion={'completed_at_local':str(datetime.datetime.now()),'all_current_authorized_stages_complete':True,
 'pickup_formal_primary_passed':False,'pickup_mechanism_diagnostic_archive_complete':True,
 'delivery_bounded_literature_and_review_complete':True,'negative_paper_bounded_closeout_complete':True,
 'original_negative_package_files_checked':len(negative['package_preservation_baseline']['files']),
 'original_negative_package_mismatches':original_mismatch,'artifact_zip':str(zip_path.relative_to(EXT)),
 'artifact_zip_sha256':sha(zip_path),'artifact_zip_files':len(files),
 'local_docs_updated':updated,'new_local_links_verified':len(links),
 'source_or_protocol_or_weights_changed':False,'new_numeric_runs':False,'remote_document_uploads':False,
 'scheduled_task_pause_required':True}
save(OUT/'LOCAL_THREE_LINE_CLOSEOUT_ACCEPTANCE_20261009.json',completion)
print(json.dumps(completion,ensure_ascii=False,indent=2))

