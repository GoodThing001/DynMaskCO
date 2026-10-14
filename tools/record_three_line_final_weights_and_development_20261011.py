"""Back up and update only16 mutable local current entries; no remote MD."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=read(R/'delivery/remote_finite_research_root_v2/training_downloaded/ACCEPTED_FIXED_TRAINING.json')
    budget=read(R/'delivery/remote_final_model_budget_bundle_root_v3/downloaded/ACCEPTED_FINAL_MODEL_BUDGET.json')
    p=read(R/'pickup/common_candidate_saa_root_v2/INDEPENDENT_ROOT_COMMON_SCORER_QA.json')
    receipt=read(R/'AUTOMATION_RESTORE_RECEIPT_20261011.json')
    assert d['failed']==p['failed']==0 and budget['independent_acceptance'] and receipt['new_automation_id']=='maskco'
    backup=R/'current_entries_before_final_weight_acceptance_and_monitor_restoration_20261011';backup.mkdir(exist_ok=False)
    entries=[DOC,E/'项目当前状态.md',E/'00_项目导航.md',E/'docs/当前规划/结果与进度/执行进度表.md',
      E/'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries +=[E/line/name for line in ('01_取货线','02_配送线','03_负结果线') for name in ('README.md','实验结果索引.md','文档索引.md')]
    entries +=[E/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    header='> **2026-10-11 当前有效更新**：配送两模型各128固定步与实际最终权重重载独立2091项通过；父子同CPU15、七臂真实总10秒及超时回退独立1764项通过，唯一16新快照开发诊断已启动，确认/动态收益尚未建立。取货原200日/影子资源回归652项通过，新共同K10相同计划评分器43项通过，实际五臂与原子截止仍待。负结果必要子集26项和183历史封存保留。原定时ID maskco-i3已不存在，新安静30分钟任务maskco已实际创建ACTIVE。共享MD仍仅本地。\n\n'
    section='''

## 2026-10-11：固定最终权重、完整资源资格和新16开发诊断

配送新有限64实例×4掩码的教师资格保持6220项0失败。两臂各128固定步训练实际退出0，256状态各两遍、同初始化/日程/SGD/学习率，偏好项0和0.2为唯一目标差异。训练壁钟1159.738秒；最终权重独立另建模型重载、234叶/76,263,424元素、实际更新及前向/重载一致性通过，根审计2091项0失败。纯重建权重SHA `08621a12b6e35cd9711a6647fed28491f005fe96e70ca9d6430993505d6c5ab5`，偏好权重SHA `a9efb20ff8061d1b5223363e6f3ed45d46d8d0fac0a8cd4420155842dfc22829`。本验收不独立重做128次梯度，逐步梯度摘要/日程链与最终实际参数核验的范围分开保留。

最终权重预算v2在两份已观察工程快照上退出0，但计算父进程未绑单核，实际0..15、子进程15，因此不能授完整资源资格。v2首次独立源码检查另有一个失败：本地准备文件写为CRLF，实际上传/执行和登记为LF。只读下载实际执行PY与JSON，证明差异仅为换行且实际执行字节与注册完全一致；`actual_executed_bytes_r2/ACTUAL_BYTE_IDENTITY.json`和第二份R2审计追加保留，第一次审计和全部原源码/产物未覆盖。v2追加数值/回退1764项0失败仍因父资源缺口不合格。

同科学参数的新隔离 `remote_final_model_budget_root_v3` 将全部计费父工作和各worker绑CPU15，已于117.192秒退出0，14结果、96候选完整重放均hard，七臂10毫秒故障注入各触发自身共同初始路线回退并回收worker，独立1764项0失败。故障10毫秒只作工程测试，科学预算仍10秒。实际两个固定最终研究模型真实激活，不把这两份工程快照数字计入研究。预算资格只支持此pre-dispatch快照包装，非动态基线或物理校准。

前置资格后，按既有科学注册 `1f19554a2790d258b7fca4f081aeae38b7250acab817142b08825cf91ab18af1` 实现16新开发快照，另在生成前封存 `fresh_development_root_v1/DEVELOPMENT_REGISTRATION.json`，SHA `47680e5008f716d39a5e2a1cdb64e33be824ed40c3b77a52e1dcb4af8a95bdb4`。数据seed202610111104、顺序seed202610111105；6臂为偏好/纯重建/原暖启动/随机未训/非学习固定8次/GLS，每臂在各顺序位置出现2或3次，全部16×6路线及候选、失败、超时与回退均保留。唯一远端目录 `research/review_3080_20261010/remote_fresh_development_root_v1`，supervisor13892、外壳上限1800秒；禁止重复启动。监督外壳不计入求解资源，计费父进程和全部计算worker均CPU15。数据生成是静态实例准备，实际特征构造/场景掩码/推理/搜索/认证/传输/选择均在请求10秒内计时；预热仅固定合成51节点、静态导入与权重载入。

16开发仅诊断，不挑checkpoint或参数、不替代三组40确认，当前无稳定MaskCO收益。完成后由 `tools/audit_delivery_fresh_development_root_v1.py` 独立再生实例/顺序/掩码、重放所有原始候选与最终路线、复核预算/身份和配对分解。固定8次机会数+共同上限不是耗时相等或ANY50，随机未训练控制也不能隔离预训练本身，GLS此处为距离提议/fullC0择优的快照对照。最终确认和动态在线仍需对应全部注册资格。

### 取货共同候选评分器：窄工程资格

原200日历史和完整能耗守恒修正继续引用607/475/652三份通过原件。新增 `pickup/common_candidate_saa_root_v1` 在读取已观察工程原件时被dataclass白名单拦截，退出1，无科学数据和评分结果，原件保持。新v2明确列入冷链合同/状态/车辆允许类型；用同一已观察公共状态、原200日历史，只抽样一次10个可能未来，候选计划独立复制完整资源状态，保持相同场景评分。公共状态中的车辆资源可影响收益，不能据候选改变需求抽样分布。

v2运行器28项通过，独立原件/历史/因果场景/均值/公共硬证书复算43项0失败。五个标签故意使用同一计划，验证评分接口一致，不能称实际模型或rolling候选五臂已通过。原子全链截止、真实不同候选、全部接/拒分支日志及五臂同实际SAA仍待，不提前生成新在线效用日。影子场景效用与真实日效用区分；不得从工程共同评分通过推断效用改善。

### 负结果与持续任务

负结果必要协议子集最终26项通过和183原历史文件保持；历史全量复现、C1联合确认/等价、MaskCO正向不因新工程通过改变。原 `maskco-i3` 已经不在应用中，旧“ACTIVE”只作历史快照；已按用户原30分钟安静推进授权创建新heartbeat `maskco`，实际状态ACTIVE，创建回执 `research/review_3080_20261010/AUTOMATION_RESTORE_RECEIPT_20261011.json`。仅阶段完成、异常或需要决策时通知，不重复已完成有限链。共享MD没有上传；未经批准不重试此前拒绝的文档同步。
'''
    manifest=[]
    for file in entries:
        if not file.exists():continue
        previous=backup/file.relative_to(E);previous.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,previous)
        old=file.read_text(encoding='utf-8')
        if file==DOC:new=old+section
        elif file.suffix=='.json':
            state=json.loads(old);state['updated']=datetime.now(timezone.utc).isoformat()
            state['delivery'].update(stage='fixed128_and_final_budget_qualified_sole_fresh16_development_running',
              final_training_independent_tensor_qualified=True,final_training_checks=2091,final_training_failures=0,
              final_weight_sha256=d['final_weights'],final_budget_resource_scope='charged_parent_and_workers_CPU15',
              final_model_budget_checks=1764,final_model_budget_failures=0,final_model_budget_and_timeout_qualified=True,
              v2_parent_resource_qualified=False,v2_first_source_check_failure_preserved=True,
              fresh_development_directory='research/review_3080_20261010/remote_fresh_development_root_v1',
              fresh_development_registration_sha256='47680e5008f716d39a5e2a1cdb64e33be824ed40c3b77a52e1dcb4af8a95bdb4',
              fresh_development_supervisor_pid=13892,confirmation_execution_enabled=False,
              next='finish sole16 diagnostic; independently replay all96 outcomes, count all timeout fallbacks; no checkpoint selection or premature dynamic claim')
            state['pickup'].update(common_candidate_scorer_checks=43,common_candidate_scorer_failures=0,
              common_candidate_scorer_scope='five identical plan labels engineering only; not five actual deployment arms',
              next='integrate actual five proposals with common original200 K10 scoring and whole transaction hard deadline; then full engineering days before registered fresh research')
            state['automation']={'id':'maskco','status':'ACTIVE','interval_minutes':30,'actual_creation_receipt':'research/review_3080_20261010/AUTOMATION_RESTORE_RECEIPT_20261011.json',
              'old_id':'maskco-i3','old_id_status':'app_reports_not_existing','routine_notifications':False}
            new=json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            if old.startswith('> **2026-10-11 三线新阶段（当前有效）**'):old=old.split('\n\n',1)[1]
            new=header+old
        file.write_text(new,encoding='utf-8');manifest.append({'path':file.relative_to(E).as_posix(),'before_sha256':sha(previous),'after_sha256':sha(file)})
    (backup/'UPDATE_IDENTITY.json').write_text(json.dumps({'files':manifest,'shared_MD_uploaded':False,'old_frozen_science_modified':False},indent=2))
    print(json.dumps({'local_current_entries_updated':len(manifest),'all_prior_bytes_preserved':True,'stable_online_gain':False}))
if __name__=='__main__':main()
