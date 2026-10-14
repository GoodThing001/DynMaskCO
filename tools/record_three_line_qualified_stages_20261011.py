"""Update mutable local entries only; snapshot prior bytes before each edit."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=read(R/'pickup/complete_history_energy_qa_root_v1/INDEPENDENT_ROOT_COMPLETE_HISTORY_ENERGY_QA.json')
    d=read(R/'delivery/remote_finite_research_root_v2/teacher_downloaded/ACCEPTED_TEACHER_STATES.json')
    b=read(R/'reproduction/formal_run_requirements_root_v2/FINAL_ROOT_ACCEPTANCE.json')
    assert p['failed']==d['failed']==b['failures']==0
    latest=sorted((R/'delivery/remote_finite_research_root_v2').glob('READONLY_TRAINING_PROGRESS_*.json'))[-1]
    training=read(latest)
    backup=R/'current_entries_before_budget_teacher_and_energy_completion_20261011'
    backup.mkdir(exist_ok=False)
    entries=[DOC,E/'项目当前状态.md',E/'00_项目导航.md',
        E/'docs/当前规划/结果与进度/执行进度表.md',
        E/'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',
        E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries +=[E/line/name for line in ('01_取货线','02_配送线','03_负结果线') for name in ('README.md','实验结果索引.md','文档索引.md')]
    entries +=[E/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    header='''> **2026-10-11 三线新阶段（当前有效）**：取货原200日历史与完整条件SAA接线回归完成，652项0失败；闭合/返仓车辆能耗遗漏已有隔离修正，该已观察日决策与效用未变。配送真实10D完整模型目标和七臂总10秒工程资格已核，64实例256教师状态独立6220项0失败，新同初始化两模型固定128步训练已启动，最终权重独立资格仍待。负结果必要设计子集修订26项0失败，183历史封存和旧负裁决保留。稳定、可归因MaskCO在线增益仍未建立；最新依据见本轮独立核查文档，文档仅本地。\n\n'''
    text='''

## 2026-10-11：完整预算、原历史、资源守恒与新有限训练

以下资格是实际工件核验，不将工程通过写成效用正向。根目录 `research/review_3080_20261010/` 为本轮唯一审计/工程产物树；旧I3、D2、旧二十轮和独立负结果包原字节保持。各段此前快照按时间理解。

### 配送：新完整训练登记与教师验收

真实冷链10维可见特征采用原三维投影零扩展，完整encoder/decoder可训练参数76,263,424、234叶。单步目标工程独立330项0失败；8状态/每臂2步的前置训练仅工程验证，独立日程/损失/完整远端字节50项通过，不能用它作最终研究权重。

七臂真实模型总10秒资格已结束0：预训练、随机、丢弃引导、非学习固定8次、纯重建、偏好目标、GLS在两份已观察工程快照上共14结果；总控独立1729项0失败，96候选和14最终路线重放全部硬可行，所属worker全部回收，准备/推理/搜索/认证/传输均计入父进程截止。此版尚缺专门超时注入。固定8次共用机会数及总时间上限，不等于实际耗时相等，也不等于ANY50持续搜索。模型臂约2.33–2.52秒，非学习约0.70–0.77秒，GLS约9.22–9.99秒；工程样本不得算新效用数据，GLS在此为快照基线。

在这些前置资格后，另立 `delivery/finite_training_research_root_v2/RESEARCH_REGISTRATION.json`，SHA256 `1f19554a2790d258b7fca4f081aeae38b7250acab817142b08825cf91ab18af1`，新seed在数据生成前登记。64新快照各4个50%掩码、共256唯一状态已经生成完毕，独立再生/完整C0轨迹与分解/掩码/保留边/逐行标签/实际源/全部分片6220项0失败；实际256标签全正，负、小差值和无效类别此次均为0。保留这些类别的协议规则不等于本批实际包含了它们；单次regret2教师也不等于最优教师。

正式有限训练引用唯一 `remote_finite_research_root_v2`，同初始化、同256状态两遍、同batch4、同SGD1e−5，两臂各固定128步，区别仅偏好项系数0与0.2。每步记录实际234叶梯度摘要与前后参数摘要，最终一步固定保存，不选checkpoint。训练正在进行或等待独立最终权重资格；以实际退出/完整128行和新进程重载为准，禁止重复启动。源树、协议、教师和权重保持原位置。

研究范围仍为pre-dispatch快照。16开发实例只是诊断；三个40实例确认队列必须另经真实最终权重/全部预算/超时注入和完整身份验收才能启动，当前执行屏障关闭。确认分四对照×三队列，第一假设族花费0.0125、每比较双侧alpha0.0010416667，注册100000次配对bootstrap而非不足的2000次尾部估计。无事后调整主门或挑均值。随机未训练臂不足以归因预训练本身；若主张预训练收益，还需随机初始化匹配训练日程。快照通过也不能替代真正动态在线证据。

资格原件：`delivery/remote_model_budget_bundle_root_v1/downloaded/INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json`；`delivery/remote_finite_research_root_v2/teacher_downloaded/ACCEPTED_TEACHER_STATES.json`。实际最终权重核验器已准备为 `tools/verify_delivery_fixed_training_tensors_root_v2.py`，须训练完整退出0后才独占启动，不训练或选择模型。

### 取货：原200日共同历史与影子资源账本

按原I3历史seed20260925再生全部200日，数据与meta谱系逐项一致；5个拟研究臂共享同K10随机场景资格607项0失败。这证明原历史和抽样接线可用，尚不证明5臂都完成了相同完整SAA或原子硬截止。

发现原 `SaaReplanner` 的资源记账缺陷：`_build_snapshot` 正确保存全车队已耗能耗，但 `_sim_states` 排除closed/returning车辆后，影子预算只加活动车辆能耗，遗漏不可再派遣车辆的已耗和固定返仓能耗。原实际终局A1评估并未因此改错；旧裁决不能改写。新隔离 `shadow_energy_correction_root_v1/energy_complete_saa.py` 保持动作集不变，将不可用车辆的固定耗能和不可改返仓在影子预算及完成账本中计回，原SAA源码未改、S3价格/阈值未调。

158条已观察公共状态账本475项0失败，9个末尾状态存在遗漏，最大706.725kWh发生在全部车辆已关闭的17.004h终局；不能把这个最大值当接单阶段可改善空间。随后在同一个已观察80订单日、原200日历史下，修正前/后各运行一次完整条件SAA，652项独立复核0失败，两臂均53接单/27拒单、效用947.405540、实际能耗579.335561kWh、hard=True、worker超时0；请求最大4.929/5.104秒。所有80个接单决定和终局轨迹效用一致，旧遗漏状态都晚于最后15.866h接单。这证实修正没有制造本样本的效用增益，不外推全体运行点惰性。

报告为 `pickup/complete_history_energy_qa_root_v1/INDEPENDENT_ROOT_COMPLETE_HISTORY_ENERGY_QA.json`。这些仍是已观察工程回归、非新独立开发日；接受请求在worker中断言20完整分支，逐拒绝请求的原始分支计数尚未导出。全部5臂同SAA评分、真实MaskCO联合候选、父进程投影/认证/安装的统一可中止10秒流程仍须闭合，之后才另立新鲜有限在线研究。此前三个人造历史日的工程结果不与原200日结果跨队列比较。

### 负结果：绑定现行主门并保留报告臂

`reproduction/formal_run_requirements_root_v2/` 的当前必要条件适配器绑定PROGRAM现行SHA和v3追加修订，而不是旧PROGRAM字节；主门为角色合格strong rolling、归因为reconstruction control，cond_hist必须保留报告臂。最终26回归0失败、实际最终适配器SHA `8504a68efeb82d193c321702f72dcf15bd315a6d5a94995ac70c2499cd914099`，旧初步hash由FINAL_ROOT_ACCEPTANCE明确取代。

此门只检查必要设计子集，完整真实源码/训练/轨迹语义、历史复现资格及稳定收益均未自动授予。历史183文件、14正式JSON和I3/D2原FAIL不变，C1联合确认失败、CI跨零不是等价、有限面板不能证明全局不可能等边界继续保留。负结果线的交付是可核验、准确的方法结论。

### 文献依据与资源/下一屏障

完整masked目标和部分邻接条件来自上表MaskCO正式论文§4；公共状态、不可撤销承诺和资源守恒按Ulmer等route-based MDP§2/5区分状态与动作；200日历史产生可能未来按Bent & Van Hentenryck §3，不能使用当前测试日真实未来。固定128步、λ0.2、batch4、10秒及固定8次均为本项目预声明选择，论文不保证这些值最优。实验评估和不确定性按Agarwal等原文，额外族花费和尾部模拟精度为本项目统计设计。物理参数仍为标来源的仿真值，不能宣称实测校准。

5090两块卡最近09:08实查95%/81%高负载，未抢占；当前新固定训练采用已资格验证的3080主机CPU15。所有共享MD仅本地维护，未远程上传或公开发布。30分钟持续任务继续ACTIVE，已完成工程只读结案，不重复采集、工程训练、D2或风险链；下一步为固定最终权重独立重载与所有臂真实预算资格，以及取货全臂共同评分/原子截止。
'''
    manifest=[]
    for file in entries:
        if not file.exists():continue
        previous=backup/file.relative_to(E);previous.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,previous)
        old=file.read_text(encoding='utf-8')
        if file==DOC:new=old+text
        elif file.suffix=='.json':
            state=json.loads(old);state['updated']=datetime.now(timezone.utc).isoformat()
            state['pickup'].update(stage='original200_conditional_SAA_and_isolated_energy_ledger_qa_complete',
                original_history_independent_checks=607,energy_ledger_independent_checks=475,
                complete_history_day_checks=652,complete_history_day_failures=0,
                complete_history_fixture_utility=947.4055395163596,ledger_correction_fixture_gain=0.,
                all_five_arm_common_scoring_qualified=False,atomic_all_in_budget_qualified=False,
                next='all five actual arms common conditional K10 scoring and atomic projection/certification/submission before fresh online research')
            state['delivery'].update(stage='fresh256_teacher_accepted_fixed128_two_arm_training_in_progress_or_final_acceptance_pending',
                actual_10D_objective_checks=330,total10s_model_budget_checks=1729,total10s_model_budget_failures=0,
                fresh_teacher_states=256,fresh_teacher_instances=64,fresh_teacher_checks=6220,fresh_teacher_failures=0,
                fresh_teacher_categories=d['categories'],training_live_observation=training,
                research_registration_sha256='1f19554a2790d258b7fca4f081aeae38b7250acab817142b08825cf91ab18af1',
                fresh_training_remote_directory='research/review_3080_20261010/remote_finite_research_root_v2',
                final_training_independent_tensor_qualified=False,confirmation_execution_enabled=False,
                next='finish sole fixed128 training; independently verify actual saved full tensors, all-arm request budget and timeout injection before fresh development16')
            state['reproduction'].update(stage='current_protocol_bound_necessary_subset_root_v2_verified',
                latest_necessary_subset_tests=26,latest_necessary_subset_failures=0,
                final_adapter_sha256=b['adapter_sha256'],cond_hist_report_arm_required=True,
                complete_scientific_qualification=False)
            state['latest_stage_receipts_root']='research/review_3080_20261010'
            new=json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            if old.startswith('> **2026-10-11 新增交接独立核查与修正'):
                old=old.split('\n\n',1)[1]
            new=header+old
        file.write_text(new,encoding='utf-8')
        manifest.append(dict(path=file.relative_to(E).as_posix(),before_sha256=sha(previous),after_sha256=sha(file)))
    write_file=backup/'UPDATE_IDENTITY.json'
    with write_file.open('x',encoding='utf-8') as f:json.dump(dict(files=manifest,shared_MD_uploaded=False,
        frozen_scientific_source_or_results_modified=False,stable_online_gain=False),f,ensure_ascii=False,indent=2)
    print(json.dumps({'mutable_local_entries_updated':len(manifest),'old_bytes_backed_up':True,'stable_gain':False}))
if __name__=='__main__':main()
