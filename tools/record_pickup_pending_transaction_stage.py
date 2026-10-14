"""Append qualified pending adapter evidence and back up current local entries."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/pending_admission_transaction_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,value):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    from qualify_pickup_joint_policy_transaction_root_v3 import codecs
    _,decode=codecs();raw=decode(read(OUT/'RESULT.json'));audit=read(OUT/'INDEPENDENT_PENDING_TRANSACTION.json');reg=read(OUT/'REGISTRATION.json')
    assert (raw['checks_count'],raw['failed'],audit['checks_count'],audit['failed'])==(71,0,350,0)
    assert audit['producer_sha256']==sha(OUT/'RESULT.json') and audit['registration_sha256']==sha(OUT/'REGISTRATION.json')
    assert all(sha(p)==h for p,h in reg['sources'].items())
    for name in ('qualify_pickup_pending_transaction_v1.py','audit_pickup_pending_transaction_v1.py','pickup_pending_transaction_v1.py','pickup_partial_common_v1.py','pickup_partial_trace_v1.py','pickup_partial_known_v1.py','pickup_partial_validation_v1.py','pickup_undecided_public_v1.py'):
        shutil.copy2(ROOT/'tools'/name,OUT/('FROZEN_'+name))
    report=E/'01_取货线/文档/结果与进度/取货_未决输入至接拒共同评分与联合提交组件_2026-10-11.md'
    text='''# 未决输入至接拒共同评分与联合提交组件（2026-10-11）

新独占pickup/pending_admission_transaction_root_v1接通此前明确未决输入与实际own_uncond_dynamic当前订单接拒、own_incumbent路线提议、原200历史共同K10/20完整分支、C0物理关闭、编译PolicyState及JointPolicyOwner整指针发布。71项运行检查与另进程350项独立复算均0失败；五请求及五新语义篡改边界全部保留。它关闭未决格式不能接入旧决定完备评分器的接口缺口，不是连续现场整请求、正式学习臂、科学收益或完整10秒资格。

输入复用此前已保存的未决组件报文：旧请求0/9/13各一个派生未决输入、实际首揭示捕获的一个封存报文，及明确派生的双未决边界。双未决边界从请求13选择ID最小的另一已拒绝订单25，事前声明移出拒绝、加入未决；不是实际同时揭示、策略历史逐项重现或新科学样本。没有重复旧投影组件或154/76/41/159/16故障/请求QA，也没有重新生成科学日。

五次实际接拒为3接受/2拒绝，双未决当前订单3被拒绝、另一订单25仍未决，不进入接受/拒绝集合或任何路线。所有公共已观察特征逐位保持，条件历史采样器可见所有观察订单，包括其显式未决状态；只对当前订单调用原接拒算法一次。原C0物理能耗、品质、货物、时间窗、承诺服务、返仓及全部车队均按原实现闭合。两个同计划候选的共同十场景完整分支逐位一致，路线未替换，无模型调用或路线收益。

## 明确的临时评价合同

旧full-day冷链评价器要求所有已观察订单已接受或拒绝，直接评价仍未决的25会报告unaccounted_order。不能把该订单写入拒绝占位、删掉它的需求/特征或把旧评价器结果改为PASS。

本组件另存pickup-public-partial-admissions-v1格式和独占临时评价器副本，唯一覆盖语义改动是将明确未决集合纳入观察订单三分分区；同时拒绝未决与已决定重叠、域外未决及未决被服务。原物理和算术主体保留，不改需求、特征、拒绝损失、正式评价规则或原评价器字节。临时评价只认证当前接受义务与既有拒绝记录，未决订单无服务义务、无拒绝罚金；输出显式provisional_not_full_day_utility=True。它不是终局完整接拒资格，正式完整日末必须未决为空并回到原完整覆盖门。

另进程不用新接拒适配器、评分函数、临时证书函数或临时评价器复算：直接原UncondDynamicReplanner重放当前接拒，原public_fleet.evaluate_known复跑全C0轨迹，核其失败只在显式剩余未决时为unaccounted_order；所有其余原物理/算术字段与临时报告逐位相同。再直接组装原EnergyCompleteSAA、条件历史采样及十场景各两个候选完整分支，全部branch和NumPy1.26.4分组mean十六进制精确一致。共享原物理模型与算术库，不是另一套物理实现。

双未决临时效用206.339215相对同状态单未决201.339215增加5，仅因为工程派生移出了25的既有拒绝罚金；这不是策略正向收益，不进入科学效用/归因。所有候选计划相同，不以临时数值挑选模型、权重或新样本。

## 时间、来源与未闭合边界

五正常请求父序列化、worker实际接拒/共同评分/C0编译、父解码/guard/整指针提交及本人worker回收墙钟为3.491200/2.864336/2.296284/3.390857/2.274879秒，均在10秒内。worker预加载历史和解释器在计时外；报文已封存，未包含真正现场公共提取与车辆/事件持续所有权。当前组件使用同步父管道写入及可信pickle解码，不能抢占这些父阶段；没有新的延迟故障或任意阶段预算保证，正常低于10秒不等完整硬实时资格。完整现场接线必须接回已有共享内存可中止传输并为解码、提交与回收预留时间。

26项所列已加载域代码、实际新适配器、输入/历史、解释器/NumPy/native身份起止一致，实际worker全部回收；不是全部stdlib/native/DLL依赖闭包。原决定完备schema、旧投影器、评价器、安装模块、历史数据和已完成QA字节未修改。没有服务器/GPU/共享MD上传、科学新数据、模型训练或新权重。

下一步实现真实事件所有权与车辆执行的连续接线，同事件依次处理时显式保存其余未决，完成末端未决为空的原完整评价；父传输/解码/提交可中止、回收余量及全依赖仍待。正式模型提议、同初始化已训CE、独立强动态接拒和完整日仍未合格，工程全部合格后才数据前登记有限新科学日。配送第2族FAIL封闭，不训练/重采/重审计/DEV/确认，第3族未启动；剩余30错配与生产动作适配工程继续。负结果31必要门/183历史封存与C1联合FAIL沿用，不重复整理。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。
'''
    create(report,text)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),runtime_checks=71,runtime_failed=0,independent_checks=350,independent_failed=0,normal_requests=5,semantic_boundaries=5,current_accepts=3,current_rejects=2,selected_sources=26,
                 pending_input_to_joint_commit_component_qualified=True,remaining_pending_preserved=True,pending_rejection_penalty_added=False,old_full_day_evaluator_modified=False,
                 provisional_evaluation_is_not_full_day_qualification=True,route_changed=False,new_scientific_data=False,full_live_request_qualified=False,continuous_live_event_qualified=False,
                 parent_transport_decode_commit_preemptible=False,full_loaded_dependency_closure_qualified=False,formal_learned_controls=False,stable_online_gain=False,shared_MD_uploaded=False,
                 result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_PENDING_TRANSACTION.json'),registration_sha256=sha(OUT/'REGISTRATION.json'))
    create(R/'reproduction/PENDING_TRANSACTION_EVIDENCE_20261011.json',receipt)
    backup=R/'current_entries_before_pending_transaction_20261011';backup.mkdir(exist_ok=False)
    entries=read(R/'current_entries_before_undecided_public_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
    note='\n> 2026-10-11取货未决至联合提交组件：五封存/派生输入实际接拒+共同K10/全C0/编译发布，71运行/350独立项0失败；双未决保留另一未决且无拒绝罚金。新临时评价显式未决分区，旧完整日覆盖门不改；3接受2拒绝/计划未换，不是收益。真实连续事件/车辆接线、父可中止与回收余量、全依赖及正式学习对照/强动态/全日10秒仍待。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(pending_input_to_joint_commit_component_qualified=True,pending_transaction_directory=str(OUT.relative_to(ROOT)),pending_transaction_runtime_checks=71,pending_transaction_independent_checks=350,remaining_pending_preserved=True,provisional_evaluation_not_full_day=True,continuous_live_event_qualified=False,full_live_request_qualified=False,
                next='Wire continuous real event ownership and vehicle execution with explicit pending preservation and empty pending set at final original full-day evaluation; connect cancellable shared transport, interruptible parent decode/commit and cleanup reserve, full dependency closure. Formal learned/same-init trained CE and independent strong admissions/full day remain required before fresh scientific preregistration.')
            s['reproduction']['pending_transaction_boundary']='research/review_3080_20261010/reproduction/PENDING_TRANSACTION_EVIDENCE_20261011.json';new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+old if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else old+note
            if f.parent==E/'01_取货线':new+='\n- [未决输入至接拒共同评分与联合提交组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');identities.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=identities,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
    print(json.dumps(dict(updated_entries=len(identities),runtime=71,independent=350,failed=0,full_live_request=False)))
if __name__=='__main__':main()
