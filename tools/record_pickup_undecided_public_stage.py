"""Append engineering evidence, preserving previous local entry bytes."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';P=R/'pickup';OUT=P/'undecided_public_input_root_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,value):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    a=read(OUT/'RESULT.json');b=read(OUT/'INDEPENDENT_UNDECIDED_PUBLIC_FIELDS.json');reg=read(OUT/'REGISTRATION.json')
    assert (a['checks_count'],a['failed'],b['checks_count'],b['failed'])==(70,0,485,0)
    assert b['producer_sha256']==sha(OUT/'RESULT.json') and all(sha(p)==h for p,h in reg['sources'].items())
    for name in ('qualify_pickup_undecided_public_v2.py','audit_pickup_undecided_public_v2.py','pickup_undecided_public_v1.py'):
        shutil.copy2(ROOT/'tools'/name,OUT/('FROZEN_'+name))
    report=E/'01_取货线/文档/结果与进度/取货_未决新订单公共输入与首揭示事件组件_2026-10-11.md'
    text='''# 未决新订单公共输入与首揭示事件组件（2026-10-11）

新独占pickup/undecided_public_input_root_v2实现独立的pickup-public-undecided-v1输入，显式区分已接受、已拒绝、已观察未决三类订单。当前订单必须属于未决集合且未服务，不允许通过向真实拒绝集合写占位来满足导出门。70项运行检查与另进程485项独立字段核验均0失败，原14项选定输入/源/解释器/NumPy/native身份起止一致。它是公共输入工程资格，不是接拒决策、学习提议、评分、安装或在线收益资格。

## 实际捕获与派生边界

三项旧请求0/9/13各做原私有日和隐藏属性/额外四个未来订单变体，共六个派生输入。派生夹具仅在导出之前移除当前订单的旧拒绝占位，不改变其他拒绝、接受、原计划、承诺、服务或轨迹；这不是重演真实历史接拒。公共导出本身没有修改任一私有状态。旧public_state.export_public及public_fleet.validate_public仍拒绝未决输入，原字节和决定完备合同保持；新格式不能直接喂给旧已决定输入评分器。

另外两个输入在实际StrictOnlineEnv.run创建车队后，经初始空规划进入原_run_loop的首个on_reveal回调，在接拒决策及该事件_plan_and_commit之前捕获。使用同一个已观察80订单日及隐藏未来变体，首订单与事件不按结果选择；观察钩子捕获后立即停止，未调用接拒算法、求解器、模型或提交，也没有继续至全日。实际第一揭示之前发生过一次空初始规划，不能写“从未规划”。这关闭实际首揭示未决输入路径，但不等连续多事件所有权或后续车辆执行已接线。

全八个报文都保持当前订单未决、不入拒绝集合、不占有路线，并与各自私有输入不变性哈希核对。四对公共报文在隐藏未来属性和数量变化后摘要逐位一致。九个篡改/边界输入被拒绝，包括未决与拒绝重叠、未决遗漏或重复、当前订单已决定或bool、私有数据通道、未决订单入路线、未揭示属性和快照摘要不一致。独立检查器不用新投影器/验证器，按原私有数据逐字段重新投影可见数据、状态分区与原快照，未调用接拒或完整物理评分。

## 首版失败保存及时间范围

v1错误假定原日零时刻有可见订单，实际没有，在选首订单时IndexError；六个派生记录、FAILURE/RUN_EXIT=1及三个冻结源码保存。v2事先声明改在实际首个揭示回调观察，沿用原私有输入、未来变体和当前订单排序；不重做旧已完成接拒/共同评分/C0/安装QA，没有科学参数或效用重试。原v1失败没有改成PASS。

导出组件八次墙钟约1.23–4.34毫秒，仅投影计时；私有恢复、哈希和离线验收均在外，不是全请求预算。该组件未经过独立可中止worker接线、父解码/提交/回收，尚无完整10秒资格。14项选定身份保持只约束所列文件，不等全部加载stdlib/native/DLL封存。

下一步将未决输入接入实际接拒响应、共同历史K10、全C0和编译联合提交，显式保存其他同事件未决订单，不能用拒绝占位或把它们收取拒绝罚金；再完成连续事件/车辆推进、父解码与提交可中止和回收预留。正式模型提议、已训同初始化CE、独立强动态接拒、全依赖及完整日仍待，工程合格后另登记新鲜有限科学日。

配送第2族原FAIL封闭、科学训练/DEV/确认关闭，四组方向结构资格沿用；不启动第3族或重复已完成QA。负结果31必要门/183历史封存与C1联合失败保持，仅追加本阶段边界。没有服务器/GPU/共享MD上传、新科学数据或收益。稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。
'''
    create(report,text)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),runtime_checks=70,runtime_failed=0,independent_field_checks=485,independent_failed=0,derived_old_state_inputs=6,actual_first_reveal_inputs=2,adversarial_boundaries=9,selected_sources=14,private_state_mutated_by_projector=False,old_rejection_placeholder_added=False,old_decided_schema_modified=False,v1_failure_preserved=True,component_qualified=True,admission_scoring_installation_qualified=False,continuous_live_event_qualified=False,full_loaded_dependency_closure_qualified=False,full_live_request_qualified=False,stable_online_gain=False,shared_MD_uploaded=False,result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_UNDECIDED_PUBLIC_FIELDS.json'),registration_sha256=sha(OUT/'REGISTRATION.json'))
    create(R/'reproduction/UNDECIDED_PUBLIC_INPUT_EVIDENCE_20261011.json',receipt)
    backup=R/'current_entries_before_undecided_public_20261011';backup.mkdir(exist_ok=False)
    entries=read(R/'current_entries_before_private_joint_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
    note='\n> 2026-10-11取货未决输入组件：新三分状态不写拒绝占位；6派生旧输入+2实际首揭示回调输入，70运行/485独立字段项0失败，9边界拒绝、隐藏未来不变。v1零时刻无可见订单夹具失败原件保留，v2首事件观察不决策、不继续全日。仅公共输入资格，连续接拒/评分安装、正式CE与全日10秒仍待，无收益。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(undecided_public_input_component_qualified=True,undecided_public_input_directory=str(OUT.relative_to(ROOT)),undecided_public_runtime_checks=70,undecided_public_independent_checks=485,actual_first_reveal_public_input_qualified=True,true_undecided_reveal_qualified=False,continuous_live_event_qualified=False,full_live_request_qualified=False,next='Integrate explicit undecided public input into admission, K10/C0/compiled joint commit without rejection placeholders or charging pending penalties; preserve other simultaneous pending orders. Then continuous event ownership/vehicle execution, interruptible parent decode/commit and cleanup reserve, full dependency closure, formal learned/same-init trained CE and independent strong admissions/full day before fresh scientific preregistration.')
            s['reproduction']['undecided_public_boundary']='research/review_3080_20261010/reproduction/UNDECIDED_PUBLIC_INPUT_EVIDENCE_20261011.json';new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+old if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else old+note
            if f.parent==E/'01_取货线':new+='\n- [未决新订单公共输入与首揭示事件组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');identities.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=identities,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
    print(json.dumps(dict(updated_entries=len(identities),runtime=70,independent=485,failed=0,full_live_request=False)))
if __name__=='__main__':main()
