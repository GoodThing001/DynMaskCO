"""Preserve original audit FAIL, append initialization recovery and prefix evidence."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/live_event_prefix_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    from qualify_pickup_joint_policy_transaction_root_v3 import codecs
    _,decode=codecs();raw=decode(read(OUT/'RESULT.json'));audit=read(OUT/'INDEPENDENT_LIVE_PREFIX.json');r2=read(OUT/'INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json');reg=read(OUT/'REGISTRATION.json')
    assert (raw['checks_count'],raw['failed'],audit['checks_count'],audit['failed'],r2['checks_count'],r2['failed'])==(105,0,999,15,91,0)
    assert audit['producer_sha256']==r2['producer_sha256']==sha(OUT/'RESULT.json') and r2['original_audit_sha256']==sha(OUT/'INDEPENDENT_LIVE_PREFIX.json')
    assert all(sha(p)==h for p,h in reg['sources'].items())
    names=('qualify_pickup_live_prefix_v1.py','audit_pickup_live_prefix_v1.py','audit_pickup_live_prefix_mode_r2.py','pickup_live_shared_transaction_v1.py','pickup_pending_transaction_v1.py','pickup_partial_common_v1.py','pickup_partial_trace_v1.py','pickup_partial_known_v1.py','pickup_partial_validation_v1.py','pickup_undecided_public_v1.py')
    for name in names:shutil.copy2(ROOT/'tools'/name,OUT/('FROZEN_'+name))
    create(OUT/'MODE_RECOVERY_SOURCE_IDENTITY_R2.json',dict(source_sha256=sha(ROOT/'tools/audit_pickup_live_prefix_mode_r2.py'),result_sha256=sha(OUT/'INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json'),original_audit_sha256=sha(OUT/'INDEPENDENT_LIVE_PREFIX.json'),production_sha256=sha(OUT/'RESULT.json'),no_solver_or_admission_or_SAA_rerun=True))
    report=E/'01_取货线/文档/结果与进度/取货_连续揭示与真实车辆执行前缀组件_2026-10-11.md'
    text='''# 连续揭示与真实车辆执行前缀组件（2026-10-11）

新独占pickup/live_event_prefix_root_v1在原已观察80订单日的实际StrictOnlineEnv.run中，预声明执行首14次揭示回调，并在第14次原_plan_and_commit之后停止。原run创建车队并连续推进，实际on_reveal开始计时，公开导出当前未决输入、own_uncond_dynamic接拒、同计划own_incumbent提议、原200历史K10/20完整分支、完整C0关闭认证、编译PolicyState及父JointPolicyOwner发布后，原环境实际规划、派车、服务。这不是旧快照恢复、首事件捕获后立即停止或封存报文输入QA；没有重复71/350、70/485、旧154/76等已完成QA。

运行105项0失败，14次实际联合发布，13接受/1拒绝；实际21次物理推进、7单服务。停止clock=3.602795362472534/event_id=21/replan_count=20，车队为committed/idle/idle，三份轨迹尚未返仓。它关闭首14揭示的连续事件/车辆所有权接线，不是全日完成、所有状态覆盖或最终交付。此前双未决的派生边界资格沿用；本前缀没有实际同刻多新单，不能升级真实同时揭示边界资格。

## 独立执行与原FAIL保留

独立新进程直接原UncondDynamicReplanner运行同一前14回调，逐事件原车队/轨迹/服务掩码、逐接拒、直接公共字段投影、逐请求原完整C0关闭轨迹及全部原SAA场景分支复算。INDEPENDENT_LIVE_PREFIX.json共999项15FAIL，984成功项精确一致；全部15失败仅14请求和末端的mode字符串身份。原报告及源码原件保存，绝不写成999项0失败。

原因是生产父状态原C0ReserveReplanner默认mode=myopic，worker构造UncondDynamicReplanner后restore_state显式保留父标签；检查器直接构造参考UncondDynamicReplanner默认mode=knn，没有恢复登记初始标签。两者实际执行均调用UncondDynamicReplanner重写的无条件动态接拒；标签不选择该重写方法。生产原mode没有被修改或放宽，C0 generic on_reveal的mode分支不是本臂实际接拒入口。

另追加INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json的91项0失败，仅诊断参考初始化假设：用原构造器/restore_state见证标签、核保留标签前后仍绑定同一原无条件handler，并遍历实际on_reveal及plan的11个self方法调用图，全部不读mode；逐14编译/前后/worker状态及末端标签一致。它没有重新调用接拒、C0求解、SAA评分或重跑前缀，没有选样本、改数字或放宽容差。资格由原984成功项加R2窄范围身份闭合组成，原999/15FAIL仍存在。新组件不称原审计整体PASS，且共享原物理模型和算术库，不是另一物理实现。

## 真实计时、共享传输与剩余缺口

14正常请求实际从回调当前订单处理开始，包括父现场公开投影、输入序列化与固定共享内存写入、worker解码/接拒/完整评分/C0认证/编译/编码、共享返回、父pickle解码与guard提交。每次墙钟2.204570–3.553065秒，worker计算窗8秒、父及回收预留2秒，总上限10秒；日初worker解释器/原历史预加载在计时外。共享输入/返回分别上限2MiB，序号单次发布，计时请求不查询管道或传大JSON。各实际请求开始报文和prepared响应原件追加保留。worker在前缀结束后回收，正常请求中没有每次重启或重复预热。

新增一项resident worker输入解码入口延迟故障，0.2秒总上限、0.08秒回收预留，实际到达stage1后取消/kill/reap，共0.137599秒；没有发布或修改原状态。这只证明这次新入口取消边界落在所留余量，不证明任意阶段/负载硬实时。当前父投影、pickle编码/解码、摘要与提交仍同步不可抢占；不能据正常小于10秒或单一故障验收声明完整10秒资格。日末worker回收并非每一请求回收耗时已实测；完整现场预算、父可中止和极端载荷仍待。

28项所列实际域代码/新源/输入/历史/解释器/NumPy/native身份起止一致，不等全部加载stdlib/native/DLL闭包。实际Python3.12.7/NumPy1.26.4/CPU7。旧已观察日仅用于工程，不用最终seed20260927/28，不生成科学新日，不上服务器/GPU，不上传共享MD。原投影器、决定完备schema、物理评价器、安装模块和历史结果字节保持。

本前缀每次当前接拒后已观察pending为空，原完整覆盖门及原全C0报告均通过，但这是已观察义务的关闭认证；真实前缀只有7单服务且无返仓，不能当完整日终局效用。两候选同计划，全14未换路线，无MaskCO模型、同初始化已训CE或独立强动态接拒，不是在线收益。

下一步补齐完整已观察日实际事件与末端交付/原完整评价的工程接线、父可中止解码/提交和回收余量/全依赖，再完善正式学习与同初始化已训CE、独立强动态接拒。全部工程满足后另登记新鲜有限科学日。配送第2族FAIL封闭不训练/重采/重审计/DEV/确认，第3族未启动；余30错配/生产动作适配仍待。负结果31必要门/183历史/C1失败沿用，仅追加本阶段边界。稳定可归因MaskCO核心在线增益、完整等耗时迭代与真实强动态证据仍未建立。
'''
    create(report,text)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),runtime_checks=105,runtime_failed=0,original_independent_checks=999,original_independent_failures=15,original_independent_successful_checks=984,initialization_recovery_checks=91,initialization_recovery_failed=0,original_FAIL_preserved=True,normal_requests=14,physical_events=21,services=7,current_accepts=13,current_rejects=1,selected_sources=28,
                 continuous_live_prefix_component_qualified=True,actual_simultaneous_reveal_qualified=False,full_day_completed=False,parent_preemptible=False,full_live_request_qualified=False,formal_learned_controls=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False,shared_MD_uploaded=False,
                 result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_LIVE_PREFIX.json'),recovery_sha256=sha(OUT/'INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json'),registration_sha256=sha(OUT/'REGISTRATION.json'))
    create(R/'reproduction/LIVE_EVENT_PREFIX_EVIDENCE_20261011.json',receipt)
    backup=R/'current_entries_before_live_prefix_20261011';backup.mkdir(exist_ok=False)
    entries=read(R/'current_entries_before_pending_transaction_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
    note='\n> 2026-10-11取货真实连续前缀：原日首14揭示实际联合提交、21物理推进/7服务、13接受1拒绝，105运行项0失败。原独立999项15标签FAIL保留；其余984精确，新91项初始化身份/实际handler闭合且无求解重跑。共享传输新入口取消0.137599秒落0.2秒预留边界；父同步阶段未可中止、无真实返仓/完整日/学习强动态或收益，完整10秒仍未资格。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(prior);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(continuous_live_prefix_component_qualified=True,live_prefix_directory=str(OUT.relative_to(ROOT)),live_prefix_runtime_checks=105,live_prefix_original_independent_checks=999,live_prefix_original_independent_failures=15,live_prefix_initialization_recovery_checks=91,live_prefix_physical_events=21,live_prefix_services=7,full_observed_day_execution_qualified=False,continuous_live_event_qualified=False,full_live_request_qualified=False,
                 next='Complete actual full observed-day event and delivery execution with original final evaluation, then interruptible parent projection/encode/decode/commit and reserved cleanup plus full dependency closure; formal learned/same-init trained CE and independent strong dynamic admissions remain prerequisites before fresh scientific preregistration. Keep original 999/15 audit FAIL and append-only 91-item label recovery; do not repeat completed prefix QA.')
            s['reproduction']['live_prefix_boundary']='research/review_3080_20261010/reproduction/LIVE_EVENT_PREFIX_EVIDENCE_20261011.json';updated=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [连续揭示与真实车辆执行前缀组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');identities.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=identities,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
    print(json.dumps(dict(updated_entries=len(identities),runtime=105,original_audit=[999,15],recovery=[91,0],full_live_request=False)))
if __name__=='__main__':main()
