"""Archive observed-day continuation evidence and preserve every prior current entry."""
import json,hashlib,shutil
import numpy as np
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';OUT=R/'pickup/observed_day_extension_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def native(x):
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,np.generic):return native(x.item())
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    return x
def create(p,x):
    data=x if isinstance(x,str) else json.dumps(native(x),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(p).open('x',encoding='utf-8') as f:f.write(data)
def main():
    from qualify_pickup_joint_policy_transaction_root_v3 import codecs
    _,decode=codecs();raw=decode(read(OUT/'RESULT.json'));audit=read(OUT/'INDEPENDENT_OBSERVED_DAY_EXTENSION.json');reg=read(OUT/'REGISTRATION.json')
    assert (raw['checks_count'],raw['failed'],audit['checks_count'],audit['failed'])==(472,0,4711,0)
    assert audit['producer_sha256']==sha(OUT/'RESULT.json') and audit['registration_sha256']==sha(OUT/'REGISTRATION.json')
    assert all(sha(p)==h for p,h in reg['sources'].items())
    assert len(raw['records'])==66 and not raw['first14_replayed'] and raw['full_day_completed'] and not raw['full_live_10s_qualified']
    assert reg['prefix_result_sha256']==sha(R/'pickup/live_event_prefix_root_v1/RESULT.json')
    for p,h in reg['sources'].items():
        if Path(p).parent==ROOT/'tools':
            target=OUT/('FROZEN_'+Path(p).name)
            if target.exists():assert sha(target)==h
            else:shutil.copy2(p,target)
    final=raw['final'];evaluation=raw['original_final_evaluation'];walls=[v['wall_s'] for v in raw['records']]
    metrics=dict(runtime_checks=472,runtime_failed=0,independent_checks=4711,independent_failed=0,extension_requests=66,extension_physical_events=137,total_reveals_with_sealed_prefix=80,total_physical_events_with_sealed_prefix=158,accepted=len(final['state']['accepted']),rejected=len(final['state']['rejected']),actual_services=sum(len(t.services) for t in final['traces']),clock=final['clock'],event_id=final['event_id'],replan_count=final['replan_count'],fleet_status=[v.status for v in final['fleet']],return_arrivals=[t.return_arrival for t in final['traces']],wall_min_s=min(walls),wall_max_s=max(walls),utility=evaluation['utility'],hard_feasible=bool(evaluation['hard_feasible']),failures=list(evaluation['failures']),selected_sources=len(reg['sources']))
    report=E/'01_取货线/文档/结果与进度/取货_封存连续前缀后续至原完整日终局工程与归档闭合_2026-10-12.md'
    text=f'''# 封存连续前缀后续至原完整日终局工程（2026-10-12）

新独占pickup/observed_day_extension_root_v1在执行前登记，以原live_event_prefix_root_v1第14次原_plan_and_commit之后的实际封存状态为唯一起点。恢复该实际车队/冷链状态、完整已执行轨迹、服务掩码、已接受/拒绝、clock/event_id/replan_count及原参数，直接进入原StrictOnlineEnv._run_loop；没有重新执行起点plan_and_commit，也没有重跑前14次已合格接拒/评分、原105/999/91 QA。普通run_resumed代表提交前决策点，不能直接用于本提交后边界。

后续66次实际揭示按原事件顺序执行，真实当前未决公共投影、own_uncond_dynamic当前-only接拒、own_incumbent同计划提议、原200历史共同K10及20完整分支、原完整C0已观察义务关闭认证、编译、共享返回、父JointPolicyOwner发布，继而原物理事件推进/派车/服务/返仓。新运行472项0失败，后续137次实际物理推进。结合原已封存14揭示/21推进，共80揭示/158推进；末端{metrics['accepted']}接受/{metrics['rejected']}拒绝，{metrics['actual_services']}单实际服务，全部已接受订单服务、全部原80订单已有互斥接拒，使用车辆均实际返仓。末端clock={metrics['clock']}/event_id={metrics['event_id']}/replan_count={metrics['replan_count']}，车辆状态{metrics['fleet_status']}，实际返仓时间{metrics['return_arrivals']}。

原完整coldchain_evaluator_a1直接评估实际全轨迹和完整真实80订单数据，hard_feasible=True、failures=[]，效用{metrics['utility']:.12f}。未使用临时partial覆盖豁免，末端没有pending。该数值仅为本已观察日、本own_uncond_dynamic/同计划臂的工程终局描述，不能与旧cond_hist日作为同策略复现，不能称MaskCO增益，也没有重跑不同策略追均值。

## 独立复算与保留边界

新的独立进程从同一封存提交后起点恢复原UncondDynamicReplanner，显式restore_state保留原mode标签，直接执行原事件循环后续，未调用新联合接拒/评分适配器来获得参考接拒。逐66实际接拒、137物理事件的车队/轨迹/服务掩码、真实公共8字段投影与未决分区、原完整C0关闭轨迹/全部报告字段、共同历史场景/全部1320分支/原NumPy分组mean、编译与发布以及末端完整原评价逐位/身份复算，INDEPENDENT_OBSERVED_DAY_EXTENSION.json共4711项0失败。

原前缀999项15标签FAIL保持原字节，先前984成功项与91项初始化恢复资格沿用；本次没有重新运行那些QA，不能把原999写成0FAIL。新后续审计共享原物理评价器/仿真器和原算术库，是独立组装与进程复算，不是另一物理实现。生产与独立分别{read(OUT/'SUPERVISION_PRODUCTION.json')['wall_s']:.3f}/{read(OUT/'SUPERVISION_AUDIT.json')['wall_s']:.3f}秒，退出0，无900秒监督超时；本人resident worker已回收。新增监督仅约束此有限工程过程，不证明请求硬实时。

## 计时与尚未满足的正式资格

66新正常请求实际{min(walls):.6f}–{max(walls):.6f}秒，包含父现场公开投影/编码、共享2MiB输入返回、worker接拒/K10/完整C0/编译/编码、父解码/guard提交。仍为8秒worker窗和2秒父/回收预留、10秒总上限；本次没有新故障注入。父投影、可信pickle编码/解码、摘要和提交同步不可抢占，正常低于10秒不等完整可中止10秒资格；正常每请求回收未实测，resident进程日末回收。日初历史/解释器预加载在外。

{len(reg['sources'])}项选定源/原前缀工件/原实际输入/原历史/解释器/NumPy/native开始结束一致，选定新工具均冻结；不等全加载stdlib/native/DLL闭包。本地Python3.12.7/NumPy1.26.4/CPU7。物理终局由原首14实际前缀加本次恢复后连续实际后续拼接，不能称重新从日初单进程完成全部80请求的独立运行资格。本协议不使用最终seed20260927/28，不是新科学日或新假设族，没有学习模型或新训练、服务器/GPU/共享MD上传。

全80订单没有实际同刻多个新单，派生双未决边界资格不能升级真实同时揭示。所有候选仍为own_incumbent同计划，本次没有模型重构与学习归因；正式五臂、同初始化已训CE、独立强动态接拒仍未齐。下一步关闭父可中止投影/编码/解码/提交及回收余量和全加载依赖，再完善正式模型与控制；全部原注册工程屏障满足后另登记有限新鲜科学日。配送第2族FAIL封闭、训练/DEV/确认关闭，第3族未启动；余30错配/生产动作适配仍待。负结果31必要门/183历史/C1失败沿用，只追加本阶段证据。

已观察日末端执行/原完整评价的工程组件已闭合。稳定可归因MaskCO核心在线增益、完整等耗时迭代和真实强动态证据仍未建立。
'''
    text+='\n归档原记录工具在写证据JSON时遇到NumPy空失败数组不可直接序列化，原工具源码、已生成完整报告及未完成JSON原件保留。R2转换空失败数组后，又因decoded clock是NumPy零维数组在JSON写入处失败，其源码、完整报告及部分JSON同样保存。R3在打开新文件前完整序列化，明确递归转换NumPy零维/数组/标量为原等值JSON数值或列表，另建归档证据与报告；没有重跑生产/独立验收或改数值/旧FAIL。\n'
    create(report,text)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(OUT.relative_to(ROOT)),metrics=metrics,first14_replayed=False,new_scientific_data=False,observed_day_suffix_execution_qualified=True,full_observed_day_execution_component_qualified=True,one_process_from_day_start_qualified=False,all_formal_arms=False,parent_preemptible=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False,shared_MD_uploaded=False,result_sha256=sha(OUT/'RESULT.json'),independent_sha256=sha(OUT/'INDEPENDENT_OBSERVED_DAY_EXTENSION.json'),registration_sha256=sha(OUT/'REGISTRATION.json'))
    create(R/'reproduction/OBSERVED_DAY_EXTENSION_EVIDENCE_R3_20261012.json',receipt)
    backup=R/'current_entries_before_observed_day_extension_20261012';backup.mkdir(exist_ok=False)
    entries=read(R/'current_entries_before_live_prefix_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
    note='\n> 2026-10-12取货封存前缀后续至原日终局：未重跑前14，后续66真实揭示/137推进，472运行与4711独立项0失败；共80订单74接受服务/6拒绝，使用车辆返仓，末端原完整C0通过。原999/15及91恢复保持；这是拼接封存实际前缀的已观察日工程，无学习增益，单进程日初运行/完整可中止10秒/全加载依赖/正式学习CE与强动态仍待。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);prior=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(prior);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(observed_day_extension_directory=str(OUT.relative_to(ROOT)),observed_day_suffix_execution_qualified=True,full_observed_day_execution_component_qualified=True,full_observed_day_execution_qualified=False,continuous_live_event_qualified=False,one_process_from_day_start_qualified=False,observed_day_extension_runtime_checks=472,observed_day_extension_independent_checks=4711,observed_day_extension_actual_services=74,full_live_request_qualified=False,
                next='Close interruptible parent projection/encode/decode/commit, per-request cleanup reserve and full loaded dependency closure; then formal learned and same-initialization trained CE controls and independent strong dynamic admissions. Observed-day terminal execution component passed by composing sealed actual first14 and exact continuation; no from-day-start single-process qualification. Do not repeat 472/4711 or earlier completed QA. Fresh scientific days only after all engineering barriers; delivery family2 remains closed FAIL.')
            s['reproduction']['observed_day_extension_boundary']='research/review_3080_20261010/reproduction/OBSERVED_DAY_EXTENSION_EVIDENCE_R3_20261012.json';updated=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            updated=note+prior if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else prior+note
            if f.parent==E/'01_取货线':updated+='\n- [封存连续前缀后续至原完整日终局工程](文档/结果与进度/'+report.name+')\n'
        f.write_text(updated,encoding='utf-8');identities.append(dict(path=entry['path'],before_sha256=sha(saved),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=identities,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
    print(json.dumps(native(dict(updated_entries=len(identities),metrics=metrics,full_live_request=False))))
if __name__=='__main__':main()
