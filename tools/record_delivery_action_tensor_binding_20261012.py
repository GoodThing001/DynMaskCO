"""Seal new tensor binding scope and update local authoritative entries with backups."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';O=D/'action_tensor_binding_root_v2';OLD=D/'action_tensor_binding_root_v1'
def main():
    r=read(O/'RESULT.json');a=read(O/'INDEPENDENT_ACTION_TENSOR_BINDING.json');assert (r['checks_count'],r['failed'],a['checks_count'],a['failed'])==(2254,0,15160,0);assert read(O/'RUN_EXIT.json')['exit_code']==0
    for n,field in [('REGISTRATION.json','inputs'),('AUDIT_REGISTRATION.json','inputs')]:assert all(sha(p)==h for p,h in read(O/n)[field].items())
    assert sha(OLD/'TENSORS.npz')==sha(O/'TENSORS.npz')=='31e8da097121f173e0f97adb1f3d6758d2caeab1ed12cdb54ee258262243c58d';assert read(OLD/'RESULT.json')['failed']==1
    names=('qualify_delivery_action_tensor_binding_v2.py','audit_delivery_action_tensor_binding_v2.py','delivery_action_tensor_binding_v1.py','recover_delivery_action_tensor_binding_v2.py');active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and any(Path(t).name in names for t in p.info['cmdline'] or [])];assert not active;frozen={}
    for n in names:
        f=O/('FROZEN_'+n);assert not f.exists();shutil.copy2(ROOT/'tools'/n,f);frozen[str(f)]=sha(f)
    create(O/'AUDIT_RUN_EXIT.json',dict(exit_code=0,actual_exec_completion=True));create(O/'ACTUAL_EXIT_RECEIPTS.json',dict(producer_actual_exit=0,independent_actual_exit=0,old_producer_actual_exit=1,owned_processes_active=active,no_teacher_solver_model_calls=True))
    report=E/'02_配送线/文档/结果与进度/配送_动作目标张量与临时锚点评分身份接口验收_2026-10-12.md'
    create(report,f'''# 动作目标张量与临时锚点评分身份接口验收（2026-10-12）

delivery/action_tensor_binding_root_v2是事前登记的新工程组件，固定全部30个已观察错配记录、旧理想动作链保存的750个临时骨架和370500条候选，不调用原或新reconstruct/action-chain/教师/solver/route_ok/matrices/C0/模型/梯度/优化器，不生成新科学数据或重做已合格163/1886864等QA。旧链中的可行标记和距离tie仅原字节转移，不重新物理认证或重算几何。

新接口每行int32保存(customer,vehicle,position,predecessor,successor)，校验当前骨架的实际前后锚点；float32保存当前订单/临时前驱/后继各9个原始可见字段（x,y,demand,temp_class,reveal_time,ready_time,tw_start,tw_end,service_time），显式车槽/位置身份单独保持，不把已知教师车槽或秩塞进特征。bool保存联合保留边且原route过滤合格，float64保存原距离tie。教师兼容标签在另一个仅训练/工程函数中生成：插入到当前车槽后整条暂时路线仍为对应教师有方向路线的子序列，label为float32的1，否则0。该表示是新工程raw9输入，不是原CVRP10D模型完整输入或已部署评分头；没有证明模型已学会、所有状态可达或可改善成本。

输入SHA绑定request key/临时step/全部当前骨架/remaining/完整旧数据；动作SHA绑定动作排列、资格标记、原距离hex。评分包必须逐行finite float32且上下文和动作顺序身份一致，仅允许observed_ideal_fixture或unqualified_numeric_fixture，learned_output_qualified始终False。拒绝伪称trained_model/已合格学习输出，不把这个诚实来源标签当密码学来源证明。消费者返回已绑定分数向量；这次没有再次运行客户regret/动作选择或完整重构。

750状态、370500动作、9750按客户候选分组全部保存为TENSORS.npz及CONTEXTS.jsonl；动作/特征/资格/原tie/标签/理想分数/状态和客户offset全保留。张量文件53808556字节，SHA31e8da097121f173e0f97adb1f3d6758d2caeab1ed12cdb54ee258262243c58d。9750个正标签全部对应已保存可行标记，这是旧已知目标的兼容动作数，不是9750独立样本或收益。float64原字段转float32最大舍入{a['max_float64_to_float32_feature_rounding']}，逐项符合登记的精确cast；原字段仍由输入SHA完整保存，不宣称float32特征无损，也不放宽原科学精确门。

9个实际API篡改边界拒绝：错误临时前驱、旧step、倒序动作、context SHA、action SHA、声称学习合格、声称trained_model、错误shape和非有限数。另一个固定线性数值向量仅证明unqualified_numeric_fixture向量可原值传递，不是实际随机/预训练/同初始化CE模型或正式控制臂。

2254运行/15160新进程独立项0失败，实际两退出0。独立不导入新binding/producer/chain/可行原语/guard/route_ok/matrices/C0/模型，用原保存骨架的临时前后节点和完整假设插入路线顺序扫描重建所有动作目标、全部raw9索引/cast/资格tie/分组/来源和身份；物理标记仅保存身份核验。生产438文件215module映射、独立437文件213module映射起止闭合，仅本地Python3.12.7/NumPy1.26.4/CPU7/BLAS单线程无模型张量负载，非3080科学训练/正式控制全依赖资格。准备、IO预热/旧记录加载/归档在外，不声明可抢占整请求10秒、任意大小或生产吞吐。

首次action_tensor_binding_root_v1实际2254项1FAIL/退出1原件与源保留。唯一失败为首次np.savez时加载6个既有zipfile源码/pyc（3module）未记起始，已有文件字节未变，750/370500目标和身份项通过。核本人进程退出后，另事前恢复登记新v2同全部输入、dtype、标签、锚点、资格和评分合同，登记前预初始化实际NPZ小接口，未改binding模块/全局依赖/科学门/容差。新v2张量与v1原张量SHA逐字一致，不能把原FAIL改成PASS。v1审计未启动，没有用失败闭包升级资格。

learned_output/full_C0/hard_request_budget/full_scientific_loaded_closure与科学新族资格均未升级，第2族FAIL封闭、三臂训练/采集/审计/DEV/确认关闭，第3族未启动。下一另事前工程将这个目标/锚点身份接入实际模型评分头及各控制，核输入/输出/保存重载/同初始化CE和同任务训练随机控制、完整C0与可中止预算，然后另登记有限科学族。不要重复本2254/15160或旧已合格QA，不据教师标签或旧成本选择模型/动作。

取货当前actual selected slot0 epoch12/callback29/order76原shared38929字节继续真正callback30，订单身份未假定；已完成19/1807及归档1183/1FAIL+R2窄8资格保持，不重做。正式学习/已训CE/独立强动态接拒/全日10秒等耗时迭代和科学完整依赖仍待。负结果31/183/C1和旧10890/4FAIL/缺13800字节原件保持；16入口先保存旧字节本地更新，本人进程退出，无服务器/GPU/共享MD上传，最终20260927/28未用。稳定可归因MaskCO核心在线增益、完整等耗时迭代及真实强动态证据仍未建立。
''')
    boundary=R/'reproduction/ACTION_TENSOR_BINDING_EVIDENCE_20261012.json'
    create(boundary,dict(created_utc=datetime.now(timezone.utc).isoformat(),directory=str(O.relative_to(ROOT)),registration_sha256=sha(O/'REGISTRATION.json'),independent_sha256=sha(O/'INDEPENDENT_ACTION_TENSOR_BINDING.json'),runtime_checks=2254,independent_checks=15160,failed=0,old_runtime_checks=2254,old_runtime_failed=1,old_FAIL_preserved=True,old_tensor_sha256=sha(OLD/'TENSORS.npz'),actual_tensor_sha256=sha(O/'TENSORS.npz'),states=750,actions=370500,customer_groups=9750,positive_targets=9750,raw9_float32_cast_rounding=a['max_float64_to_float32_feature_rounding'],scoped_target_anchor_tensor_binding_qualified=True,learned_output_qualified=False,full_C0_qualified=False,hard_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,new_scientific_family=False,stable_online_gain=False))
    backup=R/'current_entries_before_action_tensor_binding_20261012';backup.mkdir(exist_ok=False);entries=read(R/'current_entries_before_event_loss_resume_archive_r2_20261012/UPDATE_IDENTITY.json')['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note='\n> 2026-10-12配送动作张量/锚点身份：另事前工程v2固定旧750骨架/370500候选/9750客户分组，int32动作锚点/raw9 float32字段/资格/原距离tie/独立目标与评分包身份全保存，2254运行/15160独立项0失败。9实际错配边界拒绝。首次v1仅zipfile懒加载闭包1FAIL原件保持；v2同合同预热恢复、张量与v1逐字一致。仅已知教师标签/未合格数值fixture，不是实际模型、梯度、物理C0或整请求预算；第2族FAIL封闭、第3族未启动，下一实际评分模型/控制/C0/预算接线。取货actual epoch12/callback29→真正callback30及正式CE/强动态/全日10秒/依赖缺口继续，稳定在线增益未建立。\n'
    updates=[]
    for x in entries:
        f=E/x['path'];text=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            obj=json.loads(text);obj['updated']=datetime.now(timezone.utc).isoformat();obj['delivery'].update(action_tensor_binding_directory=str(O.relative_to(ROOT)),action_tensor_binding_runtime_checks=2254,action_tensor_binding_independent_checks=15160,action_tensor_binding_independent_failed=0,scoped_target_anchor_tensor_binding_qualified=True,action_tensor_binding_states=750,action_tensor_binding_rows=370500,action_tensor_binding_old_failed=1,action_tensor_binding_learned_output_qualified=False,next='Do not repeat2254/15160 tensor binding or old ideal163/1886864/closedfamily2 QA. All750 old temporary skeletons/370500 saved actions now bound to int32 action anchors/raw9 float32 observed fields/eligibility/float64 saved distance tie/known teacher compatibility labels; nine actual wrong source/context/anchors/shape/nonfinite boundaries rejected. v1 zipfile lazy closure1FAIL preserved; v2 same-contract tensors byte-identical and actual IO closure closed only this no-model workload. Next prospectively qualify actual model score head and per-action output identity/each control, same-init trained CE/task-trained random, fullC0 and interruptible whole-request budget before separately preregistering finite family3/newdata. No science/model/gradient/solver/chain rerun in tensor component. Continue pickup actual selected slot0 epoch12/callback29/order76 raw38929 to genuine callback30; no assumed identity. Formal CE/strong dynamic/full-day10s/scientific closure remain.')
            obj['reproduction']['action_tensor_binding_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(obj,ensure_ascii=False,indent=2)
        else:
            new=note+text if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if f.parent==E/'02_配送线':new+='\n- [动作目标张量与临时锚点身份接口](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(f)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False));assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    create(O/'ARCHIVE_IDENTITY.json',dict(frozen=frozen,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),owned_processes_active=active,old_FAIL_preserved=True))
    print(json.dumps(dict(entries=16,runtime=2254,independent=15160,failed=0,old_fail_preserved=1,active=active)))
if __name__=='__main__':main()
