"""Seal prospective full-model action-head engineering without running a model."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,os,shutil,importlib.metadata
import psutil
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';O=D/'actual_action_head_preparation_root_v1'
def main():
    O.mkdir(exist_ok=False)
    module=ROOT/'tools/delivery_actual_action_head_v1.py';ast.parse(module.read_text(encoding='utf-8'));shutil.copy2(module,O/'FROZEN_delivery_actual_action_head_v1.py')
    paths=[Path(__file__),module,ROOT/'MASKCO_code/models/CVRPModel.py',ROOT/'MASKCO_code/models/TSPModel.py',ROOT/'MASKCO_code/ckpts/cvrp100.ckpt',D/'actual_checkpoint_engineering_v1/inspect_checkpoint.py',D/'features_objective_qa_v1/features_objective.py',D/'action_tensor_binding_root_v2/REGISTRATION.json',D/'action_tensor_binding_root_v2/RESULT.json',D/'directed_action_chain_root_v1/ACTION_CHAINS.jsonl',D/'family2_remaining_action_diagnosis_root_v1/OBSERVED_ACTION_RECORDS.json']
    identities={str(p):sha(p) for p in paths}
    assert identities[str(ROOT/'MASKCO_code/ckpts/cvrp100.ckpt')]=='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='SOURCE_PREPARED_EXECUTION_CLOSED',inputs=identities,new_scientific_family=False,family3_started=False,training_started=False,optimizer_steps=0,model_executed=False,learned_output_qualified=False,scope='Prospective full original CVRP model10D -> decoder target=embed -> original final_norm -> ordered insertion action head. Fixed first temporary state of each of all30 old observed records, in saved order, one forward per initialization; not all750 replay and not old solver/probe/chain QA. Before any execution seal exact runtime/runner/independent auditor/closure and bounded supervisor in a NEW registration; this preparation is not execution authorization by itself.',architecture=dict(full_backbone='original234-leaf CVRP100 extended3-to10 init projection',embedding=512,head_hidden=64,head_seed=20261012,head_parameter_elements=196929,head_parameter_leaves=4,nonlinearity='tanh',ordered_inputs=['customer_embedding','current_predecessor_embedding','current_successor_embedding','current_route_mean','current_prefix_mean','current_suffix_mean','current_slot_over_max_slot','position_over_customer_count','route_length_over_customer_count'],teacher_inputs=False,vehicle_permutation_equivariance_qualified=False,current_context='undirected edges of temporary skeleton only',timestep='current customer degree sum / twice customer count',raw9_not_accepted_as10D=True,old_fixture_consumer_accepts_actual_model=False),engineering_initializations=['original_pretrained_plus_new_head_seed20261012','fresh_random_backbone888_plus_same_head_seed20261012'],formal_controls_qualified=False,same_initialized_trained_CE_qualified=False,task_trained_random_qualified=False,required_controls_future=['same_initialized_trained_CE','matched_task_trained_random','nonlearning','independentstrongdynamic'],minimum_available_ram_gib=8,cpu_only=True,no_GPU_or_server_execution=True,scientific_training_qualified=False,full_C0_qualified=False,full_request_budget_qualified=False,full_scientific_dependency_closure_qualified=False,stable_online_gain=False)
    create(O/'PREPARATION_REGISTRATION.json',reg)
    names={'qualify_pickup_resident_day_tail_v1.py','audit_pickup_resident_tail_stream_r3.py','qualify_pickup_terminal_restore_v2.py','qualify_delivery_actual_action_head_v1.py'}
    active=[p.pid for p in psutil.process_iter(['name','cmdline']) if p.pid!=os.getpid() and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe') and any(Path(x).name in names for x in p.info['cmdline'] or [])]
    available=psutil.virtual_memory().available/2**30
    versions={x:importlib.metadata.version(x) for x in ('numpy','jax','jaxlib','flax')}
    create(O/'LOCAL_RESOURCE_OBSERVATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),available_ram_gib=available,minimum_available_ram_gib=8,local_full_model_execution_allowed=False,reason='Exact execution runner/auditor/environment/supervisor not yet sealed; local RAM also below8GiB' if available<8 else 'Exact execution runner/auditor/environment/supervisor not yet sealed',installed_metadata_only=versions,no_jax_model_import=True,active_owned=active,model_executed=False,gradient_or_training_executed=False,unmodified_dependencies=True))
    report=E/'02_配送线/文档/结果与进度/配送_实际MaskCO动作评分头源码准备与执行屏障_2026-10-12.md'
    create(report,f'''# 实际MaskCO动作评分头源码准备与执行屏障（2026-10-12）

actual_action_head_preparation_root_v1仅封存新独占模块与工程前声明，没有实际模型前向、梯度、优化器、C0、solver、重构或旧QA执行；第3科学族未启动，所有执行资格仍False。原CVRP/TSP源码与checkpoint只读，原checkpoint实际SHA03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db。

新模块拟直接调用原完整模型encode/decode(target='embed')及原final_norm，把512维实际节点表示传给新4叶/196929参数的64维tanh动作头。按固定顺序连接当前客户/临时前驱/临时后继表示、当前车槽路线/前缀/后缀均值及当前槽号/插入位置/路线长度三个尺度值，输出逐动作分数。所有输入来自当前骨架和原10D完整特征；不接受旧raw9直接冒充10D，不读取教师车槽/方向秩。当前槽号是显式ordinal，车辆置换等变性未证明。上下文仍为当前骨架无向边，不把新头本身称完整有方向模型资格；旧fixture消费者仍不能接实际模型包。

拟执行工程固定旧30记录每例首临时状态、原顺序，不筛标签/成本；新头seed20261012，原预训练初始化和fresh随机888同头初始化，两者均0训练，不是假称同初始化已训CE或同任务训练随机控制。必须在实际执行前另封存确切runner/解释器与运行库/native闭包/独立核查器/有界监督/所有输出来源及未训练声明，新的执行登记不能用本准备替代。科学训练、正式控制、完整C0/动作消费者/可中止全请求仍待，不使用第2族关闭权重或旧工程两步权重。

本地只读资源观察可用RAM{available:.6f}GiB，沿用完整76M模型工程8GiB最低资源门；没有为通过而降低门或修改全局依赖。只查询已安装元数据{json.dumps(versions,ensure_ascii=False)}，没有导入JAX模型或声称此环境等同3080已验环境。下一先核实际可用资源，封存确切执行链，再在新独占目录事前登记并有界执行此实际模型工程；服务器/GPU尚未使用，共享MD未上传。

取货已选定终局64/callback80/order75，74客户全服务/三车返仓、pending空，不再重复旧日无模型正常QA。原resident整轮FAIL及缺中间raw事件ledger/R2页末FAIL仍保持，9/1164终局与31107请求复算沿用不重跑；正式学习/已训CE/独立强动态接拒/等耗时MaskCO迭代/全科学依赖预算仍待。负结果31/183/C1及原10890四FAIL/缺13800字节不重整，最终取货20260927/28未用。稳定可归因MaskCO核心在线增益未建立。
''')
    boundary=R/'reproduction/ACTUAL_ACTION_HEAD_PREPARATION_BOUNDARY_20261012.json';create(boundary,dict(preparation_registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),source_sha256=sha(module),model_executed=False,qualification=False,scientific_family_started=False,resource_execution_closed=True,previous_evidence_not_repeated=True))
    prior=R/'current_entries_before_tail_terminal_recovery_20261012/UPDATE_IDENTITY.json';entries=read(prior)['files'];assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_actual_action_head_preparation_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        dest=backup/x['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],dest)
    note='\n> 2026-10-12配送实际动作头准备：新独占实际CVRP decode embed+final_norm→顺序锚点/车槽当前子序列MLP模块与前声明已封存，仅语法/源码准备，0模型/梯度/训练/C0/旧QA执行；最低8GiB资源及确切runner/环境闭包/独立核查/有界监督未放行。拟固定旧30首状态，预训初始+fresh随机888同新头初始并非已训控制；第2族FAIL封闭、第3族未启动。取货actual终局64/pending空，旧整轮FAIL/缺ledger保持，不重复50/31107/9/1164。科学门与稳定在线增益未升级。\n'
    updates=[]
    for x in entries:
        path=E/x['path'];text=path.read_text(encoding='utf-8')
        if path.suffix=='.json':
            w=json.loads(text);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(actual_action_head_preparation_directory=str(O.relative_to(ROOT)),actual_action_head_source_prepared=True,actual_action_head_model_executed=False,actual_action_head_engineering_qualified=False,actual_action_head_execution_closed=True,next='Do not rerun old tensor/idealchain/family2 QA. Actual action-head module prepared only: original decoder embed+final_norm, ordered customer/predecessor/successor and current slot/subsequence inputs; no actual model/gradient/update/C0 output. Next seal exact runner/runtime/native/independent checker/bounded supervisor in a new execution registration after real resource check >=8GiB. Fixed all30 firststates; pretrainedinitial and random888+samehead initial are NOT trainedCE/tasktrainedrandom controls. Scientificfamily3 notstarted; fullcontrols/C0/fullrequest/MaskCOiteration remain. Pickup actualterminal64/pendingempty; no callback81 or oldday nomodelQA.')
            w['reproduction']['actual_action_head_preparation_boundary']=str(boundary.relative_to(ROOT)).replace('\\','/');new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+text if path.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if path.parent==E/'02_配送线':new+='\n- [实际动作评分头源码准备与执行屏障](文档/结果与进度/'+report.name+')\n'
        path.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(path)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates) and all(sha(p)==h for p,h in identities.items())
    create(O/'PREPARATION_IDENTITY.json',dict(inputs=identities,frozen_source_sha256=sha(O/'FROZEN_delivery_actual_action_head_v1.py'),report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),model_executed=False,qualification=False))
    print(json.dumps(dict(prepared=True,entries=16,available_ram_gib=available,model_executed=False,execution_closed=True)))
if __name__=='__main__':main()
