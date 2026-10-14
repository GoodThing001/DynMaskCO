"""Seal new full replay source preparation; actual execution remains closed."""
import ast,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';D=R/'delivery';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
O=D/'action_head_independent_replay_preparation_root_v1';PRE=D/'action_head_execution_chain_preparation_root_v1'

def main():
    previous=read(PRE/'EXECUTION_CHAIN_DRAFT.json')
    assert all(sha(PRE/n)==h for n,h in previous['package_files'].items())
    package={}
    for n in ['delivery_actual_action_head_v1.py','action_head_runtime_v1.py','run_delivery_actual_action_head_v2.py','audit_delivery_actual_action_head_v2.py','supervise_delivery_action_head_v2.py']:
        p=ROOT/'tools'/n;ast.parse(p.read_text(encoding='utf-8'));assert not (O/n).exists();shutil.copy2(p,O/n);package[n]=sha(O/n)
    for n in ['features_objective.py','INPUTS.json']:
        assert not (O/n).exists();shutil.copy2(PRE/n,O/n);package[n]=sha(O/n)
    rows=read(O/'INPUTS.json');assert len(rows)==30 and sum(len(r['actions']) for r in rows)==22500
    assert all(set(r)=={'key','data','before','remaining','actions'} for r in rows)
    assert all(not any(float(v)!=0 for v in r['data'][field]) for r in rows for field in ['reveal_time','ready_time'])
    normalization={'modules/functional/coord_transform.py':sha(O/'OBSERVED_functional.py'),'modules/functional/__init__.py':sha(O/'OBSERVED_functional_init.py')}
    assert normalization=={'modules/functional/coord_transform.py':'09de9bf90d6b21991759511bf2905700791e76cfd2f3b3986c7a7f1cff193da7','modules/functional/__init__.py':'ddccb363c3131b28c460b767231757e84c7fa963e19c06b1b8ce52cf6855e303'}
    plan={k:v for k,v in previous.items() if k not in ['created_utc','status','package_files','outstanding','numeric_bound_scope']}
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(),status='FULL_REPLAY_SOURCE_PREPARATION_ONLY',execution_authorized=False,package_files=package,normalization_sources=normalization,production_forward_records=60,independent_full_forward_records=60,backbone_comparison='exact full float32 arrays and all234 parameter identities; no relaxed latent/feature tolerance',coordinate_comparison='exact independent mean/L2/mean/divide using same JAX grouping; shared runtime, not another normalization implementation',numpy_head_max_abs_error=2e-5,numeric_bound_scope='Only preregistered new NumPy versus JAX float32 head arithmetic. Full backbone, seeded parameters and10D features require exact identity. No old scientific tolerance change.',outstanding=['Exact prospective execution registration and exclusive upload/dispatch/readonly download chain still required before any execution.','Recheck real available8GiB RAM, CPU15 load and actual process ownership before execution; no team process termination.','This source preparation has not imported JAX or performed any forward, gradient, optimizer, solver or C0; no runtime qualification.','Formal trained same-initializationCE/task-trainedrandom controls, fullC0, full10second request and equal-time dynamic iteration remain.'])
    create(O/'EXECUTION_PLAN_PREPARATION.json',plan)
    note='\n> 2026-10-12 配送独立完整重载核查源码准备：已只读取得实际coord_transform.py及coord_normalize别名，绑定原实际SHA；旧路径不存在和默认网络失败原观察/源码保留。新核查源码拟从原checkpoint独立重载234叶及随机888同3D初始化/7零扩展，独立重建全10D坐标归一/当前context/临时锚点、逐60前向精确核验final_norm输出，并另NumPy复算同头。仅AST/字节封存，未导入模型或执行任何forward；执行登记/上传派发下载链尚未封存，execution_authorized=False。2e-5仅沿用新头跨库算术界限，backbone/特征/参数要求精确，所有科学/C0/预算门仍False。第2族FAIL、第3族未启动；取货终局64/pending空无callback81；稳定可归因在线增益未建立。\n'
    report=E/'02_配送线/文档/结果与进度/配送_实际动作头独立完整重载核查源码准备_2026-10-12.md'
    create(report,'# 实际动作头独立完整重载核查源码准备\n\n'+note+'\n新源目录：research/review_3080_20261010/delivery/action_head_independent_replay_preparation_root_v1。原v1草案、模型头、30首状态和22500动作输入逐字保留；v2仅增加独立重载核查接线及准确范围描述，不改架构/输入/种子/更新次数或按教师标签、成本选动作。\n\n核查器不导入新head、forward、runner、目标binding或物理过滤组件；实际运行时拟直接原encode/decode(target=embed)/final_norm，独立原参数零扩展及fresh888初始化，比较每叶参数SHA和60完整latent精确数组；单独初始化同seed20261012的4叶196929参数头并逐字核验。坐标从全部当前可见旧快照的原coords按实际源码mean中心、L2范数均值缩放重建，连同其余8字段全10D精确核验；不复制生产坐标。NumPy头使用实际保存latent复算当前锚点/路线子序列均值/MLP，2e-5仅旧草案事前界限，不套到完整模型或旧科学门。\n\n上述是核查源码而非已通过核查：共享原JAX模型/浮点实现，不是另一Transformer或物理实现。0模型导入/前向/梯度/优化器/求解器/C0，未上传研究包/共享MD，不用GPU。先补齐准确执行登记、独占派发与有界只读下载，再复查实际8GiB门/CPU15/本人树，另新独占工程执行，900秒监督不等整请求10秒。科学第3族、新数据、正式已训CE/同任务随机及全C0/等耗时迭代/强动态资格仍关闭。\n')
    boundary=R/'reproduction/ACTION_HEAD_FULL_REPLAY_PREPARATION_BOUNDARY_20261012.json'
    create(boundary,dict(plan_sha256=sha(O/'EXECUTION_PLAN_PREPARATION.json'),source_only=True,model_executed=False,independent_backbone_executed=False,execution_authorized=False,qualification=False,scientific_family3_started=False,stable_online_gain=False))
    entries=read(R/'current_entries_before_action_head_execution_draft_20261012/UPDATE_IDENTITY.json')['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_action_head_full_replay_preparation_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    updates=[]
    for x in entries:
        p=E/x['path'];text=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(text);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(actual_action_head_full_replay_preparation_directory=str(O.relative_to(ROOT)),actual_action_head_execution_closed=True,actual_action_head_model_executed=False,actual_action_head_engineering_qualified=False,next='Full independent checkpoint reload/60 direct forwards/exact10D normalization/seeded head audit source now sealed but UNEXECUTED. Before execution seal NEW exact execution registration and exclusive upload/dispatch/bounded readonly download tools, then recheck actual3080 RAM>=8GiB/CPU15/owned processes. Do not execute draft or repeat oldQA. No model imports/forward/update/solver/C0, no family3/data. Family2FAILclosed. Trained same-initCE/tasktrainedrandom/fullC0/full10s/equal-time iteration/strongdynamic remain. Pickup authoritative terminal64/callback80/pendingempty, no callback81.')
            w['reproduction']['action_head_full_replay_preparation_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+text if p.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if p.parent==E/'02_配送线':new+='\n- [实际动作头独立完整重载核查源码准备](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    assert all(sha(O/n)==h for n,h in package.items()) and all(sha(PRE/n)==h for n,h in previous['package_files'].items())
    create(O/'PREPARATION_RECEIPT.json',dict(package_files=package,rows=30,actions=22500,AST_only=True,model_executed=False,qualification=False,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),remote_uploaded=False))
    print(json.dumps(dict(preparation_sealed=True,model_executed=False,execution_closed=True,entries=16)))

if __name__=='__main__':main()
