"""Freeze execution-chain draft and all fixed inputs without importing a model."""
import ast,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';D=R/'delivery';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';O=D/'action_head_execution_chain_preparation_root_v1'

def main():
    assert O.is_dir()
    observations=sorted(O.glob('RESOURCE_OBSERVATION_*.json'));current=read(observations[-1])
    server=current.get('server');assert server and server['available_ram_gib']>=8
    # The single reported Python process is this read-only SSH bootstrap parent.
    # Keep its full observed identity; no team process is terminated or hidden.
    own_active=[]
    for p in server['processes']:
        if any('review_3080_20261010/' in x.replace('\\','/') for x in p.get('argv') or []):own_active.append(p)
    assert not own_active
    sources=['delivery_actual_action_head_v1.py','action_head_runtime_v1.py','run_delivery_actual_action_head_v1.py','audit_delivery_actual_action_head_v1.py','supervise_delivery_action_head_v1.py']
    hashes={}
    for n in sources:
        p=ROOT/'tools'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,O/n);hashes[n]=sha(O/n)
    feature=D/'features_objective_qa_v1/features_objective.py';shutil.copy2(feature,O/'features_objective.py');hashes['features_objective.py']=sha(feature)
    records=D/'family2_remaining_action_diagnosis_root_v1/OBSERVED_ACTION_RECORDS.json';chains=D/'directed_action_chain_root_v1/ACTION_CHAINS.jsonl'
    old=read(records);mapping={(r['instance_id'],r['step']):r for r in old};rows=[]
    with chains.open(encoding='utf-8') as f:
        for line in f:
            x=json.loads(line);state=x['trace'][0];record=mapping[tuple(x['key'])]
            # Only data and current first skeleton/actions cross the model boundary.
            # No saved target, teacher route/rank, binary fixture or cost is copied.
            rows.append(dict(key=x['key'],data=record['request']['data'],before=state['before'],remaining=state['remaining'],actions=[r[:5] for c in state['customers'] for r in c['rows']]))
    assert len(rows)==len(old)==30 and len({tuple(r['key']) for r in rows})==30
    create(O/'INPUTS.json',rows);hashes['INPUTS.json']=sha(O/'INPUTS.json')
    from action_head_runtime_v1 import digest
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='EXECUTION_CHAIN_DRAFT_CLOSED',execution_authorized=False,package_files=hashes,ordered_keys_digest=digest([r['key'] for r in rows]),old_remote_root=r'C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\remote_finite_research_root_v2',checkpoint=r'C:\Users\zh\project\MASKCO-Main\MASKCO_code\ckpts\cvrp100.ckpt',checkpoint_sha256='03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db',contract_hash='ba40cd0d08fc21ef09d418c7df2bff940294ee2cfecebd2a4ba9ddde35a8268a',minimum_ram_gib=8,stage_timeout_seconds=900,reap_timeout_seconds=10,numpy_head_max_abs_error=2e-5,numeric_bound_scope='Prospective new NumPy versus JAX float32 head arithmetic only, no old scientific tolerance change. Unexecuted; this draft does not authorize forwards.',head_seed=20261012,random_seed=888,random_projection='original3D initialization then exact7 zero rows as pretrained extension',optimizer_steps=0,first_states=30,model_executed=False,scientific_family3_started=False,formal_controls_qualified=False,full_C0_qualified=False,full_request_budget_qualified=False,source_inputs={str(records):sha(records),str(chains):sha(chains)},resource_observation=str(observations[-1]),resource_observation_sha256=sha(observations[-1]),outstanding=['Independent full-backbone actual parameter reload/forward and coordinate normalization verification are not implemented by the saved-latent NumPy checker.','Prospective exact execution registration and dispatch/download/actual run closure still required.','Independent checker currently audits only head and8 mapped input fields; cannot upgrade full model qualification.','Supervisor draft is stage-bounded, not10second full request qualification.'])
    create(O/'EXECUTION_CHAIN_DRAFT.json',reg)
    note='\n> 2026-10-12 配送执行链草案：只读3080实测可用RAM11.683735GiB、CPU15负载0.8%，无本人旧研究进程；默认网络WinError10013原观察保留，经自动审核获准的同一只读核查完成。已封存runner/实际加载闭包/NumPy动作头核查器/900秒监督草案及剥离教师标签的全部30首状态输入。仅AST与身份准备，0模型/梯度/更新/求解器/C0执行；核查器尚不独立复算完整backbone及坐标归一，执行登记和派发保持关闭，不能将草案当完整模型或10秒资格。第2族FAIL、第3族未启动；取货终局64/pending空，不重跑旧QA；稳定可归因在线增益未建立。\n'
    report=E/'02_配送线/文档/结果与进度/配送_实际动作评分执行链草案与服务器资源核查_2026-10-12.md'
    create(report,'# 实际动作评分执行链草案与服务器资源核查\n\n'+note+'\n全30输入按原action-chain文件顺序提取每例首临时骨架及全部动作，不筛选标签或成本。模型输入文件只保留data/key/before/remaining/actions；不复制教师方向、车槽秩、标签和成本。原理想动作链、张量、科学第2族及所有旧原件不改、不重跑。\n\nrunner草案按原CVRP配置构造、原预训234叶加7零投影行，fresh随机888也由原3D初始化加7零行；同一196929参数动作头seed20261012。capture仅留存同一次forward的final_norm输出，不额外重跑模型。NumPy核查器从该实际latent重算顺序锚点/当前子序列均值及MLP，不能独立核验backbone计算本身；另8字段可复算而原归一坐标仍是保存的上游输出。因此新draft显式拒绝执行授权，仍需补齐实际完整独立重载/前向与归一核验，再另封存真实执行登记。2e-5仅事前未执行的新float32头跨库算术界限，不更改任何旧科学门。\n\n闭包草案纳入实际解释器、模块源/pyc、包源/native、stdlib/DLL、内存映射和注册输入；尚未实际运行，不宣称起止闭包通过。监督草案900秒/阶段、10秒回收只属整体有界工程，非在线整请求10秒。没有上传研究包或共享MD，没有GPU、训练或新科学族。\n')
    boundary=R/'reproduction/ACTION_HEAD_EXECUTION_DRAFT_BOUNDARY_20261012.json';create(boundary,dict(draft_sha256=sha(O/'EXECUTION_CHAIN_DRAFT.json'),model_executed=False,execution_authorized=False,independent_backbone_forward=False,qualification=False,scientific_family3_started=False,stable_online_gain=False))
    prior=R/'current_entries_before_actual_action_head_preparation_20261012/UPDATE_IDENTITY.json';entries=read(prior)['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_action_head_execution_draft_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    updates=[]
    for x in entries:
        p=E/x['path'];text=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(text);w['updated']=datetime.now(timezone.utc).isoformat();w['delivery'].update(actual_action_head_execution_chain_draft_directory=str(O.relative_to(ROOT)),actual_action_head_execution_closed=True,actual_action_head_model_executed=False,actual_action_head_engineering_qualified=False,next='Execution-chain draft and fixed all30 teacher-free first-state inputs sealed. Current3080 readonly RAM11.683735GiB/CPU15busy0.8%, no owned old research process; recheck before actual execution. Next implement independently reloaded full backbone and coordinate normalization audit, then seal NEW exact prospective execution registration and dispatch/download chain; existing saved-latent checker only qualifies head arithmetic and8mapped fields, not full model. No forward/gradient/update/solver/C0 executed. Family2FAILclosed; family3notstarted; formal trainedCE/tasktrainedrandom/fullC0/full10secondbudget/equal-timeiteration/strongdynamic remain. Pickup terminal64/pendingempty; no callback81/oldQA.')
            w['reproduction']['action_head_execution_draft_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+text if p.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
            if p.parent==E/'02_配送线':new+='\n- [实际动作评分执行链草案与资源核查](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');updates.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=updates,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in updates)
    assert all(sha(O/n)==h for n,h in hashes.items()) and sha(records)==reg['source_inputs'][str(records)] and sha(chains)==reg['source_inputs'][str(chains)]
    create(O/'PREPARATION_RECEIPT.json',dict(sources=hashes,rows=30,actions=sum(len(r['actions']) for r in rows),source_AST_parsed=True,model_executed=False,qualification=False,report_path=str(report),report_sha256=sha(report),boundary_path=str(boundary),boundary_sha256=sha(boundary),entry_update_sha256=sha(backup/'UPDATE_IDENTITY.json'),remote_uploaded=False))
    print(json.dumps(dict(source_draft_sealed=True,rows=30,entries=16,model_executed=False,execution_closed=True)))

if __name__=='__main__':main()
