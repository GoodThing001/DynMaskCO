"""Freeze and record source-only public-snapshot budget integration, never run it."""
import ast,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
OLD=R/'delivery/actual_score_chain_execution_root_v1';O=R/'delivery/timed_actual_request_preparation_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    old=read(OLD/'EXECUTION_REGISTRATION.json');source=OLD/'downloaded'
    names=['delivery_timed_actual_engine_v1.py','delivery_timed_shared_worker_v1.py','delivery_timed_shared_parent_v1.py']
    deps=['delivery_actual_score_chain_v1.py','delivery_feasible_action_options_v1.py','delivery_retained_edge_action_guard_v1.py','delivery_actual_action_head_v1.py','action_head_runtime_v1.py','features_objective.py','CHAIN_INPUT.json']
    for n in names:ast.parse((ROOT/'tools'/n).read_text(encoding='utf-8'))
    for n in deps:
        assert sha(source/n)==old['package_files'][n]
        if n.endswith('.py'):ast.parse((source/n).read_text(encoding='utf-8'))
    row=read(source/'CHAIN_INPUT.json')
    assert row['key']==[12,3] and len(row['remaining'])==25
    assert not any(k in row for k in ['teacher_routes','oracle','cost','target_rank','labels'])
    entries=read(R/'current_entries_before_actual_score_chain_execution_20261012/UPDATE_IDENTITY.json')['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    O.mkdir(exist_ok=False)
    for n in names:shutil.copy2(ROOT/'tools'/n,O/n)
    for n in deps:shutil.copy2(source/n,O/n)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Unexecuted actual initial model public-snapshot timed integration source; no science.',
        prior_component_registration_sha256=sha(OLD/'EXECUTION_REGISTRATION.json'),input_sha256=sha(O/'CHAIN_INPUT.json'),input_digest=old['input_digest'],
        key=[12,3],masked_count=25,arms=['pretrained_initial','random888_initial'],head_seed=20261012,optimizer_steps=0,
        package_files={n:sha(O/n) for n in names+deps},total_request_seconds=10,child_seconds=8,reap_reserve_seconds=2,
        cpu=15,minimum_available_ram_gib=8,shared_input_limit_bytes=2097152,shared_output_limit_bytes=2097152,
        old_initial_score_artifacts_reused=False,proposed_new_encodes_max=2,proposed_new_decodes_max=50,
        startup_outside_request=['libraries','checkpoint/initial-model construction','SHARED allocation','prepared public-input serialization','incumbent fallback available'],
        timed_inside_proposed_request=['bounded input shared copy','worker JSON decode/current10D projection','incumbent fullC0','encode','every current-context decode/head',
            'all candidates/route_ok/retained edges/regret/transitions and trace IO','candidate fullC0','actual loaded closure and all parameters check',
            'worker serialization/publication','caller bounded copy/SHA/JSON/source/C0 selection guard','selected raw archive','worker exit/cancel/kill/reap'],
        stage_entry_fault_candidates={1:'input decode',2:'features completed',3:'encode completed',4:'first actual score completed',5:'chain stopped',6:'fullC0 selection completed',7:'serialized before publish',8:'published before worker exit'},
        execution_authorized=False,independent_timed_request_checker_sealed=False,actual_execution_registration_sealed=False,
        exact_dispatch_supervision_download_sealed=False,model_imported=False,forward_executed=False,C0_executed=False,
        timed_request_executed=False,source_only=True,learned_output_qualified=False,formal_controls_qualified=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        limitations=['Prepared public snapshot only, not private/live event projection or vehicle installation.',
            'Parent bounded copy/SHA/JSON/archive/cleanup are synchronous; arbitrary-load/slow-disk preemption unproven.',
            'Timer stamps and 8/2 reserve are source intent, not actual measurements or hard-realtime qualification.',
            'Full requested workload may time out; must preserve the complete partial chain/actual stage/exit and fallback.',
            'Postpublication fault may have been selected before deadline; cannot relabel it uncommitted or roll back selected output.',
            'No formal trained CE/task-random/other controls/equal-time iterations/strong dynamic science.',
            'Internal source identity check is consistency, not cryptographic origin authentication.',
            'No encode/decode/modelhead/solver/C0 warmup is currently specified. Ordinal slot equivariance unproven.'])
    create(O/'PREPARATION_REGISTRATION.json',plan)
    backup=R/'current_entries_before_timed_actual_request_preparation_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    note='\n> 2026-10-12 配送计时请求源码准备：timed_actual_request_preparation_root_v1封存子进程拥有的当前10D投影/完整初始模型动作链/全C0/依赖及参数核验、2MiB共享输入返回与父8秒截止选择/2秒回收预留的新源码。固定旧首项key[12,3]/25masked及两initial臂同头0更新；拟每步真实评分，不复用旧首评分，最多2encode/50decode仅拟定。全部只有AST/字节封存，0模型导入/forward/C0/实际请求/服务器上传；独立计时核查器、准确新执行登记/监督派发下载尚缺，execution_authorized=False。仅预备公共快照，不含现场私有事件提取/实际车辆安装；父有界复制/SHA/JSON/归档仍同步，10/8/2是源码意图不是完整可中止10秒合格。不得为取得完整链放宽期限、预热旧样本、重跑旧10316或按成本选头；超时保留真实部分链和原计划，发布后已选结果不能当未提交回滚。第2族FAIL封闭、第3族未启动；取货已终局64/pending空无81。正式已训CE/同任务随机/各控制/等耗时迭代/强动态和科学完整依赖仍缺，稳定可归因核心在线增益未建立；共享MD/GPU/最终20260927/28未用。\n'
    report=E/'02_配送线/文档/结果与进度/配送_实际模型计时请求与共享取消边界源码准备_2026-10-12.md'
    create(report,'# 实际模型计时请求与共享取消边界源码准备\n'+note+'\n新组件把请求输入解码、实际可见10D特征、全模型encode与全部当前骨架decode、动作过滤/竞争、完整C0、参数和实际依赖核验、响应序列化与发布移入可终止本人worker。固定原初始模型/头和插入规则；旧合格动作链源码、结果和核查原件保持。父只处理有上限共享报文并记录截止前的选择，超时终止本人worker并保留已认证原计划。发布前取消与发布后已选故障必须分别记录，不把后者回滚成未提交。\n\n这次没有导入新模块或运行任何模型、路由或评价函数，仅解析AST和保存源码/输入字节。初始公共报文编码和模型日初加载在拟定窗口外，父仍有同步复制、SHA、JSON与小文件归档；当前源码不能据此声称现场全请求或任意负载硬实时。尚未实现独立计时/实际阶段核查器，未封存准确执行协议与有界派发下载，不允许启动草案。下一先补齐这些门，再事前注册限定工程执行；每次使用前核真实3080 CPU15/RAM至少8GiB及本人进程树，不使用GPU或旧空闲描述。\n\n不得为了完整链结果增加预算或按旧成本换头、挑状态和控制；原初始模型链计算已超过10秒的观察保持，新的8秒执行段允许如实超时/失败。新的各阶段故障若未实际到达不得按标签声称通过。即使以后这项公共快照组件验收成功，正式已训同初始化CE、同任务训练随机、各控制、等耗时MaskCO迭代、独立强动态接拒、全科学依赖与新数据前注册仍是另外的屏障。取货旧日真正终局不再接续callback81，不重跑旧无模型QA；负结果31必要门/183/C1与所有旧FAIL/缺ledger/13800保持。\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    boundary=R/'reproduction/TIMED_ACTUAL_REQUEST_PREPARATION_BOUNDARY_20261012.json'
    create(boundary,dict(directory=str(O),registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),source_only=True,
        execution_authorized=False,model_executed=False,C0_executed=False,full_request_budget_qualified=False,
        formal_controls_qualified=False,scientific_family3_started=False,stable_online_gain=False,
        old_QA_repeated=False,negative183_reorganized=False,shared_MD_uploaded=False))
    next_action='New timed_actual_request_preparation_root_v1 source-only public-snapshot child-owned model/projection/fullchain/fullC0/closure + bounded shared caller10/8/2 draft. No execution/model import; independent timed checker/exact separate execution registration/supervision/dispatch/download missing. First seal actual stage/replay/partialchain/C0/parameter/source/selection-before-deadline and reap checker; explicit synchronous parent/public-only/live-installation limitations remain. Do not blindly run draft or repeat old chains10316/68/54813. True complete10s/equal-time/formal trained same-initCE/task-random/other controls/independentstrongdynamic/science closure before separately registered family3. Pickup terminal64/pendingempty/no81; stable online gain not established.'
    changes=[]
    for x in entries:
        p=E/x['path'];oldtext=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(oldtext);w['updated']=datetime.now(timezone.utc).isoformat()
            w['delivery'].update(timed_actual_request_preparation_directory=str(O.relative_to(ROOT)),timed_actual_request_preparation_sha256=sha(O/'PREPARATION_REGISTRATION.json'),
                timed_actual_request_execution_authorized=False,timed_actual_request_model_executed=False,timed_actual_request_C0_executed=False,next=next_action)
            w['reproduction']['timed_actual_request_preparation_boundary']=str(boundary.relative_to(ROOT));new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+oldtext if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else oldtext+note
            if p.parent==E/'02_配送线':new+='\n- [实际模型计时请求源码准备](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),report=str(report),report_sha256=sha(report),
        boundary_sha256=sha(boundary),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),execution_authorized=False))
    print(json.dumps(dict(prepared_files=len(plan['package_files']),saved_entries=len(changes),registration_sha256=sha(O/'PREPARATION_REGISTRATION.json'),execution_authorized=False)))
if __name__=='__main__':main()
