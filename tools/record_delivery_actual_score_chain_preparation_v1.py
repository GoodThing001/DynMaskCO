"""Record frozen unexecuted integration draft; retain all existing entry bytes."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010'
E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';O=R/'delivery/actual_score_chain_preparation_root_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(v,str):f.write(v)
        else:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    plan=read(O/'INTEGRATION_PLAN_PREPARATION.json')
    assert plan['execution_authorized'] is False and plan['model_executed'] is False
    assert all(sha(O/n)==h for n,h in plan['package_files'].items())
    entries=read(R/'current_entries_before_action_head_module_map_detail_20261012/UPDATE_IDENTITY.json')['files']
    assert len(entries)==16 and all(sha(E/x['path'])==x['after_sha256'] for x in entries)
    backup=R/'current_entries_before_actual_score_chain_preparation_20261012';backup.mkdir(exist_ok=False)
    for x in entries:
        p=backup/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/x['path'],p)
    report=E/'02_配送线/文档/结果与进度/配送_实际评分来源接入完整动作链源码准备_2026-10-12.md'
    note='\n> 2026-10-12 配送完整动作链接线仅源码准备：actual_score_chain_preparation_root_v3固定旧30原顺序首项key[12,3]/25掩码订单，两臂仍原预训initial与fresh888、同headseed20261012/0更新；拟复用已合格两份00评分原件、每臂一次新encoder缓存及后续最多24次当前骨架decode，并接原全C0。只有新consumer/runner的AST与字节封存，无模型导入/forward/完整链/route_ok/C0/服务器执行；execution_authorized=False，独立完整链核查器/准确监督派发下载尚缺，缓存等价与预算不合格。原preparation v1误把删除产生的桥边当保留边、v2新源码括号AST失败及原源/退出1均保留；v3仅正确绑定原initial掩码后的34保留边项（删除后骨架58项并非全受保护），未改旧生产/科学门/参数。68/54813初始组件和所有已完成QA不重跑；第2族FAIL封闭、第3族未启动，取货终局64/pending空无81，科学各控制/等耗时/强动态/整请求10秒仍待；共享MD未上传，无GPU或最终20260927/28。\n'
    text='# 实际评分来源接入完整动作链源码准备\n\n'+note+'\n固定范围在执行前限定为原观察记录原顺序第一项；没有使用旧标签、成本或候选结果选范围。输入仅data/key/initial/mask/before/remaining/原保留边partial，未复制teacher_routes/oracle/目标秩或成本。完整临时动作身份与源参数摘要随每步绑定，初始未训练来源明确保持learned_output_qualified=False；消费器的字符串/摘要一致门是内部调用契约，不是密码学来源认证或正式已训资格。\n\n拟每步对所有剩余客户和当前全部槽/位置一次批量评分，再使用同资格、分数、距离次序、regret及客户ID规则。固定10D输入在链中不变，缓存仅拟复用encoder，decoder及final_norm必须依当前骨架重算。两份已合格首状态NPZ拟逐字复用，不能据此再次运行原60前向核查。两臂工程上限2个新encode和48个新decode，未执行，也未验证缓存与完整逐步前向等价。若没有可行动作保留全部失败轨迹并回退原incumbent；完整候选才走原全C0。没有正向门、重新抽样或按成本改头/参数的资格。\n\n当前仅新源码AST成功与新输入/原件身份封存。首次准备失败在创建工件树前，误要求partial等于删除后所有骨架边；只读差分确认原partial准确等于原initial中不触及masked订单的边，额外删除桥边不应冻结。第二次AST括号错误亦发生在创建执行树前；两个失败源和退出保留，v3新准备树没有远端运行。完整独立动作链核查器、准确执行登记、独占上传、隐藏有界监督及有界只读下载仍待，不直接运行此草案。\n\n下一先完成独立全链核查器，逐新骨架的完整模型重载和缓存/特征/临时锚点精确核验，逐候选物理过滤、保留边、regret、竞争及完整C0复算；封存真实完整依赖、监督和传输后另登记实际执行。使用前重新核3080 RAM至少8GiB、CPU15和本人实际树。900秒阶段监督不会自动取得在线10秒全请求资格。正式已训同初始化CE、同任务已训随机、其它控制、等耗时迭代、独立强动态接拒和全科学依赖仍未建立。稳定可归因MaskCO核心在线增益仍未建立。\n'
    create(report,text)
    boundary=R/'reproduction/ACTUAL_SCORE_CHAIN_PREPARATION_BOUNDARY_20261012.json'
    create(boundary,dict(plan_sha256=sha(O/'INTEGRATION_PLAN_PREPARATION.json'),directory=str(O),
        source_only=True,model_executed=False,chain_executed=False,C0_executed=False,
        cache_equivalence_qualified=False,full_request_budget_qualified=False,formal_controls_qualified=False,
        scientific_family3_started=False,stable_online_gain=False,old_initial_component_repeated=False,
        original_failures_preserved=True,negative183_reorganized=False,shared_MD_uploaded=False))
    next_action='New actual_score_chain_preparation_root_v3 seals unexecuted consumer/runner for fixed first old key[12,3]/25masked and pretrained_initial/random888_initial samehead0updates. Reuse old two00NPZ initial scores without repeating68/54813; proposed two newencode caches and at most48 current-context decodes, originalC0 incumbent/candidate. execution_authorized=False. Next seal independent full chain/cachedencoder/context/anchors/candidate-regret/fullC0 checker plus exact supervision/dispatch/download, then prospective exact execution registration and fresh8GiB/CPU15/actual owned tree check. No blind draft run/family3/data. Whole10s/equal-time/formal trainedCE/task-trainedrandom/strongdynamic/science closure remain. Pickup terminal64/pendingempty/no81. Stable online gain not established.'
    changes=[]
    for x in entries:
        p=E/x['path'];old=p.read_text(encoding='utf-8')
        if p.suffix=='.json':
            w=json.loads(old);w['updated']=datetime.now(timezone.utc).isoformat()
            w['delivery'].update(actual_score_chain_preparation_directory=str(O.relative_to(ROOT)),
                actual_score_chain_preparation_plan_sha256=sha(O/'INTEGRATION_PLAN_PREPARATION.json'),
                actual_score_chain_execution_authorized=False,actual_score_chain_model_executed=False,
                actual_score_chain_C0_executed=False,actual_score_chain_cache_equivalence_qualified=False,next=next_action)
            w['reproduction']['actual_score_chain_preparation_boundary']=str(boundary.relative_to(ROOT))
            new=json.dumps(w,ensure_ascii=False,indent=2)
        else:
            new=note+old if p.name in ['项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md'] else old+note
            if p.parent==E/'02_配送线':new+='\n- [实际评分完整动作链源码准备](文档/结果与进度/'+report.name+')\n'
        p.write_text(new,encoding='utf-8');changes.append(dict(path=x['path'],before_sha256=sha(backup/x['path']),after_sha256=sha(p)))
    create(backup/'UPDATE_IDENTITY.json',dict(files=changes,report=str(report.relative_to(E)),shared_MD_uploaded=False))
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in changes)
    assert all(sha(O/n)==h for n,h in plan['package_files'].items())
    create(O/'LOCAL_RECORDING_RECEIPT.json',dict(report_path=str(report),report_sha256=sha(report),
        boundary_sha256=sha(boundary),entry_manifest_sha256=sha(backup/'UPDATE_IDENTITY.json'),
        original_sources_and_results_unchanged=True,execution_authorized=False))
    print(json.dumps(dict(saved_entries=len(changes),source_files=len(plan['package_files']),execution_authorized=False)))

if __name__=='__main__':main()
