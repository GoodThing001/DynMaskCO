"""Save local evidence for a component, with full-transaction limits explicit."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';P=R/'pickup'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=P/'atomic_plan_installation_root_v2';report=json.loads((out/'INDEPENDENT_INSTALLATION_COMPONENT_REPLAY.json').read_text());assert report['failed']==0 and report['checks_count']==82
    entries=[E/x['path'] for x in json.loads((R/'current_entries_before_multistep_teacher_telemetry_20261011/UPDATE_IDENTITY.json').read_text())['files']]
    backup=R/'current_entries_before_atomic_installation_component_20261011';backup.mkdir(exist_ok=False)
    newreport=E/'01_取货线/文档/结果与进度/取货_候选认证与原子计划安装组件验收_2026-10-11.md'
    body='''# 取货候选认证与原子计划安装组件验收（2026-10-11）

已实现候选认证与现场计划发布组件，独立82项0失败。它关闭父进程重复执行完整C0认证这一组件缺口，未关闭公开投影、接拒决策、五个正式部署臂及整日事务缺口；尚无新科学日或在线增益。

## 新实现及实际验收

唯一目录为根research/review_3080_20261010/pickup/atomic_plan_installation_root_v2。固定使用已观察公共状态及此前五原型选定计划，不再推理模型或采样SAA。可中止worker完成完整C0认证及计划编译；父进程比较请求、数据、编译计划字节身份及原车队/接拒/事件状态，随后仅将实际C0ReserveReplanner的_plan指针发布。物理认证在父提交阶段被禁止调用，测试通过，接拒集合、状态实例和预算不变。

五份固定候选均在实际replanner中完成安装，组件整体耗时15.047–17.484毫秒，接收至提交44–55微秒。实际父/workerCPU7，统一10秒上限；不得将本组件耗时替代完整决策耗时。数据与源码、解释器及NumPy核心实际身份均核查，完整物理证书在该运行环境下与旧原件精确一致，没有放宽算术容差。

12种故障/边界全部保留：认证延迟、编译延迟、计划指针替换、计划原地篡改、接拒集合改变、车队状态改变、事件改变、期限过期、token重复提交、错误请求身份、未合格worker，以及编译计划字节篡改。两种10毫秒延迟故障不安装任何后到计划并回收worker；其余无效提交也保持原计划指针。独立验收核五候选原始身份、全C0证书、现场前后状态、时间、源/运行身份、故障记录及无遗留所属worker，共82项0失败；原运行自核49项0失败。

## 保留的失败与环境更正

v1在cc_ortools（Python3.12.14/NumPy2.5.3）运行，五次安装及全部故障控制通过，但五项旧物理证书逐字身份检查失败：能耗差约−0.992至−1.714微kWh、番茄品质差约3.76e−10至4.28e−10，效用与硬可行完全相同。v1结果、源码及RUNTIME_IDENTITY_FAILURE.json保留，环境兼容性不合格。

v2数据前登记固定NumPy1.26.4及实际原型环境Python3.12.7，计划安装模块字节与v1完全相同，以原运行环境恢复相同工程检查，证书逐字一致。未更改旧科学协议、物理源码或容差，没有换环境挑效用。新科学实验必须为所有臂统一并封存实际环境，不能把旧版本名当跨环境精确重现的证明。

[NumPy官方2.0迁移说明](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html)和[NEP50](https://numpy.org/neps/nep-0050-scalar-promotion.html)记录了混合标量运算类型提升变化，可导致较低精度；它是本差异的候选解释，本轮未定位造成此轨迹差异的具体算术行，故不宣称已证明唯一根因。

## 资格边界与下一步

输入公共状态已提前导出，因此此次组件验收不包含现场公开投影；计划候选已存在，不包含模型推理、接拒控制或公共场景求值。实际类来自旧状态恢复，尚未接入整日运行。单指针发布也不是操作系统严格实时保证；它只是所测流程中限时、状态一致的现场安装组件。

下一步接入实际整请求控制链，把公开投影、接拒、提议、共同SAA、认证、传输及原子安装放入同一可中止10秒计时，补完整接拒事件日志和同初始化已训重建控制，再做已观察整日工程验收。合格后才另登记新鲜有限科学日，不触碰最终留出或重跑旧I3/D2。负结果31项必要门/183文件封存、配送多步工程资格保持，本轮不重复整理。文档仅本地，无共享MD上传。
'''
    with newreport.open('x',encoding='utf-8') as f:f.write(body)
    short='\n> 2026-10-11最新：取货候选worker认证与实际计划原子安装组件独立82项通过，12故障/边界保留；公开投影、接拒、正式五臂和整日10秒事务仍未合格。配送多步9670项/负结果必要31项沿用；稳定MaskCO核心在线增益尚未建立。\n'
    section='''

## 2026-10-11 新增：取货候选现场安装组件独立验收

atomic_plan_installation_root_v2从此前固定五候选完成实际C0ReserveReplanner计划安装，父提交不再做完整C0认证；同一worker认证/编译、状态guard和单指针发布，独立82项0失败，12故障/边界全保留。v1 NumPy2.5.3与旧NumPy1.26.4证书逐字差异的五项失败保持；v2登记原环境、模块字节不变恢复，逐字一致，不放宽容差。只关闭候选安装组件，不含此前已导出的公共投影、接拒/提议/SAA或整日正式五臂。完整事务仍待，详见`01_取货线/文档/结果与进度/取货_候选认证与原子计划安装组件验收_2026-10-11.md`。未生成新科学日、训练或在线正向；配送9670和负结果31/183沿用，无服务器共享MD上传。
'''
    manifest=[]
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(candidate_live_installation_component_qualified=True,candidate_live_installation_independent_checks=82,candidate_live_installation_failed=0,candidate_live_installation_fault_cases=12,candidate_live_installation_directory=str(out.relative_to(ROOT)),candidate_live_installation_identity=sha(out/'INDEPENDENT_INSTALLATION_COMPONENT_REPLAY.json'),candidate_installation_public_projection_included=False,candidate_installation_actual_admission_included=False,candidate_installation_v1_environment_failure_preserved=True,atomic_all_in_budget_qualified=False,all_five_formal_deployment_qualified=False,next='integrate actual public projection/admission/proposals/common SAA/certification/atomic candidate publication under one interruptible 10s request; full observed-day engineering and same-init trained CE control before new science registration');new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=(short+old if f in (E/'项目当前状态.md',E/'00_项目导航.md',DOC) else old+short)
            if f==DOC:new+=section
            if f.parent==E/'01_取货线':new+='\n- [候选认证与原子计划安装组件验收](文档/结果与进度/'+newreport.name+')\n'
        f.write_text(new,encoding='utf-8');manifest.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    with (backup/'UPDATE_IDENTITY.json').open('x') as f:json.dump({'files':manifest,'new_report':str(newreport.relative_to(E)),'shared_MD_uploaded':False,'old_science_modified':False},f,indent=2)
    print(json.dumps({'entries':len(manifest),'report_created':True,'whole_transaction_qualified':False}))
if __name__=='__main__':main()
