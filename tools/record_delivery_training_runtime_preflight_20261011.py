"""Local append-only runtime evidence and current-entry updates, with backups."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
OUT=R/'delivery/training_runtime_preflight_root_v4'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    a=json.loads((OUT/'INDEPENDENT_RUNTIME_ADJUDICATION.json').read_text());assert a['failed']==0 and a['checks_count']==7334
    reg=json.loads((OUT/'REGISTRATION.json').read_text());raw=json.loads((OUT/'RESULT.json').read_text())
    write(OUT/'REGISTRATION_AMENDMENT_IMPORT_SCOPE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'registered_input_sha256':sha(OUT/'REGISTRATION.json'),'actual_program_sha256':sha(OUT/'REMOTE_CENSUS_PROGRAM.py'),
        'clarification':'v4 removes the absent optional Orbax metadata/import from v1 census; program bytes differ. Actual old NNX/pickle trainer does not import Orbax. Scientific parameters/data/model are not executed or changed.',
        'v1_transport_failure_preserved':True,'v2_inspector_self_match_preserved':True,'v3_optional_package_failure_preserved':True,
        'global_packages_installed_or_changed':False,'same_program_bytes_as_v1':False})
    report=E/'02_配送线/文档/结果与进度/配送_3080实际训练运行环境只读封存与独立验收_2026-10-11.md'
    body='''# 配送3080实际训练运行环境只读封存与独立验收（2026-10-11）

3080的实际CPU训练运行环境完成7316文件的导入前后封存和7334项独立检查，0失败。此阶段只关闭训练环境的只读预检，不代表新教师在服务器环境已合格，不代表实际完整模型梯度或新科学训练已经执行；稳定、可归因MaskCO核心在线增益仍未建立。

## 实际内容及身份

唯一合格目录为根`research/review_3080_20261010/delivery/training_runtime_preflight_root_v4`。服务器实际解释器为`D:/anaconda/envs/MASKCO_env/python.exe`，CPU15，JAX仅CPU且明确屏蔽CUDA，没有占用3080 GPU。派发前排除了已结束的本人训练/监督进程；未停止团队进程。5090同日13:44实查两卡利用率95%/96%，继续不把旧“空闲”快照当授权资源资格。

实际版本：NumPy2.2.6、JAX/JAXLIB0.6.2、Flax0.10.4、Optax0.2.4、Chex0.1.90、SciPy1.15.3、ml_dtypes0.6.0、etils1.13.0、OR-Tools9.15.6755、protobuf6.33.6、psutil7.2.2。完整Python版本、解释器路径和哈希见RESULT。不能套用本地多步教师的Python3.12.14/NumPy2.5.3/OR-Tools9.11.4210/protobuf5.26.1，或旧上游安装说明中的版本。

实际导入旧隔离上游的CVRP模型类、sol2adj、coord_normalize以及训练运行库；只运行两个标量的自动微分、三节点邻接/坐标接口和2输入1输出NNX split/merge检查。没有构建或训练7626万参数的MaskCO模型，没有读取checkpoint，没有生成新训练、开发或确认数据。两个标量得到值5和梯度[2,4]，这只是运行库诊断。

7316项包含实际导入源码/缓存字节码、相关安装包树、解释器和已加载native/DLL；同进程前后无变更、无新增未封存文件。独立进程重新读取服务器实际7316项字节哈希、包元数据、CPU后端，并独立计算两标量梯度，与下载原件精确一致，合计7334项0失败。此清单不涵盖未来完整训练可能再加载的依赖，不宣称封存了全部操作系统或未来运行。未来训练必须在自己的完整模型加载、编译及梯度步骤前后另做真实封存和新增依赖检查。

## 保存的工程失败与恢复

v1长编码SSH命令返回Channel closed，无远端完成回执；核实际进程后才恢复。v2进程过滤误把检查器自身表达式中的目录字符串当在跑任务，原进程快照和过滤错误说明保留；v3移除检查器自身和直接父进程后，在新目录发送独占PY/JSON工程包。

v3发现Orbax分发元数据不存在而停止，失败源码及stderr保留。实际旧训练实现使用pickle和NNX，不依赖Orbax。本次没有安装或升级任何全局依赖；v4只移除这项不属于实际路径的可选导入/元数据要求，实际必需运行库保持原版本，程序字节变更在REGISTRATION_AMENDMENT_IMPORT_SCOPE.json中明确绑定，不写成“与v1相同字节”。四个目录都保留，不覆盖旧结果，不以工程恢复选效用。

## 下一屏障与三线边界

配送下一阶段要在实际服务器环境封存匹配多步部署动作的教师和新训练实现，闭合完整目标与梯度、同初始化CE及同任务训练随机控制，然后为新科学族数据前登记有限训练与开发。此次预检不能替代这些屏障；旧DEV16负结果、固定最终权重和3×40未启动状态保持。不重跑旧DEV16，不自动推进旧确认。

取货82项候选安装组件资格保持，公开投影、实际接拒、提议、共同SAA和整请求10秒仍待接通；负结果31项必要门和183文件历史封存沿用，不重复整理。本轮只上传已授权的独占PY/JSON工程包及审计输入，共享MD仅本地维护，无共享MD或历史镜像同步。
'''
    with report.open('x',encoding='utf-8') as f:f.write(body)
    entries=[E/x['path'] for x in json.loads((R/'current_entries_before_atomic_installation_component_20261011/UPDATE_IDENTITY.json').read_text())['files']]
    backup=R/'current_entries_before_training_runtime_preflight_20261011';backup.mkdir(exist_ok=False)
    short='\n> 2026-10-11最新：配送3080实际CPU训练环境7316文件封存、7334项独立核查通过；仅运行库预检，未执行新完整模型训练/科学日。取货82安装组件及整请求缺口、负结果31/183保持。稳定MaskCO核心在线增益尚未建立。\n'
    section='\n\n## 2026-10-11新增：3080实际训练运行库预检独立验收\n\n配送training_runtime_preflight_root_v4完成7316项导入/包/native身份封存和7334项独立服务器复读核查，0失败；NumPy2.2.6/JAX0.6.2/Flax0.10.4/OR-Tools9.15.6755/protobuf6.33.6为本次实际3080环境。没有实际完整模型梯度、checkpoint加载或新数据/训练，不能据此写新科学训练闭包合格。v1通道故障、v2检查器误匹配、v3不必要Orbax要求失败均保留，新目录恢复不改全局依赖。下一步封存服务器多步教师及实际训练闭包、同初始化和任务训练随机控制，再另登记新族。取货整请求缺口、旧DEV16负结果、负结果31/183保持；共享MD仍仅本地。详见`02_配送线/文档/结果与进度/'+report.name+'`。\n'
    files=[]
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat()
            s['delivery'].update(training_runtime_preflight_qualified=True,training_runtime_actual_files=7316,training_runtime_independent_checks=7334,
                training_runtime_independent_failures=0,training_runtime_directory=str(OUT.relative_to(ROOT)),training_runtime_identity=sha(OUT/'INDEPENDENT_RUNTIME_ADJUDICATION.json'),
                actual_3080_runtime_versions=raw['packages'],scientific_training_closure_qualified=False,new_science_family_registered=False,
                next='seal matched multistep teacher and actual full-model training start/end in actual server runtime; qualify same-init CE and task-trained random controls; register new finite family before fresh data; no old DEV16 rerun')
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=short+old if f in (E/'项目当前状态.md',E/'00_项目导航.md',DOC) else old+short
            if f==DOC:new+=section
            if f.parent==E/'02_配送线':new+='\n- [3080实际训练运行环境只读封存与独立验收](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');files.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    write(backup/'UPDATE_IDENTITY.json',{'files':files,'new_report':str(report.relative_to(E)),'shared_MD_uploaded':False,'old_science_modified':False})
    print(json.dumps({'entries':len(files),'independent_checks':7334,'full_actual_training_qualified':False,'report_created':True}))
if __name__=='__main__':main()
