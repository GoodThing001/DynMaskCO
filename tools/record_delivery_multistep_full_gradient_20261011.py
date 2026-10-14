"""Append actual full-model runtime qualification; preserve prior evidence."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
OUT=R/'delivery/multistep_full_gradient_root_v2';AUD=R/'delivery/multistep_full_gradient_audit_root_v1'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    a=read(AUD/'downloaded/INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json');raw=read(OUT/'downloaded/RESULT.json');start=read(OUT/'downloaded/FULL_SOURCE_START.json');end=read(OUT/'downloaded/FULL_SOURCE_END.json')
    assert a['failed']==0 and a['checks_count']==18347 and raw['failed']==0 and raw['checks_count']==59
    assert read(OUT/'downloaded/RUN_EXIT.json')['exit_code']==read(AUD/'downloaded/RUN_EXIT.json')['exit_code']==0
    assert raw['optimizer_steps']==a['optimizer_steps']==0 and not raw['scientific_training_closure_qualified']
    assert start['files']==end['files'] and not end['changed'] and not end['new_files']
    for name,h in read(AUD/'AUDIT_REGISTRATION.json')['inputs'].items():
        p=OUT/'downloaded'/name
        if not p.exists():p=OUT/name
        assert sha(p)==h,name
    g0=read(OUT/'downloaded/GRADIENT_0.0.json');g1=read(OUT/'downloaded/GRADIENT_0.2.json')
    changed=sum(g0['gradient'][k]['sha256']!=g1['gradient'][k]['sha256'] for k in g0['gradient']);assert changed==234
    report=E/'02_配送线/文档/结果与进度/配送_多步教师完整模型前后向与独立复算验收_2026-10-11.md'
    body='''# 配送多步教师完整模型前后向与独立复算验收（2026-10-11）

匹配此前8个已观察多步教师状态的完整MaskCO前后向，在3080服务器CPU15完成59项运行检查及18,347项独立复算，均0失败。实际加载原CVRP权重，10维扩展后76,263,424参数、234参数叶全部参与梯度计算。本阶段关闭实际完整模型前后向与运行身份缺口；优化器更新次数为0，没有生成新科学数据或新训练权重，稳定、可归因MaskCO核心在线增益仍未建立。

## 原件、运行与身份

生产目录：根`research/review_3080_20261010/delivery/multistep_full_gradient_root_v2`；独立目录：`delivery/multistep_full_gradient_audit_root_v1`。服务器对应独占目录省略本地`delivery`层。运行登记、上传逐字校验、实际父子进程快照、退出回执、原件下载哈希均保存。生产退出0，82.473秒；独立复算退出0，56.533秒。监督1200秒上限只管理本人worker；实际计算固定CPU15、JAX CPU后端，CUDA屏蔽，未占用GPU或停止团队进程。

实际Python3.10.20、NumPy2.2.6、JAX/JAXLIB0.6.2、Flax0.10.4、Optax0.2.4、OR-Tools9.15.6755、protobuf6.33.6；完整包清单见FULL_SOURCE_START。实际完整模型加载和张量构建后、梯度编译前封存8,883项源码/缓存/包树/解释器/native与独占工程输入；前后0变更、0新增未封存文件。独立进程在自身复算前后再次读取这8,883项实际服务器字节，及绑定的11项生产输入/输出。此范围不等完整操作系统或未来科学训练的依赖闭包。

原CVRP checkpoint SHA256=`03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db`；原234叶76,259,840参数。仅新增输入投影的7个零行，原3行保留，扩展后76,263,424参数。两系数使用同一实际初始化，计算前后参数身份均不变；原checkpoint也未改变。

## 实际目标和独立复算

输入是已经验收的两份旧工程快照900000/900001各4步H部署轨迹，8状态全保留原顺序，无筛选、不生成新样本。每步使用该步真实演变的incumbent及掩码，不重复第一步上下文。重新评价incumbent与插入教师的完整C0，原车/未掩码顺序/保留边一致，customer度为2；以完整C0成本差定义标签。全部8状态正标签仅是这批旧工程输入的性质，不是正向部署结果或科学标签分布资格。旧本地教师与当前服务器C0成本复核的1e−8绝对容差已在运行前登记，未事后放宽。

10D输入按已封存字段/单位映射计算，只接受全部已公开的配送快照，不能据此证明动态可见性。CE在customer行上按度2归一；偏好目标使用教师与incumbent邻接差的行log-softmax得分、固定100分母及softplus。两系数0/.2在同一初始化、同一8状态批次上计算完整前后向。CE均值3.509559392929077，偏好损失均值0.2807796001434326；总目标分别3.509559392929077和3.5657150745391846。系数不同的目标值不能直接作性能比较。

独立程序重新写特征公式、标签/择优规则和路由邻接，重新加载原checkpoint并构造模型，计算两次完整234叶梯度；每个梯度SHA、shape、dtype和目标分项与生产精确一致，且全部有限。复算共享已封存的JAX模型、C0评价器和登记的浮点运算分组，因此是独立程序/新进程复算，不是另一物理评价器、另一自动微分实现或完整梯度有限差分证明。增加偏好系数后234叶梯度哈希发生变化，仅说明该目标在这批输入上进入梯度，不能推导部署收益或预训练归因。

## 方法依据与证据边界

JAX官方文档说明`value_and_grad`同时计算函数值与导数；本验收据此直接检查实际目标及全参数梯度，并另开进程复算。梯度不是优化器更新，更不等在线效用提升。[JAX自动微分文档](https://docs.jax.dev/en/latest/automatic-differentiation.html)

模型图和参数状态按实际Flax0.10.4 NNX的graphdef/merge接口处理，状态库存逐叶比较避免把模型重建当新随机初始化。[Flax0.10.4官方graph源码](https://github.com/google/flax/blob/v0.10.4/flax/nnx/graph.py) 这些来源支持实现选择，不支持正向增益主张。

本次没有优化器步骤、保存重载新训练模型、同任务训练随机控制、学习部署轨迹、空车骨架运行、新开发/确认日或真实动态强基线。这些仍需在各自实际执行路径关闭，随后为新科学族数据前登记，不能把本工程输出作为模型权重或将本8旧状态混入未见科学确认。旧DEV16负结果和3×40未启动状态保持，不改旧协议或重跑。

## 工程恢复及三线下一步

首次本地v1在实际资源只读查询前因网络沙箱WinError10013失败，TRANSPORT_FAILURE.json和原派发源保留；其远端未启动。自动审批随后允许授权独占工程网络操作，v2新目录登记后执行，不是效用重试，也没有重启旧DEV16。审计源在上传前删去未执行公式草稿并修正结果字段映射，上传登记字节固定后未改。

配送下一步是匹配多步目标的实际优化器、训练顺序和保存重载工程，以及同初始化CE/同任务训练随机控制，科学新族登记在新数据之前。取货154项接拒与计划联合提交历史资格沿用，公开现场提取、父传输可中止、学习提议/已训控制、真实拒绝与全日整请求10秒仍待；负结果31项必要门和183历史封存沿用，必要门不等完整复现，不重复已完成审计。完整等耗时迭代和真实强动态对照证据仍缺。

只上传已授权独占PY/JSON工程包，报告及16个共享入口只更新本地并保留先前字节。共享MD/历史镜像未上传。自动任务保持ACTIVE。
'''
    with report.open('x',encoding='utf-8') as f:f.write(body)
    entries=[E/x['path'] for x in read(R/'current_entries_before_joint_policy_transaction_20261011/UPDATE_IDENTITY.json')['files']]
    backup=R/'current_entries_before_multistep_full_gradient_20261011';backup.mkdir(exist_ok=False)
    short='\n> 2026-10-11最新：配送旧8多步状态的完整76,263,424参数前后向完成59运行/18,347独立项0失败，8,883实际文件闭包无差异；0优化器更新，只是完整梯度运行资格，未建立稳定MaskCO在线增益。取货154联合组件和负结果31/183资格沿用。\n'
    section='\n\n## 2026-10-11新增：多步教师完整模型前后向实际闭包\n\n配送multistep_full_gradient_root_v2及独立audit_root_v1均退出0，59运行/18,347独立项0失败。实际原CVRP扩展10D后76,263,424参数234叶，8旧H轨迹状态、同初始化系数0/.2的目标和全梯度精确复算；8,883实际运行文件开始结束一致。0优化器步骤、无新科学日/权重；不等完整科学训练、同任务训练随机控制或部署收益。v1资源查询网络故障原件保留，审批后独占新目录v2恢复；共享MD未上传。下一步实际优化器/保存重载与正式控制工程，再为新族数据前登记；不重跑旧DEV16或旧QA。详见`02_配送线/文档/结果与进度/'+report.name+'`。\n'
    files=[]
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat()
            s['delivery'].update(multistep_actual_full_gradient_qualified=True,multistep_full_gradient_runtime_checks=59,multistep_full_gradient_independent_checks=18347,multistep_full_gradient_independent_failures=0,
                multistep_full_gradient_runtime_files=8883,multistep_full_gradient_parameter_elements=76263424,multistep_full_gradient_optimizer_steps=0,
                multistep_full_gradient_directory=str(OUT.relative_to(ROOT)),multistep_full_gradient_independent_directory=str(AUD.relative_to(ROOT)),multistep_full_gradient_independent_identity=sha(AUD/'downloaded/INDEPENDENT_FULL_GRADIENT_ADJUDICATION.json'),
                scientific_training_closure_qualified=False,new_science_family_registered=False,MaskCO_core_online_gain=False,
                next='qualify matched multistep optimizer/order/save-reload and same-init CE/task-trained random controls; preregister new finite family before fresh scientific data; no old DEV16 rerun')
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=short+old if f in (E/'项目当前状态.md',E/'00_项目导航.md',DOC) else old+short
            if f==DOC:new+=section
            if f.parent==E/'02_配送线':new+='\n- [多步教师完整模型前后向与独立复算验收](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');files.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    write(backup/'UPDATE_IDENTITY.json',{'files':files,'new_report':str(report.relative_to(E)),'shared_MD_uploaded':False,'old_science_modified':False})
    print(json.dumps({'entries':len(files),'independent_checks':a['checks_count'],'scientific_training_qualified':False,'report_created':True}))
if __name__=='__main__':main()
