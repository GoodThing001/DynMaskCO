"""Local current-entry update with prior bytes saved; no remote MD sync."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';P=R/'pickup';OUT=P/'private_joint_integration_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    a=read(OUT/'INDEPENDENT_PRIVATE_JOINT_REPLAY.json');b=read(OUT/'INDEPENDENT_MEAN_ORDER_RECOVERY_R2.json')
    assert (a['checks_count'],a['failed'],b['checks_count'],b['failed'])==(159,2,16,0)
    assert b['qualified_component'] and b['producer_sha256']==sha(OUT/'RESULT.json') and b['original_audit_sha256']==sha(OUT/'INDEPENDENT_PRIVATE_JOINT_REPLAY.json')
    report=E/'01_取货线/文档/结果与进度/取货_私有提取至接拒评分联合提交组件_2026-10-11.md'
    text='''# 私有提取至接拒评分联合提交组件（2026-10-11）

新独占private_joint_integration_root_v1将旧私有恢复环境的公共提取、实际接拒、原计划提议、共同K10完整分支、完整C0认证、状态编译和共享返回串在同一可终止worker中，再由父端解码和原JointPolicyOwner守卫联合发布。三个旧工程请求全部提交：编号0/9接受，13拒绝，三者均未换路线。41项运行检查通过；独立原159项中157项通过、2项均值核验失败，经追加16项同分组逐位复算通过解决核验器假设，原失败保留。没有生成新研究日、调用学习模型或测量在线收益。

## 接通的范围

输入仍是原80订单完整私有日及三个旧捕获状态；可信worker在设置阶段恢复这些已观察环境，接到固定共享请求后才调用原public_state.export_public。公共数据由实际私有数组和可见性重建，随后计算只用公共报文。父请求通道只写固定控制字段并读取上限2MiB共享报文，不再把大公共状态编码进同步管道。

接拒仍为own_uncond_dynamic；提议仅own_incumbent，没有strong_rolling路线模式、预训练/随机提议或已训同初始化CE。共同场景K10、原200日历史、seed202610111301、主档(5,10,15)、合同和域算法保持既有工程值。两个相同原计划各20完整分支是接口联合证据，不能称为实际不同五臂已通过。父端禁止重做完整C0，实际接拒集合和计划由原编译状态的单指针一起发布。

编号13是原拒绝结果选择的工程边界，非新科学样本；原捕获公共状态已把待处理订单放在拒绝集合占位，所以本组件不是新事件第一次透露订单的现场证据。后续必须支持真正未决的当前新订单，并核验既有公共导出要求“全部可见已决定”的限制；不能为了导出通过而把尚未决定的订单偷偷记为实际拒绝。实时事件所有权和连续车辆执行仍未接线。

## 实际时间与故障

本地Python3.12.7/NumPy1.26.4、CPU7，三个正常请求从父守卫准备至解码、提交和本人子进程回收为3.612858、2.987087、2.534452秒，均小于10秒。设置阶段恢复旧环境、进程启动和ready身份核查不在此请求计时；该排除仅适用于声明的工程夹具，不提供实际在线每事件复制/同步状态免费的理由。

八个故障分别在实际私有提取、接拒、提议、共同评分、认证、编译、结果编码、共享输出阶段入口延迟8秒；每请求上限6秒。阶段共享标记证明相应入口被实际到达，所有请求终止并回收，未调用提交。每次6.019545至6.033355秒，回收总耗时超出6秒，不能写严格预算内回收。它们证明实际阶段入口可取消，不证明阶段内部任意指令或操作系统负载下的时延上界；原14个只支持外层取消的故障测试保持原范围。

父端可信本地响应pickle解码虽已计入正常墙钟，仍不能被同一父线程抢占；JointPolicyOwner的签名/摘要/提交也属于父端循环。普通时延通过不等完整10秒可抢占事务。下一版应把大响应解码和状态编译安排在计算端，给父端最小可验证准备状态，明确回收预留，并在现场事件推进/提交路径验证。共享内存中的发布标志也不是跨平台原子内存或硬实时形式证明。

## 两项均值核验失败的准确处置

原独立重放所有接拒状态、20分支值、原场景身份、完整C0和实际源码身份精确一致。两项失败位于请求13两个相同候选的均值：Python顺序累加为506.3336914263426，保存的NumPy均值为506.33369142634274，差−1.1368683772161603e−13。检查器错误地把两种浮点归约顺序当作应逐位相同。

[NumPy1.26 sum说明](https://numpy.org/doc/1.26/reference/generated/numpy.sum.html)说明浮点归约可使用分组求和，其精度依顺序与参数变化。[1.26.4实际源码](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/loops_utils.h.src)给出10元素对应的8路分组与尾部累加。新独立Python直接实施这个加法分组，六个候选均值的hex与原记录逐位一致，同时记录分支二进制浮点转为精确有理数后的平均值。没有增加误差容差、改变注册选择阈值或改写科学结果，亦未重新执行接拒/评分/求解器来挑均值。

原INDEPENDENT_PRIVATE_JOINT_REPLAY.json仍159项/2FAIL，原生产41项/0FAIL和源字节保持；追加INDEPENDENT_MEAN_ORDER_RECOVERY_R2.json的16项/0FAIL关闭核验器归约假设，不能把原文件改写为0失败。当前组件资格由157项原成功重放与新逐位算术资格共同支持，不是另一套独立物理实现。

## 身份与后续屏障

30项选定实际源/工具/历史/私有输入/Python/NumPy/native文件起止一致，独立当前字节再核，包括先前遗漏的传输辅助模块。仍不是全部已加载stdlib、JAX或DLL闭包，full_loaded_dependency_closure_qualified=False；科学运行需要实际完整封存。所有自有测试worker已回收，未使用GPU、服务器或上传共享MD。

下一步：实际未决新订单的可信现场公共提取和事件状态所有权、父解码及发布可中止/回收计入、正式模型提议与同初始化已训CE、独立强动态接拒、全日完整执行日志。工程合格后再登记新鲜有限科学日。配送第2族原科学屏障FAIL保持关闭，四组方向组件不等模型动作适配；负结果仅追加本阶段边界，历史183/C1联合失败/CI跨零非等价不变。稳定、可归因MaskCO核心在线增益、等耗时迭代和真实强动态对照证据仍未建立。
'''
    create(report,text)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(OUT.relative_to(ROOT)),'normal_requests':3,'stage_entry_faults':8,'runtime_checks':41,'runtime_failed':0,'original_independent_checks':159,'original_independent_failed':2,'original_FAIL_preserved':True,'arithmetic_recovery_checks':16,'arithmetic_recovery_failed':0,'mean_tolerance_relaxed':False,'normal_walls_s':[x['wall_s'] for x in a['summaries']],'true_new_reveal_qualified':False,'continuous_live_ownership_integrated':False,'parent_decode_preemptible':False,'full_loaded_dependency_closure_qualified':False,'whole_live_request_qualified':False,'stable_online_gain':False,'shared_MD_uploaded':False,'result_sha256':sha(OUT/'RESULT.json'),'original_audit_sha256':sha(OUT/'INDEPENDENT_PRIVATE_JOINT_REPLAY.json'),'recovery_sha256':sha(OUT/'INDEPENDENT_MEAN_ORDER_RECOVERY_R2.json')}
    create(R/'reproduction/PRIVATE_JOINT_INTEGRATION_EVIDENCE_20261011.json',receipt)
    backup=R/'current_entries_before_private_joint_20261011';backup.mkdir(exist_ok=False)
    entries=read(R/'current_entries_before_private_public_transport_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
    note='\n> 2026-10-11取货联合链组件：三个旧私有状态的公共提取→接拒→原计划提议→K10→完整C0→共享返回→解码/联合提交/回收已接通，41运行项通过。原独立159项两均值顺序FAIL保留，新16项同分组逐位复算关闭核验器假设；2接受/1拒绝、均未换路线，八实际阶段入口取消。父解码不可抢占、真实未决新订单/现场所有权/正式CE及全日10秒仍待，无收益。\n'
    for entry in entries:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(restored_private_joint_component_qualified=True,private_joint_directory=str(OUT.relative_to(ROOT)),private_joint_runtime_checks=41,private_joint_runtime_failed=0,private_joint_original_independent_checks=159,private_joint_original_independent_failed=2,private_joint_arithmetic_recovery_checks=16,private_joint_arithmetic_recovery_failed=0,private_joint_original_FAIL_preserved=True,private_joint_direct_stage_faults=8,true_undecided_reveal_qualified=False,live_event_ownership_integrated=False,parent_bulk_decode_qualified=False,full_live_request_qualified=False,atomic_all_in_budget_qualified=False,next='Define true undecided current-order public projection without rejection placeholder, integrate event-owned private state, interruptible decode/commit with cleanup reserve; formal learned proposals, trained same-init CE, independent strong admissions/full-day remain required. Seal actual full dependencies before preregistering fresh science.')
            s['reproduction']['private_joint_boundary']='research/review_3080_20261010/reproduction/PRIVATE_JOINT_INTEGRATION_EVIDENCE_20261011.json';new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=note+old if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else old+note
            if f.parent==E/'01_取货线':new+='\n- [私有提取至接拒评分联合提交组件](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');identities.append({'path':entry['path'],'before_sha256':sha(saved),'after_sha256':sha(f)})
    create(backup/'UPDATE_IDENTITY.json',{'files':identities,'report':str(report.relative_to(E)),'shared_MD_uploaded':False})
    assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
    print(json.dumps({'updated_entries':len(identities),'runtime_checks':41,'original_independent_FAIL_preserved':2,'arithmetic_recovery_checks':16,'recovery_failed':0,'full_live_request':False}))
if __name__=='__main__':main()
