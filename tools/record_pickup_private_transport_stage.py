"""Record the private projection/opaque transport component, not full online QA."""
import ast,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';P=R/'pickup'
OUT=P/'private_public_transport_root_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
 with Path(p).open('x',encoding='utf-8') as f:
  if isinstance(x,str):f.write(x)
  else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 a=read(OUT/'INDEPENDENT_PUBLIC_FIELD_ACCEPTANCE.json');r=read(OUT/'RESULT.json')
 assert (a['checks_count'],a['failed'],r['checks_count'],r['failed'])==(289,0,48,0)
 module=ROOT/'tools/pickup_shared_public_transport_v2.py';tree=ast.parse(module.read_text(encoding='utf-8'))
 ask=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='ask')
 calls=[ast.unparse(n.func) for n in ast.walk(ask) if isinstance(n,ast.Call)]
 assert not any(c in ('self.read_frame','available','os.read','json.loads','json.dumps') for c in calls)
 prior=read(P/'private_public_transport_root_v1/REGISTRATION.json')
 helper=ROOT/'tools/pickup_interruptible_public_transport.py'
 assert sha(helper)==prior['transport_sha256']
 static={'current_timed_ask_pipe_or_JSON_calls':False,'inherited_setup_helper_sha256':sha(helper),
         'helper_matches_prior_v1_registration':True,'helper_start_end_seal_in_v2_manifest':False,
         'full_loaded_dependency_closure_qualified':False,'reason':'selected source checks are not all interpreter/native/imported-helper closure',
         'source_sha256':sha(module),'whole_live_10s_qualified':False}
 create(OUT/'TRANSPORT_CALL_BOUNDARY.json',static)
 report=E/'01_取货线/文档/结果与进度/取货_私有状态公共提取与可中止传输组件验收_2026-10-11.md'
 text='''# 取货私有状态公共提取与可中止传输组件（2026-10-11）

新v2组件完成6个正常请求、5个直接到达目标阶段的故障请求，48项运行检查与另进程289项公共字段/原状态/实际选定源码核对均0失败。计时请求的父端不再序列化完整状态，不调用管道查询/读取或大JSON解析，返回通道使用固定上限共享内存；大状态提取和编码在可终止进程内。它只是旧真实私有环境恢复实例上的提取/传输组件，尚未接入实际连续在线模拟器的状态所有权、接拒、模型提议、SAA、认证和安装。完整整请求/五正式臂/全日10秒仍不合格，没有收益实验。

## 旧真实状态与公开边界

固定旧请求0、9、13：分别是既有首请求、已观察多可改订单状态和按既有拒绝结果选定的工程边界。编号13的选择不是新科学样本选择，未重新裁决接拒。可信提取worker在声明的工程设置阶段读取原80订单完整私有数据和原状态，恢复实际StrictOnlineEnv、C0ReserveReplanner、车辆/货物/已执行轨迹；此时没有生成公共报文。计时请求启动后，才调用原public_state.export_public，从完整私有数组中取当前可见订单字段，并重新捕获公共可恢复状态。不同于此前只从已公开packet重新投影。

每个状态另建一个隐藏属性改变且增加4个未来订单的副本。六次导出的公共状态逐项和摘要完全一致，公共数组宽度只按最大已观察ID派生。独立审计不用public_state导出函数，直接从原私有日志构造可见ID、字段零填充/仓库/观察值，并核dtype、每个公共字段、可见性、served切片、原车辆资源/承诺、接拒状态、事件/replan_count和状态哈希。没有把完整私有数据传给学习模块，本次根本没有调用学习模块。

worker可读取私有原件是可信仿真/投影边界的职责，不意味着策略有权访问未来。设置阶段恢复的是三个已观察固定测试环境，不是新在线日；不能把未来生产请求的动态状态准备、复制或重同步挪到设置阶段再声称免预算。后续必须将实时状态所有权与事件推进实际接入。

## 截止与故障

共享控制只写固定24字节的请求编号、截止时间与故障编号，控制区64字节。公共返回缓冲区按注册2MiB上限预分配，与当前图无关。worker在实际投影、JSON编码、返回阶段更新阶段证据，先写完整字节后发布完成标志。父端只读取固定头、有限原始字节及期限，不在此组件中解码公共报文。完成后的离线解析及来源审计不计入此组件，必须在后续完整事务中纳入计算worker和总预算，不能宣布省掉这部分耗时。

五故障分别在实际投影、编码、传输开始处延迟2秒，在共享输出写出128字节后延迟2秒，以及宣布超出2MiB上限。测试限时0.2秒，全部记录目标阶段已到达，父端取消、杀死并回收本人worker；半报文丢弃，过大报文不分配额外缓冲，没有任何策略安装。这是直接阶段证据，不能沿用旧14个仅外层取消标签来冒称深层阶段已验。

普通v2组件耗时3.42至8.16毫秒。四个延迟故障从开始至杀死回收完成220.10至222.34毫秒，超过0.2秒的时间包含检测/系统调度/终止回收；超大长度故障24.83毫秒。不能写严格0.2秒内回收，也不能据此宣布完整10秒硬预算合格。整链需要把取消检测与回收纳入总计时/独立核查，明确截止前计算余量；这些少量故障时延也不是所有系统负载下的上界保证。单写者/单请求共享字节完成标志工程在当前Windows环境通过，不是跨平台原子内存模型或实时系统的形式证明。

## v1保留与v2修正的依据

v1同样有48运行/289独立项通过，原源码、注册和结果保留；但父端使用同步匿名管道PeekNamedPipe轮询。[Windows官方说明](https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-peeknamedpipe)指出同步句柄在多线程环境下仍可能阻塞，因而不能将有限故障通过升级为普遍可中止I/O证明。v2据此移除计时请求内的管道操作，不修改v1结果；相同工程输入、参数和故障，只更换返回传输实现。管道仍只用于声明设置/离线来源审计，不在ask计时请求调用树内。

[Python共享内存官方接口](https://docs.python.org/3.12/library/multiprocessing.shared_memory.html)支持跨进程内存访问并要求清理生命周期。本方案使用共享控制和有限报文区，终止本人子进程后关闭/释放；这份文档不提供本项目调度时延或共享发布安全保证。2MiB、0.2秒、CPU7与固定三个状态是工程选择，不是论文确定值。

原v1通过而v2更换传输有明确工程理由，未重跑旧154接拒联合提交或76拒单QA，也没有新数据/科学族/训练。v2核对了所登记当前源、原私有/公共原件、Python/NumPy实际文件和开始结束选定源；导入的旧管道辅助模块本次只另核当前字节与v1注册一致，并没有v2自身起止封存，全部实际解释器/native/DLL闭包仍未独立合格。不能泛称完整运行环境闭包已经通过。新科学运行前须封存完整实际依赖。

## 未闭合与下一步

先把可信实时状态提取和共同接拒/提议/K10/完整C0/状态编译放到同一个可中止计算事务，避免收到公共字节后在父端再次大解析/序列化；再接入实际状态guard和联合提交。完成真实现场事件所有权、含回收开销的总10秒、正式学习提议、同初始化已训CE、独立强动态接拒、完整日接拒与执行日志后，另登记新鲜有限科学日。当前接拒两个旧模式均own_uncond_dynamic，strong_rolling仅路线模式，不升级为独立强动态基线。

配送四组方向别名及其结构目标沿用，未模型部署；第2族原失败/训练关闭。负结果仅追加本组件边界，183历史/C1联合失败/CI跨零非等价保持，31必要门不等历史全复现。所有新源码与工件在本地独占目录，未使用GPU、未上传共享MD、未公开发布。稳定、可归因MaskCO核心在线增益仍未建立。

工件：research/review_3080_20261010/pickup/private_public_transport_root_v1与root_v2，后者REGISTRATION、RESULT、INDEPENDENT_PUBLIC_FIELD_ACCEPTANCE、TRANSPORT_CALL_BOUNDARY；三个当前工具为qualify_pickup_private_public_transport_v2.py、pickup_shared_public_transport_v2.py、audit_pickup_private_public_transport_v2.py。
'''
 create(report,text)
 receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'normal_requests':6,'direct_stage_faults':5,'runtime_checks':48,'independent_checks':289,'failed':0,
          'directory':str(OUT.relative_to(ROOT)),'normal_wall_range_s':[min(a['normal_walls']),max(a['normal_walls'])],
          'restored_private_environment_only':True,'production_live_ownership_integrated':False,'parent_bulk_decode_included':False,
          'full_loaded_dependency_closure_qualified':False,'whole_live_request_qualified':False,'stable_online_gain':False,'shared_MD_uploaded':False,
          'registration_sha256':sha(OUT/'REGISTRATION.json'),'result_sha256':sha(OUT/'RESULT.json'),'independent_sha256':sha(OUT/'INDEPENDENT_PUBLIC_FIELD_ACCEPTANCE.json')}
 create(R/'reproduction/PRIVATE_PUBLIC_TRANSPORT_EVIDENCE_20261011.json',receipt)
 backup=R/'current_entries_before_private_public_transport_20261011';backup.mkdir(exist_ok=False)
 entries=read(R/'current_entries_before_orientation_diagnosis_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
 note='\n> 2026-10-11取货私有恢复环境公共提取/共享传输组件：6正常+5直接阶段故障，48运行/289独立项0失败；隐藏属性/数量变化不影响公共输入。v1同步管道限制保留，新v2计时请求无管道/大JSON父端操作。恢复旧状态组件，不等真实连续在线事务；回收超0.2秒、父解码/全依赖/正式学习CE与全日10秒仍未合格，无正向增益。\n'
 for entry in entries:
  f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
  if f.suffix=='.json':
   s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['pickup'].update(restored_private_public_transport_component_qualified=True,private_public_transport_directory=str(OUT.relative_to(ROOT)),private_public_transport_runtime_checks=48,private_public_transport_independent_checks=289,private_public_transport_failures=0,private_public_transport_direct_stage_faults=5,private_public_transport_source_full_closure=False,live_event_ownership_integrated=False,parent_bulk_decode_qualified=False,full_live_request_qualified=False,atomic_all_in_budget_qualified=False,next='Integrate actual live event-owned private projection, joint admission/proposal/K10/C0/compiled state into interruptible shared-memory transaction; charge decode/cleanup/commit and seal all actual dependencies. Formal learned proposals, same-init trained CE, independent strong dynamic admissions/full-day still absent; only then preregister finite fresh science.')
   s['reproduction']['private_public_transport_boundary']='research/review_3080_20261010/reproduction/PRIVATE_PUBLIC_TRANSPORT_EVIDENCE_20261011.json';new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
  else:
   new=note+old if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else old+note
   if f.parent==E/'01_取货线':new+='\n- [私有状态公共提取与可中止传输组件](文档/结果与进度/'+report.name+')\n'
  f.write_text(new,encoding='utf-8');identities.append({'path':entry['path'],'before_sha256':sha(saved),'after_sha256':sha(f)})
 create(backup/'UPDATE_IDENTITY.json',{'files':identities,'report':str(report.relative_to(E)),'shared_MD_uploaded':False})
 assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
 print(json.dumps({'updated_entries':len(identities),'runtime':48,'independent':289,'failed':0,'whole_live_request':False}))
if __name__=='__main__':main()
