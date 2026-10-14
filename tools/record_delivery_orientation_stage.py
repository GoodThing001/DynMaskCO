"""Record observed structural diagnosis without altering scientific family2."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
BASE=R/'delivery/family2_orientation_action_diagnosis_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
 with Path(p).open('x',encoding='utf-8') as f:
  if isinstance(x,str):f.write(x)
  else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 result=read(BASE/'RESULT.json');ind=read(BASE/'INDEPENDENT_EVENT_ARITHMETIC.json')
 assert (result['checks_count'],result['failed'])==(83,0) and (ind['checks_count'],ind['failed'])==(104,0)
 original=R/'delivery/multistep_insertable_family2_root_v1'
 source=read(original/'downloaded/SOURCE_START.json')['files'];matches=[]
 for local,suffix in [(E/'scripts/delivery/delivery_contract.py','delivery_contract.py'),(R/'delivery/multistep_full_gradient_root_v2/features_objective.py','features_objective.py')]:
  found=[(p,h) for p,h in source.items() if p.replace('\\','/').endswith('/'+suffix)];assert len(found)==1 and sha(local)==found[0][1]
  matches.append({'local':str(local),'actual_producer_path':found[0][0],'sha256':found[0][1]})
 assert sha(original/'RESEARCH_REGISTRATION.json')=='562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176'
 create(BASE/'ACTUAL_SOURCE_EQUIVALENCE.json',{'matches':matches,'family2_registration_unchanged':True,'new_training':False})
 report=E/'02_配送线/文档/结果与进度/配送_四组方向别名定位与动作目标结构验收_2026-10-11.md'
 lines=['# 配送四组方向别名：具体定位与动作目标结构验收（2026-10-11）','',
 '四组原教师/固定二值探针同无向邻接的差异，现在已定位为同一车槽内一条路线完全反向；订单所属车辆、其他车辆路线与执行轨迹均保持。不能再将这四项笼统解释为车槽分配变化。83项结构检查与另进程104项原始事件算术核查均0失败，仅旧数据诊断和目标编解码工程。第2族原注册/失败、全部工件与训练关闭不变，没有新科学族、模型权重、求解器运行或在线收益。','',
 '## 原件、差值与具体解释','',
 '从第2族96,248项独立审计的全部四组同邻接记录确定集合，顺序读取原STATES导出四组请求和原报告；不是从新收益样本选例。全部输入开始结束哈希一致。合同与旧目标模块的本地实际字节进一步匹配原服务器SOURCE_START清单，完整成本权重为距离1、能耗0.1、品质损失1。物理参数仍文献/仿真pilot。','',
 '| 状态(instance,step) | 反向车槽 | 探针−教师距离 km | 能耗 kWh | 品质价值损失 | 完整成本 |','|---|---:|---:|---:|---:|---:|']
 for p in result['pairs']:
  d=p['oracle_minus_teacher'];lines.append('| %s | %s | %.9f | %.9f | %.9f | %.9f |'%(tuple(p['key']),p['changed_vehicle'][0],d['distance_km'],d['energy_kwh'],d['quality_loss_value'],d['cost']))
 lines.extend(['',
 '独立核查未调用生产目标编解码器或物理求解器：逐事件求和距离、制冷电输入、预冷、开门恢复能耗，以及逐订单品质价值损失；与原报告的各项和完整成本差在1e−12内一致。每辆车deliver事件顺序与原路线逐项核对。此前服务器完整C0重放已经完成，本次是独立账本算术，不冒称另一热物理实现。距离项四组严格不变；(51,1)品质更差但节能足以抵消。三个反向探针比限时教师更好，第四更差，故不能将任一方向默认优胜。','',
 '这些差值很小：完整成本绝对变化0.0014至0.0780。它们证明表示确实折叠有成本意义的方向，不解释其余30项不同邻接的探针错配，也不解释第1族相对GLS的大幅距离劣势。不能把发现这一信息缺口等同于找到主要收益瓶颈。','',
 '## 隔离修正：无损目标与明确插入动作','',
 '新独占结构组件以(vehicle_slot, from_node, to_node)有向弧及显式车槽数编码路线。逐车从仓库沿唯一后继恢复，验证单一入/出度、完整仓库连通、无重复订单；空车槽不压缩。四组教师和探针分别精确往返，反向目标可区分；另有空槽和换车槽区分边界。此组件没有改原MaskCO、旧模型或生产重构器，也没有选择某个反向解。','',
 '另从既定教师路线生成(customer, vehicle, position, predecessor, successor)插入动作。先去掉掩码订单，固定未掩码原车和严格顺序，再按教师目标顺序生成与当前骨架匹配的锚点。八条既有目标逐步重放精确恢复，所有掩码订单各一次。重复订单、未掩码换车/换序、断开环和过时锚点均拒绝。','',
 '这是教师强制的结构目标编码，不是生产greedy解码器、预测器或完整C0认证。知道目标后把动作写出来不能证明模型能预测这些动作，也不证明中间插入状态均物理可行。车槽有向弧张量不是现有无向模型输出的直接替换；新增输出/特征、掩码、损失、推理资源及等预算控制需实际实现、验收并另登记。原目标的邻接差偏好分数对这四组中的两条同邻接路线恒相同；这不意味着第2族实际教师相对原计划的偏好梯度全为零，后者是另一对比较。','',
 '## 文献与推导边界','',
 '[Kool等，ICLR 2019，§3及附录](https://pure.uva.nl/ws/files/44641651/attention_learn_to_solve_routing_problems_.pdf)将路线作为顺序决策序列，解码上下文与约束掩码随问题定义改变。这支持显式保留序列和执行状态的设计原则；该论文不规定本项目的三元弧、五元插入标签或冷链权重，也不保证此修正取得收益。本文具体编码是项目数学构造及上述旧实例结构验收。','',
 '[MaskCO作者实现](https://github.com/Thinklab-SJTU/MaskCO)的参考解掩码恢复与迭代重构仍是方法范式，静态距离任务的方向等价不能直接转为本冷链完整成本等价。本次作者仓库页面读取受服务错误影响，仅沿用此前已读取的官方说明；不新增全文审读声明。原训练目标字节和原裁决不因替代表示构造改写。','',
 '## 后续及三线边界','',
 '第2族保持封闭，不训练、不重复采集/审计/开发或确认。下一配送工程应检查剩余不同邻接错配及生产插入打分如何保留方向、原车和锚点，实际模型与生产适配尚未实现；科学第3族尚未启动。先接续取货真正现场公开提取、可中止传输、正式学习/同初始化CE、独立强动态接拒和全日10秒缺口，避免重复已完成旧请求QA。负结果只追加本次明确的证据边界，31必要门不是全复现、183历史/C1失败仍保持。没有服务器共享MD上传或公开发布。稳定、可归因MaskCO核心在线正向，完整等耗时迭代和真实强动态证据仍缺。','',
 '工件：research/review_3080_20261010/delivery/family2_orientation_action_diagnosis_root_v1下ENGINEERING_SCOPE、OBSERVED_PAIRS、RESULT、INDEPENDENT_EVENT_ARITHMETIC与ACTUAL_SOURCE_EQUIVALENCE。相应两核查工具及目标组件仅在本地tools独占，未部署服务器。'])
 create(report,'\n'.join(lines)+'\n')
 receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'structure_checks':83,'independent_event_checks':104,'failed':0,'direction_only_pairs':4,'vehicle_assignment_changed':False,'new_codec_not_production_decoder':True,'new_scientific_family':False,'stable_online_gain':False,'original_family2_FAIL_unchanged':True,'result_sha256':sha(BASE/'RESULT.json'),'independent_sha256':sha(BASE/'INDEPENDENT_EVENT_ARITHMETIC.json')}
 create(R/'reproduction/FAMILY2_DIRECTION_ALIAS_EVIDENCE_20261011.json',receipt)
 backup=R/'current_entries_before_orientation_diagnosis_20261011';backup.mkdir(exist_ok=False)
 entries=read(R/'current_entries_before_family2_independent_closeout_20261011/UPDATE_IDENTITY.json')['files'];identities=[]
 note='\n> 2026-10-11四组方向别名定位完成：四组全为同车槽一条路线反向，分配/距离不变，差异来自能耗和品质；83结构/104独立事件算术项0失败。有向车槽与锚点动作目标仅无损结构组件，未接入模型或生产解码器。第2族FAIL/训练关闭保持，不证明正向或主要瓶颈，取货现场缺口继续。\n'
 for entry in entries:
  f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);text=f.read_text(encoding='utf-8')
  if f.suffix=='.json':
   s=json.loads(text);s['updated']=datetime.now(timezone.utc).isoformat();s['delivery'].update(orientation_alias_directory=str(BASE.relative_to(ROOT)),orientation_aliases_all_same_slot_reversal=True,orientation_structural_checks=83,orientation_independent_event_checks=104,orientation_target_codec_qualified=True,oriented_learned_decoder_qualified=False,orientation_gap_explains_all_failures=False,next='Continue pickup true-live all-request control gaps; delivery observed mismatches/action scoring and directed-slot target integration remain engineering only. Family2 closed FAIL, no repeat or training; new scientific family requires prospective registration.')
   s['reproduction']['direction_alias_boundary']='research/review_3080_20261010/reproduction/FAMILY2_DIRECTION_ALIAS_EVIDENCE_20261011.json';text=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
  else:
   text=note+text if f.name in ('项目当前状态.md','00_项目导航.md','三线新增交接_独立核查修正与下一阶段_2026-10-10.md') else text+note
   if f.parent==E/'02_配送线':text+='\n- [四组方向别名定位与动作目标结构验收](文档/结果与进度/'+report.name+')\n'
  f.write_text(text,encoding='utf-8');identities.append({'path':entry['path'],'before_sha256':sha(saved),'after_sha256':sha(f)})
 create(backup/'UPDATE_IDENTITY.json',{'files':identities,'report':str(report.relative_to(E)),'shared_MD_uploaded':False})
 assert all(sha(E/x['path'])==x['after_sha256'] and sha(backup/x['path'])==x['before_sha256'] for x in identities)
 print(json.dumps({'updated_entries':len(identities),'structure':83,'independent':104,'failed':0,'stable_gain':False}))
if __name__=='__main__':main()
