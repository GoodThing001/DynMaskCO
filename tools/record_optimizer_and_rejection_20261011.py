"""Record new accepted engineering evidence without changing original failures."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';P=R/'pickup'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    training=D/'multistep_optimizer_root_v1';audit=D/'multistep_optimizer_audit_root_v2';diagnostic=D/'optimizer_arithmetic_diagnosis_root_v1';reject=P/'joint_policy_rejection_root_v1'
    result=read(training/'downloaded/RESULT.json');a=read(audit/'downloaded/INDEPENDENT_OPTIMIZER_ADJUDICATION.json');n=read(diagnostic/'downloaded/NUMERICAL_DIAGNOSIS.json');q=read(reject/'INDEPENDENT_REJECTION_TRANSACTION_REPLAY.json')
    assert result['failed']==2 and result['checks_count']==45 and not result['engineering_training_path_qualified']
    assert a['failed']==0 and a['checks_count']==22769 and a['corrected_arithmetic_audit_qualified'] and a['original_failure_preserved']
    assert read(training/'downloaded/RUN_EXIT.json')['exit_code']==1 and read(audit/'downloaded/RUN_EXIT.json')['exit_code']==read(diagnostic/'downloaded/RUN_EXIT.json')['exit_code']==0
    assert q['failed']==0 and q['checks_count']==76 and len(q['summaries'])==2 and all(not s['accepted'] for s in q['summaries'])
    assert n['all_FTZ_predictions_exact'] and n['frozen_failure_unchanged'] and n['optimizer_steps']==0
    assert all(v['all234_final_hashes_equal'] and not v['research_model_eligible'] for v in a['outcomes'].values())
    for name,h in read(audit/'AUDIT_REGISTRATION.json')['inputs'].items():
        path=training/'downloaded'/name
        if not path.exists():path=training/name
        assert sha(path)==h,name
    for base,prefix in ((training,'TRAINING'),(audit,'AUDIT')):
        start=read(base/'downloaded'/(prefix+'_SOURCE_START.json'));end=read(base/'downloaded'/(prefix+'_SOURCE_END.json'))
        assert start['files']==end['files'] and not end['changed'] and not end['new_files']
    assert sha(reject/'joint_install.py')==sha(P/'joint_policy_transaction_root_v3/joint_install.py')
    distinction={'historical_input_policy':'cond_hist','new_transaction_admission_policy':'own_uncond_dynamic','route_modes':['own_incumbent','strong_rolling'],'strong_rolling_is_independent_admission_arm':False,'old_controller_source_sha256':sha(P/'complete_history_energy_qa_root_v1/complete_history_worker.py'),'new_transaction_source_sha256':sha(ROOT/'tools/qualify_pickup_joint_rejection_root_v1.py'),'same_input_is_not_same_policy':True,'historical_decision_equivalence_claimed':False,'source_evidence':'complete_history_worker.py sets config name cond_hist; new transaction sets own_uncond_dynamic. Differences in other old admission outcomes do not establish an engineering bug or a gain.'}
    create(reject/'SOURCE_POLICY_DISTINCTION.json',distinction)
    dr=E/'02_配送线/文档/结果与进度/配送_三臂固定更新保存重载与浮点核验更正_2026-10-11.md'
    pr=E/'01_取货线/文档/结果与进度/取货_真实拒单联合提交边界独立验收_2026-10-11.md'
    create(dr,'''# 配送三臂固定更新、保存重载与浮点核验更正（2026-10-11）

三臂完整模型各2个固定工程步骤已完成；新增独立程序在新进程重算6次完整梯度、独立更新算术及实际最终权重，22,769项0失败。本阶段关闭旧多步输入上的更新、顺序及保存重载工程路径，保留原生产退出1和两项核验失败。没有重训、改学习率、放宽容差、选择checkpoint或启动新开发/确认日。稳定、可归因MaskCO核心在线增益仍未建立。

## 范围与实际控制

生产原件为根`research/review_3080_20261010/delivery/multistep_optimizer_root_v1`；修正算术依据在`optimizer_arithmetic_diagnosis_root_v1`；新增独立验收在`multistep_optimizer_audit_root_v2`。服务器对应独占目录省略本地delivery层，计算固定CPU15、JAX CPU、屏蔽CUDA。实际资源及本人进程核查后派发，有界监督仅管理本人worker。未使用5090，也未停止团队进程。

输入固定为之前合格的旧8状态，次序为900000/900001各4步；两批[0,1,2,3]和[4,5,6,7]，每臂各2步。234参数叶、76,263,424参数。pretrained_CE与pretrained_paired使用逐叶相同原CVRP初始化及7个零扩展行，偏好系数分别0/.2；random_task_paired从同架构原3D构造器seed888独立初始化，再同样零扩展，偏好系数.2，与预训练偏好臂同目标/批次/步数。随机控制实际受训，尚不是科学任务训练控制资格。

固定SGD无动量、学习率0.00001。完整encoder和decoder实际更新；预训练两臂各234叶改变，随机臂229叶改变，其余5叶没有发生实际改变，未要求所有叶必改。三份新工程最终权重均明确research_model_eligible=False；这两步不是新的科学训练族或可部署模型。

## 原失败与明确修正

原生产45项检查中的2项失败，退出1，168.865秒，原JSON、源、步骤日志和3份权重全部保留。失败名是两个预训练臂首次更新的SGD_float32_formula_exact；其余43项通过。原检查直接采用NumPy渐进下溢算术比较实际JAX CPU更新，假设不适用于本运行点。

独占诊断使用同原参数、第一批旧数据及完整梯度，精确重现原失败中的梯度和实际更新。两个首次更新均只有final_norm.scale的14个元素不同；最大绝对差1.11445687257292e−38，全部属于float32次正规数，实际JAX值均为0。float32最小正规数为1.1754943508222875e−38。单独受控乘法探针也直接重现次正规输入/结果归零。NumPy核验器显式建模输入、乘积及输出的flush-to-zero后，234叶全部精确匹配；诊断退出0、58.814秒。这里有实际运行证据，不仅是从文档推测。

JAX官方文档说明部分后端会将次正规数归零；NumPy文档区分smallest_normal与其下的次正规范围。本验收采用已测服务器数值行为，仍逐位比较，无误差阈值，也不改变原SGD实现或科学参数。[JAX数值精度说明](https://docs.jax.dev/en/latest/accuracy.html)；[NumPy浮点范围定义](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)。文档支持解释，不能替代上述实际梯度和受控探针。

## 新增独立验收

独立程序在执行前登记原失败源/结果/步骤/模型卡及诊断哈希，重新从原CVRP checkpoint和随机构造器建立相同初始化；每步目标分项、234叶完整梯度、更新前后SHA/shape/dtype精确核对。更新由NumPy独立实现并显式建模实测FTZ；六步、三臂的全部最终参数逐叶匹配实际已保存权重，文件SHA及重新加载模型目标也精确一致。独立程序未保存新权重或修改原训练工件，退出0、139.942秒。原生产的两项FAIL继续存在，新增裁决单独记为corrected_arithmetic_audit_qualified=True。

生产实际8,886项源码/字节码/运行库/native/输入开始结束无差异，独立进程另封存自身8,885项运行文件无差异；新进程还在前后重新读取全部原生产文件。原checkpoint SHA仍为03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db。源闭包计数因进程及独占输入不同不应强求相等，比较的是各自前后字节和原输入绑定。

三个工程最终参数摘要：CE为8ea95659aade599e4be78a2e67ed03518936e57869a6a89c90b9ce4bd8b3f247；预训练偏好为563a6d4ac604242cf7624d10793291cc0a4874c6a93d84c75557382352a1be2f；随机任务偏好为9ed7d12c2d6e7d0632c9c9be82392a59025b60468d1236f562e17ef423949a3b。完整PKL保留服务器原目录，独立进程直接读取、重算哈希和全部参数；本地已下载完整检查JSON/日志/清单/模型卡，未声称900MB权重已下载。

共享JAX模型及先前合格张量，独立程序/算术和新进程复算不等另一自动微分或物理评价器。旧8状态全正标签、工程目标变化和参数更新不支持收益。科学训练/新数据/科学任务随机控制、学习分布覆盖、强动态对照、等耗时迭代和独立稳定确认仍未合格；旧DEV16负裁决及3×40未启动状态保持。

下一步先封存服务器实际多步教师与部署闭包、补齐科学运行屏障，再在生成新数据前登记有限新族，保留同初始化CE和同任务训练随机控制；不能将本工程权重用于确认。负结果必要门31项和183历史封存沿用，未重复原审计或公开发布。共享MD仍只更新本地。
''')
    create(pr,'''# 取货真实拒单联合提交边界独立验收（2026-10-11）

此前6个联合事务均接受新订单，未覆盖拒单。新增独占目录joint_policy_rejection_root_v1只检查旧记录第13项、订单3的真实拒绝路径：own_incumbent和strong_rolling两个路线模式均成功发布正确拒单状态。27运行项及76独立项0失败，两次2.471/2.380秒；未重复旧14故障，也未生成新科学日或在线收益。

## 登记及选择边界

原件位于根research/review_3080_20261010/pickup/joint_policy_rejection_root_v1。选择明确是工程覆盖中的结果筛选，不是科学随机样本、效用挑例或确认。旧cond_hist轨迹的拒绝记录0/6/7/13在当前own_uncond_dynamic直接决策中前三个接受、13拒绝；69也作接口诊断并接受。选择13以补验真实拒单，所有选择信息及旧输入SHA登记保存，不能拿这项样本估计收益。

上述不同结果来自不同接拒策略：旧complete_history_worker明确config name=cond_hist，新联合事务明确name=own_uncond_dynamic；来源及SHA记录在SOURCE_POLICY_DISTINCTION.json。同公开请求不等同策略，差异不支持工程bug、策略增益或历史决策逐项复现。两个路线模式共用own_uncond_dynamic接拒，strong_rolling在此仅提议路线，不等完整独立强滚动接拒臂。原先154项独立资格及原结果保持原字节。

## 实际路径

沿用字节不变的joint_install组件，实际Python3.12.7/NumPy1.26.4、CPU7；重新投影旧公开状态、实际接拒、路线提议、共用原200历史K10完整分支评分、全C0认证、编译、父guard与单指针发布。订单3没有进入accepted或任何计划路线，进入rejected；此前accepted集合与预算保持，承诺不丢失，完整认证通过。两模式均未改变路线。父提交禁止再次全C0认证的原保护保持。

独立程序从旧FULLRAW重新取请求，重新运行当前实际接拒、共同原200日场景和两个候选各10完整分支、择优规则及C0；独立核对拒单、accepted/预算、路线排除、完整编译状态、源前后字节和本人worker已回收。工件及注册绑定精确一致，无容差放宽。

此输入已预先公开，不含真正现场公开提取或可中止父序列化。27/76只关闭这一拒单联合提交边界，不等全五部署臂、已训同初始化CE、真实强滚动接拒、深层阶段延迟、整日或全请求10秒资格。旧14故障只证明外层取消/回收，原解释不升级。本阶段没有学习提议或任何收益测量；稳定MaskCO核心在线增益仍未建立。

下一步接入现场公开提取与可中止传输，完善正式学习控制与完整事件日志，再验收已观察整日工程路径，随后在新鲜研究日前另登记科学协议。物理参数仍为文献支持的仿真参数，S3禁用、最终种子禁用和旧结果保持。共享MD仅本地维护。
''')
    boundary={'created_utc':datetime.now(timezone.utc).isoformat(),'delivery_corrected_optimizer_audit_sha256':sha(audit/'downloaded/INDEPENDENT_OPTIMIZER_ADJUDICATION.json'),'pickup_rejection_audit_sha256':sha(reject/'INDEPENDENT_REJECTION_TRANSACTION_REPLAY.json'),'delivery_original_failure_preserved':True,'engineering_only':True,'new_science_family':False,'scientific_training_qualified':False,'formal_pickup_five_arms_qualified':False,'stable_MaskCO_core_online_gain':False,'necessary_subset_31_not_full_reproduction':True,'historical_183_not_modified':True,'old_numeric_court_not_changed':True,'public_release':False}
    create(R/'reproduction/OPTIMIZER_REJECTION_EVIDENCE_BOUNDARY_20261011.json',boundary)
    entries=[E/f['path'] for f in read(R/'current_entries_before_multistep_full_gradient_20261011/UPDATE_IDENTITY.json')['files']]
    backup=R/'current_entries_before_optimizer_and_rejection_20261011';backup.mkdir(exist_ok=False)
    intro='\n> 2026-10-11最新：配送三臂旧8状态各2步的真实更新与保存重载，修正次正规数核验假设后新增22,769独立项0失败；原45项中2FAIL及退出1保持。取货新增真实拒单联合提交27运行/76独立项0失败。仅工程资格，稳定可归因MaskCO在线增益仍未建立；负结果31必要门/183封存沿用。\n'
    doc=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
    files=[]
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat()
            s['delivery'].update(multistep_optimizer_engineering_qualified=True,multistep_optimizer_directory=str(training.relative_to(ROOT)),multistep_optimizer_original_checks=45,multistep_optimizer_original_failures=2,multistep_optimizer_original_exit=1,multistep_optimizer_original_failure_preserved=True,multistep_optimizer_fixed_steps_per_arm=2,multistep_optimizer_arms=['pretrained_CE','pretrained_paired','random_task_paired'],multistep_optimizer_source_files=8886,multistep_optimizer_independent_source_files=8885,multistep_optimizer_independent_checks=22769,multistep_optimizer_independent_failures=0,multistep_optimizer_independent_directory=str(audit.relative_to(ROOT)),multistep_optimizer_independent_identity=sha(audit/'downloaded/INDEPENDENT_OPTIMIZER_ADJUDICATION.json'),arithmetic_diagnosis_directory=str(diagnostic.relative_to(ROOT)),arithmetic_diagnosis_identity=sha(diagnostic/'downloaded/NUMERICAL_DIAGNOSIS.json'),engineering_weights_research_eligible=False,scientific_training_closure_qualified=False,scientific_random_control_qualified=False,new_science_family_registered=False,MaskCO_core_online_gain=False,next='seal actual server multistep teacher and deployment/learning distribution barriers; preregister finite new scientific family before fresh data; retain same-init CE/task-trained random controls; old DEV16 and engineering models ineligible')
            s['pickup'].update(joint_transaction_new_reject_path_qualified=True,joint_transaction_rejection_directory=str(reject.relative_to(ROOT)),joint_transaction_rejection_checks=27,joint_transaction_rejection_independent_checks=76,joint_transaction_rejection_failures=0,joint_transaction_rejection_identity=sha(reject/'INDEPENDENT_REJECTION_TRANSACTION_REPLAY.json'),joint_transaction_rejection_is_old_outcome_selected_engineering=True,joint_transaction_same_input_not_same_admission_policy=True,full_live_request_qualified=False,stable_online_gain=False)
            s['reproduction']['new_engineering_evidence_boundary_receipt']=str((R/'reproduction/OPTIMIZER_REJECTION_EVIDENCE_BOUNDARY_20261011.json').relative_to(ROOT))
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=intro+old if f in (doc,E/'项目当前状态.md',E/'00_项目导航.md') else old+intro
            if f==doc:new+='\n\n## 2026-10-11新增：三臂固定更新及拒单边界\n\n原配送两项核验FAIL是实测CPU次正规归零与NumPy假设差异；14个元素/臂，明确FTZ算术后全部6步、三份实际最终权重及重载目标经新进程22,769项精确复算通过。原失败和权重不修改，无训练重跑，无容差放宽。工程同初始化CE/同任务受训随机控制路径已通，不等科学资格；完整教师/部署新族仍需数据前登记。取货旧第13请求在两路线模式真实拒单，27运行/76独立项通过，组件字节不变；仅补拒单边界，不等现场完整10秒或强动态接拒臂。负结果追加证据边界记录，31必要门和183原包不重复整理。\n'
            if f.parent==E/'02_配送线':new+='\n- [三臂固定更新、保存重载与浮点核验更正](文档/结果与进度/'+dr.name+')\n'
            if f.parent==E/'01_取货线':new+='\n- [真实拒单联合提交边界独立验收](文档/结果与进度/'+pr.name+')\n'
        f.write_text(new,encoding='utf-8');files.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    create(backup/'UPDATE_IDENTITY.json',{'files':files,'new_reports':[str(dr.relative_to(E)),str(pr.relative_to(E))],'shared_MD_uploaded':False,'old_science_modified':False,'old_failure_modified':False})
    for f in files:assert sha(E/f['path'])==f['after_sha256'] and sha(backup/f['path'])==f['before_sha256']
    print(json.dumps({'entries':len(files),'independent_optimizer_checks':22769,'independent_rejection_checks':76,'failed':0,'science_qualified':False}))
if __name__=='__main__':main()
