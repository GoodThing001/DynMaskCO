"""Append accepted server boundary evidence; preserve failed runs and old science."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        if isinstance(x,str):f.write(x)
        else:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    prod=D/'server_teacher_boundary_root_v3';audit=D/'server_teacher_boundary_audit_root_v1'
    result=read(prod/'downloaded/RESULT.json');a=read(audit/'downloaded/INDEPENDENT_SERVER_TEACHER_BOUNDARY.json')
    assert result['failed']==a['failed']==0 and result['requests']==4 and result['candidates']==675
    assert result['checks_count']==1369 and a['actual_server_teacher_boundary_qualified']
    assert not a['scientific_training_qualified'] and not a['learning_distribution_qualified'] and not a['online_gain']
    assert read(prod/'downloaded/RUN_EXIT.json')['exit_code']==read(audit/'downloaded/RUN_EXIT.json')['exit_code']==0
    for name,h in read(audit/'AUDIT_REGISTRATION.json')['inputs'].items():
        p=prod/'downloaded'/name
        if not p.exists():p=prod/name
        assert sha(p)==h,name
    for base,prefix in ((prod,'SOURCE'),(audit,'AUDIT_SOURCE')):
        start=read(base/'downloaded'/(prefix+'_START.json'));end=read(base/'downloaded'/(prefix+'_END.json'))
        assert start['files']==end['files'] and not end['changed'] and not end['new_files']
    for version in (1,2):
        failed=D/('server_teacher_boundary_root_v'+str(version))
        assert read(failed/'downloaded/RUN_EXIT.json')['exit_code']==1
        raw=read(failed/'downloaded/RESULT.json')
        assert raw['failed']==1 and [c['name'] for c in raw['checks'] if not c['pass']]==['actual_environment_closed']
        assert sha(failed/'retained_gls.py')==sha(prod/'retained_gls.py')
        assert sha(failed/'REQUESTS.json')==sha(prod/'REQUESTS.json')
    assert sha(prod/'retained_gls.py')==sha(D/'multistep_insertable_teacher_engineering_root_v2/retained_gls.py')
    details=read(prod/'ENGINEERING_REGISTRATION.json')
    assert details['request_cap_s']==10 and details['teacher_search_cap_s']==3 and details['closure_initialization_before_start']
    report=E/'02_配送线/文档/结果与进度/配送_3080教师部署与空车槽边界独立验收_2026-10-11.md'
    text='''# 配送3080教师、非学习部署与空车槽边界独立验收（2026-10-11）

3080实际运行环境中四个固定工程请求已完成：1,369运行项及{audit_checks:,}独立项0失败，675个教师候选完整保留、逐个重算完整C0与动作约束。正常路径覆盖未掩码原车与顺序、保留边、空车槽及零掩码身份。此阶段关闭服务器上的教师/非学习部署边界与实际依赖封存，不等科学训练、学习轨迹覆盖、硬故障完整事务或动态在线收益。稳定、可归因MaskCO核心在线增益仍未建立。

## 固定范围与实际执行

根research/review_3080_20261010/delivery/server_teacher_boundary_root_v3为合格生产原件，server_teacher_boundary_audit_root_v1为独立验收；服务器独占目录省略delivery层。旧v1/v2失败原件、旧DEV16和原训练工件未覆盖。派发前核查实际资源与本人进程组，CPU15、屏蔽CUDA、有界监督及回收；本阶段没有使用5090或终止团队进程。

四个请求均在执行前固定并绑定哈希。第一项是已观察H轨迹的第二状态；其余三项从该旧状态的第一、第二车辆路线取订单并重编号，原路线分别放车槽1/3，槽0/2/4为空，采用奇数订单掩码、全掩码和零掩码。实际为18订单，加仓库共19节点，原两个路线长6和12。v1注册描述误写old20-order，实际输入、精确派生规则和全部运行始终为18；更正明确保存在v1/ENVIRONMENT_SEAL_DIAGNOSIS.json，新v3注册描述为18。没有更改旧注册字节或科学样本。

实际教师retained_gls.py与此前合格的本地多步教师逐字相同，生产reconstruct调用冻结的旧服务器部署源，均未修改。合同SHA ba40cd0d08fc21ef09d418c7df2bff940294ee2cfecebd2a4ba9ddde35a8268a、请求上限10秒、教师搜索上限3秒，初始路径及掩码保持。合同为预派遣快照、online_ready=False。三项派生边界是工程输入，不是新的研究日、效用筛选队列或科学族。

四请求实际耗时见原ANSWERS.jsonl与独立summaries；每项均在10秒内，教师与非学习部署最终全C0 hard/complete。全掩码允许重指派；部分掩码逐订单保持未掩码原车及严格顺序，并保持原保留边。返回始终保留五个车槽，空车编号不压缩；零掩码返回原计划。这次实际覆盖此前8个旧H状态未包含的空骨架场景，但不外推任意空车状态或学习策略分布。

## 封存失败与工程恢复

v1共1,371项中仅actual_environment_closed失败，退出1；没有源字节变化，但264个依赖文件未进入起始清单。根因来自封存函数顺序：先枚举sys.modules，随后导入JAX/Flax等包；首次调用加载的间接依赖没有在同一次起始枚举中被捕获。末尾清单包含absl、rich、toolz等此前未封存文件。全部文件与失败结果保持原字节，不能将这一轮写成通过。

第一次恢复v2的派发器误选旧teacher程序，因此再次出现相同封存失败。该错误由实际源哈希证明，记录在v2/DISPATCH_SOURCE_SELECTION_ERROR.json，原件保留。v3派发显式绑定修正程序，在不读取请求或生成候选前初始化封存所需包，然后取起始清单。teacher、请求、合同和预算均与原轮逐字一致。恢复只修正封存顺序和派发源选择，没有调参、选权重或效用重试。3秒限时搜索的候选数量会随执行不同，全部候选均保留，不要求工程恢复候选数量逐位重复。

v3实际8,793项源/字节码/解释器/依赖/native/输入开始结束一致，无新增依赖。实际NumPy2.2.6、OR-Tools9.15.6755、protobuf6.33.6；不套用此前本地NumPy2.5.3/OR-Tools9.11环境身份。总生产监督耗时33.918秒，其中包含导入与清单哈希，不能把总耗时当单个请求预算。

## 独立核查与证据边界

新进程在执行前绑定实际生产源、注册、请求、完整ANSWERS、结果和封存清单。独立重建18订单派生输入、原保留邻接、所有车槽/订单/未掩码原车与顺序，逐个重算675候选的完整C0、严格改善接纳和最终选择，再核实际非学习部署结果、零掩码身份及预算。原生产8,793文件在审计前后再次逐项读取实际字节，审计自身{audit_files:,}项文件也前后无差异。全部{audit_checks:,}项通过，退出0；不重跑求解器或教师，不重新生成候选或训练。

物理重算共享原C0评价器，不等另一物理实现；实际部署这里只运行非学习H路径，没有证明训练模型已学会教师，也未验证学习分布上的多步状态、五臂取货整日或真实强动态配送。正式新科学族仍须在新数据前登记，保留同初始化CE及同任务训练随机控制，并绑定实际教师/完整训练/预算部署闭包。此前仅两工程步的权重均research_model_eligible=False，不能用于确认；旧DEV16负裁决与3×40未启动保持。固定机会数加10秒上限仍不等等耗时ANY50。

取货76拒单联合提交及现场全请求缺口沿用；负结果仅追加本次工程边界记录，31必要门/183历史包/C1失败/CI跨零非等价保持，不重复核验。共享MD只在本地维护，没有服务器文档同步或公开发布。
'''.format(audit_checks=a['checks_count'],audit_files=a['source_files'])
    create(report,text)
    boundary={'created_utc':datetime.now(timezone.utc).isoformat(),'producer_result_sha256':sha(prod/'downloaded/RESULT.json'),'independent_adjudication_sha256':sha(audit/'downloaded/INDEPENDENT_SERVER_TEACHER_BOUNDARY.json'),'old_seal_failures_preserved':True,'engineering_only':True,'new_science_family':False,'scientific_training_qualified':False,'learning_distribution_qualified':False,'dynamic_online_qualified':False,'stable_MaskCO_core_online_gain':False,'historical_183_untouched':True,'necessary_subset_31_not_full_reproduction':True,'public_release':False}
    receipt=R/'reproduction/SERVER_TEACHER_BOUNDARY_EVIDENCE_20261011.json';create(receipt,boundary)
    manifest=read(R/'current_entries_before_optimizer_and_rejection_20261011/UPDATE_IDENTITY.json')
    backup=R/'current_entries_before_server_teacher_boundary_20261011';backup.mkdir(exist_ok=False)
    intro='\n> 2026-10-11最新：配送3080四个旧/派生请求、675候选，正常教师与非学习部署空车槽边界通过1,369运行/'+format(a['checks_count'],',')+'独立项；8,793实际文件封存闭合。v1封存顺序及v2误选旧检查器的失败原件保留。无科学训练/学习分布/动态在线正向；取货76拒单工程与负结果31/183沿用。\n'
    doc=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md';files=[]
    for entry in manifest['files']:
        f=E/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat()
            s['delivery'].update(server_teacher_boundary_qualified=True,server_teacher_boundary_directory=str(prod.relative_to(ROOT)),server_teacher_boundary_checks=result['checks_count'],server_teacher_boundary_candidates=result['candidates'],server_teacher_boundary_source_files=result['source_files'],server_teacher_boundary_independent_checks=a['checks_count'],server_teacher_boundary_independent_failures=0,server_teacher_boundary_independent_directory=str(audit.relative_to(ROOT)),server_teacher_boundary_independent_identity=sha(audit/'downloaded/INDEPENDENT_SERVER_TEACHER_BOUNDARY.json'),server_teacher_empty_slot_engineering_qualified=True,server_teacher_derived_orders=18,server_teacher_original_failures_preserved=True,scientific_training_closure_qualified=False,scientific_random_control_qualified=False,learning_distribution_qualified=False,new_science_family_registered=False,MaskCO_core_online_gain=False,next='preregister new finite multistep insertable scientific family before fresh data; bind qualified server teacher/full optimizer/runtime and actual budget deployment; retain same-init CE and task-trained random controls; close learning-distribution barrier, no old DEV16/confirm rerun')
            s['reproduction']['server_teacher_boundary_receipt']=str(receipt.relative_to(ROOT))
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=intro+old if f in (doc,E/'项目当前状态.md',E/'00_项目导航.md') else old+intro
            if f==doc:new+='\n\n## 2026-10-11新增：服务器教师与空车槽\n\n配送四个固定旧/派生请求及675候选的实际3080教师/非学习部署已独立核查；部分掩码锁车槽与顺序、全掩码及零掩码正常边界通过。原v1依赖封存顺序错误、v2误派旧源保留，v3预初始化闭包后8,793文件开始结束相同。仅工程资格，科学数据/学习轨迹和动态在线资格不升级。下一步在新数据前登记有限新族，绑定同初始化CE和同任务训练随机控制；旧DEV16与工程权重不重用确认。\n'
            if f.parent==E/'02_配送线':new+='\n- [3080教师部署与空车槽边界独立验收](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');files.append({'path':entry['path'],'before_sha256':sha(saved),'after_sha256':sha(f)})
    create(backup/'UPDATE_IDENTITY.json',{'files':files,'new_report':str(report.relative_to(E)),'shared_MD_uploaded':False,'old_science_modified':False,'old_failure_modified':False})
    for f in files:assert sha(E/f['path'])==f['after_sha256'] and sha(backup/f['path'])==f['before_sha256']
    print(json.dumps({'entries':len(files),'producer_checks':result['checks_count'],'independent_checks':a['checks_count'],'failed':0,'scientific_gain':False}))
if __name__=='__main__':main()
