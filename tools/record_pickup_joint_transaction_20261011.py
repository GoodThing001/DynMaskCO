"""Local evidence closure and backed-up current entry maintenance."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
OUT=R/'pickup/joint_policy_transaction_root_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    a=json.loads((OUT/'INDEPENDENT_JOINT_TRANSACTION_REPLAY.json').read_text(encoding='utf-8'))
    assert a['failed']==0 and a['checks_count']==154 and len(a['summaries'])==6
    assert all(x['accepted'] and not x['route_changed'] for x in a['summaries'])
    reg=json.loads((OUT/'REGISTRATION.json').read_text(encoding='utf-8'))
    write(OUT/'BASE_AND_SCOPE_CLARIFICATION.json',{'registration_sha256':sha(OUT/'REGISTRATION.json'),
        'implemented_admission_base':'own_uncond_dynamic for both route proposal modes; mode names identify route proposers, not independent admission policies',
        'actual_six_admission_outcomes':'all accepted; reject-new-order publication path not exercised',
        'actual_six_route_outcomes':'no alternative route selected; no realized gain claim',
        'delayed_target_stage_reached_qualified':False,'outer_deadline_reaping_qualified':True,
        'source_pin_defines_fixed_configuration':reg['source_sha256'],'parent_serialization_interruptible':False,
        'full_live_10s_qualified':False,'formal_five_arms_qualified':False,'stable_online_gain':False})
    report=E/'01_取货线/文档/结果与进度/取货_接拒与计划联合提交工程独立验收_2026-10-11.md'
    text='''# 取货接拒与计划联合提交工程独立验收（2026-10-11）

新目录 `research/review_3080_20261010/pickup/joint_policy_transaction_root_v3` 完成六次已观察请求的接拒、提议、共同SAA、完整认证和实际C0状态联合提交。65运行项、154独立复算项均0失败。此阶段关闭“接拒状态与计划分别更新”的组件衔接缺口；没有完成正式五臂、完整现场请求或新科学日，也没有建立稳定、可归因MaskCO核心在线增益。

## 预声明范围与实际行为

输入仍是完整账本工程回放中的旧80日之一，顺序选择记录0、1以及首个至少两项可改订单的记录9。全部请求已经观察；没有新开发日、最终留出日、采集、训练或权重选择。原200日历史池SHA为 `44064ff788b7e322294a23e6ac6de39da9f399dec326b70af7a0f7439895d3c6`，未修改。REGISTRATION在执行前写入，绑定原件、输入、程序、安装模块、CPU7、NumPy1.26.4、K10及场景种子202610111301。

两个模式为 own_incumbent 和 strong_rolling 的路线提议；两者接拒都使用实际 own_uncond_dynamic 基础策略。不能把模式名字当作两种独立接拒策略或正式强滚动臂。每次重投影原公开包，实际执行新订单接拒，再比较原计划与路线提议。共同未来只抽样一次，两候选各保留10个完整分支；认证采用唯一C0，完整制冷/品质/返回账本沿用已合格实现。

实际订单46、20、61全部被接受，两模式六次均未选用替代路线。这只验证实际接拒新增接受、原承诺保留及拒绝集合其他成员保留；“本次新订单拒绝”的发布路径尚未实际覆盖，不能声称完整接拒边界已经运行合格。未选替代路线也不能证明全局没有策略空间。

安装模块在真实 C0ReserveReplanner 子类的属性接口后放置一个 PolicyState 指针，预算、模式、已接、已拒、计划和实例身份一次发布；worker先完整认证、编译，父进程核请求、输入、状态、源身份及deadline后单指针发布。父提交阶段禁止再次调用完整物理认证，运行检查通过。现场状态变更、准备状态篡改、错误请求、先前承诺删除、过期和重放等8项不得安装；6项延迟请求回收worker并保持旧指针，合计14故障原件保留。

六次整体实测为2.874–3.438秒，从父请求起点计入请求传输、worker重新投影、接拒、提议、共同评分、认证、编译及父联合提交，共用一个10秒绝对截止时间。预热/worker建立不在请求内。传入包已经公开；真正现场的私有环境提取与公开投影、父序列化的可中止性、全日事件闭环尚未接通。故本阶段**不是**整现场请求10秒硬预算资格。Python进程调度也不是操作系统硬实时保证。

6个延迟标签不能证明延迟到达了对应深层阶段：10毫秒外部截止可能在投影或传输时已到期。这些原件只支持外层取消、回收和无迟到发布；逐阶段精确到达的延迟故障还待另补。PolicyState包含可变集合/字典，运行期间的原地变更由guard拒绝，未声称对象深不可变或支持并发发布。

## 独立复核与两个保留失败

独立进程重新读取实际源/解释器/NumPy、原输入和历史池；恢复原快照，复算实际接拒、共同场景身份、两候选120个完整分支和算术均值、选择规则及完整C0证书，均精确一致。核实原物理快照、重规划计数和揭示位置保持，联合导出与编译状态一致，所有本人worker回收。154项0失败。这里的独立性指另一验收程序从工件重新计算，仍共享物理评估器、采样器和基础接拒实现，不是另一套独立物理模型。

v1：运行65项未完成，实际记录为11项/3失败、0正式请求完成。接拒更新后派生快照沿用更新前的state_hash，完整性恢复正确拒绝。v2在新目录为派生状态重新计算哈希，65运行项0失败，但154独立项中6项失败：新建环境把历史replan_count从1/2/13变为0。接拒/评分复算一致不能替代身份合格。v3恢复真实计数和reveal_idx，并继续检查它们；65/154全部通过。三目录、失败结果、注册和程序均保留；joint_install.py在三次恢复中字节相同。未放宽身份或算术容差，原公开包及历史结果未改写。

## 文献依据与工程选择的边界

Ulmer、Goodson、Mattfeld、Thomas的 *On Modeling Stochastic Dynamic Vehicle Routing Problems*（2020，EURO Journal on Transportation and Logistics 9，100008）§2、§5将动态路由状态与计划共同建模，用计划更新描述决策。本文据此将接拒承诺和可执行计划的一致性作为实现要求；单指针提交是本项目的工程选择，论文没有规定这种Python实现，也没有证明本项目策略收益。来源：[作者全文](https://justin-goodson.com/papers/RouteBasedMDPs.pdf)、[出版社](https://www.sciencedirect.com/science/article/pii/S219243762030008X)。

共用单调截止时间采用实际Python3.12 perf_counter，Windows3.10起跨进程时钟一致；取消后显式kill/wait回收worker。依据：[Python time官方文档](https://docs.python.org/3.12/library/time.html)、[subprocess官方文档](https://docs.python.org/3.12/library/subprocess.html)。这些文档支持时钟/进程接口语义，不支持“所有父代码均可抢占”或“10秒最优”。K10、10秒和两模式是既定工程范围，不伪装成文献最优参数。

## 下一屏障及三线状态

取货还须接通真正现场公开提取、可中止传输、实际学习提议、同初始化已训CE控制及完整新订单拒绝/全日事件日志；合格后才另登记新鲜研究日。旧184原型和82计划安装组件不能替代这些屏障，本次只验新增联合事务，不重复它们。

配送7316文件/7334运行库只读预检、9670多步教师工程资格沿用；完整服务器模型梯度、匹配多步教师训练、同初始化和同任务随机训练控制仍待，旧DEV16及3×40未启动状态不改。负结果31必要门/183历史封存沿用，无新证据需要重复整理。共享MD只在本地维护；未上传、公开或投稿。
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    prior=R/'current_entries_before_training_runtime_preflight_20261011/UPDATE_IDENTITY.json'
    entries=[E/x['path'] for x in json.loads(prior.read_text(encoding='utf-8'))['files']]
    backup=R/'current_entries_before_joint_policy_transaction_20261011';backup.mkdir(exist_ok=False)
    short='\n> 2026-10-11新增：取货旧状态接拒与计划联合事务六请求、65运行/154独立项通过；两处状态身份缺陷修复且失败保留。六次全部接受、未换路线；真实现场全请求/正式五臂仍未合格，无在线正向。配送7334预检与9670教师工程、负结果31/183沿用。\n'
    doc=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
    files=[]
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            v=json.loads(old);v['updated']=datetime.now(timezone.utc).isoformat()
            v['pickup'].update(observed_joint_policy_transaction_qualified=True,joint_transaction_directory=str(OUT.relative_to(ROOT)),
                joint_transaction_independent_checks=154,joint_transaction_independent_failures=0,joint_transaction_identity=sha(OUT/'INDEPENDENT_JOINT_TRANSACTION_REPLAY.json'),
                joint_transaction_normal_requests=6,joint_transaction_new_reject_path_qualified=False,joint_transaction_targeted_stage_delay_qualified=False,
                full_live_request_qualified=False,all_five_formal_deployment_qualified=False,stable_online_gain=False,
                next='integrate actual live public extraction and cancellable transport; formal learned proposals and same-init trained CE; complete rejection/event/full-day engineering; then preregister new finite scientific days')
            new=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=short+old if f in (doc,E/'项目当前状态.md',E/'00_项目导航.md') else old+short
            if f==doc:new+='\n\n## 2026-10-11新增：已观察请求接拒与计划联合提交\n\n65运行/154独立项0失败；v1派生快照旧哈希故障、v2重规划计数重置的6身份失败原件均保留，v3恢复原计数并保持身份检查。六请求全部接受/未换路线，不能当完整拒绝边界或在线增益。六个延迟标签只支持外层回收，不能称逐阶段到达资格。现场提取、父可中止传输、正式五臂、已训同初始化控制和全日仍待。详见`01_取货线/文档/结果与进度/'+report.name+'`。\n'
            if f.parent==E/'01_取货线':new+='\n- [接拒与计划联合提交工程独立验收](文档/结果与进度/'+report.name+')\n'
        f.write_text(new,encoding='utf-8');files.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    write(backup/'UPDATE_IDENTITY.json',{'files':files,'new_report':str(report.relative_to(E)),'old_science_modified':False,'shared_MD_uploaded':False})
    print(json.dumps({'entries':len(files),'joint_independent_checks':154,'full_live_request_qualified':False,'stable_online_gain':False}))
if __name__=='__main__':main()
