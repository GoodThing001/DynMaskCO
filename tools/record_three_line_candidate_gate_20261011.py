"""Back up current entries, then append qualified scopes without changing science."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'; R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pp=R/'pickup/actual_common_candidate_root_v2/INDEPENDENT_ROOT_ACTUAL_CANDIDATE_QA.json'
    dp=R/'delivery/dev16_mechanism_diagnosis_root_v1/DIAGNOSIS.json'
    bp=R/'reproduction/formal_run_requirements_root_v4/ROOT_ACCEPTANCE.json'
    p,d,b=[json.loads(x.read_text()) for x in (pp,dp,bp)]
    assert p['checks_count']==184 and p['failed']==d['failed']==0 and d['checks_count']==270 and b['exit_code']==0 and b['parent_unchanged']
    assert 'Ran 31 tests' in (bp.parent/'TEST_RUN.log').read_text()
    entries=[DOC,E/'项目当前状态.md',E/'00_项目导航.md',E/'docs/当前规划/结果与进度/执行进度表.md',E/'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries +=[E/line/name for line in ('01_取货线','02_配送线','03_负结果线') for name in ('README.md','实验结果索引.md','文档索引.md')]
    entries +=[E/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup=R/'current_entries_before_actual_candidate_and_gate_hardening_20261011';backup.mkdir(exist_ok=False)
    header='> **2026-10-11 当前实际候选与审计资格**：取货五种真实原型共同K10/完整C0认证184项通过，预训练与随机在该状态给出同一改动，不能归因学习；父投影/现场安装及正式五臂仍未合格。配送DEV16原负结果保持，270项同队列机制复算表明对纯重建劣势97.884%来自距离；不启动确认、另做保留边教师工程。负结果当前v4必要门31项通过，原183文件与旧门保留。稳定MaskCO在线增益尚未建立，maskco ACTIVE，共享MD仅本地。\n\n'
    section='''

## 2026-10-11：实际五候选、机制分解与后续协议门加固

### 取货：实际候选与共同未来评分已经接通，正式部署尚未完成

`pickup/actual_common_candidate_root_v1` 的固定旧状态包含两条尾部订单，但其中一条是不可改的已承诺目的地，实际可改订单只有1个。它足够做相同计划接口测试，不够核验两单联合模型。v1原始输出保留；这不是策略硬失败或收益负结果。新v2按记录顺序选第一个至少2个可改订单的已观察公共状态，排除承诺目的地，不按效用挑状态；域外模型返回自己的原计划并明确零模型调用。

五种原型为原预训练完整3D模型、同架构随机未训练、同预训练推理后丢弃分数、真实rolling固定接单集合重排、自己的原计划。三模型234叶真实参数身份核验，所有静态3..13节点合成预热在请求前完成；实际候选、同原200日历史抽得的10个未来、每候选完整10分支、已知状态全C0认证、身份核验及传输纳入请求10秒。五臂共同场景哈希及原计划10个值逐项一致，独立184项0失败，正常请求4.416–5.048秒，五个10毫秒延迟故障只返回原计划且worker回收；另三模型域外边界核验不虚报模型激活。

预训练和随机候选在这一个工程状态完全相同，均被共同SAA接受，影子效用差+6.176827；丢弃分数和rolling未改变原计划。此值是可能未来的影子评分，不是真实完整日收益，不能写学习归因或稳定正向。域外状态没有变好不代表全局无空间。`INDEPENDENT_ROOT_ACTUAL_CANDIDATE_QA.json` 明确：实际原型候选评分合格，正式五臂部署、同初始化已训练纯重建控制、父公共投影/现场安装原子硬截止、真实在线效用均未合格/未测；没有生成新研究日。

下一屏障是将实际完整接/拒分支、公共投影、候选、认证和安装接入同一个可中止事务，再核完整工程日；之后另登记新有限在线日。不能把原3D/随机/丢弃原型标签冒称任务训练的正式归因臂，也不能把此前43项同标签测试写成这一184项实际候选资格。

### 配送：全16实例同队列机制分解，后续教师须尊重部署动作

`delivery/dev16_mechanism_diagnosis_root_v1/DIAGNOSIS.json` 从已经完整裁决的16实例/96最终路线和640重构候选复算270项0失败，没有新增数值实验。偏好相对同初始化纯重建的平均成本劣势8.899865中，距离为8.711549（97.884%），加权能耗0.183897、品质0.004418；16例输15例。相对非学习的劣势98.774%、相对GLS的劣势98.679%来自距离，分别输15/16、16/16。该面板六臂零超时，不能用预算失效解释此批差距。

第一步共享初始路线与掩码时，偏好仍劣于纯重建；之后各臂状态已分叉，不将后续逐步均值作同动作因果比较。完整邻接偏好代理与实际受约束插入动作可能错位，初始路线4掩码训练与8步不断改变的部署状态可能错位；两者为待验证假设，不宣称已经定位唯一根因。相同逐行边数时行softmax归一项会抵消，不能仅凭raw logit与log-softmax不同认定数学错误。

[Chen与Tian的NeurIPS 2019局部重写论文§4.2](https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf)将奖励定义为实际任务成本变化，支持检验动作对齐目标，但不保证本项目会正向。[Li、Yan与Wu的NeurIPS 2021委派论文](https://papers.neurips.cc/paper_files/paper/2021/file/dc9fa5f217a1e57b8a6adeb065560b38-Paper.pdf)把学习放在固定求解器的子问题选择上，是另一职责，需要另行归因，不能自动转作MaskCO重构证据。

当前先做已观察工程快照上的保留边强教师功能验收。无约束GLS任意换边不够作同一部分邻接输入的教师；需明确保留边方向、depot对应及可行性认证。新的科学族须在新数据前独立登记。旧128最终权重、DEV16负裁决和所有候选保留，不挑checkpoint、改旧门或重复同种子数值尝试。

### 负结果：必要门v4加固，失败版本也保留

旧v2适配器缺每线最多12族上限检查；部分原始结果、事件/调用/运行证明只在初读时验哈希，未在返回前再次核验；角色标签还能复用同一控制臂或在一族内切换处理模型。新隔离v4以PROGRAM实际12族上限检查、所有读入工件末尾重新核哈希、不同对照角色不复用以及所有比较同一处理臂补齐必要条件。31项实测0失败，包括族13、双角色同臂、双处理臂、读后原始记录/事件被改等拒绝测试；SHA `b52a75c2bdfc2086ee3da0875ade71b4ff54e54607c83a1e380a4d78efb6139c`。v2字节不变，作为历史资格保留，当前新增必要门为v4。

中间v3生成器自身递归错误且60秒外壳超时，源码、注册与故障回执保留；v4只做隔离工程恢复，未改科学结果。31项只是必要协议/原件完整性子集，不证明历史训练源码补齐、完整复现、所有运行语义资格或方法收益。历史183文件、C1联合失败和CI跨零非等价继续有效，无公开发布。

资源实查10:28两块5090分别81%/94%利用率，旧空闲叙述不作为本轮依据；未抢占或停止团队任务。新本地候选测试CPU7且各worker回收。所有以上共享MD只写本地，30分钟安静推进继续ACTIVE。
'''
    changes=[]
    for f in entries:
        old=f.read_text(encoding='utf-8');before=backup/f.relative_to(E);before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,before)
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat()
            ds=s['delivery']; legacy={k:ds.pop(k) for k in ('instances','candidate_replays','final_replays','audit_checks','actual_end_items','actual_end_mismatches','mean_costs','paired','actual_decode_calls','timeouts') if k in ds}
            ds['historical_destroy_eq8_panel']=legacy
            if 'training_live_observation' in ds:ds['historical_training_live_observation']=ds.pop('training_live_observation')
            ds.update(development_instances=16,development_final_routes=96,development_candidate_replays=640,mechanism_checks=270,mechanism_failures=0,mechanism_identity=sha(dp),mechanism_gaps=d['matched_cohort_gaps'],next='qualified retained-edge strong-teacher engineering on observed fixtures before separate new data registration; no DEV16 rerun')
            s['pickup'].update(stage='actual_five_prototype_common_K10_qualified_deployment_gaps_open',actual_candidate_checks=184,actual_candidate_failures=0,actual_prototype_candidate_scoring_qualified=True,actual_candidate_summaries=p['summaries'],actual_candidate_identity=sha(pp),all_five_formal_deployment_qualified=False,same_init_trained_reconstruction_control_present=False,atomic_all_in_budget_qualified=False,realized_online_utility_gain_measured=False,next='whole admission/projection/scoring/certification/live commit hard transaction; no fresh online days before qualification')
            s['reproduction'].update(stage='current_protocol_bound_necessary_subset_root_v4_verified',latest_necessary_subset_tests=31,latest_necessary_subset_failures=0,previous_adapter_sha256=s['reproduction']['final_adapter_sha256'],final_adapter_sha256=b['new_adapter_sha256'],current_adapter_directory='research/review_3080_20261010/reproduction/formal_run_requirements_root_v4',v3_generator_failure_preserved=True)
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            if old.startswith('> **2026-10-11 最新独立开发裁决**'):old=old.split('\n\n',1)[1]
            new=header+old+(section if f==DOC else '')
        f.write_text(new,encoding='utf-8');changes.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(before),'after_sha256':sha(f)})
    (backup/'UPDATE_IDENTITY.json').write_text(json.dumps({'files':changes,'shared_MD_uploaded':False,'frozen_science_modified':False},indent=2))
    print(json.dumps({'local_entries_updated':len(changes),'pickup_checks':184,'mechanism_checks':270,'future_gate_tests':31,'stable_online_gain':False}))
if __name__=='__main__':main()
