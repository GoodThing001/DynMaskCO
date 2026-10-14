"""Append the independently accepted DEV16 outcome to mutable local entries."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=R/'delivery/remote_fresh_development_root_v1/downloaded/INDEPENDENT_DEVELOPMENT_ADJUDICATION.json'
    a=json.loads(path.read_text());assert a['failed']==0 and a['independent_acceptance'] and a['all96_final_hard']
    backup=R/'current_entries_before_independent_dev16_adjudication_20261011';backup.mkdir(exist_ok=False)
    entries=[DOC,E/'项目当前状态.md',E/'00_项目导航.md',E/'docs/当前规划/结果与进度/执行进度表.md',E/'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries +=[E/line/name for line in ('01_取货线','02_配送线','03_负结果线') for name in ('README.md','实验结果索引.md','文档索引.md')]
    entries +=[E/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    header='> **2026-10-11 最新独立开发裁决**：配送固定128步后的16新快照/六臂96路线独立10,742项通过，全部hard、零超时；偏好相对随机改善16.620%，相对非学习/GLS/纯重建为−16.502%/−38.549%/−10.618%，尚无稳定MaskCO收益。三组40确认未启动；当前先做保留原件的机制诊断，不挑权重/改旧门。取货继续实际候选共同K10与全链硬截止资格；相同计划43项仅为接口工程通过。负结果26项子集/183封存保持。30分钟任务maskco ACTIVE，共享MD仅本地。\n\n'
    section='''

## 2026-10-11：配送16新开发快照独立终局裁决

原数据前科学/运行注册、128固定最终权重和父子CPU15/总10秒资格保持。唯一DEV16在400.969秒退出0，16唯一实例、六臂96最终路线和所有640重构候选完整保留（五个重构臂×16×8）；GLS独立搜索，未虚计固定8重构候选。根审计从种子独立再生实例/掩码/顺序，逐候选与最终路线重放manifest、交付、距离、能耗、品质和全C0成本，并核全部实际开始/结束源码、权重、合同和预算身份，10,742项0失败，全部96最终路线hard，无臂超时/删例。

| 臂 | 16实例平均完整成本 |
|---|---:|
| 偏好训练 paired_objective | 95.675348 |
| 纯重建训练 reconstruction_control | 86.775484 |
| 原预训练 pretrained | 96.580954 |
| 随机未训练 random | 115.396289 |
| 非学习固定8次 nonlearning | 82.564835 |
| GLS | 69.139971 |

使用预登记seed202610111106、100,000次整实例配对bootstrap，逐实例相对改善定义为(control−paired)/control。以下95%区间仅为开发描述性区间，不能冒充三组40实例的多重比较校正确认门：

| 偏好相对对照 | 平均相对改善 | 描述性95%区间 |
|---|---:|---:|
| 随机未训练 | +16.620% | [12.788%,20.003%] |
| 非学习固定8次 | −16.502% | [−20.774%,−12.225%] |
| GLS | −38.549% | [−42.253%,−35.268%] |
| 纯重建训练 | −10.618% | [−14.304%,−6.945%] |

未建立偏好目标的部署改善，也未建立稳定MaskCO快照或核心动态在线增益。胜未训练随机不能替代胜非学习/GLS或同初始化重建控制，不能隔离预训练。当前不启动三组40确认，先保留原件完成部署/目标错位机制诊断；这是观察到开发劣势后的投入决策，不称预先注册的数值停止规则被触发，不重写科学JSON，不另选checkpoint、参数或对照来美化本批。后续若新科学假设需独立数据前登记；旧最终留出20260927/28始终禁用。

原件下载出现大文件串行读取延迟，科学实验已先结束。核实并只停止两个自有本地只读下载进程，保留133已完成文件；新有界SFTP分块读取补齐112,812,973字节组合原件，结束源核0差异。两份恢复证据和读取回执留在原目录；没有重跑任何科学实例，也未停止服务器他人任务。组合原件与各原始记录逐项一致由独立根审计验证。共享MD未上传。

资格路径：`research/review_3080_20261010/delivery/remote_fresh_development_root_v1/downloaded/INDEPENDENT_DEVELOPMENT_ADJUDICATION.json`；其`stable_snapshot_gain`及`MaskCO_core_online_gain`均false。取货相同计划评分器43项通过的范围与实际五臂/原子截止缺口保持，负结果原183封存和C1联合失败不改。
'''
    manifest=[]
    for f in entries:
        prev=backup/f.relative_to(E);prev.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,prev)
        old=f.read_text(encoding='utf-8')
        if f==DOC:new=old+section
        elif f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat()
            s['delivery'].update(stage='fixed128_dev16_independently_qualified_no_attributable_gain',fresh_development_complete=True,
                development_independent_checks=a['checks_count'],development_independent_failed=0,development_independent_sha256=sha(path),
                development_mean_costs=a['mean_costs'],development_paired_descriptive=a['paired_descriptive'],development_timeouts=a['timeouts_all_counted'],
                confirmation_execution_enabled=False,confirmation_adjudicated=False,stable_snapshot_gain=False,MaskCO_core_online_gain=False,
                next='preserve all96 final outcomes and candidate logs; diagnose objective/deployment mismatch; no checkpoint selection or false registered stop-rule claim')
            s['delivery'].pop('fresh_development_supervisor_pid',None)
            new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            if old.startswith('> **2026-10-11 当前有效更新**'):old=old.split('\n\n',1)[1]
            new=header+old
        f.write_text(new,encoding='utf-8');manifest.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(prev),'after_sha256':sha(f)})
    (backup/'UPDATE_IDENTITY.json').write_text(json.dumps({'files':manifest,'shared_MD_uploaded':False,'frozen_science_modified':False},indent=2))
    print(json.dumps({'local_entries_updated':len(manifest),'delivery_development_failed':0,'stable_online_gain':False}))
if __name__=='__main__':main()
