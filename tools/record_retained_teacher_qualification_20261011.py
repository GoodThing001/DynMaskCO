"""Append retained-edge teacher engineering and preserve prior current entries."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    fp=R/'delivery/retained_edge_teacher_engineering_root_v2/INDEPENDENT_RETAINED_TEACHER_REPLAY.json';d=json.loads(fp.read_text());assert d['checks_count']==1968 and d['failed']==0
    backup=R/'current_entries_before_retained_teacher_engineering_20261011';backup.mkdir(exist_ok=False)
    entries=[DOC,E/'项目当前状态.md',E/'00_项目导航.md',E/'docs/当前规划/结果与进度/执行进度表.md',E/'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries +=[E/line/name for line in ('01_取货线','02_配送线','03_负结果线') for name in ('README.md','实验结果索引.md','文档索引.md')]
    entries +=[E/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    header='> **2026-10-11 当前三线资格**：取货实际五原型共同K10/完整C0184项通过，但同状态预训与随机改动相同、正式部署/父投影安装未合格。配送DEV16负结果保持，270项机制分解完成；新保留边教师仅两旧快照工程，274候选/独立1968项通过，不计研究收益或新科学族。负结果当前v4必要门31项通过、旧183封存保持。稳定MaskCO在线增益尚未建立，maskco ACTIVE，共享MD仅本地。\n\n'
    section='''

## 2026-10-11：保留边强教师功能独立验收

依据[OR-Tools官方初始路线与搜索限时接口](https://developers.google.com/optimization/routing/routing_tasks)，隔离实现`delivery/retained_edge_teacher_engineering_root_v2/retained_gls.py`。使用旧已观察seed的实例900000/900001、原mask ordinal0，以及全部边保留边界。没有新训练、开发、确认数据，也不重复旧收益实验。

约束完整复刻模型部分邻接的无向语义：customer/customer边允许任一方向，depot/customer边允许任一车辆的起点或终点；不暗中把无向边锁成有向边或原车辆。所有订单必须服务一次，求解器容量/时间整数保守约束，原始路线作为种子，无法接收原可行种子时显式失败且不筛例。GLS以距离提议，各次已找到解都按完整C0距离/能耗/品质认证，完整成本选择且不比自己的初始路线差；这是非学习教师，不是MaskCO方法收益或最优解证明。

父/worker实际CPU7，4个工程请求各3.028–3.267秒，搜索上限3秒、请求上限10秒。保留274个候选及全部物理报告，运行检查579项0失败，独立逐路线、每个候选四项成本/全部保留边及顺序择优复算1968项0失败。两个全部保留边测试没有改进，保留此边界结果；两个部分掩码工程数字只用于功能验收，不作可推广对照收益。

v1进入时本地OR-Tools环境缺可选psutil，注册前退出1，没有开始求解器或新数据。原模块/入口源码和故障回执保留；新v2使用Windows原生CPU亲和性设定与实际读取，在新目录同参数恢复。未安装或改动全局依赖，未抢占GPU或更改历史源码。

此模块已可用于下一阶段设计，但完整解释器/依赖开始结束封存、新部署状态教师生成、固定训练和等预算部署归因尚未合格。后续先补齐源闭包及实际插入动作与教师目标的对应，再在新数据前登记独立新假设族；无约束强搜索教师不能冒称相同动作空间的标签。新科学族目前未登记，原配送确认不自动启动。取货仍待完整接拒/公共投影/评分/现场安装硬事务；负结果只读31项结案，不重复整理或公开发布。

独立资格：`research/review_3080_20261010/delivery/retained_edge_teacher_engineering_root_v2/INDEPENDENT_RETAINED_TEACHER_REPLAY.json`。当前仍未建立稳定、可归因MaskCO在线增益。
'''
    manifest=[]
    for f in entries:
        before=backup/f.relative_to(E);before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,before);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['delivery'].update(retained_teacher_engineering_directory='research/review_3080_20261010/delivery/retained_edge_teacher_engineering_root_v2',retained_teacher_independent_checks=1968,retained_teacher_independent_failures=0,retained_teacher_candidate_replays=274,retained_teacher_identity=sha(fp),next_scientific_family_registered=False,next='seal full interpreter/dependency closure and insertion-action/teacher correspondence; then new pre-data scientific registration, no old DEV rerun');new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            if old.startswith('> **2026-10-11 当前实际候选与审计资格**'):old=old.split('\n\n',1)[1]
            new=header+old+(section if f==DOC else '')
        f.write_text(new,encoding='utf-8');manifest.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(before),'after_sha256':sha(f)})
    (backup/'UPDATE_IDENTITY.json').write_text(json.dumps({'files':manifest,'shared_MD_uploaded':False,'frozen_science_modified':False},indent=2))
    print(json.dumps({'updated_local_entries':len(manifest),'teacher_replay_checks':1968,'research_new_days':0,'stable_online_gain':False}))
if __name__=='__main__':main()
