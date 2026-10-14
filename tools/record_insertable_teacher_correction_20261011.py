"""Append the action-space correction; retain weaker teacher and all raw results."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=R/'delivery/insertable_teacher_engineering_root_v3/INDEPENDENT_RETAINED_TEACHER_REPLAY.json';a=json.loads(path.read_text());assert a['failed']==0 and a['checks_count']==2594 and a['all_targets_pass_structural_insertion_necessity']
    prev=R/'current_entries_before_retained_teacher_engineering_20261011/UPDATE_IDENTITY.json';entries=[E/x['path'] for x in json.loads(prev.read_text())['files']]
    backup=R/'current_entries_before_insertable_teacher_correction_20261011';backup.mkdir(exist_ok=False)
    header='> **2026-10-11 当前三线资格**：取货实际五原型共同K10/完整C0184项通过，但预训与随机同状态改动相同、正式部署/父投影安装未合格。配送DEV16负结果与270项分解保持；保留边教师v2超出部署动作的缺口已记录，新v3锁未掩码原车/顺序，316候选独立2594项通过，仅旧快照工程。负结果v4必要门31项通过、183封存保持。稳定MaskCO在线增益尚未建立，maskco ACTIVE，共享MD仅本地。\n\n'
    section='''

## 2026-10-11：保留边不等于部署动作空间，已修正教师结构约束

上一节v2的“保留无向部分邻接”工程资格本身成立，但不足以覆盖当前插入解码器。实际`reconstruct`先从各原路线删去掩码订单，然后仅将这些订单插回；所有未掩码订单的原车辆和相对顺序不改变。v2可移动或翻转未掩码片段，其两个部分掩码最优记录均未保留这个必要骨架，故不能把它们冒称部署动作可执行标签。已追加`retained_edge_teacher_engineering_root_v2/DEPLOYMENT_ACTION_MISMATCH.json`，旧1968项边保留核验及全部原件不覆盖。

新隔离`insertable_teacher_engineering_root_v3`明确注册实际掩码，并要求部分邻接恰为原路线删除掩码订单的结果；每个未掩码订单锁原车辆，严格整数OrderRank约束其原相对顺序，避免坐标重合或零服务时仅物理时间排序不够。原无向保留边约束仍在。掩码订单允许多车重新插入，全部客户必须服务、容量/时间及全C0择优保持。

同两份已经观察的工程快照，4个请求3.029–3.281秒；保留316候选，运行663项0失败，独立全候选边/原车/原顺序/四项成本/物理资格和顺序择优2594项0失败。全部边保留边界仍未改善。这个新增约束修正了已证明的结构越界，并非重跑同一科学假设挑均值；没有新增科学日、任务训练或确认数据。

严格限定：这些标签通过“只插回掩码订单”结构必要条件，不证明现有特定贪心解码器在可学习logit下必能生成它们，也不证明最优、预训练归因或真实动态增益。下一步封存完整源码/解释器闭包并做实际插入动作可达性与多步状态教师资格，再在新数据前登记新有限科学族。先前v2工程数字不作为本次标签质量比较或新收益主张，旧DEV16负裁决仍保留。
'''
    manifest=[]
    for f in entries:
        old=f.read_text(encoding='utf-8');saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved)
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['delivery'].update(retained_teacher_engineering_directory='research/review_3080_20261010/delivery/insertable_teacher_engineering_root_v3',retained_teacher_independent_checks=2594,retained_teacher_independent_failures=0,retained_teacher_candidate_replays=316,retained_teacher_identity=sha(path),previous_retained_teacher_action_mismatch_preserved=True,teacher_structural_insertion_necessity_qualified=True,specific_learned_greedy_decoder_teacher_reachability_qualified=False,next='seal full source/interpreter closure; test actual insertion action reachability and multistep teacher-state qualification; new scientific registration before data');new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            if old.startswith('> **2026-10-11 当前三线资格**'):old=old.split('\n\n',1)[1]
            new=header+old+(section if f==DOC else '')
        f.write_text(new,encoding='utf-8');manifest.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    (backup/'UPDATE_IDENTITY.json').write_text(json.dumps({'files':manifest,'shared_MD_uploaded':False,'frozen_science_modified':False},indent=2));print(json.dumps({'local_entries_updated':len(manifest),'independent_teacher_checks':2594,'stable_online_gain':False}))
if __name__=='__main__':main()
