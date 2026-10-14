"""Record completed engineering QA in mutable local entries; preserve prior bytes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
R = ROOT / 'research/review_3080_20261010'
DOC = E / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'

def main():
    audit = R / 'delivery/remote_actual_bundle_v2/downloaded/INDEPENDENT_ROOT_ACTUAL_FULLSTEP_AUDIT.json'
    result = json.loads(audit.read_text(encoding='utf-8'))
    assert result['n_checks'] == 528 and result['n_failures'] == 0
    backup = R / 'current_entries_before_actual_model_completion_20261011'
    backup.mkdir(exist_ok=False)
    entries = [DOC, E/'项目当前状态.md', E/'00_项目导航.md',
        E/'docs/当前规划/结果与进度/执行进度表.md',
        E/'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',
        E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries += [E/line/name for line in ['01_取货线','02_配送线','03_负结果线']
                for name in ['README.md','实验结果索引.md','文档索引.md']]
    entries.append(E/'results/research_20261010_continuation_v2/WORK_STATUS.json')
    manifest = []
    for p in entries:
        if not p.exists():
            continue
        q = backup / p.relative_to(E)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        old = p.read_text(encoding='utf-8')
        if p == DOC:
            new = old + '''

## 2026-10-11：真实完整模型工程阶段完成

3080 主机新隔离 `remote_actual_fullstep_v2` 的固定一次全网络更新已结束，实际退出码0、外层61.334秒。原CVRP checkpoint 的76,259,840个参数、234叶全部实际更新；encoder/decoder均有有限非零梯度，掩码上下文改变输出，完整候选C0硬可行并保留未掩码边，保存重载输出逐位一致。总控另从原工件独立再生实例、重算路线/成本和实际保存权重及38源码身份，528项、0失败。资格文件位于根目录 `research/review_3080_20261010/delivery/remote_actual_bundle_v2/downloaded/INDEPENDENT_ROOT_ACTUAL_FULLSTEP_AUDIT.json`。

首个Windows后台尝试被SSH作业环境中断，未进入模型步骤，空日志和进程证据保存在v1。v2使用系统进程创建接口隔离SSH作业，另开独占目录、同输入和固定工程参数，未覆盖原尝试。没有重复正式数值实验。

此模型仅工程QA，不是可用研究权重；一次CE更新、旧已看教师样本和代理loss下降不能作为训练效用或在线增益。下一步正在验证10维可见冷链字段、原三维投影零扩展及同初始化CE-only与CE+偏好目标，然后才能登记新的有限训练。取货公共状态模型已实际激活，正在补共同历史、完整日和硬截止接线；负结果线继续补实际源码/语义重放验证。

只读检查发现本地attention函数忽略softcap参数，但该原CVRP checkpoint的实际softcap和final_softcap均为null；因此不能将这个潜在非空配置缺陷误报为本次checkpoint故障。其他配置的算子兼容仍须独立验证，上游保持不改。所有工程阶段仍不授予稳定、可归因的MaskCO在线增益。
'''
        elif p.suffix == '.json':
            status = json.loads(old)
            status['updated'] = datetime.now(timezone.utc).isoformat()
            status['delivery'].update(
                stage='actual_full_checkpoint_engineering_pass_coldchain_objective_qualification_in_progress',
                actual_full_checkpoint_engineering_exit=0,
                actual_full_checkpoint_independent_checks=528,
                actual_full_checkpoint_independent_failures=0,
                actual_full_parameter_leaves_updated=234,
                actual_full_checkpoint_scientific_training_qualified=False,
                actual_full_checkpoint_remote_directory='research/review_3080_20261010/remote_actual_fullstep_v2',
                next='visible 10D feature mapping, identical initialization CE-only versus CE-plus-preference, C0 teacher validation before fresh finite training')
            new = json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False)
        else:
            new = old.replace('真实76M参数工程正在资格验证。',
                              '真实76M参数工程已结束0、独立528项0失败；冷链输入与目标对照验证继续，仍非研究增益。', 1)
        p.write_text(new, encoding='utf-8')
        sha = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
        manifest.append({'path':p.relative_to(E).as_posix(), 'before_sha256':sha(q), 'after_sha256':sha(p)})
    with (backup/'UPDATE_IDENTITY.json').open('x', encoding='utf-8') as f:
        json.dump({'engineering_stage_only':True, 'old_protocol_or_result_modified':False,
                   'server_shared_MD_uploaded':False, 'files':manifest}, f, ensure_ascii=False, indent=2)
    print(json.dumps({'updated_local_current_entries':len(manifest), 'actual_checks':528, 'failures':0}))

if __name__ == '__main__':
    main()
