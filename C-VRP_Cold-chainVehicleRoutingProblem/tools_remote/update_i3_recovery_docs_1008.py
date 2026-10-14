"""Current entries only; retain dated history and original protocol bytes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEC = ROOT / 'docs/当前规划/决策与审计'
NAME = '方向A_I3采集工程故障修复与同协议重启_2026-10-08.md'
TAG = '**2026-10-08 14:46 工程恢复实查**'
notice = '''> **2026-10-08 14:46 工程恢复实查**：v3第3日教师保存重载因无序集合身份哈希失败，
> 旧链及两个观察链已精确停止、全部工件保留，旧分片不进入正式训练。
> 新v4已14:41启动工程QA，332项封存一致、原protocol SHA不变，正式重新采集尚待QA。
> 同时修正异步日任务CPU冲突，增加全部128状态独立验收后才开训的资格屏障。
> 两模型192步及四臂40日在线尚未开始，MaskCO核心正向仍未建立；最终27/28未碰。
> 以下较早时间条目为历史，不覆盖本条。详情见[工程修复与新链]({link})。
'''


def prepend(path, link):
    text = path.read_text(encoding='utf-8')
    if TAG not in text:
        title, rest = text.split('\n', 1)
        text = title + '\n\n' + notice.format(link=link) + '\n' + rest.lstrip('\n')
    path.write_bytes(text.encode('utf-8'))


prepend(ROOT / '项目当前状态.md', 'docs/当前规划/决策与审计/' + NAME)
prepend(ROOT / 'docs/当前规划/结果与进度/执行进度表.md', '../决策与审计/' + NAME)
for name in ['方向A_总核查与下一阶段路线图_2026-09-30.md',
             'I3固定教师决策训练_执行预声明_2026-10-08.md',
             '方向A_I3教师数据独立验收与失败诊断_2026-10-08.md',
             'I3定时跟进与自动裁决_2026-10-08.md']:
    prepend(DEC / name, NAME)

roadmap = DEC / '方向A_总核查与下一阶段路线图_2026-09-30.md'
text = roadmap.read_text(encoding='utf-8')
text = text.replace('| 可信底座/资格 | I3 QA通过，322项封存一致；原A3完整日/身份/均值CI复核闭环 |',
                    '| 可信底座/资格 | 原A3复核闭环；新I3 v4启动332项封存一致，QA在跑；v3教师整轮工程资格失败 |')
text = text.replace('| I3正式教师 | 13:10激活，32日128状态，13:19为8worker且尚无完整日落盘 |',
                    '| I3正式教师 | v3工程失败留档；v4按原32日128状态全部重采集，14:46待QA |')
text = text.replace('| 教师数据独立验收 | 终局轨迹再评价及标签质量汇总已实现，本地/服务器反例通过，13:36有限链启动 |',
                    '| 教师数据独立验收 | 新v2链已启动，逐日核验后完整128状态SHA匹配，作为开训资格屏障；v1历史留档 |')
text = text.replace('| 当前：正式采集 |', '| 当前：QA→正式采集 |')
text = text.replace('有限链`i3_registered_1008_v3`已自动越过QA和原A3失败依赖，正在教师采集，随后\n训练→首门→过门才双复验，无需因计算等待反复人工“继续”。独立风险链只读固定\n权重并产生诊断。源/工程失败停止保留；数值失败不自动调参或换种子重试。',
'''当前有限链`i3_registered_1008_v4`正在新QA，随后按原协议重新采集全部128状态，
完整独立验收→两模型192步→首门→过门才双复验，无需因计算等待反复人工“继续”。
v3因教师无序集合身份哈希工程故障结案，所有旧状态保留但不混用。新数据/风险链
为v2；当前对话定时任务已更新这些路径。源/工程失败停止保留；数值失败不自动调参。''')
roadmap.write_bytes(text.encode('utf-8'))

predecl = DEC / 'I3固定教师决策训练_执行预声明_2026-10-08.md'
text = predecl.read_text(encoding='utf-8').replace(
    '冻结执行目录为 `results/i3_registered_1008_v3`，有限监督链已启动。',
    '当前冻结执行目录为 `results/i3_registered_1008_v4`，有限监督链14:41启动新QA；v3因归档身份工程故障结案，全部正式状态重新采集。科学参数与机器协议未改，下方v2/v3条目为当时历史记录。')
predecl.write_bytes(text.encode('utf-8'))

timer = DEC / 'I3定时跟进与自动裁决_2026-10-08.md'
text = timer.read_text(encoding='utf-8')
text = text.replace('最近13:53实查仍在正式教师采集，完整日\n尚未落盘，训练/在线未开始，当前没有MaskCO核心正向。',
                    '最近14:46实查新v4处于工程QA，332项封存一致；原v3已工程失败留档，全部状态重新采集，正式训练/在线未开始，当前没有MaskCO核心正向。')
text = text.replace('results/i3_registered_1008_v3', 'results/i3_registered_1008_v4').replace(
    'results/i3_teacher_data_audit_1008_v1', 'results/i3_teacher_data_audit_1008_v2').replace(
    'results/i3_teacher_risk_1008_v1', 'results/i3_teacher_risk_1008_v2')
text = text.replace('终局收入/燃油/拒绝损失独立核对。',
                    '终局收入/燃油/拒绝损失独立核对；完整128状态独立验收报告须与主链全部分片SHA匹配，才开训。')
text = text.replace('先查实际进程和落盘证据，再决定行动。',
                    '先查实际进程、每个teacher_job退出/日志及collection_failure，不能只看主链失败文件，再决定行动。')
timer.write_bytes(text.encode('utf-8'))
current = roadmap.read_text(encoding='utf-8')
assert '| I3正式教师 | v3工程失败留档；v4' in current
assert '当前有限链`i3_registered_1008_v4`' in current
print('UPDATED_SIX_CURRENT_DOCS; PROTOCOL/OLD_RESULT_BYTES_UNTOUCHED')
