"""Record verified v4 QA activation, leaving numerical protocol and history intact."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEC = ROOT / 'docs/当前规划/决策与审计'
TAG = '**2026-10-08 15:25 QA独立验收与正式采集**'
block = '''> **2026-10-08 15:25 QA独立验收与正式采集**：v4新QA已完成，RUN/SEAL均0、332项
> 封存一致。实际3步工程权重已下载独立核对，2次actor更新+1次无效教师重建、
> 保存重载通过，完整权重字节与修复前工程QA相同；不代表正式192步训练或在线增益。
> 正式128状态采集已于15:04:28自动启动，当前8任务分别使用CPU8..15，无失败；
> 完整教师日暂为0/32，独立验收等待分片，风险诊断工程通过并等待两正式模型。
> 科学参数及主门不变，MaskCO核心正向仍未建立，最终27/28未碰。
> [独立QA证据]({audit})。下方较早条目为历史。
'''


def update(path, audit):
    text = path.read_text(encoding='utf-8')
    if TAG not in text:
        title, rest = text.split('\n', 1)
        text = title + '\n\n' + block.format(audit=audit) + '\n' + rest.lstrip('\n')
    path.write_bytes(text.encode('utf-8'))


update(ROOT / '项目当前状态.md', 'results/i3_registered_1008_v4/INDEPENDENT_QA_RELEASE_AUDIT.json')
update(ROOT / 'docs/当前规划/结果与进度/执行进度表.md', '../../../results/i3_registered_1008_v4/INDEPENDENT_QA_RELEASE_AUDIT.json')
for name in ['方向A_总核查与下一阶段路线图_2026-09-30.md',
             'I3固定教师决策训练_执行预声明_2026-10-08.md',
             'I3定时跟进与自动裁决_2026-10-08.md',
             '方向A_I3采集工程故障修复与同协议重启_2026-10-08.md']:
    update(DEC / name, '../../../results/i3_registered_1008_v4/INDEPENDENT_QA_RELEASE_AUDIT.json')
path = DEC / '方向A_总核查与下一阶段路线图_2026-09-30.md'
text = path.read_text(encoding='utf-8').replace(
    '原A3复核闭环；新I3 v4启动332项封存一致，QA在跑；v3教师整轮工程资格失败',
    '原A3复核闭环；新I3 v4 QA/权重独立验收通过、332项一致；v3教师整轮工程资格失败')
text = text.replace('v3工程失败留档；v4按原32日128状态全部重采集，14:46待QA',
                    'v3工程失败留档；v4已15:04按原32日128状态重新采集，15:25为8任务/0完整日')
text = text.replace('| 当前：QA→正式采集 |', '| 当前：正式采集 |')
text = text.replace('当前有限链`i3_registered_1008_v4`正在新QA，随后按原协议重新采集全部128状态，',
                    '当前有限链`i3_registered_1008_v4`已通过新QA，正在按原协议重新采集全部128状态，')
path.write_bytes(text.encode('utf-8'))
path = DEC / 'I3固定教师决策训练_执行预声明_2026-10-08.md'
text = path.read_text(encoding='utf-8').replace('有限监督链14:41启动新QA；v3因归档身份工程故障结案',
    '有限监督链14:41启动、新QA已独立验收、15:04正式采集；v3因归档身份工程故障结案')
path.write_bytes(text.encode('utf-8'))
path = DEC / 'I3定时跟进与自动裁决_2026-10-08.md'
text = path.read_text(encoding='utf-8').replace(
    '最近14:46实查新v4处于工程QA，332项封存一致；原v3已工程失败留档，全部状态重新采集，正式训练/在线未开始，当前没有MaskCO核心正向。',
    '最近15:25实查新v4已通过工程QA及完整权重独立核对，332项封存一致；15:04开始全部128状态重新采集，8任务CPU无冲突，原v3工程失败留档，正式训练/在线未开始，当前没有MaskCO核心正向。')
path.write_bytes(text.encode('utf-8'))
print('RECORDED_QA_VERIFIED_AND_COLLECTION_ACTIVE; SCIENTIFIC_PROTOCOL_UNTOUCHED')
