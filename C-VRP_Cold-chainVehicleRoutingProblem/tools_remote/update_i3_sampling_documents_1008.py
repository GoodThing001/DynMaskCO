"""Record human approval and the fresh v5 execution without rewriting old evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/当前规划/决策与审计'
common = '''> **2026-10-08 17:53 批准修订与新冻结链**：第12日原轨迹 hard=True、0失败，
> 但仅150个有效订单。每日200是最大槽数，原固定序号174在12/25/29日不存在，
> 独立完整数组核验旧设计最多125/128状态；这属于采集设计不可完成，不是I3效用失败。
> 用户已明确批准按各日订单数12.5%/37.5%/62.5%/87.5%位置整轮重采。
> 新协议v2已登记，v5于17:50启动有限QA及条件监督链，337项冻结核验通过；
> 315个脚本字节不变，SAA/C0/模型/损失/优化器/在线口径保持原值。
> 低订单日真实四状态验收须通过，才自动重采32日128状态；再独立验收才开训。
> v4及观察链已精确停止、旧48状态保留但不复用；两正式模型和在线尚未开始。
> 当前仍无MaskCO核心正向，最终27/28不碰。下方较早状态属于历史。
> [获批修订及实现](LINK)。

'''
for p in [ROOT / '项目当前状态.md', DOCS / '方向A_总核查与下一阶段路线图_2026-09-30.md',
          DOCS / 'I3固定教师决策训练_执行预声明_2026-10-08.md', DOCS / 'I3定时跟进与自动裁决_2026-10-08.md']:
    text = p.read_text(encoding='utf-8')
    link = ('docs/当前规划/决策与审计/' if p.parent == ROOT else '') + '方向A_I3采集位置修订提案_2026-10-08.md'
    first, rest = text.split('\n', 1)
    p.write_text(first + '\n\n' + common.replace('LINK', link) + rest.lstrip('\n'), encoding='utf-8')

proposal = DOCS / '方向A_I3采集位置修订提案_2026-10-08.md'
text = proposal.read_text(encoding='utf-8')
text = text.replace('# I3 采集位置修订提案（2026-10-08，待批准）', '# I3 采集位置修订与执行（2026-10-08，已批准）')
text = text.replace('**状态：原注册采集设计已证实不可完成；本提案尚未获准执行。**',
    '**状态：用户已明确批准“批准修订并整轮重采（推荐）”；新科学协议v2生效。**\n\n'
    '原提案草案保留；获批版见 [protocol v2](../../../configs/i3_20261008_v2/protocol.json) 与'
    ' [批准及变更记录](../../../configs/i3_20261008_v2/APPROVAL_AND_CHANGE_RECORD.json)。\n'
    '新执行目录为 `results/i3_registered_1008_v5`，17:50启动；17:53等待工程QA。')
text = text.replace('全部32日映射已明确写入[协议草案]', '审批前全部32日映射已明确写入[协议草案]')
text = text.replace('用户此前要求严格按冻结protocol执行；改变采集位置将改变训练样本，因此需批准\n本项明确修订后才激活新协议。批准前不启动新教师、训练或在线比较。',
    '用户此前要求严格按冻结protocol执行；采集位置属于训练样本设计，因此本项经明确批准才激活。\n'
    '批准只覆盖四分位采集位置和整轮重采，训练/在线门、业务取货方向均未改变。\n\n'
    '## 新冻结链与验收范围\n\n'
    '- v5源码从可信v4冻结副本构建，六处脚本变更/新增仅为逐日映射与对应验收；315个脚本不变。\n'
    '- 337项清单包括新协议、暖启动、CVRP、运行闭包及批准记录；启动和17:53核验均通过。\n'
    '- QA增加真实第12日N=150、K=2的四状态工程采集与独立终局核验；其状态不能进入正式训练。\n'
    '- 独立数据链为 `results/i3_teacher_data_audit_1008_v3`；风险链为 `results/i3_teacher_risk_1008_v3`。\n'
    '- 正式仍须128唯一状态、全部分支对账、独立验收匹配，之后两模型各192步、四臂新鲜40日。\n'
    '- 首开发种子20260926裁决完成即报告；正式过门才20260930/20261001双种子复验。\n'
    '- 新配送研究只属于独立设计建议，不能写入本取货protocol或直接启动第二条正式实验。\n')
proposal.write_text(text, encoding='utf-8')

pre = DOCS / 'I3固定教师决策训练_执行预声明_2026-10-08.md'
text = pre.read_text(encoding='utf-8').replace('机器协议为 [protocol.json](../../../configs/i3_20261008_v1/protocol.json)，所有参数在\n正式教师采集、训练和在线收益观察之前固定。',
    '当前机器协议为 [protocol v2](../../../configs/i3_20261008_v2/protocol.json)。用户批准的变更仅为\n'
    '按实际日订单数取四分位采集位置、整轮重采；其余训练、在线和主门在新正式采集前固定。\n'
    '原v1及旧尝试保持原样，不以旧状态回填新实验。')
old = '当前冻结执行目录为 `results/i3_registered_1008_v4`；新QA独立验收后15:04正式采集，但第12日资格检查失败，停止派发剩余任务，17:01完整10日/40状态。原配置只读诊断正在新目录运行；未完成128状态独立验收前不得开训。科学参数与机器协议未改，下方v2/v3正常运行条目为当时历史记录。'
text = text.replace(old, '当前冻结执行目录为 `results/i3_registered_1008_v5`，17:50开始工程QA。\n'
    '第12日诊断已证明固定序号缺失，用户批准逐日四分位整轮重采。原v4已停止，48旧状态不能复用。\n'
    '新QA（含N150四状态）及完整128状态独立验收之前不得开训；下方v2/v3记录属于历史。')
text = text.replace('| 采集 | 全局 day_id=0..31；每日日程固定第 24/74/124/174 决定，共 128 状态 |',
    '| 采集 | day_id=0..31；每日日程为 ceil(q×N_day)−1，q=1/8、3/8、5/8、7/8；逐日映射见protocol v2，共128状态 |')
pre.write_text(text, encoding='utf-8')

road = DOCS / '方向A_总核查与下一阶段路线图_2026-09-30.md'
text = road.read_text(encoding='utf-8')
text = text.replace('| I3正式教师 | v4第12日资格检查失败；17:01完整10日/40状态、前8日独立验收；同配置诊断在跑 |',
    '| I3正式教师 | 已证明旧采集位置不存在；用户批准protocol v2，v5等待工程QA后整轮重采 |')
text = text.replace('| 教师数据独立验收 | 新v2链已启动，逐日核验后完整128状态SHA匹配，作为开训资格屏障；v1历史留档 |',
    '| 教师数据独立验收 | 新v3观察链等待v5分片；完整128状态SHA匹配才开训；旧轮次全部留档 |')
text = text.replace('| I3固定训练 | 尚未启动；须先解除第12日资格故障并完成128状态独立验收 |',
    '| I3固定训练 | 尚未启动；须先通过真实低订单日QA及128状态独立验收 |')
text += '\n## 2026-10-08任务范围补充：取货与配送\n\n'
text += '当前取货I3继续，新配送线只做独立设计建议。优先候选为仓库装货的动态波次配送，\n'
text += '用于连接原始MaskCO路线解重构和公开基准；不改变当前合同、主门或种子。\n'
text += '详见[任务选择与有投入边界的扩展计划](动态取货与配送_任务选择及MaskCO适配建议_2026-10-08.md)。\n'
road.write_text(text, encoding='utf-8')

guide = DOCS / 'I3定时跟进与自动裁决_2026-10-08.md'
text = guide.read_text(encoding='utf-8')
for old, new in [('`results/i3_registered_1008_v4`：正式教师', '`results/i3_registered_1008_v5`：正式教师'),
    ('`results/i3_teacher_data_audit_1008_v2`：每日日分片', '`results/i3_teacher_data_audit_1008_v3`：每日日分片'),
    ('`results/i3_teacher_risk_1008_v2`：两模型', '`results/i3_teacher_risk_1008_v3`：两模型')]:
    text = text.replace(old, new)
text = text.replace('正式科学参数继续由`configs/i3_20261008_v1/protocol.json`确定，SHA为\n`345604e3c3ed0ff2c40e049ded3c4a5a6784fcde1f464fe221dbc98850d8f5a5`。\n定时任务创建不修改该协议、任何在跑冻结源码、现有工件或保留最终种子。',
    '当前正式协议为`configs/i3_20261008_v2/protocol.json`，SHA为\n'
    '`4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c`。\n'
    '用户批准逐日采集位置修订并整轮重采；其余科学参数不变，不修改任何在跑冻结源码、旧工件或最终种子。')
guide.write_text(text, encoding='utf-8')

diagnosis = DOCS / '方向A_I3第12日采集拦截与诊断_2026-10-08.md'
text = diagnosis.read_text(encoding='utf-8')
text += '\n## 17:25诊断完成与获批决策\n\n'
text += '原轨迹hard=True、0失败，只有150次决定/3捕获；诊断RUN/SEAL均0、332项源稳定。\n'
text += '原始200历史日数组下载独立解码：预定32日中12/25/29日无序号174，旧规则最多125/128。\n'
text += '原v4收尾后12完整日/48状态保留但无正式128资格；17:31–17:32按实际进程组停止主链及观察链。\n'
text += '用户已批准按各日有效订单数四分位位置整轮重采，其余训练和在线参数保持原值。\n'
text += '新protocol v2及v5冻结链17:50激活，尚待真实低订单日QA；未训练正式模型或取得在线效用。\n'
diagnosis.write_text(text, encoding='utf-8')

progress = ROOT / 'docs/当前规划/结果与进度/执行进度表.md'
text = progress.read_text(encoding='utf-8')
first, rest = text.split('\n', 1)
entry = '''
## 2026-10-08 17:53：采集结构失败定位、用户批准v2、v5启动

- 原第12日hard=True、0失败，但实际150单；预定32日旧采集设计最多125/128。原v4及两个观察链停止并保留48状态，禁止复用。
- 用户明确批准各日12.5%/37.5%/62.5%/87.5%整轮重采。protocol v2登记，科学变更仅离线采集位置；192步/四臂40日/K10/总10s/主门不变。
- v5与数据/风险v3观察链17:50启动；337项冻结清单17:53复验通过，315个脚本字节不变。工程QA增加真实N150四捕获与独立终局对账，正式采集等待QA。
- 尚无新正式训练/在线结果，MaskCO核心正向未建立，最终27/28不碰。
- 配送仅增加独立研究设计备忘录，当前取货实验不更换业务合同。

'''
progress.write_text(first + '\n' + entry + rest, encoding='utf-8')
print('UPDATED_APPROVED_PROTOCOL_V2_AND_CURRENT_V5_DOCS')
