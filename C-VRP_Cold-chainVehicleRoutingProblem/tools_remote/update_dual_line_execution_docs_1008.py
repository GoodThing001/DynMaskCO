"""Record dual-line authorization and qualified engineering separately from gains."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/当前规划/决策与审计'
status='''> **2026-10-08 19:00 双线定时推进已落实**：同一每30分钟任务已扩展为取货与配送独立实验，取货结案后仍继续配送。
> P线：v5 QA于18:41通过并自动整轮重采32日128状态；337项源码、实际工程权重字节和低订单日四状态已独立核验，当前无完整教师日，正式192步训练/四臂40日仍待完整128屏障。
> D线：独立配送路线快照合同、认证器与OR-Tools/大邻域搜索18:39真实工程通过；完整MaskCO encoder+decoder18:50工程通过。配送独立协议v1已在看开发收益前登记，18:58启动有限链：先实际/随机同架构51节点预热核验，再32全新快照六臂同初始路线、同总10秒比较，结束后独立裁决。
> 配送首pilot采用宽时间窗和已备货快照，只检验路线迁移；正式通过才另登记真实波次/车队/货物在线合同及强对照，不把快照改善称在线收益。当前仍未建立MaskCO核心正向，取货最终27/28不碰。
> [双线执行与方向决策](LINK_PREFIX取货与配送双线独立实验_执行与方向决策_2026-10-08.md) · [配送预声明](CONFIG_LINK)。下方旧状态为历史。

'''


def prepend(path,block):
    old=path.read_text(encoding='utf-8')
    title,rest=old.split('\n',1)
    path.write_text(title+'\n\n'+block+rest.lstrip('\n'),encoding='utf-8')


def main():
    for path,prefix,config in [
        (ROOT/'项目当前状态.md','docs/当前规划/决策与审计/','configs/delivery_20261008_v1/README.md'),
        (DOC/'方向A_总核查与下一阶段路线图_2026-09-30.md','','../../../configs/delivery_20261008_v1/README.md'),
        (DOC/'I3定时跟进与自动裁决_2026-10-08.md','','../../../configs/delivery_20261008_v1/README.md')]:
        prepend(path,status.replace('LINK_PREFIX',prefix).replace('CONFIG_LINK',config))
    path=DOC/'I3定时跟进与自动裁决_2026-10-08.md'
    txt=path.read_text(encoding='utf-8').replace('| 任务名称 | 推进 MaskCO I3 正式实验与裁决 |','| 任务名称 | 推进 MaskCO 取货与配送独立实验 |')
    txt=txt.replace('| 运行边界 | 本轮正式链裁决、独立核验及归档完成后暂停 |',
        '| 运行边界 | 取货有限链与配送已授权有限链分别验收；两线结案归档后暂停，取货先结案时继续配送 |')
    txt=txt.replace('暂停此定时任务。若首门数值未过','取货线停止数值阶段，配送仍按独立协议继续。若首门数值未过')
    txt+='''
## 配送线的同任务推进

用户再次明确要求定时推进两线。自动任务不在取货结案时提前暂停；配送工程、pilot与通过后的在线全量属于授权范围。配送已运行工程目录为`results/delivery_engineering_1008_v1`和`results/delivery_provider_1008_v1`；首路线pilot唯一目录为`results/delivery_pilot_1008_v1`，协议`configs/delivery_20261008_v1/protocol.json`。

检查配送`pipeline_state.json`、`failure.json`、`run.exit`、`adjudication.exit`、`SOURCE_END.json`、`pilot/MODEL_ENGINEERING_PASS.json`、`pilot/INITIAL_ROUTES_PASS.json`、`pilot/progress.jsonl`、`PILOT_ADJUDICATION.json`及实际进程。模型工程屏障在产生配送开发数据之前；不得以tmux消失或CUDA插件警告推断运行失败。CPU54、单线程、无GPU，有限链最多2小时，不重复启动。

原始逐实例结果须独立核对完整32唯一ID、六臂硬可行、共享初始路线/数据、所有源码清单与实际权重起止SHA、总10秒/超时回退和模型控制身份，再复算CI。自动runner报告不是唯一资格证据。正式过路线门才登记真实配送波次在线合同、强动态对照、开发/最终划分；未过保留全部结果并诊断，不自动改门槛、换模型或换种子。首路线裁决立即报告；只有两线本轮授权阶段均裁决归档且无后续已授权阶段在途时才暂停。
'''
    path.write_text(txt,encoding='utf-8')
    path=DOC/'取货与配送双线独立实验_执行与方向决策_2026-10-08.md'
    txt=path.read_text(encoding='utf-8')
    txt=txt.replace('用户已授权启动准备；合同/基线/数值协议尚未完成',
        '路线快照合同/两搜索基线/真实MaskCO工程已过，32快照六臂有限链18:58启动')
    txt=txt.replace('当前未登记具体阈值，不启动数值筛选。',
        '首版具体规则已在读取任何配送开发收益前登记为`configs/delivery_20261008_v1/protocol.json`，见下方执行记录；后续不得据结果改门。')
    txt=txt.replace('- P线：18:06实查v5仍在工程QA；337项封存合格，正式128采集/192步训练/四臂在线尚未启动，无新增方法正向。',
        '- P线：18:41 v5 QA通过并开始整轮重新采集；337项封存、实际工程权重与低订单日四状态已独立核验，完整128/192步/四臂在线尚待，无新增方法正向。')
    txt=txt.replace('- D线：用户本轮授权双线推进；本文件建立执行边界。现有C0仅支持取货，旧EURO接口不具备官方波次控制器资格；配送合同和强基线属于接下来必须完成的工程项。',
        '- D线：路线快照合同与认证器、OR-Tools GLS/随机destroy-regret-2 LNS和真正完整MaskCO encoder+decoder工程均已通过；18:58唯一配送pilot有限链启动，先模型/随机控制工程屏障再开发实例。正式在线运营合同尚未实现，不称EURO官方适配或在线增益。')
    txt=txt.replace('- 下一动作：取货QA完成后独立核验并继续有限链；配送先完成D0/D1，数值pilot按自身完整预声明启动。两条线分别汇报资格、正负结果和待补证据。',
        '- 下一动作：取货逐日独立教师验收→完整128→两模型固定192步→首开发四臂40日；配送完成模型屏障、32快照六臂及原始逐例独立裁决。两条线分别汇报，配送只有正式过路线门才登记在线全量。')
    txt+='''
## 六、2026-10-08 配送执行记录与固定pilot规则

同一自动任务`maskco-i3`已改名为“推进 MaskCO 取货与配送独立实验”，每30分钟继续；常规计算保持安静，阶段/异常/裁决报告。取货结束不停止配送，配送不占用取货或团队GPU。

| 工件 | 实查与资格 |
|---|---|
| `scripts/delivery/delivery_contract.py`、`delivery_evaluator.py` | 独立仓库装货→客户卸货；订单级manifest守恒，品质从装货完成到交付完成，载重递减，独立距离/预冷/制冷/开门/品质计费 |
| `results/delivery_engineering_1008_v1` | 本地及服务器合同反例通过，6项源码起止封存；真实OR-Tools与regret-2 LNS同初始路线、1秒工程调用约0.862/0.966秒，零超时 |
| `results/delivery_provider_1008_v1` | 完整CVRP encoder/decoder使用，部分邻接输入使实际logit变化约1.188；固定toy上单步/迭代均硬可行并保留未掩码边；不属于方法收益 |
| 工程时限线索 | 旧toy迭代2秒调用实际2.075秒，如实计超时并回退共同初始路线。正式pilot封存前增加依据固定合成预热解码时长的保守留时；未改掩码比例、评分或主门，旧工程输出保留 |
| `configs/delivery_20261008_v1/protocol.json` | SHA256 `8e40c705e57a6b6f2ed2244fc9c73108b43db7757596ff8ed03283f110cc3642`；无配送开发收益读入后调整 |
| `results/delivery_pilot_1008_v1` | 18:58在CPU54开始有限链，72项源清单，仅读原始CVRP权重SHA固定；先真实/随机同架构51节点预热与独立初始路线验收，再数值比较 |

**首pilot的固定设计**：32个独立新快照、50客户、坐标[0,10]²公里、容量50、五车、8小时、宽时间窗[0,8]；所有订单已释放/备货。五个客户ID块各10单形成所有臂共享初始路线；不筛掉坏例或重新生成。物理参数为pilot参考，不宣称真实车辆校准；仓库保管不由路线策略控制，品质自装货完成起算，所有温舱预冷计费。

六臂：共同初始路线、OR-Tools、regret-2 LNS、MaskCO单步、同权重迭代、同架构随机单步。改善臂同CPU/单线程/10秒，含特征、准备、推理、搜索、认证和提交；静态权重加载、库导入及固定合成51节点JIT预热规则事前统一。真实权重为原CVRP完整encoder+decoder，不使用取货I3 actor、无微调。随机控制同架构seed42。两个搜索法目前是可运行经典搜索候选，不称HGS/ALNS或经验证的在线最强。

**正式进入下一阶段的主门**：MaskCO单步相对OR-Tools和LNS分别平均成本改善≥1%、2000次实例配对bootstrap 95%CI下界>0；另须胜同架构随机单步（均值与CI下界>0）。六臂均完整32唯一有限实例、hard=1、身份与源码/权重起止合格。迭代是同模型同预算消融，不能在单步失败后事后换作主门。超时保留并按共同初始路线回退，不能筛例。

**仍未完成的配送项**：模型/随机控制正式启动屏障、32快照正式裁决；真实动态波次、有限车队返仓再用、装货与承诺后缀、库存/延期/拒单、在线强对照及独立最终留出须在路线门通过后另登记实现。路线快照正向只表示具备进入在线验证的线索。
'''
    path.write_text(txt,encoding='utf-8')
    progress=ROOT/'docs/当前规划/结果与进度/执行进度表.md'
    prepend(progress,'''## 2026-10-08 19:00：取货采集恢复，配送独立pilot已调度

- 取货v5低订单日四状态/实际工程权重/337项封存独立核验通过，18:41开始32日128状态整轮重采；完整日暂0/32，正式192步和四臂在线未开始。工程报告不能代替正向。
- 同一每30分钟任务扩展双线；取货结案后继续配送。配送合同/认证器与两搜索法工程18:39通过、真实完整CVRP encoder+decoder18:50通过。旧toy迭代2.075秒如实超时回退；新有限链在封存前补保守解码留时，不改科学参数。
- 配送主协议v1在收益前登记：32全新50客户快照、六臂、同初始路线/同10秒/同CPU54；单步分别胜两搜索法≥1%且CI下界>0，另胜随机控制，全部完整/硬可行/身份才过门。18:58启动`results/delivery_pilot_1008_v1`，先实际/随机预热屏障，源72项。
- 两线均不动取货最终27/28；配送尚无正式收益结果。通过路线门才登记真实配送在线全量，未过如实归档/诊断。

''')
    print('UPDATED_DUAL_LINE_STATUS_ROADMAP_PROTOCOL_AND_AUTOMATION_GUIDE')


if __name__=='__main__':main()
