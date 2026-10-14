"""Navigation, physical-document indices and Windows folder-view specifications."""
from pathlib import Path
import json
import os

ROOT=Path(__file__).resolve().parents[1]/'C-VRP_Cold-chainVehicleRoutingProblem'
LOG=ROOT/'docs/目录整理记录/2026-10-08'
LINES=['01_取货线','02_配送线','03_负结果线']
def save(p,t):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8')
def link(p,base,label=None):
    rel=os.path.relpath(p,base.parent).replace('\\','/')
    return '['+(label or p.name)+']('+('<'+rel+'>' if ' ' in rel else rel)+')'


def inventory():
    records=[]
    for kind in ['scripts','configs','results','tools_remote','logs','data','ckpts']:
        for p in sorted((ROOT/kind).iterdir()):
            if p.name=='__pycache__':continue
            name=p.name.lower()
            if 'delivery' in name:group='02_配送线'
            elif kind=='results' and name in ('jf2','jf2_closure','hfr','m0_scale','jf15_b1','jf15_b4','jf15_b8','jf15_b16'):group='03_负结果线'
            elif kind in ('results','configs','tools_remote') and (name.startswith(('i3','a1','a3','tools_remote_a1','tools_remote_i3','tools_remote_s3','tools_remote_rr3')) or 'i3_' in name):group='01_取货线'
            else:group='共享或历史基础'
            records.append({'path':p.relative_to(ROOT).as_posix(),'direction':group,'kind':kind,
                'type':'directory' if p.is_dir() else 'file',
                'qualification':'classification only; use original identity/adjudication for scientific claims'})
    records.append({'path':'negative_result_paper','direction':'03_负结果线','kind':'independent_reproduction_package',
                    'type':'directory','qualification':'kept intact, see original audit'})
    save(LOG/'分类资产清单.json',json.dumps(records,indent=2,ensure_ascii=False))
    return records


def main():
    records=inventory()
    descriptions={
        '01_取货线':('动态冷链取货','128状态及独立验收 → I3/纯重建各192步 → 四臂40日首门 → 过门才双种子。',
            '空载发车、取货入舱、返仓卸货。当前权威协议为I3 v2，服务器唯一正式链v5。尚未建立MaskCO核心在线正向。'),
        '02_配送线':('动态冷链配送','合同与搜索基线 → 同初始路线/同总预算重构pilot → 有合格线索才在线全量。',
            '仓库装货、客户卸货。独立路线快照合同与真实MaskCO工程已通过，32快照六臂pilot在跑；真实波次在线合同与全量属于后续阶段。'),
        '03_负结果线':('负结果与迁移失效机制','独立复现包、论文初稿、机制证据和已收束分支归档。',
            '用于研究迁移失效边界与解释全部尝试，不以工程故障充当方法负结果。独立论文包及release原字节/路径保留；是否可引用以原审计为准。')}
    for line,(title,flow,note) in descriptions.items():
        base=ROOT/line;readme=base/'README.md'
        docs=sorted((base/'文档').rglob('*.md'))
        docindex=base/'文档索引.md'
        txt='# '+title+'：文档索引\n\n2026-10-08按方向整理，正文已实际迁入本方向；原位置仅保留兼容跳转。\n\n'
        for folder in sorted(set(p.parent for p in docs)):
            txt+='## '+folder.relative_to(base).as_posix()+'\n\n'
            txt+='\n'.join('- '+link(p,docindex) for p in docs if p.parent==folder)+'\n\n'
        if line=='03_负结果线':
            txt+='## 独立论文与复现包\n\n'+link(ROOT/'negative_result_paper/README.md',docindex,'独立负结果论文包')+' · '+link(ROOT/'negative_result_paper/docs',docindex,'论文与审计')+' · '+link(ROOT/'negative_result_paper/release',docindex,'封存复现包')+'\n'
        if line=='02_配送线':
            txt+='## 全课题设计\n\n'+link(ROOT/'docs/当前规划/决策与审计/取货与配送双线独立实验_执行与方向决策_2026-10-08.md',docindex,'双线执行与方向决策')+'\n'
        save(docindex,txt)
        for kind,filename in [('results','实验结果索引.md'),('tools_remote','脚本与配置索引.md')]:
            path=base/filename
            selected=[r for r in records if r['direction']==line and (r['kind']==kind or (kind=='tools_remote' and r['kind']=='configs'))]
            body='# '+title+'：'+('实验结果' if kind=='results' else '脚本与配置')+'索引\n\n'
            body+='原运行路径保持稳定；通过资源入口打开原目录。本地可能是部分同步快照，服务器在途状态以当前状态与实际阶段报告为准。分类不自动赋予结果资格。\n\n'
            body+='| 原路径 | 类型 |\n|---|---|\n'
            for r in selected:body+='| '+link(ROOT/r['path'],path,r['path'])+' | '+r['type']+' |\n'
            if kind=='tools_remote':
                if line=='02_配送线':
                    body+='\n配送执行代码：'+link(ROOT/'scripts/delivery',path)+'；工程测试：'+link(ROOT/'scripts/tests/test_delivery_contract.py',path)+'、'+link(ROOT/'scripts/tests/test_delivery_maskco.py',path)+'。\n'
                elif line=='01_取货线':
                    body+='\n取货涉及共享SAA、模型、训练与评价，入口见'+link(ROOT/'scripts/evaluation',path)+'、'+link(ROOT/'scripts/training',path)+'、'+link(ROOT/'scripts/models',path)+'；详见本方向文档与当前冻结协议。\n'
                else:body+='\n负结果独立入口脚本：'+link(ROOT/'negative_result_paper/scripts',path)+'；源冻结复现：'+link(ROOT/'negative_result_paper/release',path)+'。\n'
            save(path,body)
        body='# '+title+'\n\n'+flow+'\n\n'+note+'\n\n'
        body+='## 阅读入口\n\n'
        body+='- '+link(ROOT/'00_项目导航.md',readme,'项目总导航')+'\n- '+link(ROOT/'项目当前状态.md',readme,'实时当前状态')+'\n'
        body+='- '+link(docindex,readme)+'\n- '+link(base/'脚本与配置索引.md',readme)+'\n- '+link(base/'实验结果索引.md',readme)+'\n'
        body+='\n`资源入口/`中的Windows快捷方式会打开原始工作目录，源码和结果没有复制出第二套。运行仍按已登记协议和当前状态指定的原始入口执行。\n'
        save(readme,body)
    nav=ROOT/'00_项目导航.md'
    save(nav,'''# MaskCO 动态冷链项目：三个研究方向总导航

2026-10-08。日常从下面三个文件夹进入；实时状态与全课题计划只有一个维护入口。

| 方向 | 目标 | 当前任务 |
|---|---|---|
| [01_取货线](01_取货线/README.md) | 建立MaskCO在动态取货接单中的核心在线增益 | I3正式采集、固定训练、四臂40日裁决 |
| [02_配送线](02_配送线/README.md) | 检验路线重构迁移，再验证动态配送在线收益 | 独立合同与基线已工程通过，32快照六臂pilot在跑 |
| [03_负结果线](03_负结果线/README.md) | 迁移失效边界、机制诊断与负结果论文证据 | 独立论文/复现包和收束分支归档 |

## 总控与共享基础

- [项目当前状态](项目当前状态.md)：唯一实时状态。
- [总路线图](docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md)：总体目标、证据门和下一阶段。
- [执行进度表](docs/当前规划/结果与进度/执行进度表.md)：所有尝试与阶段记录。
- [双线定时任务](docs/当前规划/决策与审计/I3定时跟进与自动裁决_2026-10-08.md)：每30分钟推进两性能线，阶段/异常/裁决报告。
- [共享文档](docs/README.md)、[公共代码](scripts/README.md)、[结果总索引](results/README.md)、[远程入口索引](tools_remote/README.md)。
- `scripts/`、`configs/`、`data/`、`ckpts/`、`results/`、`logs/`：保留已注册运行布局；各方向索引和资源快捷方式直接指向这些原始位置。
- `negative_result_paper/`：完整独立论文复现包；`archive/`：历史封存。两者不拆包、不覆写。

## 放置规则

1. 单方向可编辑文档写入对应方向的`文档/`；跨方向计划、全局状态与统一进度继续写总控入口。
2. 新运行仍按当前冻结协议创建独立代码快照/结果目录，登记到相应索引；在途目录、权重、配置和身份记录不因目录整理迁移。
3. 合格数值负结果归科研证据；崩溃、缺日、版本污染和身份缺失单独注明工程资格，不能写成方法失败。
4. 旧文档路径提供跳转，后续修改新正文位置，避免两套正文。
5. 当前取货最终20260927/28保持未访问。目录分类不改变实验门槛、科学参数或发表结论。

整理记录：[迁移清单](docs/目录整理记录/2026-10-08/迁移清单.json) · [分类资产清单](docs/目录整理记录/2026-10-08/分类资产清单.json)。
''')
    # Only editable index pages are augmented; sealed packets and runtime files stay untouched.
    for p,text in [
        (ROOT/'README.md','> **2026-10-08 当前入口**：项目已按[取货线、配送线、负结果线](00_项目导航.md)整理。两性能线独立协议与数据，负结果机制论文包单独追溯；实时状态见[项目当前状态](项目当前状态.md)。下文较早阶段为历史背景。\n\n'),
        (ROOT/'docs/README.md','> **2026-10-08 三线导航**：日常从[项目总导航](../00_项目导航.md)进入。可编辑方向文档已迁入`../01_取货线/文档/`、`../02_配送线/文档/`与`../03_负结果线/文档/`，原路径保留跳转；共享总控继续在此维护。\n\n'),
        (ROOT/'results/README.md','> **2026-10-08 当前索引**：按[取货](../01_取货线/实验结果索引.md)、[配送](../02_配送线/实验结果索引.md)和[负结果](../03_负结果线/实验结果索引.md)查看。结果目录与身份保持原位；主线为取货I3和独立配送路线pilot。各合同的效用/成本分别定义，以下旧O0与纯距离规则属于历史协议。\n\n'),
        (ROOT/'tools_remote/README.md','> **2026-10-08 按方向查找**：[取货脚本与配置](../01_取货线/脚本与配置索引.md) · [配送脚本与配置](../02_配送线/脚本与配置索引.md) · [负结果复现](../03_负结果线/脚本与配置索引.md)。在跑入口、`__file__`相对路径和冻结目录保持原位；后续新入口按协议登记，不从历史脚本标题推断当前任务。\n\n'),
        (ROOT/'项目当前状态.md','> **2026-10-08 目录整理完成**：请从[三线总导航](00_项目导航.md)进入；取货/配送/负结果方向正文已分目录，旧链接保留跳转。当前配置、源码、权重、服务器在途目录及封存结果保持原路径；定时任务仍继续推进两性能线。\n\n')]:
        old=p.read_text(encoding='utf-8');first,rest=old.split('\n',1);save(p,first+'\n\n'+text+rest.lstrip('\n'))
    p=ROOT/'README.md';t=p.read_text(encoding='utf-8')
    t=t.replace('> **Execution warning:** upstream MaskCO has moved to `../MASKCO_code/`, while existing extension entry points still contain the old root-level import assumption. Do not run training/decoding as a valid experiment until P0-M is complete.',
        '> **Layout status:** P0-M is complete. Current finite experiments use their registered frozen source trees; upstream `../MASKCO_code/` remains read-only. Historical reproduction commands below are not current I3 or delivery launchers.')
    t=t.replace('## Current Research Status','## Historical Research Status (before the current dual-line study)');save(p,t)
    views=[]
    targets={
        '01_取货线':[('01_当前I3配置','configs/i3_20261008_v2'),('02_取货训练代码','scripts/training'),
                  ('03_取货评估裁决代码','scripts/evaluation'),('04_当前I3本地快照','results/i3_registered_1008_v5')],
        '02_配送线':[('01_配送配置','configs/delivery_20261008_v1'),('02_配送执行代码','scripts/delivery'),
                  ('03_配送pilot本地快照','results/delivery_pilot_1008_v1'),('04_配送合同工程证据','results/delivery_engineering_1008_v1')],
        '03_负结果线':[('01_独立论文完整包','negative_result_paper'),('02_独立论文与审计','negative_result_paper/docs'),
                    ('03_负结果复现入口','negative_result_paper/scripts'),('04_封存release','negative_result_paper/release')]}
    for line,items in targets.items():
        folder=ROOT/line/'资源入口';folder.mkdir(exist_ok=True)
        for name,target in items:
            p=ROOT/target
            if p.exists():views.append({'shortcut':str(folder/(name+'.lnk')),'target':str(p.resolve())})
    save(LOG/'Windows目录入口.json',json.dumps(views,indent=2,ensure_ascii=False))
    print(json.dumps({'three_line_indices_written':True,'assets_classified':len(records),'windows_folder_shortcuts':len(views)},ensure_ascii=False))


if __name__=='__main__':main()
