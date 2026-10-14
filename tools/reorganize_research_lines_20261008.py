"""Classify editable documentation without relocating runtime or frozen artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]/'C-VRP_Cold-chainVehicleRoutingProblem'
LOG=ROOT/'docs/目录整理记录/2026-10-08'
LINK=re.compile(r'(!?\[[^\]\n]*\])\((<[^>\n]+>|[^)\n]+)\)')
KEEP={'方向A_总核查与下一阶段路线图_2026-09-30.md','I3定时跟进与自动裁决_2026-10-08.md',
      '取货与配送双线独立实验_执行与方向决策_2026-10-08.md','风险与解决预案.md','决策记录.md','R3封存清单_2026-09-28.md'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(text.encode('utf-8'))
def relative(target,source):return os.path.relpath(target,source.parent).replace('\\','/')
def safe(p):
    p=p.resolve()
    if ROOT.resolve() not in p.parents:raise ValueError('Outside extension workspace: '+str(p))
    return p


def migration_plan():
    plan={}
    base=ROOT/'docs/当前规划'
    for p in (base/'决策与审计').glob('*.md'):
        if p.name in KEEP:continue
        if p.name.startswith(('方向A','A-v1','A1','A3','I3','S3')):
            plan[p]=ROOT/'01_取货线/文档/协议与审计'/p.name
        elif p.name=='动态取货与配送_任务选择及MaskCO适配建议_2026-10-08.md':
            plan[p]=ROOT/'02_配送线/文档/研究设计'/p.name
    for section,names in {
        '工程实现':['动态承诺与条件场景前瞻_A_工作包.md','MaskCO场景生成_最小本体设计_2026-09-29.md'],
        '研究设计':['Step3_掩码条件场景生成设计.md','步骤3可行性文献核查_2026-09-25.md'],
        '文献调研':['四杠杆文献调研_动态冷链接单_2026-09-26.md'],
        '结果与进度':['方向A外部对比_开发集预对比值与RRNCO_StageB_2026-09-26.md',
            'A3在线比较正式裁决_2026-10-06.md','A1臂2_V模型先行探针结果_2026-10-03.md',
            'A1强动态对照正式裁决_2026-10-05.md']}.items():
        for name in names:
            p=base/section/name
            if p.is_file():plan[p]=ROOT/'01_取货线/文档'/section/name
    for p in (base/'历史已收束').rglob('*.md'):
        plan[p]=ROOT/'03_负结果线/文档/历史已收束'/p.relative_to(base/'历史已收束')
    for name in ['因果承诺评价器v4_三路对比结果_2026-09-23.md','三路候选实验汇总_2026-09-23.md']:
        p=base/'结果与进度'/name
        if p.exists():plan[p]=ROOT/'03_负结果线/文档/历史结果'/name
    p=ROOT/'configs/delivery_20261008_v1/README.md'
    if p.exists():plan[p]=ROOT/'02_配送线/文档/执行协议/配送路线重构pilot_v1_2026-10-08.md'
    for a,b in plan.items():safe(a);safe(b)
    if len(set(plan.values()))!=len(plan):raise ValueError('Duplicate migration targets')
    return plan


def protected_snapshot():
    result={}
    for folder in ['scripts','configs','results','data','ckpts','negative_result_paper','archive']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file():continue
            if folder=='configs' and p.name=='README.md':continue
            s=p.stat();result[p.relative_to(ROOT).as_posix()]=[s.st_size,s.st_mtime_ns]
    return result


def rewrite_links(text,old,new,mapping,changed):
    def replace(m):
        target=m.group(2);angle=target.startswith('<')
        raw=target[1:-1] if angle else target
        if not raw or raw.startswith('#') or re.match(r'^[a-zA-Z][\w+.-]*:',raw):return m.group(0)
        path,sep,anchor=raw.partition('#')
        original=(old.parent/urllib.parse.unquote(path)).resolve()
        relocated=mapping.get(original,original)
        if not original.exists() and original not in mapping:return m.group(0)
        if original==relocated and old==new:return m.group(0)
        updated=relative(relocated,new)+(sep+anchor if sep else '')
        # Preserve angle form for targets containing spaces.
        formatted='<'+updated+'>' if angle or ' ' in updated else updated
        changed.append({'document':new.relative_to(ROOT).as_posix(),'target':str(relocated)})
        return m.group(1)+'('+formatted+')'
    return LINK.sub(replace,text)


def main(apply):
    plan=migration_plan()
    if not apply:
        print(json.dumps({'documents_to_move':len(plan),'by_line':{line:sum(line in str(p) for p in plan.values())
              for line in ['01_取货线','02_配送线','03_负结果线']},
              'runtime_data_weights_frozen_results_relocated':False},ensure_ascii=False))
        return
    LOG.mkdir(parents=True,exist_ok=False)
    before=protected_snapshot()
    # Take originals before any moves so relative references resolve correctly.
    docs=[p for p in (ROOT/'docs').rglob('*.md') if '目录整理记录' not in p.parts]
    originals={p:p.read_bytes().decode('utf-8-sig') for p in docs}
    originals.update({p:p.read_bytes().decode('utf-8-sig') for p in plan if p not in originals})
    mapping={a.resolve():b.resolve() for a,b in plan.items()}
    records=[]
    for old,new in plan.items():
        if new.exists():raise FileExistsError(str(new))
        new.parent.mkdir(parents=True,exist_ok=True)
        record={'old':old.relative_to(ROOT).as_posix(),'new':new.relative_to(ROOT).as_posix(),
                'original_sha256':sha(old)}
        shutil.move(str(safe(old)),str(safe(new)))
        records.append(record)
    changed=[]
    for old,text in originals.items():
        new=plan.get(old,old)
        updated=rewrite_links(text,old,new,mapping,changed)
        if updated!=text:
            # Only link destinations may change in migrated scientific records.
            assert LINK.sub(r'\1(<TARGET>)',text)==LINK.sub(r'\1(<TARGET>)',updated)
            save(new,updated)
    # Thin compatibility documents preserve user bookmarks and historical references.
    for record in records:
        old=ROOT/record['old'];new=ROOT/record['new']
        save(old,'# 此文档已按研究方向归档\n\n2026-10-08目录整理。正文请见 ['+new.stem+']('+relative(new,old)+')。\n\n'
             '后续维护新位置；本入口仅保留旧链接兼容。实验配置、源码和数值结果未在此处改动。\n')
        record['new_sha256']=sha(new)
    for item in changed:
        if not Path(item['target']).exists():raise ValueError('New link target missing: '+str(item))
    after=protected_snapshot()
    differences=[p for p,v in before.items() if after.get(p)!=v]
    if differences:raise RuntimeError('Protected runtime artifacts changed: '+str(differences[:20]))
    manifest={'moves':records,'moved_documents':len(records),'updated_link_destinations':len(changed),
        'validated_changed_targets':len(changed),'protected_runtime_files_checked':len(before),
        'runtime_artifacts_unchanged':True,'negative_reproduction_packet_unchanged':True,
        'scientific_body_changes_in_moved_documents':'none; only relative link destinations rewritten',
        'compatibility':'old editable document paths contain pointers to new canonical locations'}
    save(LOG/'迁移清单.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in manifest.items() if k!='moves'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args();main(a.apply)
