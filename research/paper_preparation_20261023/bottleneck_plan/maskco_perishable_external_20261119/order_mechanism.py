"""Describe audited split activity and paired order-versus-insertion interactions."""
from pathlib import Path
from statistics import mean
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"maskco_heatmap_calibration_20261119"))
from assess import describe
from phase_plan import read
SOURCES=("VRPTWQ-C2","PDPCDPG-D100","XTW-p100","GH-p100")

def stage(phase):
    external=phase=="reserve02"
    root=HERE/"reserved_validation" if external else HERE
    folder=root/"results"/phase
    cfg=read(root/(phase+".json"));audit=read(folder/"independent_audit.json")
    expected=1053 if external else 312
    if audit["status"]!="PASS" or audit["endpoints"]!=expected:raise ValueError("Incomplete route audit")
    meta=read(root/"heldout_sources.json" if external else HERE.parent/"maskco_cross_dataset_20261119/matrix.json")["cases"]
    data={}
    for i,b in enumerate(cfg["blocks"]):
        arms=read(folder/f"out/{i}.json")["arms"];pair=b["case"],b["seed"]
        h,d=(arms[m] for m in ("ORDER_SPLIT","ORDER_SPLIT_D"))
        if h["forwards"]<=0 or d["forwards"]!=0 or h["split_orders"]==0 or d["split_orders"]==0:raise ValueError("Missing matched reconstruction")
        row={}
        for name,r in (("heat",h),("dist",d)):
            for k in ("split_blocks","split_orders","split_offers","forwards","forward_s","iterations"):
                row[name+"_"+k]=r[k]
            row[name+"_had_offer"]=float(r["split_offers"]>0)
        counts=h["ranking_counts"]
        if len(counts)!=3 or sum(counts)==0:
            raise ValueError("No completed ranking trials")
        for label,count in zip(("distance_share","raw_share","calibrated_share"),counts):
            row[label]=count/sum(counts)
        row["regret_share"]=h["regret_epochs"]/sum(counts)
        if external:
            oldh,oldd=(arms[m] for m in ("HYBRID","HYBRID_D"))
            row["version_by_heat"]=(h["U"]-d["U"])-(oldh["U"]-oldd["U"])
        data[pair]=row
    lines=["# 访问顺序和经济分车的作用","","每次搜索 60 秒。‘提高纪录’指分车步骤当时直接提高历史最好净效益。先对同一几何中的案例及种子平均，再对几何等权。","",
           "| 来源 | 排序 | 每次求解的分车组数 | 尝试访问序列数 | 分车直接提高纪录次数 | 至少一次提高纪录的比例 | 神经前向秒 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    contrasts={}
    choices={}
    for src in SOURCES:
        pairs=[k for k in data if meta[k[0]]["dataset"]==src]
        if not pairs:continue
        def stat(key):
            return describe({k:data[k][key] for k in pairs},meta)
        for name,label in (("heat","MASKCO"),("dist","同规则纯距离")):
            x={k:stat(name+"_"+k)["mean"] for k in ("split_blocks","split_orders","split_offers","had_offer","forward_s")}
            lines.append(f"| {src} | {label} | {x['split_blocks']:.1f} | {x['split_orders']:,.0f} | {x['split_offers']:.2f} | {x['had_offer']:.1%} | {x['forward_s']:.2f} |")
        if external:contrasts[src]=stat("version_by_heat")
        choices[src]=[stat(k)["mean"] for k in ("distance_share","raw_share","calibrated_share","regret_share")]
    lines+=["","| 来源 | 普通重插选距离 | 选原热图 | 选校准热图 | 选 regret-2 插位 |","|---|---:|---:|---:|---:|"]
    for src,values in choices.items():
        lines.append("| "+src+" | "+" | ".join(f"{v:.1%}" for v in values)+" |")
    lines+=["","前三项是完成试选的重构时段中三类边排序被选中的比例，包含按既定轮换次序处理的平局。最后一项按插位算子统计，与排序类型交叉。跨路线分车单独使用该臂的实际边分数。"]
    if contrasts:
        lines+=["","两版热图增量之差为（新神经版 − 新距离版）−（旧神经版 − 旧距离版）。正值表示引入分车步骤后，神经信息相对同版纯距离的贡献增加。区间以几何为整簇重采样，固定三种子。","",
                "| 来源 | 两版热图增量之差 U | 95% 描述区间 U | 正向几何 | 三个种子 U |","|---|---:|---|---:|---|"]
        for src,x in contrasts.items():
            lo,hi=x["ci95"]
            lines.append(f"| {src} | {x['mean']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {x['wins']}/{x['geometries']} | {' / '.join(f'{v:+.2f}' for v in x['by_seed'].values())} |")
    (folder/"分车机制.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines[4:]))
if __name__=="__main__":stage(sys.argv[1])
