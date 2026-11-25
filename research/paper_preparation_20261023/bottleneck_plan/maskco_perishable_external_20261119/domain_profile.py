"""Describe the physical inputs seen by the frozen first-fold MASKCO and external cases."""
import json
from pathlib import Path
from statistics import mean
from itertools import combinations
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import audit_delivery
from mechanism_summary import source_means
from phase_plan import read
sys.path.insert(0,str(audit_delivery.ROOT/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"))
import profit_core as core

def features(rows):
    a=np.array(rows,dtype=float);p=a[1:,1:3]
    d=np.sqrt(np.square(p[:,None,:]-p[None,:,:]).sum(axis=2))
    np.fill_diagonal(d,np.inf)
    depot=np.linalg.norm(p-a[0,1:3],axis=1)
    loads=a[1:,3]/20000
    neighbors=np.argsort(d,axis=1,kind="stable")[:,:8]
    j,k=np.array(list(combinations(range(8),2))).T
    pair_excess=(loads[:,None]+loads[neighbors])>1+1e-9
    triple_excess=(loads[:,None]+loads[neighbors[:,j]]+loads[neighbors[:,k]])>1+1e-9
    return dict(median_depot_distance_km=float(np.median(depot)),median_nearest_customer_km=float(np.median(np.min(d,axis=1))),
                median_window_minutes=float(np.median(a[1:,5]-a[1:,4])),mean_capacity_share=float(np.mean(a[1:,3]/20000)),
                mean_service_minutes=float(np.mean(a[1:,6])),near_pair_capacity_rejection=float(np.mean(pair_excess)),
                near_triple_capacity_rejection=float(np.mean(triple_excess)))
def main():
    meta={}
    fields={}
    conf=core.load_config()
    for case in core.fold_definition(0)["train"]:
        r=core.load_original(case,conf)
        meta[case]=dict(dataset="VRPTWQ train",geometry=r["geometry_sha256"])
        fields[case]=features(r["nodes"])
    for label,folder,original in (("Development",PREV.parent/"maskco_cross_dataset_20261119",True),
                                  ("First validation",PREV,False),("External replication",HERE,False),
                                  ("Reserved validation",HERE/"reserved_validation",False),
                                  ("Additional low-overlap XTW",HERE/"xtw_low_overlap",False)):
        data=read(folder/("matrix.json" if original else "heldout_sources.json"))
        for c,m in data["cases"].items():
            key=label+" / "+c
            meta[key]=dict(dataset=label+" / "+m["dataset"],geometry=m["geometry"])
            if c in data["derived_instances"]:
                rows=read(folder/data["derived_instances"][c]["file"])["nodes"]
            else:rows=core.load_original(c,conf)["nodes"]
            fields[key]=features(rows)
    totals=source_means(fields,meta)
    (HERE/"domain_profile.json").write_text(json.dumps(totals,indent=2),encoding="utf8")
    lines=["# 冷链输入的来源差异","","先在单例上统计客户，再在每个来源内将同几何时窗变体取平均、对几何等权。距离与时间使用统一冷链模型单位。需求占比是单个客户占车辆容量的比例。","",
        "| 来源 | 仓到客户距离中位数 km | 最近客户距离中位数 km | 时窗宽度中位数 min | 平均需求占比 | 平均服务时长 min |",
        "|---|---:|---:|---:|---:|---:|"]
    for label,d in totals.items():
        lines.append(f"| {label} | {d['median_depot_distance_km']:.1f} | {d['median_nearest_customer_km']:.1f} | {d['median_window_minutes']:.1f} | {100*d['mean_capacity_share']:.1f}% | {d['mean_service_minutes']:.1f} |")
    dk=totals["Reserved validation / PDPCDPG-D100"];train=totals["VRPTWQ train"]
    lines+=["","| 来源 | 与近邻组成双客即超载 | 与两近邻组成三客即超载 |","|---|---:|---:|"]
    for label,d in totals.items():
        lines.append(f"| {label} | {d['near_pair_capacity_rejection']:.1%} | {d['near_triple_capacity_rejection']:.1%} |")
    lines+=["","每名客户取欧氏距离最近的八名客户，枚举其 8 个双客组合及 28 个三客组合。表中比例只用货量即可判定，时窗和品质可能进一步排除路线。"]
    lines+=["",f"丹麦预留请求平均占车辆容量的 {dk['mean_capacity_share']:.1%}，训练客户平均占 {train['mean_capacity_share']:.1%}。两批客户服务时长分别为 {dk['mean_service_minutes']:.0f} 和 {train['mean_service_minutes']:.0f} 分钟。这里的零分钟取自原配送阶段的时间递推，每托盘装卸耗时属于转运仓。多项输入同时变化，名次差不能归因于其中某一项。[原始模型、单位和时序](https://backend.orbit.dtu.dk/ws/files/206144611/Wen_2009_Vehicle_routing_with_cross_docking.pdf) · [作者易腐品基准 v2](https://data.mendeley.com/datasets/93w8ynrk59/2)"]
    audit=HERE/"reserved_validation/source_audit.json"
    if audit.is_file():
        counts=read(audit).get("perishable_unique_location_demand_tuples_outside_five_200_request_parents")
        if counts:
            lines+=["",f"按配送坐标、取货坐标及托盘数合并比对全套文件，五个 200 请求母实例之外，静态目录只多出 {counts['Instances_static']} 个不同请求，滚动目录只多出 {counts['Instances_Rolling_Horizon']} 个。同一母实例的两个 100 请求子集可检验新订单，但两者仍共享配送背景。[原件逐客户核对](reserved_validation/source_audit.json)"]
    (HERE/"来源输入对照.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines[5:-1]))
if __name__=="__main__":main()
