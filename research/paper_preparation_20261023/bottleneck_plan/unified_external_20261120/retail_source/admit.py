"""Project author-provided retailer instances and check the common feasible initial solution."""
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from urllib.parse import quote

from fetch_sources import CITIES, REV

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DEV=STUDY.parent/"maskco_perishable_external_20261119"
sys.path.insert(0,str(DEV))
import prepare_panel
from freight_admission import DAYS, arc, delivery_day


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def center(nodes):
    hub=nodes[0]
    return Counter((round(r[1]-hub[1],5),round(r[2]-hub[2],5)) for r in nodes[1:])


def common_counts(new,known):
    return max((sum((new & value).values()) for value in known.values()),default=0)


def main():
    locations={}
    for country in ("BE","NL"):
        for day in DAYS:
            for n in (0,1):
                name=f"{country}_{day}_{n}_F"
                locations[name]=Counter(delivery_day(name)[2][1:])
    canonical={}
    for folder in (DEV,DEV/"reserved_validation",STUDY/"new_panel",DEV.parent/"maskco_heatmap_calibration_20261119"):
        catalog=read(folder/"heldout_sources.json")
        for c,e in catalog["derived_instances"].items():
            canonical[c]=center(read(folder/e["file"])["nodes"])
    sealed=read(DEV/"future_validation/heldout_sources.json")
    protected_geometry={r["geometry"] for r in sealed["cases"].values()}
    entries=[];metadata={}; diagnostics={}
    for city,tag in CITIES.items():
        origin=HERE/"raw/data"/city
        rows=read(origin/"locations.json")
        vehicles=read(origin/"vehicles.json")
        if len(rows)!=1+int(tag[2:]) or [r["id"] for r in rows]!=list(range(len(rows))):
            raise ValueError("Source customers differ from the registered variant: "+city)
        capacities={v["capacity"] for v in vehicles}
        shifts={tuple(v["driver_tw"]) for v in vehicles}
        if len(capacities)!=1 or len(shifts)!=1:
            raise ValueError("A homogeneous shift or vehicle capacity is undefined: "+city)
        cap,shift=capacities.pop(),shifts.pop()
        hub=rows[0];service=hub["service_time"]
        if hub["location_type"]!="depot" or hub["demand"] or not 0<service or set(shift)!=set(hub["time_window"]):
            raise ValueError("Unaligned driver and depot hours: "+city)
        left=max(shift[0],hub["time_window"][0])+service
        right=min(shift[1]-service,hub["time_window"][1])
        customers=rows[1:101]
        ll=[(r["y_coor"],r["x_coor"]) for r in [hub,*customers]]
        def map_xy(point):
            distance,bearing=arc(ll[0],point)
            return distance*math.sin(bearing),distance*math.cos(bearing)
        xy=[map_xy(point) for point in ll]
        err=max(abs(math.dist(xy[i],xy[j])/arc(ll[i],ll[j])[0]-1)
                for i,j in combinations(range(101),2) if ll[i]!=ll[j])
        nodes=[[0,0.,0.,0.,left/60,right/60,0.]]
        for i,c in enumerate(customers,1):
            assert 0<c["demand"]<=cap and c["location_type"] in ("supermarket","convenience")
            nodes.append([i,*xy[i],20000*c["demand"]/cap,c["time_window"][0]/60,
                          c["time_window"][1]/60,c["service_time"]/60])
        case=f"rw_{tag.lower()}_p100"
        raw_counts=Counter(ll[1:])
        matching=common_counts(raw_counts,locations)
        canonical_matches=common_counts(center(nodes),canonical)
        locations[city]=raw_counts
        canonical[case]=center(nodes)
        url="https://github.com/unizg-fer-lares/RWVRP_benchmark/blob/"+REV+"/data/"+quote(city)+"/locations.json"
        entries.append((case,"RWVRP-p100",nodes,tuple(sorted(ll)),url))
        metadata[case]=dict(city=city,origin=f"RW-{tag}-VRP",evidence_role="author_modelled_grocery_demand_real_store_geography")
        diagnostics[case]=dict(original_customers=len(rows)-1,source_ids=[1,100],capacity_pallets=cap,
                               original_source_vehicles=len(vehicles),source_shift_seconds=shift,warehouse_service_seconds_each_end=service,
                               model_dispatch_return_window_minutes=[left/60,right/60],
                               store_types=dict(Counter(c["location_type"] for c in customers)),
                               max_projection_relative_error=err,raw_customer_coordinate_overlap=matching,
                               centered_customer_coordinate_overlap=canonical_matches,raw_source=url)
    prepare_panel.HERE=HERE
    catalog,checks,_=prepare_panel.keep(entries,set())
    for c,info in metadata.items():
        catalog["cases"][c].update(info)
    valid=(len(catalog["cases"])==8 and len({r["geometry"] for r in catalog["cases"].values()})==8 and
           not (set(catalog["cases"]) & set(sealed["cases"])) and
           all(r["geometry"] not in protected_geometry for r in catalog["cases"].values()) and
           all(not v["infeasible_singletons"] and v.get("initializer_routes",0)>0 for v in checks.values()) and
           all(z["max_projection_relative_error"]<=.001 and max(z["raw_customer_coordinate_overlap"],z["centered_customer_coordinate_overlap"])<=10 for z in diagnostics.values()))
    status="PASS" if valid else "STOP_ADMISSION"
    write("heldout_sources.json",catalog)
    write("heldout_admission.json",dict(status=status,cases=checks,by_source={"RWVRP-p100":{"cases":8,"groups":8}}))
    write("selection.json",dict(status=status,solver_endpoints_at_admission=0,source_commit=REV,cases=diagnostics))
    print(status,[(c,k.get("initializer_routes"),list(k["infeasible_singletons"])) for c,k in checks.items()])
    lines=["# 商超配送基准的八城准入","",f"**{status}**。实际门店位置来自 OpenStreetMap，货量与收货时窗由原作者按门店类型建模。"
           "每城取原编号前 100 客户，基准同质车辆载量为 18 托盘。统一模型按容量占比换算到 20 吨；仓库出发与回仓各预留作者规定的 30 分钟作业。"
           "冷链效益从出发计至回仓。真实位置经以仓库为中心的公里投影进入统一模型。","",
           "| 作者实例 | 城市或区域 | 原客户数 | 百客中超市 / 便利店 | 共同初解车辆 | 不可服务客户 |",
           "|---|---|---:|---:|---:|---:|"]
    for c,r in diagnostics.items():
        m=metadata[c];t=r["store_types"]
        lines.append(f"| {m['origin']} | {m['city']} | {r['original_customers']} | {t.get('supermarket',0)} / {t.get('convenience',0)} | {checks[c].get('initializer_routes','—')} | {len(checks[c]['infeasible_singletons'])} |")
    lines+=["",f"单客可行性按原始 800 单检查，各城共同初解独立复算。公里投影对大圆距离的最大相对偏差为 {100*max(r['max_projection_relative_error'] for r in diagnostics.values()):.5f}%。"
            f"与已登记的可比经纬度或仓库中心相对坐标比较，最多完全重合 {max(max(r['raw_customer_coordinate_overlap'],r['centered_customer_coordinate_overlap']) for r in diagnostics.values())} 个客户。"
            "作者的道路时间矩阵、原车队数量和交通模式是原比较口径；本研究使用统一速度、直线公里和车辆数上限。","",
            "研究组公开说明允许按应用选取子集开展算法比较，并给出论文引用时的实例命名。门店位置来源按 OpenStreetMap 贡献者署名；文档引用原作者，不另行发布原始库或原始商业道路矩阵。"
            "[作者固定版本](https://github.com/unizg-fer-lares/RWVRP_benchmark/tree/"+REV+") · [OpenStreetMap 署名及许可](https://www.openstreetmap.org/copyright)","",
            "[输入预登记](输入选择预登记.md) · [几何和字段逐例记录](selection.json) · [单客与初解审计](heldout_admission.json)"]
    (HERE/"准入.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if not valid: raise SystemExit(2)


if __name__=="__main__":
    main()
