"""Render the paper overview from the complete paired experiments and audit receipts."""
import csv
import json
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEV = HERE.parent / "maskco_followup_dev_20261120/results"
OUR, MATCH = "ORDER_SPLIT", "ORDER_SPLIT_D"
HIST = ("PDPCDPG-D100", "XTW-p100", "GH-p100")
NEW = ("VRPPDL-p100", "XTW-p100")
EXTRA = "GH-p100-block2"
RETAIL = "RWVRP-p100"
LEGACY_LINK = "最新证据见[十二个外部方法的同期比较、前瞻验证与论文结论](../unified_external_20261120/论文实验总览.md)。"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def formatted(number):
    return "缺测" if number is None else f"{number:+,.2f}"


def paired_summary(phase):
    folder = RESULTS / phase
    art = read(folder / "analysis.json")
    audit = read(folder / "independent_audit.json")
    if art["status"] != "COMPLETE" or audit["status"] != "PASS" or art["endpoints"] != audit["endpoints"]:
        raise ValueError(f"{phase}: results or independent replay incomplete")
    return art, audit


def rows_for_figure(benchmark, prospective, retail):
    return [
        ("既有丹麦", "DK perishable", 0, benchmark["panels"][HIST[0]]),
        ("既有 XTW", "XTW", 0, benchmark["panels"][HIST[1]]),
        ("既有 GH", "GH", 0, benchmark["panels"][HIST[2]]),
        ("既有 XTW 低重叠复核", "XTW low overlap", 0, benchmark["panels"]["XTW 低重叠"]),
        ("新荷比配送日", "BE/NL order days", 1, prospective["panels"][NEW[0]]),
        ("新 XTW 几何", "XTW geometries", 1, prospective["panels"][NEW[1]]),
        ("新商超城市", "RWVRP retail cities", 1, retail["panels"][RETAIL]),
        ("GH 同母例新客户块", "GH customer blocks", 2, prospective["panels"][EXTRA]),
    ]


def figure(panels):
    colors = ("#394E60", "#0072B2", "#A44C16")
    markers = ("o", "s", "D")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.65), sharey=True)
    yy = (8.5, 7.5, 6.5, 5.5, 4, 3, 2, .5)
    for ax, title, contrast in zip(axes, ("Strongest external method", "Matched distance ablation"), ("external", MATCH)):
        for y, (_, _, index, result) in zip(yy, panels):
            m = result["strongest_external"] if contrast == "external" else contrast
            effect = result["contrasts"][m]
            lower, upper = effect["ci95"]
            ax.plot((lower, upper), (y, y), color=colors[index], lw=1.6)
            ax.plot(effect["mean"], y, marker=markers[index], ms=5.2, color=colors[index])
        ax.set_title(title,loc="left",fontsize=11,fontweight="semibold")
        ax.axvline(0,ls="--",lw=.9,color="#555555")
        for boundary in (4.75,1.25):
            ax.axhline(boundary,color="#CCCCCC",lw=.65,zorder=0)
        ax.set_xlabel("Paired net benefit difference (U)", fontsize=9)
        ax.tick_params(axis="both",labelsize=8,length=3)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.grid(axis="x",color="#dddddd",lw=.55)
        ax.set_axisbelow(True)
        ax.margins(x=.09)
    axes[0].set_yticks(yy,[p[1] for p in panels])
    axes[0].set_ylim(0,9.05)
    fig.subplots_adjust(left=.18,right=.985,bottom=.16,top=.90,wspace=.16)
    fig.savefig(HERE/"paired_benefit.png",dpi=380,bbox_inches="tight",pad_inches=.10,facecolor="white")
    plt.close(fig)


def ranks_table(panels,methods):
    selected = [panels[i] for i in (0,1,2,4,5,6,7)]
    out = ["| 方法 | 既有丹麦 | 既有 XTW | 既有 GH | 新配送日 | 新 XTW | 新商超城市 | GH 压力 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in [OUR,*methods]:
        ranks=[p[3]["external_plus_candidate_ranks"][m] for p in selected]
        out.append("| "+("**MASKCO**" if m == OUR else m)+" | "+" | ".join(map(str,ranks))+" |")
    return out


def trajectories(report):
    folder = RESULTS / "trajectory01"
    paired = csv_records(folder/"paired_seeds.csv")
    if report["status"]!="COMPLETE" or report["endpoints"]!=360:
        raise ValueError("Incomplete search trajectory")
    lines = ["| 来源 | 对照 | 搜索 5 秒 | 15 秒 | 30 秒 | 60 秒 | 60 秒胜种子 /10 | 从进程启动计 60 秒 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in HIST:
        for m in (MATCH,"ALNS"):
            stats = report["sources"][s][m]
            series = []
            for t in (5,15,30,60):
                v=[r["paired_delta_U"] for r in paired if r["source"]==s and r["control"]==m and int(r["readout_s"])==t]
                if len(v)!=30:
                    raise ValueError("Incomplete paired trajectory matrix")
                series.append(fmean(map(float,v)) if all(x!="" for x in v) else None)
            wins=sum(v>1e-6 for v in stats["by_seed_at_60"].values())
            cold=report["cold_start"][s][m]
            lines.append(f"| {s} | {m} | "+" | ".join(formatted(v) for v in series)+
                         f" | {wins} | {formatted(cold['delta_U'])}（{cold['paired']}/30） |")
    return lines


def scenario_ranges():
    rows=csv_records(HERE/"economic_sensitivity/cost_scenarios.csv")
    limits={}
    for s in HIST:
        for m in (MATCH,"ALNS"):
            corners=[float(r["mean_delta_U"]) for r in rows if r["panel"]==s and r["control"]==m and
                     all(float(r[key]) in (.8,1.2) for key in ("quality_price_factor","distance_price_factor","cold_price_factor"))]
            if len(corners)!=8:
                raise ValueError("Incomplete eight-corner economic stress")
            limits[s,m]=f"{formatted(min(corners))} 至 {formatted(max(corners))}"
    return limits


def main():
    old,oa=paired_summary("benchmark01")
    new,na=paired_summary("prospective01")
    retail,ra=paired_summary("retail01")
    joint=read(RESULTS/"three_sources.json")
    tracks=read(RESULTS/"trajectory01/analysis.json")
    coverage=read(RESULTS/"prospective01/rank_coverage.json")
    p=retail["panels"][RETAIL]
    top=p["strongest_external"]
    cg=p["contrasts"][top]["by_geometry"]
    city_ranks=[1+sum(p["contrasts"][m]["by_geometry"][g]<=1e-6 for m in retail["external_methods"]) for g in cg]
    full=csv_records(RESULTS/"retail01/endpoints.csv")
    expenses=[sum(float(r[f]) for f in ("quality_loss_U","distance_km","refrigeration_U")) for r in full if r["method"]==top]
    if len(cg)!=8 or len(expenses)!=24:
        raise ValueError("Missing retail cities or seeds")
    difference=p["contrasts"][top]["mean"]
    city_delta = {row["city"]: float(row["delta_panel_best"]) for row in csv_records(RESULTS/"retail01/cities.csv")}
    bay = city_delta["SF Bay Area"]
    other_cities = fmean(value for city, value in city_delta.items() if city != "SF Bay Area")
    coverage[RETAIL]=dict(geometries=len(cg),top3_13=sum(n<=3 for n in city_ranks),first_13=city_ranks.count(1),
                          strongest_external=top,mean_delta_U=difference,controllable_loss_reduction_percent=100*difference/fmean(expenses))
    col=read(DEV/"columnprobe01/analysis.json")
    degree=read(DEV/"degree01/analysis.json")
    if not (col["status"]==degree["status"]=="COMPLETE" and len(old["external_methods"])==12 and new["external_methods"]==old["external_methods"]):
        raise ValueError("Missing mechanism delivery or mismatched comparators")
    case_table=HERE/"实例与方法覆盖.md"
    if not case_table.is_file():
        raise ValueError("Missing instance coverage inventory")
    panels=rows_for_figure(old,new,retail)
    main_new=[new["panels"][s] for s in NEW]+[p]
    r_new=[v["external_plus_candidate_ranks"][OUR] for v in main_new]
    r_old=[old["panels"][s]["external_plus_candidate_ranks"][OUR] for s in HIST]
    forward_coverage = [coverage[name] for name in (*NEW, RETAIL)]
    units = sum(c["geometries"] for c in forward_coverage)
    top3_units = sum(c["top3_13"] for c in forward_coverage)
    first_units = sum(c["first_13"] for c in forward_coverage)
    lines=["# MASKCO 静态冷链掩码重构：论文实验总览","",
           f"锁版 ORDER_SPLIT 与 **12 个外部方法**逐例配对，使用同一机器、逐块固定的物理核、相同可行初解和 60 秒在线预算。"
           f"既有丹麦、XTW、GH 的来源均值名次依次为 {'、'.join(str(rank)+'/13' for rank in r_old)}。在新荷比配送日、新 XTW 几何和新商超城市，来源均值分别排 **{r_new[0]}/13、{r_new[1]}/13、{r_new[2]}/13**。"
           f"逐例有 **{top3_units}/{units}** 进入前三，其中 **{first_units}/{units}** 排第一。加上三项内部消融后，三个前瞻来源的均值名次依次为 {'、'.join(str(v['all_ranks'][OUR])+'/16' for v in main_new)}。","",
           "## 方法与评价","",
           "每份输入有 100 个客户。所有客户恰好配送一次，并满足 20 吨容量、时间窗、回仓和最低品质 0.8 的硬约束。"
           "车队净效益 U 等于品质折算交付价值减去里程与制冷费用。固定的 MASKCO 读取 40 客户掩码和剩余路线邻接，生成客户边分。"
           "边分用于提出客户插回和跨路线访问顺序，动态规划按真实经济目标选择分车点，整车队经严格复算后才接受。"
           "同规则距离消融 ORDER_SPLIT_D 保留相同的掩码、插回、分车和接受机制，以距离排序构造候选。[模型与旧版消融](../maskco_perishable_external_20261119/总报告.md)","",
           "## 同批对比","",
           "| 面板 | 实例 / 几何或订单日 | 名次 /13 | 同期最强外部法 | 对它 ΔU | 相对 % | 对同规则距离 ΔU | 热图胜几何或日 |",
           "|---|---:|---:|---|---:|---:|---:|---:|"]
    for label,_,_,p in panels:
        m=p["strongest_external"]; external=p["contrasts"][m]; ab=p["contrasts"][MATCH]
        lines.append(f"| {label} | {p['cases']}/{p['geometries']} | {p['external_plus_candidate_ranks'][OUR]} | {m} | {formatted(external['mean'])} | {external['pct_relative']:+.3f}% | {formatted(ab['mean'])} | {ab['wins']}/{p['geometries']} |")
    lines+=["", "![锁定 MASKCO 相对最强外部法及同规则距离的几何配对差和 95% 区间](paired_benefit.png)","",
            "图中点为各面板平均配对差，横线为固定三次种子、整几何或配送日重采样 20,000 次的 95% 描述区间；正值表示 MASKCO 净效益更高。"
            "上四行为已测案例复测，主 XTW 与低重叠复核共享 7 例；方形是新订单日、新几何与新城市，菱形是 GH 压力测试。每行的最强外部法见表。","",
            "### 十二个外部方法的同场名次","",*ranks_table(panels,old["external_methods"]),"",
            "ALNS、VNS 采用冷链搜索算子，NDS-Econ 用作者权重选择破坏集合并做经济重插。CaR、DeepACO、POMO、AttentionModel 使用项目冷链训练权重。"
            "RouteFinder、CaDA、MVMoE、RRNCO 使用作者预训练权重和冷链可行性掩码；PyVRP 的 HGS 内层使用里程、制冷及品质损失下界，产出的可行车队按完整冷链目标择优。"
            "MASKCO 使用既有 1,440 步冷链适配权重。表中排名对应这些实现和资源；三项内参照 ORDER_SPLIT_D、DIST、DIST_SLOT 另见十六法全表。"
            "[既有面板全表](results/benchmark01/RESULTS.md) · [新面板全表](results/prospective01/RESULTS.md) · [商超八城全表](results/retail01/RESULTS.md) · [近年方法和协议出处](../recent5_20261120/实验登记.md)","",
            "### 新配送日与新几何的覆盖","",
            "| 前瞻分组 | 单位数 | 逐几何或日进入前三 /13 | 第一 /13 | 同期最强外部法 | 对它 ΔU | 可控经济损失减少 % |",
            "|---|---:|---:|---:|---|---:|---:|"]
    for s in (NEW[0],"比利时配送日","荷兰配送日",NEW[1],RETAIL,EXTRA):
        c=coverage[s]
        lines.append(f"| {s} | {c['geometries']} | {c['top3_13']} | {c['first_13']} | {c['strongest_external']} | {formatted(c['mean_delta_U'])} | {c['controllable_loss_reduction_percent']:+.2f}% |")
    lines+=["", "可控经济损失是品质折价、里程与制冷三项之和。荷比两国来自同一一般货运运营商，本实验把订单投影到统一冷链参数。"
            "XTW 新面板与已登记实例及彼此重合的客户坐标均不超过 10 个。商超八城使用 OpenStreetMap 实际门店位置及作者建模的货量与收货时窗。GH 六例使用原母例第 101–200 个客户，合并为五组。[新实例准入](new_panel/准入.md) · [商超来源与字段](retail_source/准入.md)","",
            f"商超八城与 {top} 逐城比较为 {sum(v>0 for v in city_delta.values())} 胜、{sum(v<0 for v in city_delta.values())} 负，旧金山湾区的差额为 {formatted(bay)} U。描述性留出该城，余七城平均为 {formatted(other_cities)} U。[逐城市差额](results/retail01/CITY_RESULTS.md)","",
            "## 收益来自哪里","",
            "| 三个前瞻来源等权 | MASKCO 减对照 U | 95% 几何簇区间 U |","|---|---:|---:|"]
    for method in (MATCH,"DIST","DIST_SLOT","ALNS","PyVRP"):
        diff=joint["equal_source"][method]
        lo,hi=diff["interval_U"]
        lines.append(f"| {method} | {formatted(diff['mean_U'])} | [{formatted(lo)}, {formatted(hi)}] |")
    lines+=["","ORDER_SPLIT_D 的差额衡量用热图提出候选所取得的总增量，包含神经预测占用时间后减少的搜索机会；DIST 与 DIST_SLOT 的差额还包括经济重构和分车机制。","",
            "原始冷链适配 MASKCO 另有八例四臂同批开发比较：锁版在 7/7 个几何上高于原始适配；另保留 26 个旧案例的分批桥接结果。"
            "原模型计作内部参照。[原始 MASKCO 与机制消融](原始MASKCO与机制消融.md)","",
            "| 既有开发实验 | 比较 | 丹麦 ΔU | XTW ΔU | GH ΔU | 预登记结论 |",
            "|---|---|---:|---:|---:|---|"]
    for label,detail,a,contrast,url in (
        ("保留首次神经定价全部路线","神经全列 − 距离全列",col,"distance","columnprobe01"),
        ("按剩余入度校准热图","校准热图 − 校准距离",degree,"distance_balanced","degree01")):
        vals=[a["sources"][s]["contrasts"][contrast]["mean"] for s in HIST]
        lines.append(f"| [{label}](../maskco_followup_dev_20261120/results/{url}/RESULTS.md) | {detail} | "+" | ".join(formatted(v) for v in vals)+f" | {'晋级' if a['passed'] else '停止'} |")
    limits=scenario_ranges()
    lines+=["", "两项开发各取 11 个旧实例，分别独立复算 88 个重组终点和 220 个方法终点。"
            "首次定价保留的额外路线未改善强终态。剩余度校准改变约三至四成候选，优于同规则距离的几何数为 1/5、0/3、1/3。","",
            "另在早先同批四法的 360 个已选车队终点上固定路线与时刻，将品质、里程、制冷三个单价各改变 ±20%。表内区间取八种组合的极值。[成本分解和情景结果](economic_sensitivity/RESULTS.md)","",
            "| 固定方案净收益差 U | 丹麦 | XTW | GH |","|---|---:|---:|---:|"]
    for m in (MATCH,"ALNS"):
        lines.append(f"| MASKCO − {m} | "+" | ".join(limits[s,m] for s in HIST)+" |")
    gaps={r["source"]:r for r in csv_records(HERE/"bounds/source_gaps.csv")
          if r["batch"] in ("prospective01","retail01") and r["method"]==OUR}
    lines+=["","## 净效益还能提高多少","",
            "以本项目净效益作界：纳入可能邻边的乐观品质、里程和制冷损失，求解允许子圈的最小费用指派。"
            "当前可行解与松弛上界之差给出仍可能改进的上限；当批最佳可行解取同例十六法、三种子的已实现最高值。","",
            "| 前瞻来源 | 距当批最佳可行 U | 距乐观净效益上界 U | 上界差占已发生损失及费用 |","|---|---:|---:|---:|"]
    for s in (*NEW,RETAIL,EXTRA):
        g=gaps[s]
        lines.append(f"| {s} | {float(g['to_best_feasible_U']):,.2f} | {float(g['to_upper_U']):,.2f} | {float(g['to_upper_pct_of_achieved_loss_cost']):.2f}% |")
    case_bounds=csv_records(HERE/"bounds/case_bounds.csv")
    n716=next(r for r in case_bounds if r["case"]=="xtw_n716_p100")
    lines+=["",f"{len(case_bounds)} 个百客实例均按相同经济目标构造上界；每个既有来源另取一个五客户子问题，枚举最优值核对界的方向。"
            f"XTW-n716 的名义货值为 {float(n716['nominal_U']):,.2f} U，在必须全覆盖的条件下，最优净效益的上界也只有 {formatted(float(n716['net_upper_U']))} U。"
            "这例的算法节约体现为降低亏损。[全体界限与逐方法差距](bounds/RESULTS.md)"]
    lines+=["","## 重复性与首次调用","",
            "每个既有来源取三个代表几何，每个几何运行十颗种子。下表为 MASKCO 减对照的来源平均 U。冷启动列纳入导入、加载和编译，括号给出完整配对数。","",
            *trajectories(tracks),"",
            f"共 {tracks['endpoints']} 个终点，独立复算 {tracks['audited_snapshots']:,} 个已接受时点车队。"
            f"三个十六法批次分别完成 {oa['endpoints']:,}、{na['endpoints']:,} 和 {ra['endpoints']:,} 个终点，逐条复算 {oa['routes']+na['routes']+ra['routes']:,} 条路线。"
            "[完整轨迹和冷启动](results/trajectory01/RESULTS.md) · [全部实例覆盖和未评分样本](实例与方法覆盖.md)","",
            "单 CPU 新进程的交叉开发试验中，复用编译产物将训练例 r101 从脚本入口到两次真实热图的墙钟中位数从 61.45 秒缩至 13.02 秒（−78.81%）。"
            "六次测量中对应边分的最大偏差为 0，前八名次完全一致；一次性编译耗时 65.05 秒。"
            "另一项 24 终点的独立进程开发复核，原流程和复用编译流程的前 60 秒可交付率分别为 3/6、6/6，加载至搜索由 59.88 秒降至 12.36 秒。"
            "复用版在这组开发几何上较同批 ALNS 高 37.33 U，较无需加载的同规则距离低 3.53 U。主排名采用锁版流程。"
            "[首次调用计时和开发复核](../maskco_followup_dev_20261120/startup/RESULTS.md)","",
            "## 论文结论",""]
    passed=[s for s,a in joint["primary"].items() if a["stable_heat"]]
    heat=joint["equal_source"][MATCH]
    alns=joint["equal_source"]["ALNS"]
    mean_best="、".join(formatted(v["contrasts"][v["strongest_external"]]["mean"]) for v in main_new)
    lines += [f"**外部比较。** 三个前瞻主面板上，MASKCO 分别较各自最强外部法 {mean_best} U。"
              f"三个来源等权时，整套方法比 ALNS 高 {formatted(alns['mean_U'])} U，95% 描述区间 [{formatted(alns['interval_U'][0])}, {formatted(alns['interval_U'][1])}] U。"
              f"固定同一套修复和分车规则，仅使用热图替换距离排序的增量是 {formatted(heat['mean_U'])} U，95% 描述区间 [{formatted(heat['interval_U'][0])}, {formatted(heat['interval_U'][1])}] U。"
              f"前瞻稳定热图增量的均值、严格多数几何与至少两种子条件通过 {len(passed)}/3 个主面板"
              + ("（"+"、".join(passed)+"）。" if passed else "。"),"",
              "**适用范围。** 论文可以分别报告锁版算法在 100 客户、给定冷链参数和单核时限下的排名，以及 MASKCO 提出掩码重构候选的配对增量。"
              "新 XTW 检验跨客户几何，荷比配送日检验同运营商的跨日迁移，RWVRP 检验向真实城市门店布局和作者建模订单的迁移。GH 另检验同母例的跨客户块迁移。","",
              "[三来源独立判定与区间](results/三来源前瞻结果.md) · [实验设计及研究范式](实验设计与文献依据.md) · [原目标、参数与历次消融](../maskco_perishable_external_20261119/总报告.md)"]
    figure(panels)
    (HERE/"论文实验总览.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    legacy=HERE.parent/"maskco_perishable_external_20261119/总报告.md"
    text=legacy.read_text(encoding="utf-8")
    if LEGACY_LINK not in text:
        title,rest=text.split("\n",1)
        legacy.write_text(title+"\n\n"+LEGACY_LINK+"\n"+rest,encoding="utf-8")


if __name__=="__main__":
    main()
