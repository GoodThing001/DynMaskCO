"""Index completed source-level comparisons, mechanism ablations and research costs."""
from datetime import datetime
from pathlib import Path
import json
from statistics import mean
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def f(x):return f"{x:+.2f}"


def recent_comparison(directory):
    summary=directory/"recent_summary.json"
    if not summary.is_file() or not (directory/"RESULTS.md").is_file():
        return [], ""
    study=read(summary)
    if study["status"]!="COMPLETE" or study["endpoints"]!=540 or study["independent_audit"]["status"]!="PASS" or study["independent_audit"]["endpoints"]!=540:
        return [], ""
    method="ORDER_SPLIT"
    primary=("PDPCDPG-D100","XTW-p100","GH-p100")
    low="XTW 低重叠"
    panels=study["panels"]
    lines=["","## 五项近期方法的同批比较","",
           "RouteFinder、CaDA、MVMoE、RRNCO 与 PyVRP 同锁版 MASKCO 在原案例、原种子和相同初解上重新计时。"
           "三个主来源共 27 例，低重叠 XTW 十例含其中七例。六法共 540 个终点通过独立复算。","",
           "| 面板 | 同期名次 /6 | 最强同期对照 | 相对差 U | 扩展名次 /18 | 扩展表最高对照 | 相对差 U |",
           "|---|---:|---|---:|---:|---|---:|"]
    for source in (*primary,low):
        panel=panels[source]
        scores=panel["scores"]
        counterparts=[]
        for family in (panel["ranks_recent"],panel["ranks_expanded"]):
            best=max(scores[m] for m in family if m!=method)
            names="、".join(m for m in family if m!=method and scores[m]>=best-1e-6)
            counterparts.append((names,f(scores[method]-best)))
        (new_name,new_gap),(all_name,all_gap)=counterparts
        lines.append(f"| {source} | {panel['ranks_recent'][method]} | {new_name} | {new_gap} | "
                     f"{panel['ranks_expanded'][method]} | {all_name} | {all_gap} |")
    base="../recent5_20261120/results/benchmark01"
    lines+=["","净效益差为 MASKCO 减该列最高对照，等分列出全部方法。十八法扩展表采用本轮六法与历史十二法的分批测量。",
            "",f"![各新对照的配对净效益差与几何整簇区间]({base}/paired_recent.png)","",
            f"[十八法得分、逐种子差值、初解回退与计算费用]({base}/RESULTS.md)"]
    strict_wins=sum(all(v["mean"]>1e-6 for v in panels[s]["paired"].values()) for s in primary)
    top_recent=sum(panels[s]["ranks_recent"][method]<=3 for s in primary)
    top_expanded=sum(panels[s]["ranks_expanded"][method]<=3 for s in primary)
    conclusion=(f"在已有的三个主来源上，MASKCO 的平均净效益在 {strict_wins}/3 个来源高于全部五项近期对照，"
                f"同期六法有 {top_recent}/3 个来源列前三，分批扩展至十八法后有 {top_expanded}/3 个来源列前三。"
                f"低重叠 XTW 十例对应名次为 {panels[low]['ranks_recent'][method]}/6 和 {panels[low]['ranks_expanded'][method]}/18。"
                "这些结果衡量整套求解器在已测面板上的表现；热图自身的净贡献以同规则距离和置乱消融为依据。")
    return lines,conclusion


def main():
    base=read(PREV/"results/confirm01/analysis.json")
    second=read(HERE/"results/replicate01/analysis.json")
    tests=[("首次","HYBRID",base,"VRPTWQ-C1"),("补充","HYBRID",second,"PDPCDPG-D100")]
    results=HERE/"reserved_validation/results/reserve02/analysis.json"
    if results.is_file() and (HERE/"reserved_validation/RESULTS.md").is_file():
        newer=read(results)
        method=read(HERE/"reserved_validation/locked_method.json")["candidate"]
        tests.append(("预留",method,newer,"PDPCDPG-D100"))
    latest=tests[-1];passed=latest[2]["passes_numerical_criteria"][latest[1]]
    supplement=HERE/"xtw_low_overlap/analysis.json"
    strict=read(supplement) if supplement.is_file() and len(tests)==3 else None
    verdict=(f"锁版 {latest[1]} 的三来源主验证与低重叠复核联合门槛{'通过' if strict['joint_pass'] else '未通过'}。"
             if strict else f"最近一次独立比较的三来源联合门槛{'通过' if passed else '未通过'}。")
    lines=["# MASKCO 冷链掩码重构与三来源外部验证","",verdict+"三来源均按同一规则比较可行冷链车队的总净效益，每格单物理核 60 秒，先对同几何的种子和实例取均值，再对几何等权。","",
           "| 历史外测 | 方法 | 数据来源 | 几何 | 名次 / 当批方法数 | 对同规则距离 U | 改善几何 | 对原 DIST U | 对原 DIST_SLOT U | 对 ALNS U |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for title,m,study,one in tests:
        for src in (one,"XTW-p100","GH-p100"):
            v=study["sources"][src];x=v["contrasts"][m];d=x[m+"_D"]
            lines.append(f"| {title} | {m} | {src} | {d['geometries']} | {v['ranks'][m]}/{len(v['scores'])} | {f(d['mean'])} | {d['wins']}/{d['geometries']} | {f(x['DIST']['mean'])} | {f(x['DIST_SLOT']['mean'])} | {f(v['external_contrasts']['ALNS']['mean'])} |")
    if strict:
        z=strict["panels"]["低重叠十例"];x=z["contrasts"];m=latest[1];d=x[m+"_D"]
        lines+=[f"| 客户低重叠复核 | {m} | XTW-p100 | 10 | {z['rank']}/13 | {f(d['mean'])} | {d['wins']}/10 | {f(x['DIST']['mean'])} | {f(x['DIST_SLOT']['mean'])} | {f(x['ALNS']['mean'])} |","",
                f"三来源主门槛与客户低重叠附加门槛合并判定为{'通过' if strict['joint_pass'] else '未通过'}。原 XTW 十例中三例与先前实例共享 71–100 个客户位置。补充验证保留七例并按官方顺序加入三个低重叠实例，固定模型、种子、13 法和资源。[低重叠复核](xtw_low_overlap/RESULTS.md)"]
        if (HERE/"reserved_validation/paired_effects.png").is_file():
            lines+=["","![锁定版本的热图增量、几何区间与十三法名次](reserved_validation/paired_effects.png)"]
    recent,conclusion=recent_comparison(HERE.parent/"recent5_20261120/results/benchmark01")
    lines+=recent
    lines+=["","## 目标、数据与方法","",
            "统一目标为品质折算交付价值减里程及制冷成本，单位 U。客户覆盖、容量、时间窗、回仓和最低品质为硬约束。项目比较使用 100 客户、20 吨、品质保留率至少 0.8；还原每条入选路线的最优合法发车和服务时刻后独立复算。","",
            "训练使用原 VRPTWQ 的 fold0 冻结 MASKCO 权重。每轮隐藏 40 个客户，将残缺路线的邻接矩阵和时间步交给模型。用同一严格经济目标从距离、原始热图、训练校准热图以及贪心、regret-2 重插中择优。一次预测可供该时段的多次重构复用。纯距离消融运行相同的掩码、插位、接受及计时规则。","",
            "VRPTWQ 的四个原始几何已用于训练或历次验证。新增 PDPCDPG 取作者静态易腐品数据的配送阶段。原坐标由米转公里，配送时窗沿用原值，托盘按 33 托盘占比转换成统一容量。XTW 和 GH 取仓库与前 100 个客户，按原时域与容量映射。每次确认保留原预定实例与种子。[作者数据 v2](https://data.mendeley.com/datasets/93w8ynrk59/2) · [DTU 字段说明](https://orbit.dtu.dk/en/datasets/vehicle-routing-with-cross-docking/) · [各来源输入与物理量](来源输入对照.md)","",
            "## 机制与消融","",
        "| 旧版固定插位 | 同样 60 秒，热图 − 距离 U | 同样重构次数，热图 − 距离 U | 距离额外迭代带来的收益 U | 热图前向占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (PREV/"DIAGNOSIS.md").read_text(encoding="utf8").splitlines():
        cols=[s.strip() for s in row.strip("|").split("|")]
        if len(cols)==6 and cols[0] in ("VRPTWQ-C2","XTW-p100","GH-p100") and "%" in cols[4]:
            lines.append("| "+" | ".join([cols[0],cols[1],cols[2].split("（")[0],cols[3],cols[4]])+" |")
    lines+=["","固定插位后，每次神经重构的质量更高，但每次都预测会消耗大部分时间。纯距离能在一分钟内完成约九至十倍的重构，抵消单次质量优势。新版本让多次重构复用一次热图，并用同一热图提出跨路线客户顺序。[完整轨迹配对和同掩码对照](../maskco_heatmap_calibration_20261119/DIAGNOSIS.md)","",
            "| 开发试验 | 来源 | 热图减同规则距离 U | 改善几何 | 逐种子 U |","|---|---|---:|---:|---|"]
    for phase,m in (("cachescreen01","CACHEHYBRID"),("staticscreen01","STATIC"),("disscreen01","DISAGREE"),("disdev01","DISAGREE"),("orderscreen01","ORDER_SPLIT"),("orderdev01","ORDER_SPLIT"),("workscreen01","WORKMASK"),("workdev01","WORKMASK"),("refreshscreen01","REFRESH"),("refreshdev01","REFRESH")):
        target=HERE/"results"/phase/"analysis.json"
        if not target.exists():continue
        study=read(target)
        for src in ("VRPTWQ-C2","XTW-p100","GH-p100"):
            d=study["sources"][src]["contrasts"][m][m+"_D"]
            lines.append(f"| {phase} / {m} | {src} | {f(d['mean'])} | {d['wins']}/{d['geometries']} | {' / '.join(f(v) for v in d['by_seed'].values())} |")
    lines+=["","CACHEHYBRID 在循环掩码上复用编码；STATIC 在每个实例上只编码一次，每轮仍用实际缺边解码。DISAGREE 保留试选的另一个可行解，优先遮去两解之间的不同邻接，同时对路线连通分量作经济择优。ORDER_SPLIT 在完整路线组内用热图提出访问序列，再按冷链经济收益选择分车点。",""]
    fac=HERE/"results/factorial01/效应分解.md"
    if fac.exists():lines+=["[掩码、路线组合和热图的八臂同批消融](results/factorial01/效应分解.md) · [固定案例和种子的重复性及内层试选诊断](重复性对照.md)",""]
    mask=HERE/"results/maskvalue01/analysis.json"
    if mask.exists():
        values=read(mask)
        lines+=["固定插回算子，仅由热图或距离在同一批 64 个十客掩码中选八个。数值是一次移除并重新插回后，能够保留的最佳正增益。每个来源先对几何等权聚合。","",
                "| 来源 | 共同初解，热图选掩码减距离 U | 强距离终态，同一差值 U | 强距离终态，遍历 64 个掩码的最大可达增益 U |","|---|---:|---:|---:|"]
        for src in ("VRPTWQ-C2","XTW-p100","GH-p100"):
            lines.append(f"| {src} | {f(values['initial_routes'][src]['delta_U'])} | {f(values['routes'][src]['delta_U'])} | {values['routes'][src]['oracle']:.2f} |")
        lines+=[""]
    costprior=HERE/"results/economicprobe01/analysis.json"
    if costprior.exists():
        z=read(costprior)["crossfit_train_to_other_geometry"]
        lines+=["在 1,440 个训练插位上，用可行双客路线相对两条单客路线的净效益构造有向先验。按一个训练几何选系数，在另一个几何评价。表中为与逐位置严格求优相比错失的净效益，越低越好。","",
                "| 拟合 → 评价 | 距离 U | 原校准热图 U | 有向经济先验 U | 加入热图后的 U | 选中热图权重 |","|---|---:|---:|---:|---:|---:|"]
        for src,other in (("R","RC"),("RC","R")):
            v=z[src];lines.append(f"| {src} → {other} | {v['distance']:.2f} | {v['legacy']:.2f} | {v['0.0']:.2f} | {v['chosen']:.2f} | {v['alpha']:.3g} |")
        lines+=["","局部双客经济量解释了大部分插位改进；这项诊断未建立经济先验之上的神经收益。",""]
    orientation=HERE/"directional_evidence.json"
    if orientation.exists():
        ds=read(orientation)["summary"]
        lines+=[f"基础 MASKCO 对称内积产生的插位边分，无法区分反向经过同一组三个客户的顺序。在旧 R、RC 训练几何的可行双客组合中，分别有 {ds['R']['share_one_given_any']:.1%}、{ds['RC']['share_one_given_any']:.1%} 只有一个方向可行；两向均可行时，两种顺序的平均绝对净效益差分别为 {ds['R']['mean_directional_gap_U']:.2f}、{ds['RC']['mean_directional_gap_U']:.2f} U。ORDER_SPLIT 对各有向连续段使用真实冷链收益。",""]
    orient_test=HERE/"orientation_ablation.json"
    if orient_test.exists():
        data=read(orient_test)["training_geometries"]
        lines+=["在共同初解的另外 1,440 个插位上，同一短名单分别引入无向双客经济值、方向可行性、方向经济值。下表为平均错失净效益 U。","",
                "| 训练几何 | 距离 | 无向双客经济值 | 加方向可行性 | 加方向经济值 |","|---|---:|---:|---:|---:|"]
        for s,z in data.items():
            lines.append("| "+s+" | "+" | ".join(f"{z[k]:.2f}" for k in ("distance","undirected_savings","directed_feasibility","directed_savings"))+" |")
        lines+=[""]
    if (HERE/"results/orderdev01/分车机制.md").exists():
        lines+=["每三轮选择 2–4 条相近的完整路线，合计至多 24 客，再补齐 40 客的掩码。MASKCO 给出可能相连的客户，生成多条访问顺序。对每个顺序，令 F(j) 为前 j 客分车后的最大净效益，枚举末条可行路线起点 i+1，取 F(i) 加该路线净效益的最大值。这样在每条给定顺序内求得最优分车，同时保留旧可行解。每个分组最多分配 0.30 秒。[开发面板分车次数与直接提高纪录的频率](results/orderdev01/分车机制.md)",""]
    head=HERE/"results/directedprobe01/analysis.json"
    lines+=["掩码预测缺边沿用 MaskCO，给定访问顺序后求最优分车沿用经典 Split。这里研究的问题是，在相同在线时间内，把一次掩码预测用于多次冷链重构，并在临近路线之间按真实品质与能耗重新划分，能否带来足以覆盖预测开销的净收益。[MaskCO 原论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html) · [Beasley 的顺序分车方法](https://people.brunel.ac.uk/~mastjjb/jeb/papers/omega_83.pdf)",""]
    if head.exists():
        z=read(head)["crossfit"]
        lines+=["| 方向特征交叉检验 | 原有向经济先验 U | 加距离邻居特征 U | 加 MASKCO 邻居特征 U | 神经优于距离的案例 |",
                "|---|---:|---:|---:|---:|"]
        for g,t in (("R","RC"),("RC","R")):
            r=z[g];v=r["loss"]
            lines.append(f"| {g} → {t} | {v['prior']:.3f} | {v['spatial']:.3f} | {v['neural']:.3f} | {r['neural_beats_spatial_cases']}/6 |")
        lines+=["","两臂使用相同的六系数校准，数值为短名单平均遗憾，越低越好。",""]
    placebo=HERE/"results/placebodev01/analysis.json"
    if placebo.exists():
        if (HERE/"development_effects.png").is_file():
            lines+=["![固定插位的时间代价与新方法的客户边置乱消融](development_effects.png)",""]
        study=read(placebo);lines+=["| 边分对应关系消融 | 真热图 − 置乱 U | 改善几何 | 置乱 − 距离 U |","|---|---:|---:|---:|"]
        for s in ("VRPTWQ-C2","XTW-p100","GH-p100"):
            z=study["sources"][s];x=z["aligned_shuffled"]
            lines.append(f"| {s} | {f(x['mean'])} | {x['wins']}/{x['geometries']} | {f(z['shuffled_distance']['mean'])} |")
        lines+=["","在相同前向成本、距离分层、掩码与插位规则下比较具体客户边的对应关系。[三臂完整开发归因](results/placebodev01/RESULTS.md)",""]
    operator=HERE/"results/opprobe01/analysis.json"
    if operator.exists():
        paired=read(operator)
        lines+=["在相同初解、早期解和强距离终态，先保留一组纯距离候选，再将第二组从独立随机种子的纯距离改成热图。每格是等工作量配对的几何平均净效益差。","",
                "| 第二组候选用途 | 丹麦 U | XTW U | GH U |","|---|---:|---:|---:|"]
        for op,name in (("insertion","普通重插"),("order","跨路线分车")):
            lines.append("| "+name+" | "+" | ".join(f(paired["operators"][op][s]["mean"]) for s in ("PDPCDPG-D100","XTW-p100","GH-p100"))+" |")
        lines+=["","两种用途在丹麦均未达到晋级门槛。[27 例固定工作量对照](results/opprobe01/RESULTS.md)",""]
    load_result=HERE/"results/loadprobe01/analysis.json"
    if load_result.exists():
        result=read(load_result)
        vals=[result["summary"][s]["means"] for s in ("PDPCDPG-D100","XTW-p100","GH-p100")]
        lines+=["加入接收路线载重和时窗后，真实边分相对纯物理特征在丹麦、XTW、GH 的单客短名单增量依次为 "+
                "、".join(f"{d['real']-d['physical']:+.3f}" for d in vals)+" U。按几何留组的晋级门槛"+("通过。" if result["passed"] else "未通过。")+"[载重交互与置乱对照](results/loadprobe01/RESULTS.md)",""]
        dk=vals[0]
        lines+=[f"丹麦每次重插中，物理排序六个额外插位取得 {dk['physical']:.3f} U，遍历其余全部插位也仅取得 {dk['oracle']:.3f} U。该短名单内留给进一步排序的平均空间为 {dk['oracle']-dk['physical']:.3f} U。",""]
    if (HERE/"freight_admission.json").is_file():
        freight=read(HERE/"freight_admission.json")
        data=freight["source_cases"]
        lines+=[f"另核对荷兰、比利时各五个配送日的原始零担订单。按固定仓库、备货时刻和订单顺序取 1,000 个客户，平面公里距离相对大圆距离的最大误差为 {max(v['max_projection_relative_error'] for v in data.values()):.4%}。这批一般货运样本已完成输入准入，求解终点数为 {freight['solver_endpoints']}。[来源依据与适用范围](补充来源与准入.md)",""]
    tail_file=HERE/"results/tailprobe01/analysis.json"
    if tail_file.is_file():
        tail=read(tail_file)
        lines+=["从距离排序的前两个路线对出发，给另外两个位置使用热图。在强距离终态，丹麦、XTW、GH 的可行尾段互换收益相对纯距离分别为 "+
                "、".join(f(row["means"]["real"]-row["means"]["distance"]) for row in (tail["sources"][s]["terminal"] for s in ("PDPCDPG-D100","XTW-p100","GH-p100")))+
                " U，三来源晋级门槛"+("通过。" if tail["passed"] else "未通过。")+"[固定四个路线对的逐来源比较与算子上界](results/tailprobe01/RESULTS.md)",""]
    sources=("PDPCDPG-D100","XTW-p100","GH-p100")
    cycle=HERE/"results/cycleprobe01/analysis.json"
    region=HERE/"results/regionprobe01/analysis.json"
    mix=HERE/"results/headcapture01/customer_mixture.json"
    if cycle.exists() and region.exists() and mix.exists():
        lines+=["## 热图能改进哪一步","","| 旧开发面板上的配对结果 | 丹麦 | XTW | GH |","|---|---:|---:|---:|"]
        for path,name in ((cycle,"强距离终态，三车各轮换一客，热图减距离 U"),(region,"强距离终态，54–60 客联合重构，热图减距离 U")):
            means=[read(path)["sources"][s]["terminal"]["means"] for s in sources]
            lines.append("| "+name+" | "+" | ".join(f(v["real"]-v["distance"]) for v in means)+" |")
        data=read(mix)["sources"]
        for arm,title in (("physical","纯物理"),("shuffled","分层置乱"),("mixture","MASKCO 客户混合")):
            vals=[data[s]["scores"][arm] for s in sources]
            lines.append("| "+title+"，客户后继交叉熵 / 前八命中 | "+" | ".join(f"{loss:.3f} / {hit:.1%}" for loss,hit in vals)+" |")
        lines+=["","在五折按几何留出的 12,960 个遮蔽客户上，使用物理模型估计回仓概率，再由 MASKCO 与物理模型共同预测下一位客户，三个来源都优于纯物理和置乱对照。预测指标单列，实际保留的车队净效益由同批计时求解衡量。",
                "","[回仓误差分解](results/headcapture01/回仓与客户后继.md) · [有向后继消融](results/headcapture01/客户混合.md) · [三路线轮换](results/cycleprobe01/RESULTS.md) · [区域重构](results/regionprobe01/RESULTS.md) · [单核前向](results/fpprobe01/RESULTS.md)",""]
        for stage,name in (("mixscreen01","11 例完整 60 秒开发筛查"),("mixdev01","27 例完整 60 秒开发复核")):
            path=HERE/"results"/stage/"analysis.json"
            if path.exists():
                data=read(path)
                lines += [f"| {name}，MASKCO 减参照 U | 丹麦 | XTW | GH |","|---|---:|---:|---:|"]
                for m,title in (("ORDER_SPLIT_D","同规则纯距离"),("PHYSICAL","纯物理头"),("ORDER_SPLIT","原掩码分车"),("DIST","DIST"),("DIST_SLOT","DIST_SLOT")):
                    lines.append("| "+title+" | "+" | ".join(f(data["sources"][s][m]["mean"]) for s in sources)+" |")
                lines += ["",f"{data['total']} 个终点通过独立复算。晋级门槛"+("通过。" if data["passed"] else "未通过。")+f"[逐几何和种子](results/{stage}/RESULTS.md)",""]
    if (HERE/"results/mixscreen01/danish_economics.json").is_file():
        lines+=["丹麦 a、b 的品质损失分别比同规则距离少 20.40 和 44.85 U，但里程费用分别多出 149.43 和 118.66 U；两例均在两个种子上落后。[逐实例净效益分解](results/mixscreen01/丹麦差额.md)",""]
        if (HERE/"results/mixscreen01/配对净效益.png").is_file():
            lines+=["![单核 60 秒的逐种子收益与丹麦五个实例的成本分解](results/mixscreen01/配对净效益.png)",""]
    if (HERE/"results/headcapture01/incident_slot.json").is_file():
        vals=read(HERE/"results/headcapture01/incident_slot.json")
        lines+=["从完整教师路线删去一名客户，在其余容量允许的位置中找回原位置。丹麦绕行距离的前八覆盖 100.00%，双向 MASKCO 覆盖 99.79%；XTW 分别为 90.54%、86.85%，GH 分别为 80.06%、82.00%。[25,270 个非单客车插位查询](results/headcapture01/双向插位.md)",""]
    for phase,key,title in (("repairpriority01","advance","按前后继预测确定首客"),("forkcredit01","passed","按完整重构的净效益学习首客")):
        target=HERE/"results"/phase/"analysis.json"
        if not target.is_file():continue
        v=read(target);scores=[v["sources"][s]["means"] for s in sources]
        lines += [f"{title}，相对严格经济 regret-2 的首客决策净效益差在丹麦、XTW、GH 分别为 "+"、".join(f(t["neural"]-t["economic"]) for t in scores)
                 +f" U。晋级门槛"+("通过。" if v[key] else "未通过。")+f"[全部对照](results/{phase}/RESULTS.md)",""]
    if (HERE/"future_validation/selection.json").is_file():
        lines+=["下一批 10 个荷比配送日、10 个 XTW 实例和 12 个 GH 实例已完成输入准入，其中 GH 有 5 个不同客户几何。XTW 每例与历史任一实例最多重合 8 个客户位置，GH 最多重合 1 个。配送得分与入选过程分开。[未来确认名单与来源](下一次验证样本.md)",""]
    if (HERE/"results/columns01/analysis.json").is_file():
        z=read(HERE/"results/columns01/analysis.json")
        m=z["means"]["terminal"]
        lines+=["丹麦高载重终态中，允许跨多条路线重新配车，每个区域给相同数目的有向路线候选。在十个强终态区域，热图相对距离的均值为 "+f(m["neural"]-m["distance"])+" U。[按容量重组与路线短名单](results/columns01/RESULTS.md)",""]
        full=HERE/"results/columns01/full_opportunity.json"
        if full.exists():
            rows=read(full)["rows"]
            lines+=["| 丹麦强距离终态，遍历区域内四客及以下路线 | 完整池可行净增 U | 净增上界 U | 距离短名单 U | 热图短名单 U |","|---|---:|---:|---:|---:|"]
            for key,label in (("neighbour","结合空间邻近"),("max_spare","最大化余量")):
                r=[v for v in rows if v["group"]==key]
                lines.append("| "+label+" | "+f(mean(v["gain"] for v in r))+" | "+f(mean(v["upper_gain"] for v in r))+" | "+f(mean(v["shortlist"]["distance"] for v in r))+" | "+f(mean(v["shortlist"]["neural"] for v in r))+" |")
            lines+=["","[完整枚举及其可达收益](results/columns01/完整路线池.md)",""]
        if (HERE/"results/columns01/complements.json").is_file():
            lines+=["距离池补全诊断中，丹麦 a 的一个区域需同时补回两条缺失路线才提高 106.62 U，c 的一个区域需三条一起补回才能提高 21.95 U。给每个客户和路线长度增加同数目的神经候选，十个邻近强终态的收益仍为零。[路线互补性](results/columns01/路线互补.md)",""]
        path=HERE/"results/columns01/residual_partition.json"
        if path.exists():
            v=read(path)
            r=[x for x in v["rows"] if x["group"]=="neighbour" and x["state"]=="terminal"]
            lines+=["提出跨路线客户组合后，按严格经济目标同步补全剩余客户。丹麦邻近强终态的热图减距离为 "+f(mean(x["gains"]["neural"]-x["gains"]["distance"] for x in r))+" U。扩大开发的门槛"+("通过。" if v["passed"] else "未通过。")+"[同步补全和直接保留余下路线的配对](results/columns01/残余配车.md)",""]
    joint=HERE/"results/jointcapture01/analysis.json"
    utility=HERE/"results/utilitycapture01/analysis.json"
    if joint.is_file():
        j=read(joint);s=j["stats"]
        lines+=["丹麦三个已知的整组换车收益分别要求至少同步移动 9、6、10 名客户。给定相应掩码，重构分别实现 +106.62、+21.95、+8.00 U。每例的六个旧随机掩码均遗漏必要变动。[最小移动集合](results/columns01/最小联合掩码.md)","",
                f"用整区 MASKCO 热图选择可移动客户，相对同规则距离在原十个邻近终态提高 {s['原邻近终态']['neural']-s['原邻近终态']['distance']:+.3f} U，在新增二十个锚点提高 {s['新增邻近锚点']['neural']-s['新增邻近锚点']['distance']:+.3f} U。邻近区域合计 {j['wins']}/5 个几何改善，纯物理排序的均值仍高于热图。[五臂对照与车队审计](results/jointcapture01/RESULTS.md)",""]
    if utility.is_file():
        u=read(utility);m=u["overall"]
        lines+=[f"改用车队净效益训练掩码排序头后，留一几何开发检验中，MASKCO 相对仅距离特征 {m['neural']-m['distance']:+.3f} U，相对载重、时窗与距离 {m['neural']-m['physical']:+.3f} U，相对置乱 {m['neural']-m['shuffled']:+.3f} U。[共同候选池与逐几何结果](results/utilitycapture01/RESULTS.md)",""]
        area=utility.parent/"area_selection.json"
        if area.exists():
            z=read(area)["levels"]
            lines+=[f"把热图改为选择路线区域，在共同初解相对距离 {z['initial']['neural']-z['initial']['distance']:+.2f} U，在强距离终态 {z['terminal']['neural']-z['terminal']['distance']:+.2f} U。[42 个互异区域的配对](results/utilitycapture01/区域选择.md)",""]
    context=HERE/"results/skeletoncontext01/analysis.json"
    if context.is_file():
        t=read(context)["levels"]
        lines+=[f"仅在存在两种不同正收益换车解的上下文，固定可行车队表，保留可见路线骨架的热图相对距离在初解、终态分别为 {t['initial']['means']['narrow']-t['initial']['means']['distance']:+.2f}、{t['terminal']['means']['narrow']-t['terminal']['means']['distance']:+.2f} U；终态可比较 {t['terminal']['geometries']}/5 个几何。[真实掩码边界对照](results/skeletoncontext01/RESULTS.md)",""]
    groups=HERE/"results/whole_group01/analysis.json"
    if groups.is_file():
        m=read(groups)["means"]
        lines+=["| 丹麦开发阶段 | 共同初解，热图减距离 U | 原距离终态 U | 第二种子距离终态 U |","|---|---:|---:|---:|"]
        segment=read(HERE/"results/segment_assignment01/analysis.json")["by_state"]
        lines.append(f"| 每车交换一个片段 | {segment['initial']['neural']-segment['initial']['distance']:+.2f} | {segment['terminal']['neural']-segment['terminal']['distance']:+.2f} | 未运行 |")
        lines.append("| 整组重新配车 | "+" | ".join(f"{m[s]['neural']-m[s]['distance']:+.2f}" for s in ("initial","neighbour","new_terminal"))+" |")
        lines+=["","已知三次强终态改善中，一条新路线分别汇集多达 4、3、4 辆旧车的客户。给定相应车组，整组重配可完整复现三次收益。十个第二种子终态区域中，原后继热图九次与纯物理模型选中相同车组。[同步片段交换](results/segment_assignment01/RESULTS.md) · [整车组重配](results/whole_group01/RESULTS.md)",""]
    partner=HERE/"results/vehicle_partner01/analysis.json"
    if partner.is_file():
        t=read(partner)
        lines+=["跨车同车伙伴预测以两次独立求解的高收益车队为监督，询问分属不同车辆的客户在经济解中是否会共同配送。按几何留出的三来源预测门槛"+("通过。" if t["passed"] else "未通过。")+"[交叉熵、前八召回和逐几何结果](results/vehicle_partner01/RESULTS.md)",""]
    views=HERE/"results/viewprobe01/analysis.json"
    pair=HERE/"results/viewpair01/analysis.json"
    carry=HERE/"results/carryprobe01/analysis.json"
    if views.exists() and pair.exists():
        view=read(views)["sources"];dual=read(pair)["sources"];follow=read(carry)["sources"] if carry.exists() else {}
        lines+=["| 开发来源 | 八视图平均热图减距离 U | 原向和反向两车队减双距离 U | 双车队改善几何 | 去掉收益最大几何 U | 一次八视图起点经 256 次距离重构后 U |","|---|---:|---:|---:|---:|---:|"]
        for s in ("PDPCDPG-D100","XTW-p100","GH-p100"):
            m=view[s]["means"];q=dual[s];last=follow[s]["stages"]["256"] if follow else None
            tail=f(last["eight"]-last["distance"]) if last else "未出结果"
            lines.append(f"| {s} | {f(m['eight']-m['distance'])} | {f(q['delta']['DD'])} | {q['wins']}/{len(q['geometries'])} | {f(q['leave_best_out'])} | {tail} |")
        lines+=["","原实例与冷链评价保持相同，仅改变传给 MASKCO 的坐标轴。前三列是相同候选次数的开发对照，最后一列使用相同后续移除名单及两个新种子。[等距视图](results/viewprobe01/RESULTS.md) · [等预算双车队](results/viewpair01/RESULTS.md)"+(" · [收益保留](results/carryprobe01/RESULTS.md)" if follow else ""),""] 
    route_path=HERE/"results/trajectory_routes01/analysis.json"
    choice_path=HERE/"results/trajectory_choice01/analysis.json"
    refine_path=HERE/"results/remaskprobe01/analysis.json"
    if route_path.exists() and choice_path.exists():
        route=read(route_path)["sources"];choice=read(choice_path)["sources"]
        refine=read(refine_path)["sources"] if refine_path.exists() else {}
        lines+=["| 开发来源 | 两轨迹整条路线重组 U | 第 64 步选轨迹 U | 混合轨迹事后择优对实际双距离 U | 二次掩码热图减同规则距离 U |",
                "|---|---:|---:|---:|---:|"]
        for s in ("PDPCDPG-D100","XTW-p100","GH-p100"):
            last=f(refine[s]["mean"]) if refine else "未出结果"
            lines.append(f"| {s} | {f(route[s]['mean'])} | {f(choice[s]['mean'])} | {f(choice[s]['oracle_over_distance'])} | {last} |")
        lines+=["","前三项在同两枚种子的存档轨迹上检验。二次掩码另用新种子和同一插位规则。"
                "[整条路线重组](results/trajectory_routes01/RESULTS.md) · [中途选择](results/trajectory_choice01/RESULTS.md)"
                +(" · [二次掩码](results/remaskprobe01/RESULTS.md)" if refine else ""),""]
    stages=(("固定距离插位，仅用热图选二次遮蔽客户","selectprobe01","primary","by_geometry"),
            ("为未插回客户保留可行邻边","remaining_edges01","mean","geometries"),
            ("全 100 客访问序列重排并重新分车","global_split01","mean","geometries"),
            ("模型只遮住实际移动的 20 客","contextprobe01","mean","geometries"))
    current=[(label,stage,field,gs) for label,stage,field,gs in stages if (HERE/f"results/{stage}/analysis.json").is_file()]
    if current:
        lines+=["| 开发消融，对同规则距离的净增量 U（改善几何） | 丹麦 | XTW | GH |","|---|---:|---:|---:|"]
        for label,stage,field,gs in current:
            ds=read(HERE/f"results/{stage}/analysis.json")["sources"]
            ordered=[ds[name] for name in ("PDPCDPG-D100","XTW-p100","GH-p100")]
            cells=[f"{v[field]:+.2f}（{v['wins']}/{len(v[gs])}）" for v in ordered]
            lines.append("| "+label+" | "+" | ".join(cells)+" |")
        lines+=[""," · ".join(f"[{label}](results/{stage}/RESULTS.md)" for label,stage,_,_ in current),""]
        size_result=HERE/"results/contextprobe01/analysis.json"
        if size_result.is_file():
            ds=read(size_result)["sources"]
            gains="、".join(f(ds[s]["padding"]) for s in ("PDPCDPG-D100","XTW-p100","GH-p100"))
            lines+=[f"冻结权重的 1,440 次训练更新均遮住 20 客。在完全相同的 20 客移动上，模型只遮这 20 客相对额外遮住 20 客，在丹麦、XTW、GH 的净增量依次为 {gains} U。",""]
    bank=HERE/"results/order_bank01/analysis.json"
    if bank.is_file():
        data=read(bank)["sources"]
        lines+=["| 同一批 48 个跨车顺序选六个 | 热图减距离 U | 热图减物理头 U | 距离已达全池最好 |",
                "|---|---:|---:|---:|"]
        for src in ("PDPCDPG-D100","XTW-p100","GH-p100"):
            z=data[src];ctrl=z["controls"]
            lines.append(f"| {src} | {f(ctrl['distance']['mean'])} | {f(ctrl['physical']['mean'])} | {z['distance_attains_pool_best']}/{z['contexts']} |")
        lines+=["","模型在其他客户几何上学习完整分车后的净收益排序。丹麦 20 个评价上下文中只有 4 个包含优于共同起点的顺序。"
                "[全池机会、固定六选消融与独立复算](results/order_bank01/RESULTS.md)",""]
    valley=HERE/"results/order_valley01/analysis.json"
    if valley.is_file():
        v=read(valley)["sources"]
        gains="、".join(f(v[s]["controls"]["distance"]["mean"]) for s in ("PDPCDPG-D100","XTW-p100","GH-p100"))
        wins="、".join(str(v[s]["controls"]["distance"]["wins"])+"/"+str(v[s]["groups"]) for s in ("PDPCDPG-D100","XTW-p100","GH-p100"))
        lines+=[f"给暂时下降的车队各 512 次后续操作，丹麦、XTW、GH 中热图挑选中间车队相对距离的净增量为 {gains} U，改善几何数为 {wins}。"
                "[延后接受与中间车队选择](results/order_valley01/RESULTS.md)",""]
    fleet_conclusion=""
    for phase,title in (("fleetscreen01","11 例筛查"),("fleetdev01","27 例完整开发")):
        folder=HERE/"results"/phase
        if not (folder/"RESULTS.md").is_file() or not (folder/"analysis.json").is_file():
            continue
        study=read(folder/"analysis.json")
        if study["status"]!="COMPLETE":
            continue
        lines+=["",f"### 全车队首次定价的 60 秒消融（{title}）","",
                "三种配对均使用相同冷链目标和限时。前缀距离对照只把首次定价的热图改为距离，随后使用相同的神经掩码重构。","",
                "| 开发来源 | 热图减前缀距离：首次定价后 U | 热图减前缀距离：60 秒 U | 热图减全程同规则距离：60 秒 U | 改善几何 | 新候选减锁版 ORDER_SPLIT：60 秒 U |",
                "|---|---:|---:|---:|---:|---:|"]
        for src in ("PDPCDPG-D100","XTW-p100","GH-p100"):
            item=study["sources"][src]
            contrasts=item["contrasts"]
            paired=contrasts["FLEET_PRICE_D"]
            lines.append(f"| {src} | {f(item['prefix_only_delta']['mean'])} | {f(contrasts['PREFIX_DISTANCE']['mean'])} | "
                         f"{f(paired['mean'])} | {paired['wins']}/{paired['geometries']} | {f(contrasts['ORDER_SPLIT']['mean'])} |")
        margin=study["equal_source"]["PREFIX_DISTANCE"]["mean"]
        immediate=mean(v["prefix_only_delta"]["mean"] for v in study["sources"].values())
        fleet_conclusion=(f"在{title}中，把首次定价从纯距离换为 MASKCO 边分与距离的联合排序，"
                          f"三个来源等权的净效益差从首次定价后的 {f(immediate)} U 变为 60 秒终点的 {f(margin)} U。"
                          f"全车队定价候选{'通过' if study['passed'] else '未通过'}该阶段预设门槛。")
        if not study["passed"]:
            fleet_conclusion+=("按预登记条件，27 例完整开发和 32 例独立验证均为 NOT_RUN。" if phase=="fleetscreen01"
                               else "按预登记条件，32 例独立验证为 NOT_RUN。")
        lines+=["",f"[逐种子、逐几何配对和门槛](results/{phase}/RESULTS.md)",""]
    panel_note=("下一次独立验证已登记 32 个案例、25 个评价组。荷比 VRPPDL 的十个配送日用于检验一般货运订单在统一冷链目标上的迁移。"
                "原 13 法、三枚种子、每格 60 秒，共 1,248 个预留终点。")
    if (HERE/"future_validation/近期五法扩展预登记.json").is_file():
        panel_note+=("新增五个同期对照后共 18 法、1,728 个预留终点，原 13 法门槛与扩展名次分别报告。"
                     "[五法扩展预登记](future_validation/近期五法扩展预登记.json) · ")
    lines+=[panel_note+"[固定名单与开封门槛](下一次验证样本.md)","",
            "## 费用","","| 已回收阶段 | 单元身份 | 完成块 | 求解配额（核秒） | 阶段墙钟（分钟） |","|---|---|---:|---:|---:|"]
    for folder in [HERE/"results",HERE/"reserved_validation/results",HERE/"xtw_low_overlap/results"]:
        if not folder.exists():continue
        for d in sorted(folder.iterdir()):
            stat=d/"status.json";cfg=HERE/(d.name+".json")
            if not stat.exists() or not cfg.exists():continue
            t=read(stat);b=read(cfg)
            if not t["end"]:continue
            wall=(datetime.fromisoformat(t["end"])-datetime.fromisoformat(t["start"])).total_seconds()
            typ="独立比较" if b["purpose"]=="locked independent comparison" else "开发" if b["purpose"] in ("development screening","full development","paired development screening","complete paired development") else "机制诊断或验收"
            budget=(str(sum(len(row.get("arms",[]))*row.get("seconds",b["seconds"]) for row in b["blocks"])) if "seconds" in b else "按诊断工作量")
            lines.append(f"| {d.name} | {typ} | {t['complete']}/{t['total']} | {budget} | {wall/60:.1f} |")
    lines+=["","不同阶段可使用互不重叠的物理核，墙钟分阶段列示。","",
            "[首次独立结果](../maskco_heatmap_calibration_20261119/RESULTS.md) · [易腐品复核、11 法完整对比与经济分解](RESULTS.md)"]
    if len(tests)==3:
        lines+=["","[新版本预留面板结果](reserved_validation/RESULTS.md)"]
    if (HERE/"名次稳定性.md").is_file():lines+=["","[几何重采样后进入前三的比例](名次稳定性.md)"]
    if strict:
        m=latest[1];study=latest[2];ordered=("PDPCDPG-D100","XTW-p100","GH-p100")
        external=("NDS-Econ","ALNS","VNS","CaR","DeepACO","POMO","AttentionModel")
        lines+=["","## 论文结论","","| 历史预留样本 | 当批最强外部对比法 | 相对它的净效益 U | 对它改善的几何 | 相对同规则纯距离 U | 原 13 法名次 |","|---|---|---:|---:|---:|---:|"]
        for src in ordered:
            v=study["sources"][src];scores=v["scores"];best=max(external,key=scores.get)
            paired=v["external_contrasts"][best]
            lines.append(f"| {src} | {best} | {f(scores[m]-scores[best])} | {paired['wins']}/{paired['geometries']} | {f(scores[m]-scores[m+'_D'])} | {v['ranks'][m]} |")
        lines+=["","MASKCO 读取掩码与当前路线，产生客户相邻的分数，参与选择重插位置和跨路线访问顺序。容量、时窗、品质、发车时间和最终分车收益由统一的冷链评价器决定。NDS-Econ 用作者 VRPTW100 权重选择移除客户；CaR、DeepACO、POMO、AttentionModel 用各自的 11,400 步冷链适配权重。MASKCO 用 1,440 步冻结权重。名次衡量这些实现的单核 60 秒结果。",
                "","丹麦两次检验使用同一组母实例中的不同请求，XTW 和 GH 为统一的 100 客户配送投影。"]
        in_top3=all(study["sources"][src]["ranks"][m]<=3 for src in ordered)
        lines+=["",f"历史十三法同期比较中，三个主来源{'均进入' if in_top3 else '未全部进入'}前三。三个主来源与低重叠 XTW 的预设热图收益和名次联合门槛{'通过' if strict['joint_pass'] else '未通过'}。"]
        overall=mean(study["sources"][src]["contrasts"][m][m+"_D"]["mean"] for src in ordered)
        lo,hi=study["three_source_equal_weight_ci95"]
        lines.append(f"预留主面板的三来源等权热图增量为 {f(overall)} U，固定种子并按几何整簇重采样的 95% 描述区间为 [{f(lo)}, {f(hi)}] U。")
        low_lo,low_hi=strict["low_overlap_three_source_ci95"]
        lines.append(f"采用客户坐标低重叠的 XTW 十例后，三来源等权增量为 {f(strict['low_overlap_three_source_mean'])} U，同口径区间为 [{f(low_lo)}, {f(low_hi)}] U。")
        if placebo.exists():
            aligned=read(placebo)["alignment_supported_on_all"]
            lines.append("同距离分层置乱对照"+("支持三个开发来源均从正确的客户边对应关系中获益。" if aligned else "显示具体客户边的对应关系尚未在三个开发来源上均表现出稳定增益。"))
        if tail_file.is_file() and load_result.exists():
            lines+=["","固定 60 秒预算的 MASKCO 掩码重构与冷链经济分车在历史十三法比较中有 18/20 个几何超过同批 ALNS。热图净贡献随来源而变。丹麦的 600 个单客插位中，只有 12 个存在比六个最近插位和单客路线更好的选择；在强距离终态上，纯距离选中的四个路线对已取到本次尾段互换遍历的全部平均收益。"]
        if (HERE/"results/mixscreen01/analysis.json").is_file():
            lines+=["","同规则纯距离沿用掩码、重构、分车和总时限，在内层只试距离排序下的贪心与 regret-2。热图版还会试原始及校准边分。两者的差额计入神经预测用时及这些候选的机会成本。六法开发另用相同流程的纯物理输出头作对照。"]
    if conclusion:
        lines+=["",conclusion]
    if fleet_conclusion:
        lines+=["",fleet_conclusion]
    if (HERE.parent/"unified_external_20261120/论文实验总览.md").is_file():
        lines[2:2]=["最新证据见[十二个外部方法的同期比较、前瞻验证与论文结论](../unified_external_20261120/论文实验总览.md)。",""]
    (HERE/"总报告.md").write_text("\n".join(lines)+"\n",encoding="utf8")
if __name__=="__main__":main()
