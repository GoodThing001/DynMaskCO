"""Compare complete comparable solutions to an instance-specific relaxation."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/"bounds"


def main():
    bounds = pd.read_csv(OUT/"case_bounds.csv")
    reports = []
    sign_summaries = []
    for batch in bounds["batch"].unique():
        scores = pd.read_csv(HERE/"results"/batch/"endpoints.csv")
        metadata=bounds.loc[bounds["batch"]==batch].rename(columns={"source":"bound_source","geometry":"geometry_group"})
        study = scores.merge(metadata, on="case", validate="many_to_one")
        if len(study)!=len(scores) or (study["U"]>study["net_upper_U"]+1e-6).any():
            raise ValueError("A feasible endpoint exceeds the relaxation, or input coverage differs")
        best = study.groupby("case")["U"].transform("max")
        study["to_best_feasible_U"] = best-study["U"]
        study["to_upper_U"] = study["net_upper_U"]-study["U"]
        study["to_upper_pct_of_achieved_profit"] = study["to_upper_U"]/study["U"].abs()*100
        study["to_upper_pct_of_achieved_loss_cost"] = study["to_upper_U"]/(study["nominal_U"]-study["U"])*100
        for source,rows in study[study["method"]=="ORDER_SPLIT"].groupby("bound_source"):
            case_means=rows.groupby("case")["U"].mean()
            case_bounds=rows.groupby("case")["net_upper_U"].first()
            sign_summaries.append((batch,source,len(case_means),int((case_means>1e-6).sum()),
                                   int((case_means < -1e-6).sum()),int((case_bounds < -1e-6).sum())))
        metrics = ["U","to_best_feasible_U","to_upper_U","to_upper_pct_of_achieved_profit","to_upper_pct_of_achieved_loss_cost"]
        per_case=study.groupby(["batch","bound_source","geometry_group","case","method"])[metrics].mean()
        per_case.to_csv(OUT/(batch+"_case_gaps.csv"), encoding="utf-8-sig")
        per_group=per_case.groupby(["batch","bound_source","geometry_group","method"]).mean()
        reports.append(per_group.groupby(["batch","bound_source","method"]).mean().reset_index().rename(columns={"bound_source":"source"}))
    summary=pd.concat(reports, ignore_index=True)
    summary.to_csv(OUT/"source_gaps.csv", index=False, encoding="utf-8-sig")
    lines=["# 统一冷链目标的解质量界限", "",
           f"公开距离最优值与本项目的净效益单位不同。这里在本次 {len(bounds)} 个百客案例上，逐例计算一个乐观的净效益上界，同时保留本次 16 法三种子中实际达到的最高可行净效益。",
           "", "| 面板 | 来源 | 方法 | 距当批最佳可行解 U | 距净效益上界 U | 占已达净效益 % | 占已发生品质损失与运行费 % |",
           "|---|---|---|---:|---:|---:|---:|"]
    for row in summary[summary["method"].isin(["ORDER_SPLIT","ORDER_SPLIT_D","ALNS","PyVRP"])].itertuples():
        lines.append(f"| {row.batch} | {row.source} | {row.method} | {row.to_best_feasible_U:,.2f} | "
                     f"{row.to_upper_U:,.2f} | {row.to_upper_pct_of_achieved_profit:.3f}% | {row.to_upper_pct_of_achieved_loss_cost:.2f}% |")
    lines+=["","表内差距越小越好。每行先平均同例种子，再平均同一几何内各例，最后对来源内几何等权。已测 XTW 行含全部 13 个不同实例。上界差给出真实最优值与当前解之间可能剩余的最大改进空间。净效益包含名义货值，接近零时百分比不稳定，因此同时报告与品质损失、里程和制冷总成本相比的差距。",
            "", "## 绝对净效益", "",
            "| 面板 | 来源 | 案例数 | MASKCO 三种子均值正 | 均值负 | 可证明最优净效益也为负 |",
            "|---|---|---:|---:|---:|---:|"]
    for batch,source,n,pos,neg,certified in sign_summaries:
        lines.append(f"| {batch} | {source} | {n} | {pos} | {neg} | {certified} |")
    for r in bounds[bounds.net_upper_U < -1e-6].itertuples():
        lines.append(f"\n{r.case} 的名义货值为 {r.nominal_U:,.2f} U，全覆盖条件下的品质与运营总损失至少为 {r.loss_lower_U:,.2f} U，故任何可行算法的净效益最多为 {r.net_upper_U:,.2f} U。")
    lines+=["",
            "## 界限的构造", "",
            "将净效益写成名义货值减总损失。若车从客户 i 到达 j，j 的货龄至少包含仓库直达 i、i 的服务、i 到 j 及 j 的服务。直接从仓库到 j 时只计算直达时间和 j 的服务。冷藏车全部行驶和全部服务时间必然发生，忽略额外等待。由此得到每条有向边的乐观损失。对明显违反容量、时窗、回仓或最低品质的客户对删边。",
            "", "每个客户各需要一条入边和一条出边，仓库复制为与客户数相同的节点，强制至少容纳总需求所需的最少车辆从仓库发车。求解二分图最小费用指派，并加全体客户服务期间的制冷成本。任意可行车队都能嵌入该指派；松弛还允许客户子圈，并忽略部分载荷和等待。因此指派最低损失是全问题下界，名义货值减去它给出净效益上界。", "",
            "三个既有来源各取一个五客户子问题，枚举全部客户排列与分车，均满足精确最优值不高于该界；全部本轮可行终点另行检验同一关系。",
            "", "[逐百客案例的上下界](case_bounds.csv) · [逐方法、来源与差距](source_gaps.csv) · [五客枚举](exact_small.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__=="__main__":
    main()
