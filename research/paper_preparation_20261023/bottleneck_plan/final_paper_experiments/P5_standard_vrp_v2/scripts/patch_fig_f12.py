import ast
from pathlib import Path

p = Path("scripts/p5v2_figures.py")
t = p.read_text(encoding="utf-8")
old_start = t.index("    # F12：计时与预测器")
old = t[old_start:t.index("def main():")]
new = '''    # F12：固定候选数量诊断（四种排序的插位质量）
    d = load_json(V2 / "tables/analysis_diag.json")
    if d:
        agg = d["per_variant"]
        names = ["distance", "heat_raw", "heat_calibrated", "heat_norm"]
        label = {"distance": "距离", "heat_raw": "原始热图",
                 "heat_calibrated": "校准热图", "heat_norm": "归一化热图"}
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        xs = range(len(names))
        axes[0].bar(xs, [agg[n]["mean_recall1"] for n in names],
                    color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"])
        axes[0].set_xticks(list(xs))
        axes[0].set_xticklabels([label[n] for n in names], fontsize=7)
        axes[0].set_ylabel("最优插位召回率 @1")
        axes[0].set_title("插位选择质量")
        axes[1].bar(xs, [agg[n]["mean_regret"] for n in names],
                    color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"])
        axes[1].set_xticks(list(xs))
        axes[1].set_xticklabels([label[n] for n in names], fontsize=7)
        axes[1].set_ylabel("平均后悔值（越小越好）")
        axes[1].set_title("插位后悔值")
        fig.suptitle("F12 固定候选数量诊断（同候选集/同插入顺序，仅排序不同）")
        save(fig, "F12_fixed_candidate_diagnostics")


'''
t = t[:old_start] + new + t[t.index("def main():"):]
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("F12 replaced with diagnostics figure")
