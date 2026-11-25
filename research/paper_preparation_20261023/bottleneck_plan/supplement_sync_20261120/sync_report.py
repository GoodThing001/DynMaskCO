"""Independently audit and summarize the four-method paired replication."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RECENT = HERE.parent / "recent5_20261120"
METHODS = ("ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS", "PyVRP")
SEEDS = (33001, 33002, 33003)
SOURCES = ("PDPCDPG-D100", "XTW-p100", "GH-p100")
LOW = "XTW 低重叠"
EPS = 1e-6
DRAWS = 20000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_module():
    spec = importlib.util.spec_from_file_location("recent_independent_replay", RECENT / "recent_independent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_and_load(stage):
    require(__debug__, "Independent schedule checks require Python assertions")
    checker = independent_module()
    inputs, _ = checker.original_panel()
    phase, meta = read(HERE / "phase.json"), read(HERE / "metadata.json")
    require(meta == read(RECENT / "metadata.json"), "Panel metadata differs from the existing audited comparison")
    expected = {(c, s) for c in inputs for s in SEEDS}
    require(phase["seconds"] == 60 and len(phase["blocks"]) == 90
            and {(b["case"], b["seed"]) for b in phase["blocks"]} == expected,
            "The 90 registered case/seed pairs differ")
    require(set(meta["cases"]) == set(inputs), "Unexpected input roster")
    for block in phase["blocks"]:
        require(len(block["methods"]) == 4 and set(block["methods"]) == set(METHODS),
                "Expected four registered methods per pair")
    for method in METHODS:
        counts = [sum(b["methods"][i] == method for b in phase["blocks"]) for i in range(4)]
        require(max(counts) - min(counts) <= 1, "Unbalanced method positions")
    files = {f"{i}_{m}.json" for i, b in enumerate(phase["blocks"]) for m in b["methods"]}
    received = {p.name for p in (stage / "out").glob("*.json")}
    require(received == files, f"Endpoint files: {len(files-received)} missing, {len(received-files)} unexpected")
    state = read(stage / "status.json")
    require(state.get("end") and not state["active"] and not state["failed"]
            and (state["complete"], state["total"], state["cells"], state["total_cells"]) == (90, 90, 360, 360),
            "Controller has not completed all 90 blocks and 360 endpoints")
    audit = dict(status="PASS", cases=30, blocks=0, endpoints=0, routes=0,
                 maximum_absolute_U_error=0., maximum_initial_U_error=0.)
    fresh = {}
    for index, block in enumerate(phase["blocks"]):
        key = block["case"], block["seed"]
        model = checker.original_model(inputs[key[0]], key[0])
        rows = {}
        for method in block["methods"]:
            name = f"{index}_{method}.json"
            row = read(stage / "out" / name)
            try:
                count, error, initial_error = checker.replay(row, key, method, *model)
            except (AssertionError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{name}: {type(exc).__name__}: {exc}") from exc
            require(row["budget_s"] == 60, f"Search budget differs: {name}")
            require(math.isfinite(row["setup_s"]) and row["setup_s"] >= 0, f"Invalid preparation timing: {name}")
            if "controller_process_s" in row:
                require(math.isfinite(row["controller_process_s"]) and row["controller_process_s"] > 0,
                        f"Invalid process timing: {name}")
            audit["routes"] += count
            audit["endpoints"] += 1
            audit["maximum_absolute_U_error"] = max(audit["maximum_absolute_U_error"], error)
            audit["maximum_initial_U_error"] = max(audit["maximum_initial_U_error"], initial_error)
            rows[method] = row
        require(len({tuple(r["cpu"]) for r in rows.values()}) == 1,
                f"The four methods did not share one physical-core lane: {key}")
        reference = rows[METHODS[0]]
        require(all(r["initial_routes"] == reference["initial_routes"]
                    and abs(r["initial_U"] - reference["initial_U"]) <= EPS for r in rows.values()),
                f"Common initial fleet differs: {key}")
        fresh[key] = rows
        audit["blocks"] += 1
    return meta, fresh, state, audit


def group_matrix(values, panel_cases, metadata):
    groups = defaultdict(list)
    for case in sorted(panel_cases):
        groups[metadata[case]["geometry"]].append([values[case, seed] for seed in SEEDS])
    names = sorted(groups)
    return names, np.asarray([np.mean(groups[name], axis=0) for name in names], dtype=float)


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def summary(meta, fresh):
    panels = {source: sorted(c for c in meta["primary"] if meta["cases"][c]["dataset"] == source)
              for source in SOURCES}
    panels[LOW] = sorted(meta["low_overlap"])
    result, draws, geometry_rows = {}, {}, []
    rng = np.random.default_rng(20261120)
    for name, cases in panels.items():
        matrices = {}
        for method in METHODS:
            geometries, matrices[method] = group_matrix({k: v[method]["U"] for k, v in fresh.items()},
                                                        cases, meta["cases"])
        n = len(geometries)
        resampled = rng.integers(n, size=(DRAWS, n))
        scores = {m: float(a.mean()) for m, a in matrices.items()}
        ranks = {m: 1 + sum(v >= scores[m] - EPS for other, v in scores.items() if other != m)
                 for m in METHODS}
        paired, draws[name] = {}, {}
        for method in METHODS[1:]:
            delta = matrices[METHODS[0]] - matrices[method]
            means = delta.mean(axis=1)
            boot = means[resampled].mean(axis=1)
            draws[name][method] = boot
            paired[method] = dict(mean=float(means.mean()), ci95=interval(boot),
                                  wins=int((means > EPS).sum()), ties=int((abs(means) <= EPS).sum()),
                                  losses=int((means < -EPS).sum()), seeds=delta.mean(axis=0).tolist())
            for i, geometry in enumerate(geometries):
                geometry_rows.append(dict(panel=name, geometry=geometry, comparison=method,
                    delta_U=float(means[i]), **{f"seed_{seed}": float(delta[i, j]) for j, seed in enumerate(SEEDS)}))
        result[name] = dict(cases=len(cases), geometries=n, scores=scores, ranks=ranks, paired=paired)
    equal = {}
    for method in METHODS[1:]:
        equal[method] = dict(mean=fmean(result[s]["paired"][method]["mean"] for s in SOURCES),
                            ci95=interval(np.mean([draws[s][method] for s in SOURCES], axis=0)),
                            wins=sum(result[s]["paired"][method]["wins"] for s in SOURCES),
                            ties=sum(result[s]["paired"][method]["ties"] for s in SOURCES),
                            losses=sum(result[s]["paired"][method]["losses"] for s in SOURCES),
                            geometries=sum(result[s]["geometries"] for s in SOURCES))
    equal_scores = {m: fmean(result[s]["scores"][m] for s in SOURCES) for m in METHODS}
    return result, equal, equal_scores, geometry_rows


def endpoint_rows(meta, fresh):
    rows = []
    for (case, seed), group in sorted(fresh.items()):
        for method, row in group.items():
            rows.append(dict(case=case, seed=seed, method=method,
                dataset=meta["cases"][case]["dataset"], geometry=meta["cases"][case]["geometry"],
                primary=int(case in meta["primary"]), low_overlap=int(case in meta["low_overlap"]),
                U=row["U"], initial_U=row["initial_U"], vehicles=len(row["routes"]),
                selected_s=row["selected_s"], setup_s=row["setup_s"],
                controller_process_s=row.get("controller_process_s"), worker_execute_s=row.get("worker_execute_s"),
                measured_search_wall_s=row.get("wall_s", row.get("timing", {}).get("total_measured_s")),
                search_cpu_s=row["cpu_s"], core=row["cpu"][0]))
    return rows


def timing_summary(rows):
    output = []
    for method in METHODS:
        cells = [r for r in rows if r["method"] == method]
        data = dict(method=method, cells=len(cells))
        for field in ("setup_s", "controller_process_s", "worker_execute_s", "measured_search_wall_s", "search_cpu_s"):
            values = [r[field] for r in cells]
            present = [v for v in values if v is not None]
            data[field + "_available"] = len(present)
            data[field + "_mean"] = fmean(present) if len(present) == len(cells) else None
            data[field + "_total"] = math.fsum(present) if len(present) == len(cells) else None
        output.append(data)
    return output


def figures(stage, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("Denmark", "XTW", "GH", "XTW low overlap")
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 3.4), sharey=True)
    for axis, label, panel in zip(axes, labels, panels.values()):
        axis.axvline(0, color="#9AAEC5", lw=.8)
        for y, method in enumerate(METHODS[1:]):
            value = panel["paired"][method]
            axis.hlines(y, *value["ci95"], color="#5585B5", linewidth=2.2)
            axis.plot(value["mean"], y, "o", color="#173D66", markersize=5)
        axis.set_title(label, fontsize=11)
        axis.set_yticks(range(3), ["Distance ablation", "ALNS", "PyVRP-HGS"])
        axis.tick_params(labelsize=9)
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].invert_yaxis()
    figure.supxlabel("MASKCO minus comparator (U); geometry-cluster 95% descriptive interval", fontsize=10)
    figure.tight_layout()
    figure.savefig(stage / "paired_sync.png", dpi=220, bbox_inches="tight")
    figure.savefig(stage / "paired_sync.pdf", bbox_inches="tight")
    plt.close(figure)


def number(value):
    return f"{value:+,.2f}"


def markdown(panels, equal, timing, audit):
    lines = ["# 锁版 MASKCO 四法同期复核", "",
        "ORDER_SPLIT、同规则纯距离 ORDER_SPLIT_D、ALNS 和 PyVRP 在同一批次完成 30 个案例、3 个种子的配对比较。"
        "每个区组内四法共用同一初解和物理核，方法次序轮换，每法在线搜索预算为 60 秒。"
        f"全部 {audit['endpoints']} 个终点及 {audit['routes']:,} 条路线通过独立复算。", "",
        "主面板为 27 个案例、20 个几何；低重叠 XTW 面板为 10 例，其中 7 例与主面板共用。"
        "每例先对三个种子平均，同一几何内案例等权，再对来源内几何等权。三个主来源等权结果不重复计入低重叠面板。", "",
        "| 方法 | 丹麦 U | XTW U | GH U | 低重叠 XTW U |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        lines.append("| " + method + " | " + " | ".join(f"{p['scores'][method]:,.2f}（{p['ranks'][method]}/4）"
                                                              for p in panels.values()) + " |")
    lines += ["", "括号为同期四法名次；净效益 U 越高越好，1×10⁻⁶ U 内并列时采用并列组末位名次。", "",
              "## 配对差额", "", "差值方向为 MASKCO 减对照；区间采用几何整簇重采样 20,000 次，描述已测几何之间的不确定性。", "",
              "| 面板 | 对照 | ΔU | 95% 描述区间 U | 改善几何 |", "|---|---|---:|---:|---:|"]
    for name, panel in panels.items():
        for method, effect in panel["paired"].items():
            lo, hi = effect["ci95"]
            lines.append(f"| {name} | {method} | {number(effect['mean'])} | [{number(lo)}, {number(hi)}] | {effect['wins']}/{panel['geometries']} |")
    for method, effect in equal.items():
        lo, hi = effect["ci95"]
        lines.append(f"| 三个主来源等权 | {method} | {number(effect['mean'])} | [{number(lo)}, {number(hi)}] | {effect['wins']}/{effect['geometries']} |")
    lines += ["", "![四法配对差额与几何区间](paired_sync.png)", "",
              "## 计算时间", "", "| 方法 | 终点数 | 原适配准备均值 s | 独立进程总墙钟均值 s |", "|---|---:|---:|---:|"]
    for row in timing:
        process = row["controller_process_s_mean"]
        lines.append(f"| {row['method']} | {row['cells']} | {row['setup_s_mean']:.3f} | "
                     + (f"{process:.3f}" if process is not None else "未记录") + " |")
    lines += ["", "原适配准备时间沿用既有实验口径；距离消融沿用含权重加载的同一初始化入口，但搜索不执行神经前向。控制器记录的进程总墙钟包含启动、准备、搜索和结果输出，"
              "并非各方法等额的端到端时限。在线解选择仍统一截止于 60 秒。", "",
              "## 解释范围", "", "本轮沿用既有案例、种子和锁版权重，是同期重复评估。"
              "MASKCO−ORDER_SPLIT_D 用于评价热图在完整搜索及其计算机会成本下的净增量；MASKCO−ALNS 或 PyVRP 衡量整套求解器差额。"
              "PyVRP 的内部搜索采用冷链代理成本，最终车队均按完整冷链目标评价。原近期四项神经方法的直接迁移结果及历史十八法汇总保留原批次身份。", "",
              "[逐终点结果](endpoints.csv) · [逐几何与逐种子差额](paired_geometries.csv) · [独立复算](independent_audit.json) · [汇总数据](sync_summary.json)"]
    return "\n".join(lines) + "\n"


def check_historical():
    checker = independent_module()
    inputs, _ = checker.original_panel()
    previous = RECENT.parent / "maskco_perishable_external_20261119/reserved_validation"
    plan = read(previous / "reserve02.json")
    block = plan["blocks"][0]
    key = block["case"], block["seed"]
    common = checker.original_model(inputs[key[0]], key[0])
    rows = read(previous / "results/reserve02/out/0.json")["arms"]
    checks = {}
    for method in METHODS[:3]:
        checks[method] = checker.replay(rows[method], key, method, *common)
    recent = read(RECENT / "matrix.json")
    index = next(i for i, b in enumerate(recent["blocks"]) if (b["case"], b["seed"]) == key)
    row = read(RECENT / f"results/benchmark01/out/{index}_PyVRP.json")
    checks["PyVRP"] = checker.replay(row, key, "PyVRP", *common)
    print(json.dumps(dict(status="HISTORICAL_DECODER_CHECK", case=key[0], seed=key[1],
                          methods=checks, new_experiment_endpoints=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/sync01")
    parser.add_argument("--check-historical", action="store_true")
    args = parser.parse_args()
    if args.check_historical:
        check_historical()
        return
    stage = args.results
    try:
        meta, fresh, state, audit = audit_and_load(stage)
    except (AssertionError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        if stage.is_dir():
            write(stage / "independent_audit.json", dict(status="NOT_COMPLETE" if isinstance(exc, FileNotFoundError) else "FAIL",
                                                         reason=f"{type(exc).__name__}: {exc}"))
        raise
    write(stage / "independent_audit.json", audit)
    panels, equal, equal_scores, geometries = summary(meta, fresh)
    rows = endpoint_rows(meta, fresh)
    timing = timing_summary(rows)
    write_csv(stage / "endpoints.csv", rows)
    write_csv(stage / "paired_geometries.csv", geometries)
    write_csv(stage / "timing.csv", timing)
    write(stage / "sync_summary.json", dict(panels=panels, equal_source=equal, equal_source_scores=equal_scores,
                                           timing=timing, independent_audit=audit, controller=state))
    figures(stage, panels)
    (stage / "RESULTS.md").write_text(markdown(panels, equal, timing, audit), encoding="utf-8")
    print(json.dumps(dict(status="PASS", endpoints=audit["endpoints"], routes=audit["routes"],
                          equal_source=equal, results=str(stage)), ensure_ascii=False))


if __name__ == "__main__":
    main()
