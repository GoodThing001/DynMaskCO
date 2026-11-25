"""Independently evaluate the prespecified fleet-pricing development stages."""
import csv
from datetime import datetime
import json
from itertools import product
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recent5_20261120"))
from recent_independent import (TOLERANCE, difference, independent_audit,
                                original_model, read, replay, require)
sys.path.insert(0, str(HERE.parent / "maskco_heatmap_calibration_20261119"))
from assess import describe

CANDIDATE = "FLEET_PRICE"
MATCH = "FLEET_PRICE_D"
PREFIX = "PREFIX_DISTANCE"
ARMS = (CANDIDATE, MATCH, PREFIX, "ORDER_SPLIT", "DIST", "DIST_SLOT")
PREFIX_ARMS = ARMS[:3]
SOURCES = ("PDPCDPG-D100", "XTW-p100", "GH-p100")
STAGES = {"fleetscreen01": ("mixscreen01", (56831, 56832), 11),
          "fleetdev01": ("mixdev01", (56851, 56852, 56853), 27)}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def test_gate(stage, sources, equal):
    gates = {}
    for source, result in sources.items():
        diff = result["contrasts"]
        matched = diff[MATCH]
        tests = {"positive_matched_mean": matched["mean"] > TOLERANCE,
                 "strict_majority_geometries": 2 * matched["wins"] > matched["geometries"],
                 "positive_DIST_mean": diff["DIST"]["mean"] > TOLERANCE,
                 "positive_DIST_SLOT_mean": diff["DIST_SLOT"]["mean"] > TOLERANCE}
        if stage == "fleetdev01":
            tests["two_positive_matched_seeds"] = matched["positive_seeds"] >= 2
        gates[source] = tests
    if stage == "fleetscreen01":
        gates["equal_source"] = {
            "positive_matched_all_seeds": all(v > TOLERANCE for v in equal[MATCH]["by_seed"].values()),
            "positive_prefix_all_seeds": all(v > TOLERANCE for v in equal[PREFIX]["by_seed"].values())}
    else:
        gates["equal_source"] = {"positive_prefix_mean": equal[PREFIX]["mean"] > TOLERANCE}
    return gates, all(all(tests.values()) for tests in gates.values())


def stage_records(stage):
    previous, seeds, count = STAGES[stage]
    directory = HERE / "results" / stage
    plan = read(HERE / (stage + ".json"))
    state = read(directory / "status.json")
    registry = read(HERE / "heldout_sources.json")
    cases = {b["case"] for b in read(HERE / (previous + ".json"))["blocks"]}
    pairs = set(product(cases, seeds))
    require(len(cases) == count and len(plan["blocks"]) == len(pairs)
            and pairs == {(b["case"], b["seed"]) for b in plan["blocks"]}
            and plan["seconds"] == 60, "Case, seed or time differs from registered stage")
    require(state["end"] and not state["failed"] and not state["active"]
            and state["complete"] == state["total"] == len(pairs), "Controller has not completed all pairs")
    require(__debug__, "Independent evaluator uses assertions")
    audit = dict(status="PASS", cases=count, blocks=len(pairs), endpoints=0, routes=0,
                 maximum_absolute_U_error=0., maximum_initial_U_error=0., prefix_fleets=0)
    records = {}
    for i, b in enumerate(plan["blocks"]):
        key = b["case"], b["seed"]
        path = directory / "out"
        group = read(path / f"{i}.json")
        require((group["case"], group["seed"]) == key and len(b["arms"]) == len(ARMS)
                and set(group["arms"]) == set(b["arms"]) == set(ARMS), f"Block mismatch {key}")
        model, initial, initial_u = original_model(HERE / registry["derived_instances"][key[0]]["file"], key[0])
        rows = group["arms"]
        masks = []
        for method, r in rows.items():
            identity = f"{key}/{method}"
            require(r == read(path / f"{i}_{method}.cell.json"), f"Block differs from individual receipt {identity}")
            routes, uerr, initial_err = replay(r, key, method, model, initial, initial_u)
            audit["endpoints"] += 1
            audit["routes"] += routes
            audit["maximum_absolute_U_error"] = max(audit["maximum_absolute_U_error"], uerr)
            audit["maximum_initial_U_error"] = max(audit["maximum_initial_U_error"], initial_err)
            if method not in PREFIX_ARMS:
                continue
            prefix = r["prefix"]
            difference(prefix["initial_U"], initial_u, "prefix start")
            accepted = prefix["proposed_U"] if prefix["accepted"] else initial_u
            difference(prefix["accepted_U"], accepted, "accepted prefix")
            require(r["U"] >= accepted - TOLERANCE and prefix["common_U"] >= initial_u - TOLERANCE
                    and prefix["proposed_U"] >= prefix["common_U"] - TOLERANCE, f"Incumbent order {identity}")
            score = independent_audit(model, model.solution(prefix["proposed_routes"]), True)
            difference(prefix["proposed_U"], score, "proposed fleet")
            audit["prefix_fleets"] += 1
            mask = prefix["mask"]
            require(len(set(mask)) == len(mask) == 40 and min(mask) >= 1 and max(mask) <= 100, identity)
            masks.append(mask)
        require(len(masks) == 3 and masks[1:] == masks[:1] * 2, f"Different prefix masks {key}")
        require(len({tuple(r["cpu"]) for r in rows.values()}) == 1, f"Pairing on different cores {key}")
        records[key] = rows
    return records, audit, state, registry["cases"]


def statistics(stage, records, meta):
    _, seeds, _ = STAGES[stage]
    sources = {}
    for source in SOURCES:
        panel = {k: r for k, r in sorted(records.items()) if meta[k[0]]["dataset"] == source}

        def measure(fn):
            return describe({k: fn(r) for k, r in panel.items()}, meta)

        contrasts = {a: measure(lambda rows: rows[CANDIDATE]["U"] - rows[a]["U"]) for a in ARMS[1:]}
        prefix = {a: measure(lambda rows: rows[a]["prefix"]["accepted_U"] - rows[a]["initial_U"])
                  for a in PREFIX_ARMS}
        sources[source] = dict(contrasts=contrasts,
            scores={a: measure(lambda rows: rows[a]["U"])["mean"] for a in ARMS},
            accepted_prefix_gain=prefix, cases=len({c for c, s in panel}),
            prefix_only_delta=measure(lambda r: r[CANDIDATE]["prefix"]["accepted_U"] -
                                               r[PREFIX]["prefix"]["accepted_U"]),
            change_after_prefix=measure(lambda r: (r[CANDIDATE]["U"] - r[CANDIDATE]["prefix"]["accepted_U"]) -
                                                  (r[PREFIX]["U"] - r[PREFIX]["prefix"]["accepted_U"])))
    equal = {a: {"mean": mean(s["contrasts"][a]["mean"] for s in sources.values()),
                 "by_seed": {seed: mean(s["contrasts"][a]["by_seed"][seed] for s in sources.values())
                             for seed in seeds}} for a in ARMS[1:]}
    gates, passed = test_gate(stage, sources, equal)
    return sources, equal, gates, passed


def accounting(records, state):
    costs = {}
    for method in ARMS:
        rows = [arms[method] for arms in records.values()]
        c = dict(endpoints=len(rows), cpu_s=sum(r["cpu_s"] for r in rows),
                 online_wall_s=sum(r["timing"]["total_measured_s"] for r in rows),
                 setup_s=sum(r["setup_s"] for r in rows), iterations=sum(r["iterations"] for r in rows),
                 forwards=sum(r["forwards"] for r in rows), late_candidates=sum(
                     r["timing"]["rejected_late_candidates"] for r in rows),
                 native_route_requests=sum(r.get("native_route_requests", 0) for r in rows),
                 python_search_requests=sum(r.get("requests", 0) for r in rows))
        if method in PREFIX_ARMS:
            p = [r["prefix"] for r in rows]
            c["prefix"] = {k: sum(v[k] for v in p) for k in
                           ("accepted", "elapsed_s", "forwards", "forward_s", "route_requests",
                            "unique_route_requests", "attempts", "columns", "common_columns")}
        c["all_forwards"] = c["forwards"] + c.get("prefix", {}).get("forwards", 0)
        costs[method] = c
    return dict(by_method=costs, cpu_s=sum(c["cpu_s"] for c in costs.values()),
                allocated_search_core_s=60*len(records)*len(ARMS),
                online_wall_s=sum(c["online_wall_s"] for c in costs.values()),
                setup_s=sum(c["setup_s"] for c in costs.values()),
                wall_s=(datetime.fromisoformat(state["end"])-datetime.fromisoformat(state["start"])).total_seconds())


def pair_table(path, records, meta):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source","geometry","case","seed","initial_U",*ARMS,
                         "prefix_neural_accepted_U","prefix_distance_accepted_U","neural_prefix_delta_U",
                         "final_prefix_delta_U","final_matched_delta_U"])
        for (case, seed), r in sorted(records.items()):
            h = r[CANDIDATE]
            p = r[PREFIX]
            a, b = h["prefix"]["accepted_U"], p["prefix"]["accepted_U"]
            writer.writerow([meta[case]["dataset"],meta[case]["geometry"],case,seed,h["initial_U"],
                             *(r[m]["U"] for m in ARMS),a,b,a-b,h["U"]-p["U"],h["U"]-r[MATCH]["U"]])


def signed(v):
    return f"{v:+,.2f}"


def report(stage):
    records, audit, run, meta = stage_records(stage)
    sources, equal, gates, passed = statistics(stage, records, meta)
    costs = accounting(records, run)
    folder = HERE / "results" / stage
    decision = (("RUN_FULL_DEVELOPMENT" if stage == "fleetscreen01" else "LOCK_FOR_VALIDATION")
                if passed else "STOP_FIXED_CANDIDATE")
    write(folder/"independent_audit.json", audit)
    write(folder/"analysis.json", dict(status="COMPLETE", phase=stage, passed=passed, decision=decision,
                                       audit=audit, costs=costs, sources=sources, equal_source=equal, gates=gates))
    pair_table(folder / "paired.csv", records, meta)
    title = "等时筛查" if stage == "fleetscreen01" else "完整限时开发"
    lines=[f"# FLEET_PRICE {title}", "",
           f"{audit['cases']} 例、{audit['blocks']} 个案例种子块、六种方法，{audit['endpoints']} 个终点全部通过独立复算。"
           f"复核 {audit['routes']:,} 条终点路线及 {audit['prefix_fleets']} 个定价前缀提出的车队，"
           f"终点净效益最大绝对差 {audit['maximum_absolute_U_error']:.2e} U。"
           f"预定门槛{'通过' if passed else '未通过'}（{decision}）。", "",
           "| 来源 | 热图 − 同规则距离 U | 改善几何 | 逐种子热图增量 U | 对原 DIST U | 对原 DIST_SLOT U | 对原 ORDER_SPLIT U |",
           "|---|---:|---:|---|---:|---:|---:|"]
    for source, z in sources.items():
        d=z["contrasts"]; x=d[MATCH]
        lines.append(f"| {source} | {signed(x['mean'])} | {x['wins']}/{x['geometries']} | "
                     + " / ".join(map(signed,x["by_seed"].values()))+" | "
                     + " | ".join(signed(d[k]["mean"]) for k in ("DIST","DIST_SLOT","ORDER_SPLIT"))+" |")
    lines += ["", "| 来源 | 热图首次定价比共同初解增加 U | 首次用距离比初解增加 U | 两者首次差 U | 两者 60 秒终态差 U | 差额从首次到终态的变化 U | 逐种子终态差 U |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for source, z in sources.items():
        g=z["accepted_prefix_gain"]; p=z["contrasts"][PREFIX]
        lines.append(f"| {source} | {signed(g[CANDIDATE]['mean'])} | {signed(g[PREFIX]['mean'])} | "
                     f"{signed(z['prefix_only_delta']['mean'])} | {signed(p['mean'])} | {signed(z['change_after_prefix']['mean'])} | "
                     + " / ".join(map(signed,p["by_seed"].values())) + " |")
    lines += ["", "首次差取两种开局已接纳车队的净效益差。两条轨迹的后续搜索均使用 ORDER_SPLIT，终态差计入后续轨迹变化与搜索时间。"
              "先平均同例种子，再在同一几何内平均不同实例，最后按来源内几何等权。", "",
              "| 三来源等权 | 均值 U | 各种子 U |", "|---|---:|---|"]
    for a in ARMS[1:]:
        lines.append(f"| FLEET_PRICE − {a} | {signed(equal[a]['mean'])} | "
                     + " / ".join(map(signed,equal[a]["by_seed"].values())) + " |")
    lines += ["", "## 门槛与费用", ""]
    if not passed:
        descriptions = dict(positive_matched_mean="同规则热图均值须为正", strict_majority_geometries="同规则严格多数几何改善",
            positive_DIST_mean="优于 DIST", positive_DIST_SLOT_mean="优于 DIST_SLOT",
            two_positive_matched_seeds="同规则至少两枚种子为正", positive_matched_all_seeds="每枚种子对同规则距离均为正",
            positive_prefix_all_seeds="每枚种子对首次纯距离均为正", positive_prefix_mean="对首次纯距离均值为正")
        for s, tests in gates.items():
            failed=[descriptions[k] for k, ok in tests.items() if not ok]
            if failed:
                lines.append(f"- {s if s!='equal_source' else '三来源等权'} 未通过{'、'.join(failed)}。")
        lines.append("")
    lines += [f"在线搜索配额 {costs['allocated_search_core_s']:,} 核秒，实际 CPU（含终点复算）{costs['cpu_s']:,.2f} 秒，"
              f"实测在线墙钟累加 {costs['online_wall_s']:,.2f} 秒，进程启动后的准备计时累加 {costs['setup_s']:,.2f} 秒，"
              f"并行控制器墙钟 {costs['wall_s']/60:.2f} 分钟。", "",
              "| 方法 | 前向次数（首次 / 后续） | 重构迭代 | 原生路线请求 | Python 搜索请求 | 前缀严格路线请求 / 缓存未命中 | 前缀扩展尝试 | 前缀接纳 | 前缀墙钟合计秒 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a,c in costs["by_method"].items():
        prefix=c.get("prefix")
        details=[f"{prefix['route_requests']:,} / {prefix['unique_route_requests']:,}",
                 f"{prefix['attempts']:,}",str(prefix['accepted']),f"{prefix['elapsed_s']:,.2f}"] if prefix else ["—"]*4
        lines.append(f"| {a} | {prefix['forwards'] if prefix else 0} / {c['forwards']:,} | {c['iterations']:,} | {c['native_route_requests']:,} | "
                     f"{c['python_search_requests']:,} | "+" | ".join(details)+" |")
    lines += ["", "前缀严格路线请求含缓存命中与未命中，原生搜索另计。", "",
              "[逐例逐种子收益](paired.csv) · [各几何、各种子与门槛](analysis.json) · [独立复算记录](independent_audit.json)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print(decision, audit["endpoints"], costs["allocated_search_core_s"])


if __name__ == "__main__":
    report(sys.argv[1])

