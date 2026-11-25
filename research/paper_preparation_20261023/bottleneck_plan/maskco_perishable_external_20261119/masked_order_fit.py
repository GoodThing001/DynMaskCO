"""Cross-fit utility rankings of shared masked permutations with disjoint customer geometries."""
from collections import Counter, defaultdict
import json
from statistics import mean
import numpy as np
from scipy.stats import rankdata
from directed_decoder_fit import HERE, SOURCES, read
from masked_order_bank import OUT, MASKS

DESIGNS = {"physical": list(range(36)), "neural": list(range(54)), "shuffled": [*range(36), *range(54, 72)]}
CONTROLS = ("distance", "physical", "shuffled", "random")


def fit(rows, cols):
    geometries = {s: {r["geometry"] for r in rows if r["dataset"] == s} for s in SOURCES}
    cases = defaultdict(set)
    counts = Counter(r["case"] for r in rows)
    for r in rows:
        cases[r["dataset"], r["geometry"]].add(r["case"])
    x, y, w = [], [], []
    for r in rows:
        a = r["features"][:, cols]
        x.append(a - a.mean(axis=0))
        y.append((rankdata(r["labels"].round(6)) - (len(a)+1)/2) / len(a))
        group = r["dataset"], r["geometry"]
        w.append(1. / (3 * len(geometries[group[0]]) * len(cases[group]) * counts[r["case"]] * len(a)))
    w = np.repeat(w, [len(a) for a in x])
    if abs(w.sum() - 1.) > 1.e-9:
        raise ValueError("Training geometry weighting differs")
    x = np.concatenate(x)
    sd = np.sqrt(np.maximum(np.average(x ** 2, axis=0, weights=w), 1.e-8))
    x = np.clip(x / sd, -8., 8.)
    wx = x.T * w
    coef = np.linalg.solve(wx @ x + .1 * np.eye(len(sd)), wx @ np.concatenate(y))
    return sd, coef


def group_effect(rows, control):
    groups = defaultdict(list)
    for r in rows:
        groups[r["geometry"]].append(r["scores"]["neural"] - r["scores"][control])
    return {g: mean(z) for g, z in groups.items()}


def main():
    meta = read(HERE / "heldout_sources.json")["cases"]
    blocks = read(HERE / "headcapture01.json")["blocks"]
    rows, bills = [], Counter()
    for i, b in enumerate(blocks):
        case = b["case"]
        doc = read(OUT / f"{i}.json")
        if doc["case"] != case or doc["audits"] != 294 or tuple(r["mask_index"] for r in doc["rows"]) != MASKS:
            raise ValueError("Incomplete price bank")
        for name in ("prices", "audits", "segment_requests", "cpu_s", "wall_s"):
            bills[name] += doc[name]
        for r in doc["rows"]:
            a = np.asarray(r["features"]); labels = np.asarray(r["U"][1:])
            if a.shape != (48, 72) or labels.shape != (48,) or not np.isfinite(a).all() or not np.isfinite(labels).all():
                raise ValueError("Missing features or utility")
            rows.append(dict(**meta[case], case=case, case_index=i, k=r["mask_index"], features=a, labels=labels, base=r["U"][0]))
    folds = {(src, g): i % 5 for src in SOURCES for i, g in enumerate(sorted({r["geometry"] for r in rows if r["dataset"] == src}))}
    fold = lambda r: folds[r["dataset"], r["geometry"]]
    results = []
    for k in range(5):
        train = [r for r in rows if fold(r) != k and r["k"] in (0, 6)]
        test = [r for r in rows if fold(r) == k and r["k"] not in (0, 6)]
        if ({r["geometry"] for r in train} & {r["geometry"] for r in test}):
            raise ValueError("Same geometry used for labels and test")
        weights = {m: fit(train, cols) for m, cols in DESIGNS.items()}
        for r in test:
            X = r["features"]
            rng = np.random.default_rng([56411, r["case_index"], r["k"]])
            priorities = {"distance": X[:, 0], "random": rng.random(48)}
            for m, cols in DESIGNS.items():
                a = X[:, cols]
                sd, coef = weights[m]
                priorities[m] = np.clip((a - a.mean(axis=0)) / sd, -8., 8.) @ coef
            selected = {m: np.argsort(-v, kind="stable")[:6].tolist() for m, v in priorities.items()}
            scores = {m: max(r["base"], float(r["labels"][v].max())) for m, v in selected.items()}
            scores["base"] = r["base"]
            scores["oracle"] = max(r["base"], float(r["labels"].max()))
            row = {key: r[key] for key in ("geometry", "dataset", "case", "k")}
            results.append(dict(**row, scores=scores, selected=selected, fold=k))
    sources = {}
    for src in SOURCES:
        part = [r for r in results if r["dataset"] == src]
        controls = {}
        for m in (*CONTROLS, "base", "oracle"):
            geos = group_effect(part, m)
            controls[m] = dict(mean=mean(geos.values()), wins=sum(v > 1.e-6 for v in geos.values()), geometries=geos)
        reps = [mean(group_effect([r for r in part if r["k"] % 6 == rep], "physical").values()) for rep in (4, 5)]
        attained = sum(r["scores"]["oracle"] - r["scores"]["distance"] <= 1.e-6 for r in part)
        available = sum(r["scores"]["oracle"] - r["scores"]["base"] > 1.e-6 for r in part)
        sources[src] = dict(controls=controls, repeats=reps, groups=len(controls["physical"]["geometries"]),
                            contexts=len(part), distance_attains_pool_best=attained, bank_can_improve=available)
    ok = all(z["controls"][m]["mean"] > 1.e-6 and 2*z["controls"][m]["wins"] > z["groups"]
             for z in sources.values() for m in ("distance", "physical"))
    ok = ok and all(mean(z["repeats"][i] for z in sources.values()) > 1.e-6 for i in (0, 1))
    ok = ok and mean(z["controls"]["shuffled"]["mean"] for z in sources.values()) > 1.e-6
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=sources, passed=ok, costs=bills, test_contexts=len(results), fits=5), indent=2), encoding="utf8")
    (OUT/"selections.json").write_text(json.dumps(results, separators=(",", ":")), encoding="utf8")
    lines = ["# 同池跨车顺序择优的收益", "",
             "每行是 48 个相同候选中提取六个的结果。单位为几何等权的车队净效益 U；整池机会是事后全 48 个择优相对共同起点的增量。", "",
             "| 开发来源 | 热图减距离 | 改善几何 | 热图减纯物理头 | 改善几何 | 热图减置乱 | 热图减随机 | 整池机会 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for src, z in sources.items():
        b = z["controls"]
        total = z["groups"]
        lines.append(f"| {src} | {b['distance']['mean']:+.2f} | {b['distance']['wins']}/{total} | {b['physical']['mean']:+.2f} | "
                     + f"{b['physical']['wins']}/{total} | {b['shuffled']['mean']:+.2f} | {b['random']['mean']:+.2f} | "
                     + f"{b['base']['mean']-b['oracle']['mean']:+.2f} |")
    lines += ["", "| 来源 | 存在更优顺序的上下文 | 距离前六已取到全池最好成绩 |", "|---|---:|---:|"]
    for src, z in sources.items():
        lines.append(f"| {src} | {z['bank_can_improve']}/{z['contexts']} | {z['distance_attains_pool_best']}/{z['contexts']} |")
    lines += ["", f"5 折按几何隔离，54 个拟合上下文、{len(results)} 个评价上下文。"
              + f"独立复算 {bills['audits']:,} 个车队，严格路线定价 {bills['prices']:,} 次，分车数据生成累计 CPU {bills['cpu_s']:.1f} 秒。"
              + ("单核 60 秒开发门槛通过。" if ok else "单核 60 秒开发门槛未通过。"),
              "", "[预定门槛](../../同池跨车顺序的净效益学习.md) · [逐几何与两组掩码](analysis.json) · [每组六个顺序及收益](selections.json)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
