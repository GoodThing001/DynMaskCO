# -*- coding: utf-8 -*-
"""P5-B 标准 VRPTW 分析 + 数据泄漏审计（按 P5_预登记 §8）。

- 分类汇总：C1/C2/R1/R2/RC1/RC2 平均 K 与平均 D（3 种子平均）；
- 逐实例字典序 (K,D) 两两胜负（种子配对比较）；
- BKS 参考：bks_reference.json（CVRPLIB .sol 路线本地复算，48/56 双精度可行 + 8 例截断口径，逐例标注）；
- 泄漏审计：56 例逐一标注冷链角色——训练几何（f0_r0：r101–r112、r201–r211、rc101–rc108、
  rc201–rc204、rc206–rc208，共 38；rc205 不在训练集）、验证几何（C2）、折外测试几何（C1）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "P5_standard_vrp"
sys.path.insert(0, str(P5))

FAMILY = {}
for i in range(1, 10):
    FAMILY["C1%02d" % i] = "C1"
for i in range(1, 9):
    FAMILY["C2%02d" % i] = "C2"
for i in range(1, 13):
    FAMILY["R1%02d" % i] = "R1"
for i in range(1, 12):
    FAMILY["R2%02d" % i] = "R2"
for i in range(1, 9):
    FAMILY["RC1%02d" % i] = "RC1"
    FAMILY["RC2%02d" % i] = "RC2"

TRAIN = set()
for i in range(1, 13):
    TRAIN.add("R1%02d" % i)
for i in range(1, 12):
    TRAIN.add("R2%02d" % i)
for i in range(1, 9):
    TRAIN.add("RC1%02d" % i)
for i in (1, 2, 3, 4, 6, 7, 8):
    TRAIN.add("RC2%02d" % i)
assert len(TRAIN) == 38, len(TRAIN)
VALIDATION = {("C2%02d" % i) for i in range(1, 9)}
FOLDTEST = {("C1%02d" % i) for i in range(1, 10)}


def role_of(name):
    if name in TRAIN:
        return "train(R/RC)"
    if name in VALIDATION:
        return "validation(C2)"
    if name in FOLDTEST:
        return "fold0_test(C1)"
    return "RC205-untrained"


def lex_better(k1, d1, k2, d2, eps=1e-9):
    if k1 < k2:
        return 1
    if k1 > k2:
        return -1
    if d1 < d2 - eps:
        return 1
    if d1 > d2 + eps:
        return -1
    return 0


def load_results(dirs):
    rows = []
    for d in dirs:
        base = Path(d)
        out = base / "out"
        if not out.exists():
            out = base / "endpoints"
        if not out.exists():
            continue
        for f in sorted(out.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("objective") == "vrptw" and r.get("status") in ("AUDITED", "TRUNCATED_OK"):
                rows.append(r)
    return rows


def main(dirs, out_json):
    rows = load_results(dirs)
    by_cell = {(r["case"], r["method"], r["seed"]): r for r in rows}
    instances = sorted({r["case"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    seeds = (69001, 69002, 69003)
    # 每 (实例,方法) 汇总
    per = {}
    for n in instances:
        for m in methods:
            cells = [by_cell[(n, m, s)] for s in seeds if (n, m, s) in by_cell]
            if not cells:
                continue
            ks = [c["vehicles"] for c in cells]
            ds = [c["distance"] for c in cells]
            per[(n, m)] = dict(mean_k=sum(ks) / len(ks), mean_d=sum(ds) / len(ds),
                               ks=ks, ds=ds, role=role_of(n), family=FAMILY.get(n, "?"))
    # 类别汇总
    print("== P5-B VRPTW 分类汇总（3 种子平均）==")
    print("%-14s %10s %10s %10s %10s %10s %10s %8s" % ("method", "C1", "C2", "R1", "R2", "RC1", "RC2", "overall"))
    for m in methods:
        line = [m]
        for fam in ("C1", "C2", "R1", "R2", "RC1", "RC2"):
            cells = [per[(n, m)] for n in instances if (n, m) in per and FAMILY.get(n) == fam]
            if cells:
                line.append("%.1f/%.1f" % (sum(c["mean_k"] for c in cells) / len(cells),
                                           sum(c["mean_d"] for c in cells) / len(cells)))
            else:
                line.append("-")
        cells = [per[(n, m)] for n in instances if (n, m) in per]
        line.append("%.1f/%.1f" % (sum(c["mean_k"] for c in cells) / len(cells),
                                   sum(c["mean_d"] for c in cells) / len(cells)))
        print("%-14s %10s %10s %10s %10s %10s %10s %8s" % tuple(line))
    # 逐实例两两字典序胜负（种子配对）
    wins = {m: {} for m in methods}
    for n in instances:
        for a in methods:
            for b in methods:
                if a >= b or (n, a) not in per or (n, b) not in per:
                    continue
                w = l = t = 0
                for s in seeds:
                    if (n, a, s) not in by_cell or (n, b, s) not in by_cell:
                        continue
                    ra, rb = by_cell[(n, a, s)], by_cell[(n, b, s)]
                    r = lex_better(ra["vehicles"], ra["distance"], rb["vehicles"], rb["distance"])
                    w += r == 1
                    l += r == -1
                    t += r == 0
                wins[a].setdefault(b, [0, 0, 0])
                wins[a][b][0] += w
                wins[a][b][1] += l
                wins[a][b][2] += t
    # 泄漏审计标注表
    print("\n== 泄漏审计标注（56 例角色分布）==")
    from collections import Counter
    cnt = Counter(role_of(n) for n in instances)
    for role, c in sorted(cnt.items()):
        print("  %-18s %d 例" % (role, c))
    out = P5 / out_json
    out.write_text(json.dumps(dict(
        per={("%s|%s" % k): v for k, v in per.items()},
        wins=wins, methods=methods, instances=instances,
        train=sorted(TRAIN), validation=sorted(VALIDATION), foldtest=sorted(FOLDTEST)),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, "cells:", len(rows))


if __name__ == "__main__":
    dirs = sys.argv[1].split(",") if len(sys.argv) > 1 else [str(P5 / "results_vrptw_full")]
    out = sys.argv[2] if len(sys.argv) > 2 else "analysis_p5b.json"
    main(dirs, out)
