"""A-v1 对比方法臂汇总（2026-09-30，主线助手）。

从 results/a1_ext_extra/ 下各臂 gate.json 拉取：utility mean/CI、vs cond_hist 配对差、
hfr、adjudicable、timeouts、fallback 率、solver_meta 关键项，渲染 Markdown 表。
用法（服务器，a1_ext python）：
  cd C-VRP_Cold-chainVehicleRoutingProblem
  /home/hzeng/envs/a1_ext/bin/python ../CC_Compare/tools/collect_arms.py [--formal|--diagnostic|--all]
"""
import argparse
import glob
import json
import os
import re


def stat(u):
    if not u:
        return None
    return {k: u.get(k) for k in ("mean", "ci_lo", "ci_hi")}


def arm_brief(d, pname):
    r = d.get("results", {}).get("20260926", {}).get(pname)
    if not r:
        return None
    arms = r.get("arms", {})
    armkey = next((k for k in arms if k.endswith("_accept")), None)
    a = arms.get(armkey) if armkey else None
    if a is None:
        return None
    u = a.get("utility_all_days", {})
    pd = r.get("%s_minus_cond_hist" % armkey, {})
    m = r.get("solver_meta", {})
    n_solves = m.get("n_solves") or 0
    n_fail = m.get("n_fail") or 0
    return dict(
        util=stat(u), vs_cond=stat(pd),
        hfr=a.get("hard_feasible_rate"),
        adjud=u.get("adjudicable"),
        to=a.get("mean_timeouts"),
        fails=dict(a.get("fail_counts", {})),
        fallback_pct=(100.0 * n_fail / n_solves if n_solves else None),
        max_decide=m.get("max_decide_s"),
        reasons=m.get("reject_reasons"),
        paired_adjud=pd.get("adjudicable"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "formal", "diagnostic"], default="all")
    args = ap.parse_args()
    root = "results/a1_ext_extra"
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*/gate.json"))):
        name = os.path.basename(os.path.dirname(f))
        formal = name.endswith("_f")
        if args.mode == "formal" and not formal:
            continue
        if args.mode == "diagnostic" and formal:
            continue
        d = json.load(open(f, encoding="utf-8"))
        for pname in ("p_c=0", "p_c=(5,10,15)"):
            b = arm_brief(d, pname)
            if b is None:
                continue
            rows.append((name, pname, b))
    print("| 臂 | p_c | utility | vs cond_hist | hfr | fallback% | timeout/天 | max_decide | paired adjud |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for name, pname, b in rows:
        u, v = b["util"] or {}, b["vs_cond"] or {}
        fu = ("%.1f [%.1f, %.1f]" % (u["mean"], u["ci_lo"], u["ci_hi"])
              if u.get("mean") is not None else "—")
        fv = ("%.1f [%.1f, %.1f]" % (v["mean"], v["ci_lo"], v["ci_hi"])
              if v.get("mean") is not None else "—")
        fb = ("%.1f%%" % b["fallback_pct"]) if b["fallback_pct"] is not None else "—"
        md = ("%.2f" % b["max_decide"]) if b["max_decide"] is not None else "—"
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, pname, fu, fv, b["hfr"], fb, b["to"], md, b["paired_adjud"]))
    print()
    print("failures / reject_reasons（全量）:")
    for name, pname, b in rows:
        if b["fails"] or b["reasons"]:
            print("  %-22s %-14s fails=%s reasons=%s" % (name, pname, b["fails"], b["reasons"]))


if __name__ == "__main__":
    main()
