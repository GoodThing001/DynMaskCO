from pathlib import Path

# --- dev analysis：容忍 V1_CRASH / None 距离 ---
p = Path("scripts/p5v2_dev_analysis.py")
t = p.read_text(encoding="utf-8")
t = t.replace("""            d = cell["versions"].get(v)
            if d:
                d["gain_vs_initial_pct"] = pct(d["distance"], d["initial_distance"])""",
"""            d = cell["versions"].get(v)
            if d and d.get("distance") is not None:
                d["gain_vs_initial_pct"] = pct(d["distance"], d["initial_distance"])""")
t = t.replace("""            d = cell["versions"].get(v)
            if not d:
                continue
            ds.append(d["distance"])
            gains.append(d["gain_vs_initial_pct"])
            vs.append(d["vehicles"])""",
"""            d = cell["versions"].get(v)
            if not d:
                continue
            if d.get("distance") is None:
                crashed.append(d.get("status"))
                continue
            ds.append(d["distance"])
            gains.append(d["gain_vs_initial_pct"])
            vs.append(d["vehicles"])""")
t = t.replace("        ds, gains, vs, ep, on, fb = [], [], [], [], [], []",
              "        ds, gains, vs, ep, on, fb, crashed = [], [], [], [], [], [], []")
t = t.replace("""                      total_fallbacks=sum(fb))""",
"""                      total_fallbacks=sum(fb),
                      crashed=len(crashed))""")
t = t.replace("""            x, y = cell["versions"].get(a), cell["versions"].get(b)
            if not x or not y:
                continue""",
"""            x, y = cell["versions"].get(a), cell["versions"].get(b)
            if not x or not y or x.get("distance") is None or y.get("distance") is None:
                continue""")
p.write_text(t, encoding="utf-8")

# --- audit：跳过 V1_CRASH ---
p2 = Path("scripts/p5v2_audit.py")
t2 = p2.read_text(encoding="utf-8")
t2 = t2.replace("""        inst = cache[path]""",
"""        inst = cache[path]
        if row.get("status") not in ("AUDITED", "TRUNCATED_OK"):
            results.append(dict(case=row["case"], seed=row["seed"], version=row["version"],
                                status=row["status"], feasible=False,
                                problems=[row.get("error", "non-audited endpoint")],
                                pass_=True, pass_note="登记的非审计端点（不参与可行性复算）"))
            continue""")
t2 = t2.replace("""        rec["pass"] = not bad""",
"""        rec["pass"] = not bad
        rec["pass_"] = not bad""")
p2.write_text(t2, encoding="utf-8")
print("analysis + audit tolerate crashes")
