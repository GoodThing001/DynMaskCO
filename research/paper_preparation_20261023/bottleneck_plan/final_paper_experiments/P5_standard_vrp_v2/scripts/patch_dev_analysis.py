from pathlib import Path
p = Path("scripts/p5v2_dev_analysis.py")
t = p.read_text(encoding="utf-8")
old = """    mech = {}
    for v in ORDER:
        rc, mm, re_ = {}, {}, 0
        for cell in cells.values():
            d = cell["versions"].get(v) or {}
            for k, n in (d.get("ranking_counts") or {}).items():
                rc[k] = rc.get(k, 0) + n
            for k, n in (d.get("mask_mode_counts") or {}).items():
                mm[k] = mm.get(k, 0) + n
            re_ += d.get("regret_epochs") or 0
        mech[v] = dict(ranking_counts=rc, mask_mode_counts=mm, regret_epochs=re_)"""
new = """    mech = {}
    for v in ORDER:
        rc, mm, re_ = {}, {}, 0
        for cell in cells.values():
            d = cell["versions"].get(v) or {}
            for src, dst in ((d.get("ranking_counts"), rc), (d.get("mask_mode_counts"), mm)):
                if isinstance(src, dict):
                    for k, n in src.items():
                        dst[str(k)] = dst.get(str(k), 0) + n
                elif isinstance(src, (list, tuple)):
                    for i, n in enumerate(src):
                        dst[str(i)] = dst.get(str(i), 0) + n
            re_ += d.get("regret_epochs") or 0
        mech[v] = dict(ranking_counts=rc, mask_mode_counts=mm, regret_epochs=re_)"""
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("patched")
