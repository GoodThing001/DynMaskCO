from pathlib import Path
p = Path("standard_vrp/tests/test_masked_repair_v2.py")
t = p.read_text(encoding="utf-8")
t = t.replace("""    orig = mr.choose_slot

    def counting(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)

    mr.choose_slot = counting""",
"""    orig = mr._choose_slot

    def counting(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)

    mr._choose_slot = counting""")
t = t.replace("    finally:\n        mr.choose_slot = orig", "    finally:\n        mr._choose_slot = orig")
p.write_text(t, encoding="utf-8")
print("d3 patched")
