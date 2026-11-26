from pathlib import Path
p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
t = t.replace("from standard_vrp.solver.masked_repair import masked_repair, _TimeUp",
              "from standard_vrp.solver.masked_repair import (_TimeUp, masked_repair,\n                                              repair_batch)")
t = t.replace("TEMPERATURE_BASE = 2.5",
              "TEMPERATURE_BASE = 2.5\nTRIAL_ALLOWANCE_S = 0.10\nTRIAL_TOLERANCE = 1.5")
p.write_text(t, encoding="utf-8")
print("constants patched")
