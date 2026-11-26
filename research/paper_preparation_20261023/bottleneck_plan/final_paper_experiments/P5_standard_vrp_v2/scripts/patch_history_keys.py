from pathlib import Path
p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
t = t.replace("                history[cp] = dict(distance=val_c, vehicles=k_c, at=round(elapsed, 3))",
              "                history[str(int(cp))] = dict(distance=val_c, vehicles=k_c,\n                                             at=round(elapsed, 3))")
t = t.replace("            history[cp] = dict(distance=val, vehicles=k, at=round(time.perf_counter() - started, 3))",
              "            history[str(int(cp))] = dict(distance=val, vehicles=k,\n                                        at=round(time.perf_counter() - started, 3))")
p.write_text(t, encoding="utf-8")
print("history keys normalized")
