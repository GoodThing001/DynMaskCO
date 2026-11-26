from pathlib import Path
p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
t = t.replace("""                k = min(len(working_pool), 10 if op_choice == "regret2" else 6 + rng.randrange(12))
                mask = [rng.choice(list(working_pool)) for _ in range(k)]
                try:
                    child, _ = masked_repair(adapter, [tuple(r) for r in state], set(mask),
                                             ranking_for(adapter, state, set(mask), 0, None),
                                             op_choice, heat=None, slot=slot,""",
"""                k = min(len(working_pool), 10 if op_choice == "regret2" else 6 + rng.randrange(12))
                mask = set(rng.sample(list(working_pool), k))
                opt_b = heat_choice if heat_choice is not None else None
                try:
                    child, _ = masked_repair(adapter, [tuple(r) for r in state], mask,
                                             ranking_for(adapter, state, mask, 0, opt_b),
                                             op_choice, heat=opt_b, slot=slot,""")
t = t.replace("""        op_choice, ranking_choice = "greedy", 0
        options = [None]""",
"""        op_choice, ranking_choice = "greedy", 0
        heat_choice = None
        options = [None]""")
p.write_text(t, encoding="utf-8")
print("bugfix inner loop fixed")
