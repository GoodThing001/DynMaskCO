from pathlib import Path
p = Path("standard_vrp/solver/masked_repair.py")
t = p.read_text(encoding="utf-8")

t = t.replace("def reconstruct(adapter, parent, mask, heat=None, regret=False, log=None):",
              "def reconstruct(adapter, parent, mask, heat=None, regret=False, log=None,\n                deadline=None):")
t = t.replace("""        pending = sorted(mask)
        while pending:
            highest, winner, chosen = BAD, None, None""",
"""        pending = sorted(mask)
        while pending:
            _guard(deadline)
            highest, winner, chosen = BAD, None, None""")
t = t.replace("""        for cust in mask:
            move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)""",
"""        for cust in mask:
            _guard(deadline)
            move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)""")
# repair_batch: 硬截止 + 捕获 _TimeUp
t = t.replace("""def repair_batch(adapter, state, best, group, heat=None, seed=0, seconds=0.1,
                 tolerance=1.5, related=False, regret=False, iteration_cap=0,
                 deadline=None):""",
"""def repair_batch(adapter, state, best, group, heat=None, seed=0, seconds=0.1,
                 tolerance=1.5, related=False, regret=False, iteration_cap=0,
                 deadline=None, hard_deadline=None):""")
t = t.replace("""        nxt = reconstruct(adapter, state, mask, heat=heat, regret=regret)
        calls += 1""",
"""        try:
            nxt = reconstruct(adapter, state, mask, heat=heat, regret=regret,
                              deadline=hard_deadline)
        except _TimeUp:
            break
        calls += 1""")
p.write_text(t, encoding="utf-8")

# pool_search：所有调用传硬截止
p2 = Path("standard_vrp/solver/pool_search.py")
t2 = p2.read_text(encoding="utf-8")
t2 = t2.replace("    started = time.perf_counter()\n    deadline = started + seconds - 0.8",
                "    started = time.perf_counter()\n    deadline = started + seconds - 0.8\n    hard = started + seconds - 0.05")
t2 = t2.replace("""                                       related=bool(active_mode), regret=bool(g),
                                       deadline=deadline)""",
"""                                       related=bool(active_mode), regret=bool(g),
                                       deadline=deadline, hard_deadline=hard)""")
t2 = t2.replace("""                               tolerance=temperature, related=bool(active_mode),
                               regret=(op_choice == "regret2"), deadline=deadline)""",
"""                               tolerance=temperature, related=bool(active_mode),
                               regret=(op_choice == "regret2"), deadline=deadline,
                               hard_deadline=hard)""")
t2 = t2.replace("""                                             ranking_for(adapter, state, set(mask), 0, None),
                                             op_choice, heat=None, slot=slot,
                                             deadline=time.perf_counter() + allowance)""",
"""                                             ranking_for(adapter, state, set(mask), 0, None),
                                             op_choice, heat=None, slot=slot,
                                             deadline=min(time.perf_counter() + allowance, hard))""")
p2.write_text(t2, encoding="utf-8")
print("hard deadline wired")
