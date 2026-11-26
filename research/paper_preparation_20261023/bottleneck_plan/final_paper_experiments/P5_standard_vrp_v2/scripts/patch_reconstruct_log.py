from pathlib import Path
p = Path("standard_vrp/solver/masked_repair.py")
t = p.read_text(encoding="utf-8")
t = t.replace("def reconstruct(adapter, parent, mask, heat=None, regret=False):",
              "def reconstruct(adapter, parent, mask, heat=None, regret=False, log=None):")
t = t.replace("""            insert_customer(adapter, rebuilt, values, pending[winner], chosen)
            pending.pop(winner)""",
"""            insert_customer(adapter, rebuilt, values, pending[winner], chosen)
            if log is not None:
                log.append(dict(cust=int(pending[winner]), regret=float(highest),
                                move=tuple(int(x) for x in chosen)))
            pending.pop(winner)""")
t = t.replace("""            move, _ = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)
            insert_customer(adapter, rebuilt, values, cust, move)""",
"""            move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)
            insert_customer(adapter, rebuilt, values, cust, move)
            if log is not None:
                log.append(dict(cust=int(cust), regret=float(reg),
                                move=tuple(int(x) for x in move)))""")
t = t.replace("""    out = reconstruct(adapter, [list(r) for r in parent], mask, heat=heat, regret=regret)
    if log is not None:
        log.append(dict(mask=mask, regret=regret))""",
"""    out = reconstruct(adapter, [list(r) for r in parent], mask, heat=heat, regret=regret,
                      log=log)""")
p.write_text(t, encoding="utf-8")
print("log support added")
