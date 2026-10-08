import glob
import json

for f in sorted(glob.glob("results/a1_ext_extra/*_f/gate.json")):
    name = f.split("/")[2]
    d = json.load(open(f))
    r = d["results"]["20260926"]
    pi = r.get("_pairing_identity", {})
    for pk in ("p_c=0", "p_c=(5,10,15)"):
        blk = r.get(pk)
        if not blk:
            continue
        rows = blk.get("per_day", [])
        ids = sorted(x["i"] for x in rows)
        miss = [i for i in range(40) if i not in set(ids)]
        pairk = [k for k in blk if "minus_cond_hist" in k]
        fa = blk[pairk[0]].get("formal_adjudication") if pairk else None
        print("%-22s %-14s days=%d uniq=%d miss=%s formal_adjudication=%s identity_verified=%s" % (
            name, pk, len(ids), len(set(ids)), miss[:6], fa,
            pi.get("identity_verified")))
