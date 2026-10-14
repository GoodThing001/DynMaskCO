import glob, json, os
for p in sorted(glob.glob('results/a1_ext_formal2/*/gate.json')):
    name = os.path.basename(os.path.dirname(p))
    if not any(name.startswith(t) for t in ('mvmoe','routefinder','cada','rrnco')):
        continue
    d = json.load(open(p, encoding='utf-8'))
    for yr, yb in (d.get('results') or {}).items():
        pid = yb.get('_pairing_identity') or {}
        for pname, blk in yb.items():
            if pname.startswith('_') or not isinstance(blk, dict): continue
            for an, ab in (blk.get('arms') or {}).items():
                if an in ('cond_hist','uncond_hist','explicit_feat'): continue
                u = ab.get('utility_all_days') or {}
                print('  %-18s %-16s util=%-9s to=%-6s hfr=%s paired=%s' % (
                    name, pname, ('%.1f'%u['mean']) if u.get('mean') is not None else 'n/a',
                    ab.get('mean_timeouts'), ab.get('hard_feasible_rate'),
                    pid.get('identity_verified')))
