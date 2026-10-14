"""Pre-flight the CO-enriched / MAPT providers in the interpreter that will run them.

Why this exists: cpuC already burned hours by running an interpreter that lacked a
dependency (ortools/pyvrp), and the 3080's cada died on a missing `entmax`. The same
class of failure would strike the CO-enriched / MAPT stage *after* the four legs finish,
i.e. ~10 h from now. So test the imports NOW.

Deliberately does NOT go through `ordering_providers._REGISTRY`: wiring that file
(`wire_extra_providers.py`) while the F1 legs are still running would change
`ordering_providers.py` on disk and flip the running arms' `source_stable` to False.
So we import the provider modules directly by path instead.

Usage (on 1003, with the env that will run them):
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/tools/_preflight_extra_providers.py
"""
import os
import sys
import traceback

ROOT = '/home/hzeng/project/MASKCO-Main'

PROVIDERS = [
    ('coenrich', os.path.join(ROOT, 'CC_Compare/CO-enriched-ML/a1_accept_v1'),
     'coenrich_provider', 'COEnrichedMLProvider',
     os.path.join(ROOT, 'CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts_a1')),
    ('mapt', os.path.join(ROOT, 'CC_Compare/MAPT_code/a1_accept_v1'),
     'mapt_provider', 'MAPTProvider',
     os.path.join(ROOT, 'CC_Compare/MAPT_code/a1_accept_v1/mapt_a1.pt')),
]

rc = 0
for name, d, mod, cls, ckpt in PROVIDERS:
    print('=== %s ===' % name)
    print('  provider dir : %s  exists=%s' % (d, os.path.isdir(d)))
    print('  checkpoint   : %s  exists=%s' % (ckpt, os.path.exists(ckpt)))
    if not os.path.isdir(d):
        rc = 1
        print('  SKIP: provider dir missing')
        continue
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        m = __import__(mod)
    except Exception:
        rc = 1
        print('  IMPORT FAILED:')
        traceback.print_exc()
        continue
    print('  import OK -> %s' % cls)
    try:
        obj = getattr(m, cls)(ckpt, device='cpu')
    except Exception:
        rc = 1
        print('  CONSTRUCT FAILED:')
        traceback.print_exc()
        continue
    print('  construct OK: %s' % type(obj).__name__)
    for attr in ('order', 'propose'):
        if hasattr(obj, attr):
            print('  has .%s()' % attr)

print('PREFLIGHT_RC %d' % rc)
sys.exit(rc)
