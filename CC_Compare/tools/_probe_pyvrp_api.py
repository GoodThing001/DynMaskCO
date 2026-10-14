"""Which pyvrp build can actually build a depot the way solver_accept_replanner expects?

solver_accept_replanner._solve_pyvrp (line 307) calls:
    m.add_depot(x, y, tw_early=0, tw_late=<int>)
and documents pyvrp **0.11.3** semantics (line 319). Under the v2 run in `a1_ext`
(pyvrp 0.14.0) that call raised:
    TypeError: Model.add_depot() got multiple values for argument 'tw_early'
120 times, so the whole PyVRP arm silently rejected every order
(util=0.00 at p_c=0, -1943.12 at main tier) -- a VOID arm, not a method result.

This probe replicates exactly that call and reports the installed signature.
Run it in each candidate env:  <env>/bin/python _probe_pyvrp_api.py
"""
import inspect

import importlib.metadata as md

try:
    from pyvrp import Model
except Exception as exc:  # noqa: BLE001
    print('IMPORT_FAIL %s: %s' % (type(exc).__name__, exc))
    raise SystemExit(0)

print('pyvrp_dist_version = %s' % md.version('pyvrp'))
try:
    print('add_depot signature = %s' % inspect.signature(Model.add_depot))
except Exception as exc:  # noqa: BLE001
    print('signature unavailable: %s' % exc)

m = Model()
try:
    d = m.add_depot(0, 0, tw_early=0, tw_late=100000)
    print('CALL_OK depot=%r' % (d,))
except Exception as exc:  # noqa: BLE001
    print('CALL_FAIL %s: %s' % (type(exc).__name__, exc))
