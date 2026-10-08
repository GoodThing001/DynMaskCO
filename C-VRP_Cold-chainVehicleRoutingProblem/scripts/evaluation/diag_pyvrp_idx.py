# -*- coding: utf-8 -*-
"""pyvrp 0.11.3 Client/Depot 索引语义探针。"""
from pyvrp import Model

m = Model()
d0 = m.add_depot(0, 0)
c1 = m.add_client(2, 2, pickup=1)
c2 = m.add_client(3, 3, pickup=1)
d1 = m.add_depot(5, 5)
print('client dir:', [a for a in dir(c1) if not a.startswith('_')])
print('c1.idx:', getattr(c1, 'idx', 'MISSING'))
print('m.locations type/len:', type(m.locations), len(m.locations))
print('location types:', [type(x).__name__ for x in m.locations])
print('identity map:', [(i, x is c1, x is c2, x is d0, x is d1)
                        for i, x in enumerate(m.locations)])
print('c1 == c1:', c1 == c1)
try:
    print('c1 == locations[1]:', c1 == m.locations[1])
except Exception as e:
    print('eq err:', e)
try:
    print('c1 in locations:', c1 in m.locations)
except Exception as e:
    print('in err:', e)
