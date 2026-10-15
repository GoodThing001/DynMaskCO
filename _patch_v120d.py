# -*- coding: utf-8 -*-
p = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
     r'\scripts\tests\d3_l1_v120_bypass_acceptance.py')
s = open(p, encoding='utf-8').read()
old = "len(list(run_dir.glob('EXECUTION_CONTEXT_[0-9a-f]*.json')))"
new = "len(context_records_on_disk(run_dir))"
n = s.count(old)
assert n == 2, n
s = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('replaced', n)
