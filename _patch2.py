p = r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\scripts\tests\d3_l1_v118_synthetic_worker.py'
s = open(p, encoding='utf-8').read()
old = "        return ['all_finite', 'first_argmax_rule'], a, b"
new = "        return ['all_finite'], a, b"
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('worker defect expectation fixed')
