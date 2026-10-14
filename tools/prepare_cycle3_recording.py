from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parents[1]
original = root / 'tools/record_dual_track_cycle2_20261013.py'
out = root / 'tools/record_dual_track_cycle3_20261013.py'
t = original.read_text(encoding='utf-8')
for before, after in [
    ('cycle2', 'cycle3'), ('CYCLE2', 'CYCLE3'),
    ('双线第二轮_精确调度与增量补救资格_2026-10-13.md', '双线第三轮_实际依赖守卫与精确距离缓存_2026-10-13.md'),
    ('2026-10-13 双线第二轮：精确调度与增量场景补救新候选均完成实际工程执行', '2026-10-13 双线第三轮：实际依赖守卫与精确距离缓存新候选均完成实际工程执行'),
    ('[第二轮记录]', '[第三轮记录]'),
    ('新增PyVRP原论文方法依据；工程循环2/12', '沿用PyVRP原论文缓存机制依据；工程循环3/12'),
    ("'cycle': 2", "'cycle': 3"),
]:
    assert before in t, before
    t = t.replace(before, after)
ast.parse(t)
with out.open('x', encoding='utf-8', newline='\n') as f:
    f.write(t)
p = root / 'research/review_3080_20261010/dual_track_optimization_20261013_root_v1/cycle3_20261013_root_v1'
reg = {'recording_tool_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'literature_sha256': hashlib.sha256((p/'LITERATURE_AND_CONTRACT.md').read_bytes()).hexdigest(), 'scope': 'collect completed evidence only; no scientific upgrades or reruns', 'cycle': 3}
with (p / 'RECORDING_REGISTRATION.json').open('x', encoding='utf-8') as f:
    f.write(json.dumps(reg, indent=2) + '\n')
print('cycle3 recording source AST and registration sealed')
