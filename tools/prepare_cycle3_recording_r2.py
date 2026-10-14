from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parents[1]
old = root / 'tools/record_dual_track_cycle3_20261013.py'
new = root / 'tools/record_dual_track_cycle3_20261013_r2.py'
t = old.read_text(encoding='utf-8')
needle = "            status['latest_literature_record'] = report.relative_to(EXT).as_posix()\n"
assert t.count(needle) == 1
t = t.replace(needle, needle + "            status['latest_delivery_stage_receipts_root'] = spec['tracks']['delivery']['directory']\n            status['latest_stage_receipts_root'] = spec['tracks']['pickup']['production_directory']\n")
ast.parse(t)
with new.open('x', encoding='utf-8', newline='\n') as f:
    f.write(t)
p = root / 'research/review_3080_20261010/dual_track_optimization_20261013_root_v1/cycle3_20261013_root_v1'
reg = {'original_recording_tool_sha256': hashlib.sha256(old.read_bytes()).hexdigest(), 'new_recording_tool_sha256': hashlib.sha256(new.read_bytes()).hexdigest(), 'original_tool_preserved': True, 'scope': 'Add current stage selectors only; preserve historical scientific/terminal state', 'cycle': 3}
with (p / 'RECORDING_TOOL_REGISTRATION_R2.json').open('x', encoding='utf-8') as f:
    f.write(json.dumps(reg, indent=2) + '\n')
print('R2 recording source sealed; original source preserved')
