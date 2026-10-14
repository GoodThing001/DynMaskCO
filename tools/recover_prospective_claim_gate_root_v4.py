"""Isolated recovery of the new gate generator; preserve its failed v3 bytes."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'tools/harden_prospective_claim_gate_root_v3.py'
new=ROOT/'tools/harden_prospective_claim_gate_root_v4.py'
s=old.read_text(encoding='utf-8').replace("OUT=B/'formal_run_requirements_root_v3'","OUT=B/'formal_run_requirements_root_v4'")
anchor='    s=s[:start]+body+s[end:];ast.parse(s)'
assert s.count(anchor)==1
s=s.replace(anchor,"    body=body.replace('p = bind(root, ref);bound.append','p = artifact(root, ref);bound.append')\n"+anchor)
s=s.replace('class FiniteAndImmutableRegression(NewRunGateTests):','''class FiniteAndImmutableRegression(unittest.TestCase):
    setUp=NewRunGateTests.setUp
    tearDown=NewRunGateTests.tearDown
    ref=NewRunGateTests.ref
    run_audit=NewRunGateTests.run_audit''')
s=s.replace('capture_output=True,text=True,timeout=60)','capture_output=True,text=True,timeout=180)')
ast.parse(s)
with new.open('x',encoding='utf-8') as f:f.write(s)
print('v4 isolated generator prepared; old v2/v3 unchanged')
