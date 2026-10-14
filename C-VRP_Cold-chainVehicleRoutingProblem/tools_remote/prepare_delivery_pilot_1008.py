"""Freeze the registered route pilot before any development instance is generated."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
paths=['scripts/delivery/delivery_contract.py','scripts/delivery/delivery_evaluator.py',
    'scripts/delivery/delivery_baselines.py','scripts/delivery/delivery_maskco.py',
    'scripts/delivery/run_delivery_pilot.py','scripts/delivery/adjudicate_delivery_pilot.py',
    'scripts/coldchain/coldchain_contract.py','scripts/coldchain/coldchain_state.py',
    'scripts/tests/test_delivery_contract.py','scripts/tests/test_delivery_maskco.py',
    'scripts/tests/validate_delivery_pretrained.py','configs/delivery_20261008_v1/protocol.json']
manifest={}
with zipfile.ZipFile(ROOT/'tools_remote/delivery_pilot_1008_v1.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        raw=(ROOT/p).read_bytes()
        if p.endswith('.py'):ast.parse(raw)
        manifest[p]=hashlib.sha256(raw).hexdigest();z.writestr(p,raw)
    z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2))
out=ROOT/'results/delivery_pilot_1008_v1';out.mkdir(exist_ok=False)
(out/'LOCAL_RELEASE.json').write_text(json.dumps({'expected_extension_sources':manifest,
    'archive_sha256':hashlib.sha256((ROOT/'tools_remote/delivery_pilot_1008_v1.zip').read_bytes()).hexdigest(),
    'development_data_generated':False,'predeclared':True},indent=2))
print(json.dumps({'sealed_extension_files':len(manifest),'development_data_generated':False}))
