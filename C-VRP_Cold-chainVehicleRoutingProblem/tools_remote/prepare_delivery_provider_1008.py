"""Package route-adapter engineering without changing any running release."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
paths=['scripts/delivery/delivery_contract.py','scripts/delivery/delivery_evaluator.py',
    'scripts/delivery/delivery_baselines.py','scripts/delivery/delivery_maskco.py',
    'scripts/coldchain/coldchain_contract.py','scripts/coldchain/coldchain_state.py',
    'scripts/tests/test_delivery_contract.py','scripts/tests/test_delivery_maskco.py',
    'scripts/tests/validate_delivery_pretrained.py']
manifest={}
with zipfile.ZipFile(ROOT/'tools_remote/delivery_provider_1008_v1.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        raw=(ROOT/p).read_bytes();ast.parse(raw)
        manifest[p]=hashlib.sha256(raw).hexdigest();z.writestr(p,raw)
    z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2))
print(json.dumps({'extension_files':len(manifest),'engineering_only':True}))
