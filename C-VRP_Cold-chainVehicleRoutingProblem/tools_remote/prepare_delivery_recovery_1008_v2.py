"""Seal the proven compilation-reserve fix, retaining the v1 science verbatim."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
updates = ['scripts/delivery/delivery_maskco.py', 'scripts/delivery/adjudicate_delivery_pilot.py',
           'scripts/tests/test_delivery_precompile_reserve.py',
           'scripts/tests/validate_delivery_runtime_activation.py']
old = ROOT / 'tools_remote/delivery_pilot_1008_v1.zip'
new = ROOT / 'tools_remote/delivery_pilot_1008_v2.zip'
record = {'kind': 'engineering_recovery_not_new_hypothesis',
          'old_run': 'results/delivery_pilot_1008_v1',
          'new_run': 'results/delivery_pilot_1008_v2',
          'proof': 'All 96 model-instance runs had zero cycles; fixed synthetic audit measured first JIT 13.9367s and old reserve 27.8735s > total 10s',
          'old_records_preserved': True, 'old_result_is_active_method_negative': False,
          'protocol_sha256': '8e40c705e57a6b6f2ed2244fc9c73108b43db7757596ff8ed03283f110cc3642',
          'unchanged': ['all 32 instance IDs/data seed/generator/common initial routes',
                        'all six arms and their fixed order/seeds', 'checkpoint and random seed',
                        'mask_fraction .3 and guide_weight 1', 'same CPU54 and total 10s',
                        'contract/cost weights', 'primary single method and 1%/CI gates'],
          'changed': ['exclude first compilation from runtime reserve by fixed synthetic postcompile measurements',
                      'decode/completed-candidate counters', 'actual 51-node runtime QA barrier',
                      'report active-method qualification separately from record completeness'],
          'rerun_all_six_arms_fresh': True, 'reuse_old_numeric_rows': False,
          'engineering_QA_only_before_running': True,
          'do_not_run_online_full_without_original_pilot_pass': True}
with zipfile.ZipFile(old) as source:
    raw = {n: source.read(n) for n in source.namelist() if n != 'SOURCE_MANIFEST.json'}
for path in updates:
    raw[path] = (ROOT / path).read_bytes()
    ast.parse(raw[path])
assert hashlib.sha256(raw['configs/delivery_20261008_v1/protocol.json']).hexdigest() == record['protocol_sha256']
raw['ENGINEERING_RECOVERY_RECORD.json'] = json.dumps(record, indent=2).encode()
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
with zipfile.ZipFile(new, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in raw.items():
        archive.writestr(name, data)
    archive.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
out = ROOT / 'results/delivery_pilot_1008_v2'
out.mkdir(exist_ok=False)
(out / 'LOCAL_RELEASE.json').write_text(json.dumps(
    {'extension_sources': manifest, 'archive_sha256': hashlib.sha256(new.read_bytes()).hexdigest(),
     'science_protocol_unchanged': True, 'numerical_run_started': False}, indent=2))
(out / 'ENGINEERING_RECOVERY_RECORD.json').write_text(json.dumps(record, indent=2))
print(json.dumps({'sealed_recovery_files': len(manifest), 'protocol_unchanged': True,
                  'all_old_outputs_retained': True}))
