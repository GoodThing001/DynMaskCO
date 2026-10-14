"""Qualification-only release; preserve every registered scientific choice."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
original = ROOT / 'tools_remote/i3_release_1008_v1'
target = ROOT / 'tools_remote/i3_release_1008_v2'
manifest = json.loads((original / 'release.json').read_text(encoding='utf-8'))
replacements = {'scripts/evaluation/diag_i3_online_adjudicate.py',
                'scripts/tests/validate_i3_registered_release.py'}
addition = 'scripts/tests/test_i3_online_adjudication.py'
target.mkdir(exist_ok=False)
new = {}
for rel, digest in manifest.items():
    old = original / rel
    assert hashlib.sha256(old.read_bytes()).hexdigest() == digest
    source = ROOT / rel if rel in replacements else old
    dest = target / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    new[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
dest = target / addition
shutil.copy2(ROOT / addition, dest)
new[addition] = hashlib.sha256(dest.read_bytes()).hexdigest()
assert {key for key in manifest if manifest[key] != new[key]} == replacements
(target / 'release.json').write_text(json.dumps(new, indent=2), encoding='utf-8')
(target / 'qualification_revision.json').write_text(json.dumps({
    'changed': sorted(replacements), 'added': [addition],
    'all_actor_teacher_optimizer_online_driver_bytes_unchanged': True,
    'scientific_parameters_changed': False}, indent=2), encoding='utf-8')
archive = ROOT / 'tools_remote/i3_release_1008_v2.zip'
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
    for file in target.rglob('*'):
        if file.is_file():
            stream.write(file, file.relative_to(target).as_posix())
launcher = (ROOT / 'tools_remote/start_i3_registered_1008_v2.sh').read_text(encoding='utf-8')
launcher = launcher.replace('i3_registered_1008_v2', 'i3_registered_1008_v3')
launcher = launcher.replace('i3_release_qa_1008_v2', 'i3_release_qa_1008_v3')
launcher = launcher.replace('i3_release_1008_v1.zip', 'i3_release_1008_v2.zip')
(ROOT / 'tools_remote/start_i3_registered_1008_v3.sh').write_bytes(launcher.encode('utf-8'))
print('STAGED_QUALIFICATION_ONLY', len(new))
