"""Approved sampling amendment overlay; preserve all other v4 frozen code."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools_remote'
STAGE = TOOLS / 'i3_sampling_release_1008_v1'
PROTOCOL = ROOT / 'configs/i3_20261008_v2/protocol.json'
new_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
reg = json.loads(PROTOCOL.read_text())
assert reg['execution_authorized'] and reg['predeclared']
STAGE.mkdir(exist_ok=False)
changed = ['scripts/training/i3_capture_schedule.py', 'scripts/training/collect_i3_teachers.py',
    'scripts/training/merge_i3_teachers.py', 'scripts/tests/test_i3_capture_schedule.py',
    'scripts/tests/test_i3_collection.py', 'scripts/tests/validate_i3_registered_release.py']
manifest = {}
for rel in changed:
    target = STAGE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (ROOT / rel).read_bytes()
    ast.parse(raw)
    target.write_bytes(raw)
    manifest[rel] = hashlib.sha256(raw).hexdigest()

pipeline = (TOOLS / 'i3_registered_pipeline_1008_v4.py').read_text(encoding='utf-8')
pipeline = pipeline.replace('i3_teacher_data_audit_1008_v2', 'i3_teacher_data_audit_1008_v3')
pipeline = pipeline.replace("'scripts/training/i3_counterfactual_teacher.py', 'scripts/training/i3_teacher_archive.py',",
    "'scripts/training/i3_counterfactual_teacher.py', 'scripts/training/i3_teacher_archive.py',\n        'scripts/training/i3_capture_schedule.py',")
(STAGE / 'pipeline.py').write_text(pipeline, encoding='utf-8')
shutil.copy2(TOOLS / 'i3_teacher_data_audit_1008.py', STAGE / 'teacher_data_audit.py')
chain = (TOOLS / 'i3_teacher_data_audit_chain_1008_v2.py').read_text(encoding='utf-8')
chain = chain.replace('i3_registered_1008_v4', 'i3_registered_1008_v5')
chain = chain.replace('345604e3c3ed0ff2c40e049ded3c4a5a6784fcde1f464fe221dbc98850d8f5a5', new_sha)
chain = chain.replace("for ordinal in reg['teacher_collection']['capture_ordinals']]",
    "for ordinal in reg['teacher_collection']['capture_ordinals_by_day'][str(day)]]")
chain = chain.replace("if (main / 'pipeline_failure.json').exists():", "if (main / 'pipeline_failure.json').exists() or (main / 'collection_failure.json').exists():")
chain = chain.replace("failure=read(main / 'pipeline_failure.json')", "failure=read(main / ('pipeline_failure.json' if (main / 'pipeline_failure.json').exists() else 'collection_failure.json'))")
(STAGE / 'teacher_data_audit_chain.py').write_text(chain, encoding='utf-8')
shutil.copy2(TOOLS / 'i3_teacher_risk_diagnostic_1008.py', STAGE / 'teacher_risk_diagnostic.py')
risk = (TOOLS / 'i3_teacher_risk_chain_1008_v2.py').read_text(encoding='utf-8')
risk = risk.replace('i3_registered_1008_v4', 'i3_registered_1008_v5')
risk = risk.replace("for ordinal in addendum['ordinals']]", "for ordinal in addendum['ordinals_by_day'][str(day)]]")
risk = risk.replace("if (main / 'pipeline_failure.json').exists():", "if (main / 'pipeline_failure.json').exists() or (main / 'collection_failure.json').exists():")
risk = risk.replace("explanation=read(main / 'pipeline_failure.json')", "explanation=read(main / ('pipeline_failure.json' if (main / 'pipeline_failure.json').exists() else 'collection_failure.json'))")
(STAGE / 'teacher_risk_chain.py').write_text(risk, encoding='utf-8')
for filename in ('i3_collection_cpu_leases_1008.py', 'test_i3_collection_cpu_leases_1008.py'):
    shutil.copy2(TOOLS / filename, STAGE / filename)
for filename in ('protocol.json', 'teacher_risk_addendum.json', 'APPROVAL_AND_CHANGE_RECORD.json'):
    shutil.copy2(PROTOCOL.parent / filename, STAGE / filename)
revision = {'old_attempt': 'results/i3_registered_1008_v4',
    'new_attempt': 'results/i3_registered_1008_v5', 'protocol_sha256': new_sha,
    'approved_sampling_amendment': True, 'changed_or_added_script_paths': changed,
    'scientific_change': 'offline capture positions only',
    'settings_online_budget_saa_c0_actor_model_optimizer_unchanged': True,
    'all_formal_states_recollected': True, 'old_formal_states_reused': False,
    'teacher_objective_holdout_remains_32_states_8_days': True}
(STAGE / 'sampling_revision.json').write_text(json.dumps(revision, indent=2), encoding='utf-8')
for p in STAGE.rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'))
manifest = {p.relative_to(STAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in STAGE.rglob('*') if p.is_file()}
(STAGE / 'release.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with zipfile.ZipFile(TOOLS / 'i3_sampling_release_1008_v1.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
    for p in STAGE.rglob('*'):
        if p.is_file():
            archive.write(p, p.relative_to(STAGE).as_posix())
print('PREPARED_APPROVED_SAMPLING_RELEASE; original actor/SAA/C0/model/optimizer untouched')
