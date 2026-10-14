"""New sealed attempt for unordered identity/CPU leases; protocol bytes unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools_remote'
old, new = TOOLS / 'i3_release_1008_v2', TOOLS / 'i3_release_1008_v3'
manifest = json.loads((old / 'release.json').read_text(encoding='utf-8'))
new.mkdir(exist_ok=False)
changed = {'scripts/training/i3_counterfactual_teacher.py',
           'scripts/tests/test_i3_teacher_archive.py',
           'scripts/tests/validate_i3_registered_release.py'}
addition = 'scripts/evaluation/diag_i3_teacher_probe.py'
new_manifest = {}
for rel, digest in manifest.items():
    original = old / rel
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    src = ROOT / rel if rel in changed else original
    dst = new / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    new_manifest[rel] = hashlib.sha256(dst.read_bytes()).hexdigest()
for rel in sorted((changed - set(manifest)) | {addition}):
    dst = new / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)
    new_manifest[rel] = hashlib.sha256(dst.read_bytes()).hexdigest()
(new / 'release.json').write_text(json.dumps(new_manifest, indent=2), encoding='utf-8')
(new / 'engineering_revision.json').write_text(json.dumps({
    'old_attempt': 'results/i3_registered_1008_v3', 'new_attempt': 'results/i3_registered_1008_v4',
    'changed_release_files': sorted(k for k in manifest if new_manifest[k] != manifest[k]),
    'added_release_files': sorted(set(new_manifest) - set(manifest)),
    'fixes': ['canonical unordered teacher membership identity', 'distinct dynamic CPU leases',
              'independent all-state teacher audit before training', 'fresh engineering archives in QA'],
    'scientific_parameters_changed': False, 'old_formal_states_reused': False,
    'model_actor_optimizer_saa_online_driver_unchanged': True,
    'protocol_sha256': hashlib.sha256((ROOT / 'configs/i3_20261008_v1/protocol.json').read_bytes()).hexdigest()
}, indent=2), encoding='utf-8')
with zipfile.ZipFile(TOOLS / 'i3_release_1008_v3.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
    for path in new.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(new).as_posix())

pipeline = (TOOLS / 'i3_registered_pipeline_1008.py').read_text(encoding='utf-8')
pipeline = pipeline.replace('import time\n', 'import time\nfrom i3_collection_cpu_leases_1008 import collect_shards\n')
pipeline = pipeline.replace('time.monotonic() + 3600', 'time.monotonic() + 7200')
pipeline = pipeline.replace('def collect(day):', 'def collect(day, cpu):')
pipeline = pipeline.replace('cpus=[allowed[8 + day % 8]])', 'cpus=[cpu])')
before = '''    # Workers consume days in order; all 32 are required, including hard-infeasible labels.
    with ThreadPoolExecutor(max_workers=8) as pool:
        shards = list(pool.map(collect, cfg['collection']['days']))'''
after = '''    def collection_failure(day, exc):
        write(out / 'collection_failure.json', {'day_id': day, 'error_type': type(exc).__name__,
            'error': str(exc), 'scientific_parameters_changed': False, 'online_gain_established': False})
    shards = collect_shards(cfg['collection']['days'], allowed[8:16], collect, collection_failure)'''
assert before in pipeline
pipeline = pipeline.replace(before, after)
before = "    settings = reg['settings']\n"
after = '''    # Read-only independent audit is a qualification barrier, not a label or
    # reward-selection gate. Every registered state remains in the experiment.
    audit_out = project / 'results/i3_teacher_data_audit_1008_v2'
    audit_path = audit_out / 'TEACHER_DATA_AUDIT.json'
    state('waiting_independent_teacher_audit', states=128)
    audit_deadline = time.monotonic() + 1800
    while not audit_path.exists():
        if (audit_out / 'failure.json').exists():
            raise RuntimeError('Independent teacher qualification audit failed: ' + str(read(audit_out / 'failure.json')))
        if time.monotonic() > audit_deadline:
            raise RuntimeError('Independent teacher audit unavailable after fixed 30min qualification wait')
        time.sleep(10)
    audit = read(audit_path)
    expected_inputs = {(row['report_sha256'], sha(row['archive'])) for row in shards}
    observed_inputs = {(row['report_sha256'], row['archive_sha256']) for row in audit['inputs']}
    if (audit['main_output'] != str(out) or audit['protocol_sha256'] != protocol_sha or
            audit['audit_source_sha256'] != sha(out / 'teacher_data_audit.py') or
            not audit['terminal_trace_reevaluation_complete'] or
            (audit['summary']['states'], audit['summary']['days']) != (128, 32) or
            audit['split']['training']['states'] != 96 or
            audit['split']['teacher_objective_holdout']['states'] != 32 or
            len(audit['inputs']) != 32 or expected_inputs != observed_inputs or
            sorted((row['day_id'], row['ordinal']) for row in audit['rows']) !=
            sorted(tuple(key) for key in merged['expected_keys'])):
        raise RuntimeError('Independent teacher audit does not match every sealed registered shard')
    seal()
    write(out / 'TEACHER_AUDIT_ACCEPTED.json', {'audit_path': str(audit_path),
        'audit_sha256': sha(audit_path), 'all_registered_states': 128, 'qualification_verified': True})
    settings = reg['settings']
'''
assert before in pipeline
pipeline = pipeline.replace(before, after)
(TOOLS / 'i3_registered_pipeline_1008_v4.py').write_bytes(pipeline.encode('utf-8'))

audit = (TOOLS / 'i3_teacher_data_audit_chain_1008.py').read_text(encoding='utf-8')
audit = audit.replace('i3_registered_1008_v3', 'i3_registered_1008_v4')
audit = audit.replace("json.dump({'summary': summary, 'split': split, 'rows': rows,",
                      "json.dump({'main_output': str(main), 'summary': summary, 'split': split, 'rows': rows,")
(TOOLS / 'i3_teacher_data_audit_chain_1008_v2.py').write_bytes(audit.encode('utf-8'))
risk = (TOOLS / 'i3_teacher_risk_chain_1008.py').read_text(encoding='utf-8')
risk = risk.replace('i3_registered_1008_v3', 'i3_registered_1008_v4')
needle = "    state('engineering_real_teacher_smoke')"
replacement = '''    state('waiting_release_qa')
    qa_deadline = time.monotonic() + 7200
    while not (main / 'qa.exit').exists():
        if time.monotonic() > qa_deadline:
            raise RuntimeError('Recovery QA unavailable after fixed 2h wait')
        time.sleep(30)
    if not {'RUN_EXIT=0', 'SEAL_EXIT=0'} <= set((main / 'qa.exit').read_text().splitlines()):
        raise RuntimeError('Recovery QA failed; no teacher risk evaluation')
    state('engineering_real_teacher_smoke')'''
assert needle in risk
risk = risk.replace(needle, replacement)
(TOOLS / 'i3_teacher_risk_chain_1008_v2.py').write_bytes(risk.encode('utf-8'))

launcher = (TOOLS / 'start_i3_registered_1008_v3.sh').read_text(encoding='utf-8')
launcher = launcher.replace('i3_registered_1008_v3', 'i3_registered_1008_v4')
launcher = launcher.replace('i3_release_qa_1008_v3', 'i3_release_qa_1008_v4')
launcher = launcher.replace('i3_release_1008_v2.zip', 'i3_release_1008_v3.zip')
launcher = launcher.replace('tools_remote/i3_registered_pipeline_1008.py', 'tools_remote/i3_registered_pipeline_1008_v4.py')
needle = 'find "$EXT/scripts" -type f'
copies = '''cp "$ROOT/tools_remote/i3_collection_cpu_leases_1008.py" "$OUT/i3_collection_cpu_leases_1008.py"
cp "$ROOT/tools_remote/test_i3_collection_cpu_leases_1008.py" "$OUT/test_i3_collection_cpu_leases_1008.py"
cp "$ROOT/tools_remote/i3_teacher_data_audit_1008.py" "$OUT/teacher_data_audit.py"
cp "$ROOT/tools_remote/i3_teacher_data_audit_chain_1008_v2.py" "$OUT/teacher_data_audit_chain.py"
cp "$ROOT/tools_remote/i3_teacher_risk_diagnostic_1008.py" "$OUT/teacher_risk_diagnostic.py"
cp "$ROOT/tools_remote/i3_teacher_risk_chain_1008_v2.py" "$OUT/teacher_risk_chain.py"
cp "$ROOT/configs/i3_20261008_v1/teacher_risk_addendum.json" "$OUT/teacher_risk_addendum.json"
cp "$OUT/overlay/engineering_revision.json" "$OUT/engineering_revision.json"
'''
assert needle in launcher
launcher = launcher.replace(needle, copies + needle)
needle = 'chmod -R a-w "$EXT" "$OUT/warmstart"'
helper_seal = '''sha256sum "$OUT/i3_collection_cpu_leases_1008.py" "$OUT/test_i3_collection_cpu_leases_1008.py" "$OUT/teacher_data_audit.py" "$OUT/teacher_data_audit_chain.py" "$OUT/teacher_risk_diagnostic.py" "$OUT/teacher_risk_chain.py" "$OUT/teacher_risk_addendum.json" "$OUT/engineering_revision.json" >> "$OUT/SOURCE_MANIFEST.sha256"
chmod a-w "$OUT/i3_collection_cpu_leases_1008.py" "$OUT/test_i3_collection_cpu_leases_1008.py" "$OUT/teacher_data_audit.py" "$OUT/teacher_data_audit_chain.py" "$OUT/teacher_risk_diagnostic.py" "$OUT/teacher_risk_chain.py" "$OUT/teacher_risk_addendum.json" "$OUT/engineering_revision.json"
'''
launcher = launcher.replace(needle, helper_seal + needle)
launcher = launcher.replace('timeout 1800s', 'timeout 7200s')
needle = 'date > "$OUT/qa_started_at.txt"'
launcher = launcher.replace(needle, needle + '\n"$PY" "$OUT/test_i3_collection_cpu_leases_1008.py" > "$OUT/cpu_lease_test.log" 2>&1 || exit 1')
(TOOLS / 'start_i3_registered_1008_v4.sh').write_bytes(launcher.encode('utf-8'))
for name, src in [('data_audit', 'start_i3_teacher_data_audit_1008.sh'), ('risk', 'start_i3_teacher_risk_1008.sh')]:
    text = (TOOLS / src).read_text(encoding='utf-8')
    if name == 'data_audit':
        text = text.replace('i3_teacher_data_audit_1008_v1', 'i3_teacher_data_audit_1008_v2')
        text = text.replace('$ROOT/tools_remote/i3_teacher_data_audit_1008.py', '$ROOT/results/i3_registered_1008_v4/teacher_data_audit.py')
        text = text.replace('$ROOT/tools_remote/i3_teacher_data_audit_chain_1008.py', '$ROOT/results/i3_registered_1008_v4/teacher_data_audit_chain.py')
        dest = 'start_i3_teacher_data_audit_1008_v2.sh'
    else:
        text = text.replace('i3_teacher_risk_1008_v1', 'i3_teacher_risk_1008_v2')
        text = text.replace('$ROOT/tools_remote/i3_teacher_risk_diagnostic_1008.py', '$ROOT/results/i3_registered_1008_v4/teacher_risk_diagnostic.py')
        text = text.replace('$ROOT/tools_remote/i3_teacher_risk_chain_1008.py', '$ROOT/results/i3_registered_1008_v4/teacher_risk_chain.py')
        text = text.replace('$ROOT/configs/i3_20261008_v1/teacher_risk_addendum.json', '$ROOT/results/i3_registered_1008_v4/teacher_risk_addendum.json')
        dest = 'start_i3_teacher_risk_1008_v2.sh'
    (TOOLS / dest).write_bytes(text.encode('utf-8'))
print('PREPARED_ENGINEERING_RECOVERY; NO SCIENTIFIC PARAMETER CHANGE')
