"""New append-only necessary package gate; no historical package mutation."""
import ast,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'research/review_3080_20261010/reproduction'
OLD=B/'formal_run_requirements_root_v2';OUT=B/'formal_run_requirements_root_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    s=(OLD/'new_run_gate_adapter.py').read_text(encoding='utf-8')
    anchor='    reg, registry, bundle = load(registration_path), load(registry_path), load(bundle_path)'
    s=s.replace(anchor,"""    # Bind every input before interpreting it. Recheck all actual bytes at end.
    bound = [(p, sha(p)) for p in (registration_path, registry_path, bundle_path)]
    def bind(root, ref):
        p = artifact(root, ref);bound.append((p, ref['sha256']));return p
"""+anchor)
    start=s.index('def audit(');end=s.index('\n\ndef main():',start)
    body=s[start:end]
    # Leave the bind helper's call to the original strict path/hash validator.
    body=body.replace(' = artifact(root,',' = bind(root,').replace('{k: artifact(root,','{k: bind(root,').replace('        artifact(root,','        bind(root,')
    body=body.replace("    cohorts = reg.get('cohorts')","""    maximum = load(program).get('new_bounded_cycles_per_line')
    if type(maximum) is not int or not 1 <= j <= maximum:
        reject('PROGRAM finite family limit exceeded')
    cohorts = reg.get('cohorts')""")
    body=body.replace('    alpha = .05 / (j * (j + 1)) / 2 / (len(contrasts) * 3)',"""    if len({c['treatment'] for c in contrasts}) != 1:
        reject('single frozen candidate per confirmation family required')
    required = ({'same_architecture_random','qualified_strong_nonlearned_rolling'} if line == 'delivery'
                else {'qualified_strong_nonlearned_rolling','same_init_reconstruction_control'})
    role_controls = [c['control'] for c in contrasts if c.get('control_role') in required]
    if len(role_controls) != len(required) or len(set(role_controls)) != len(required):
        reject('required comparison roles need distinct uniquely registered control arms')
    alpha = .05 / (j * (j + 1)) / 2 / (len(contrasts) * 3)""")
    body=body.replace("    return {'kind': 'registered_confirmation_raw_package_subset_gate_v1',","""    for p, expected in bound:
        if sha(p) != expected:
            reject('bound input/raw/artifact changed during audit')
    return {'kind': 'registered_confirmation_raw_package_subset_gate_root_v3',""")
    s=s[:start]+body+s[end:];ast.parse(s)
    (OUT/'new_run_gate_adapter.py').write_text(s,encoding='utf-8')
    tests=(OLD/'test_new_run_gate.py').read_text(encoding='utf-8')
    tests=tests.replace("if __name__=='__main__':unittest.main()",'''class FiniteAndImmutableRegression(NewRunGateTests):
    def test_family13_rejected_even_with_adequate_adjusted_tail(self):
        self.reg['family_index']=13
        alpha=.05/(13*14)/2/(2*3)
        self.reg['statistics'].update(two_sided_alpha=alpha,bootstrap_replicates=3000000,
            percentiles=[alpha/2*100,(1-alpha/2)*100])
        with self.assertRaisesRegex(ValueError,'finite family limit'):self.run_audit()

    def test_shared_control_cannot_impersonate_two_required_roles(self):
        self.reg['contrasts'][1]['control']='random'
        with self.assertRaisesRegex(ValueError,'distinct uniquely'):self.run_audit()

    def test_two_treatments_cannot_masquerade_as_one_frozen_candidate(self):
        self.reg['arms'].append('another_model');self.reg['contrasts'][1]['treatment']='another_model'
        with self.assertRaisesRegex(ValueError,'single frozen'):self.run_audit()

    def test_raw_changed_after_read_rejected(self):
        original=g.load
        def read_then_change(path):
            value=original(path)
            if path.name.endswith('_raw.json'):
                path.write_text('changed after its actual accepted read')
            return value
        from unittest.mock import patch
        with patch.object(g,'load',side_effect=read_then_change):
            with self.assertRaisesRegex(ValueError,'changed during audit'):self.run_audit()

    def test_event_artifact_changed_later_rejected(self):
        original=g.load;changed=[False]
        def read_and_change_later(path):
            value=original(path)
            if path.name.endswith('_raw.json') and not changed[0]:
                changed[0]=True;self.target=self.root/value['event_journal']['path']
            elif path.name.endswith('_raw.json') and changed[0]:
                self.target.write_text('event bytes changed after its accepted hash')
            return value
        from unittest.mock import patch
        with patch.object(g,'load',side_effect=read_and_change_later):
            with self.assertRaisesRegex(ValueError,'changed during audit'):self.run_audit()

if __name__=='__main__':unittest.main()
''')
    ast.parse(tests);(OUT/'test_new_run_gate.py').write_text(tests,encoding='utf-8')
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'engineering necessary subset hardening only; no formal result or historical release change',
      'parent_adapter_sha256':sha(OLD/'new_run_gate_adapter.py'),'parent_test_sha256':sha(OLD/'test_new_run_gate.py'),
      'new_adapter_sha256':sha(OUT/'new_run_gate_adapter.py'),'new_test_sha256':sha(OUT/'test_new_run_gate.py'),
      'formal_online_gain':False,'historical_full_reproduction':False}
    (OUT/'REGISTRATION.json').write_text(json.dumps(reg,indent=2))
    p=subprocess.run([sys.executable,'-B','-m','unittest','test_new_run_gate','-v'],cwd=OUT,capture_output=True,text=True,timeout=60)
    (OUT/'TEST_RUN.log').write_text(p.stdout+p.stderr,encoding='utf-8')
    result={'exit_code':p.returncode,'parent_unchanged':sha(OLD/'new_run_gate_adapter.py')==reg['parent_adapter_sha256'],
      'new_adapter_sha256':sha(OUT/'new_run_gate_adapter.py'),'formal_online_qualification':False,'stable_online_gain':False}
    (OUT/'ROOT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result));print((p.stdout+p.stderr)[-1500:]);return p.returncode
if __name__=='__main__':raise SystemExit(main())
