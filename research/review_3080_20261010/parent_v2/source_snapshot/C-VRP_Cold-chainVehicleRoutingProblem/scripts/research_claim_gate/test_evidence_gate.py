"""Adversarial development checks use synthetic data, never research instances."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from evidence_gate import audit_bundle, EvidenceError, load_json, paired_bootstrap


class ClaimGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.arms = ['model', 'strong', 'random', 'heuristic']
        self.roles = ['source', 'weights', 'data_generator', 'contract', 'budget', 'pool', 'training']
        self.identity = {role: self.write(role + '.json', {'role': role}) for role in self.roles}
        self.runner_entry = self.write('runner_source.json', {'synthetic_source': 'runner'})
        self.auditor_entry = self.write('auditor_source.json', {'synthetic_source': 'independent_auditor'})
        self.identity['source'] = self.write('source.json', {'files': [self.runner_entry]})
        self.keys = {str(seed): [str(seed) + ':' + str(i) for i in range(4)] for seed in (1, 2, 3, 4)}
        mechanisms = {k: True for k in ('native_masked_reconstruction', 'event_mask', 'frozen_prefix',
                                        'equal_total_budget', 'equal_iteration_schedule_controls')}
        self.protocol = {
            'schema': 'maskco_claim_protocol_v1', 'registered_before_outcomes': True,
            'frozen_selection_before_confirmation': True, 'arms': self.arms, 'target_arm': 'model',
            'units_per_run': 4, 'resampling_unit': 'day', 'metric_direction': 'higher',
            'bootstrap_repetitions': 2000, 'bootstrap_seed_offset': 777,
            'comparisons': [{'arm': arm, 'role': role, 'scale': 'absolute', 'minimum_mean': 1}
                            for arm, role in zip(self.arms[1:], ['strong_primary', 'random_same_schedule',
                                                              'nonlearning_same_schedule'])],
            'protected_seeds': [20260927, 20260928], 'confirmation_runs': 3,
            'identity_roles': self.roles, 'identity_hashes': {k: v['sha256'] for k, v in self.identity.items()},
            'selection_sha256': 'a' * 64, 'expected_units_by_seed': self.keys,
            'stage_by_seed': {str(seed): ('development' if seed == 1 else 'confirmation')
                              for seed in (1, 2, 3, 4)},
            'scope': 'dynamic_online', 'method_semantics': mechanisms}
        self.bundle = {'schema': 'maskco_claim_bundle_v1', 'runner_source': self.runner_entry,
                       'auditor_sources': {key: self.auditor_entry for key in
                                          ('independent_identity_audit', 'independent_trajectory_audit')},
                       'runs': [], 'data_objects': {}}
        for seed in (1, 2, 3, 4):
            data = self.write('data_%s.json' % seed, {'synthetic_seed': seed})
            self.bundle['data_objects'][data['sha256']] = data
            rows = [{'unit_id': key, 'arm': arm, 'value': (15 if arm == 'model' else 10),
                     'hard': True, 'complete': True, 'timeout': False, 'wall_seconds': 1,
                     'completed_candidates': 1, 'model_calls': (1 if arm == 'model' else 0),
                     'data_sha256': data['sha256']} for key in self.keys[str(seed)] for arm in self.arms]
            entry = self.write('rows_%s.json' % seed, rows)
            self.bundle['runs'].append({'seed': seed, 'stage': self.protocol['stage_by_seed'][str(seed)],
                                        'exit_code': 0, 'selection_sha256': 'a' * 64,
                                        'identity': copy.deepcopy(self.identity), 'rows': entry})
        self.refresh_protocol()
        for name in ('independent_identity_audit', 'independent_trajectory_audit'):
            self.bundle[name] = self.write(name + '.json', {
                'schema': 'maskco_independent_audit_v1', 'passed': True,
                'protocol_sha256': self.bundle['protocol']['sha256'],
                'auditor_source_sha256': self.auditor_entry['sha256'],
                'rows_sha256': [r['rows']['sha256'] for r in self.bundle['runs']],
                'unit_arm_rows': 64, 'mechanism_checks': mechanisms})

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def refresh_protocol(self):
        self.bundle['protocol'] = self.write('protocol.json', self.protocol)
        for name in ('independent_identity_audit', 'independent_trajectory_audit'):
            if name in self.bundle:
                proof = load_json(self.root / self.bundle[name]['path'])
                proof['protocol_sha256'] = self.bundle['protocol']['sha256']
                self.bundle[name] = self.write(name + '.json', proof)

    def mutate_rows(self, function, seed=1):
        run = self.bundle['runs'][seed - 1]
        rows = load_json(self.root / run['rows']['path'])
        function(rows)
        old = run['rows']['sha256']
        run['rows'] = self.write('rows_%s.json' % seed, rows)
        for name in ('independent_identity_audit', 'independent_trajectory_audit'):
            proof = load_json(self.root / self.bundle[name]['path'])
            proof['rows_sha256'] = [run['rows']['sha256'] if x == old else x for x in proof['rows_sha256']]
            self.bundle[name] = self.write(name + '.json', proof)

    def audit(self):
        self.write('bundle.json', self.bundle)
        return audit_bundle(self.root / 'bundle.json')

    def reject(self):
        with self.assertRaises((EvidenceError, KeyError)):
            self.audit()

    def test_positive_synthetic_fixture(self):
        result = self.audit()
        self.assertTrue(result['candidate_gain_under_supplied_v1_protocol'])
        self.assertFalse(result['stable_maskco_core_online_gain'])
        self.assertFalse(result['full_from_zero_reproduction'])

    def test_forged_boolean(self):
        self.mutate_rows(lambda rows: rows[0].update(hard='true'))
        self.reject()

    def test_missing_rows(self):
        self.mutate_rows(lambda rows: rows.pop())
        self.reject()

    def test_duplicate_rows(self):
        self.mutate_rows(lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
        self.reject()

    def test_nan_raw_value(self):
        self.mutate_rows(lambda rows: rows[0].update(value=float('nan')))
        self.reject()

    def test_boolean_is_not_numeric(self):
        self.mutate_rows(lambda rows: rows[0].update(value=True))
        self.reject()

    def test_changed_split(self):
        self.bundle['runs'][1]['stage'] = 'development'
        self.reject()

    def test_changed_selection(self):
        self.bundle['runs'][2]['selection_sha256'] = 'd' * 64
        self.reject()

    def test_changed_file(self):
        (self.root / 'source.json').write_text('{}', encoding='utf-8')
        self.reject()

    def test_missing_count(self):
        self.mutate_rows(lambda rows: rows[0].pop('model_calls'))
        self.reject()

    def test_zero_activation_prevents_core_gain(self):
        self.mutate_rows(lambda rows: rows[0].update(model_calls=0))
        self.assertFalse(self.audit()['candidate_core_online_gain_under_supplied_v1_protocol'])

    def test_ci_cross_zero_not_equivalence(self):
        def mutate(rows):
            for row in rows:
                if row['arm'] == 'model':
                    row['value'] = 1 if row['unit_id'].endswith(('0', '1')) else 19
        self.mutate_rows(mutate)
        result = self.audit()
        self.assertFalse(result['stable_maskco_core_online_gain'])
        self.assertFalse(result['equivalence_established'])
        self.assertFalse(result['runs'][0]['comparisons']['strong']['equivalence_established'])

    def test_snapshot_gain_never_online(self):
        self.protocol['scope'] = 'snapshot'
        self.refresh_protocol()
        result = self.audit()
        self.assertTrue(result['candidate_gain_under_supplied_v1_protocol'])
        self.assertFalse(result['stable_gain_in_registered_setting'])
        self.assertFalse(result['stable_maskco_core_online_gain'])

    def test_same_runner_auditor_rejected(self):
        proof = load_json(self.root / self.bundle['independent_identity_audit']['path'])
        proof['auditor_source_sha256'] = self.runner_entry['sha256']
        self.bundle['auditor_sources']['independent_identity_audit'] = self.runner_entry
        self.bundle['independent_identity_audit'] = self.write('independent_identity_audit.json', proof)
        self.reject()

    def test_source_closure_actual_bytes_changed(self):
        (self.root / self.runner_entry['path']).write_text('{"changed": true}', encoding='utf-8')
        self.reject()

    def test_auditor_actual_bytes_changed(self):
        (self.root / self.auditor_entry['path']).write_text('{}', encoding='utf-8')
        self.reject()

    def test_missing_independent_mechanism_cannot_promote_core(self):
        proof = load_json(self.root / self.bundle['independent_trajectory_audit']['path'])
        proof.pop('mechanism_checks')
        self.bundle['independent_trajectory_audit'] = self.write('independent_trajectory_audit.json', proof)
        self.assertFalse(self.audit()['candidate_core_online_gain_under_supplied_v1_protocol'])

    def test_unequal_completed_schedule_not_core(self):
        self.mutate_rows(lambda rows: rows[0].update(completed_candidates=2))
        self.assertFalse(self.audit()['candidate_core_online_gain_under_supplied_v1_protocol'])

    def test_same_data_reused_in_confirmation(self):
        old = self.bundle['runs'][0]['rows']
        rows = load_json(self.root / old['path'])
        copied_hash = rows[0]['data_sha256']
        self.mutate_rows(lambda rows: [row.update(data_sha256=copied_hash) for row in rows], seed=2)
        self.reject()

    def test_protected_seed(self):
        self.bundle['runs'][0]['seed'] = 20260927
        self.reject()

    def test_pair_data_changed(self):
        self.mutate_rows(lambda rows: rows[0].update(data_sha256='f' * 64))
        self.reject()

    def test_unregistered_day(self):
        self.mutate_rows(lambda rows: rows[0].update(unit_id='unregistered'))
        self.reject()

    def test_retained_timeout(self):
        self.mutate_rows(lambda rows: rows[0].update(timeout=True))
        result = self.audit()
        self.assertEqual(result['runs'][0]['timeouts_by_arm']['model'], 1)

    def test_no_random_control_cannot_promote_core(self):
        self.protocol['comparisons'] = self.protocol['comparisons'][:1]
        self.refresh_protocol()
        self.assertFalse(self.audit()['candidate_core_online_gain_under_supplied_v1_protocol'])

    def test_negative_gain(self):
        self.mutate_rows(lambda rows: [r.update(value=5) for r in rows if r['arm'] == 'model'])
        self.assertEqual(self.audit()['runs'][0]['comparisons']['strong']['interpretation'], 'negative_effect')

    def test_pair_bootstrap_closed_form_constant(self):
        self.assertEqual(paired_bootstrap([2, 2, 2], 99)['ci95'], [2, 2])

    def test_unbound_95_percent_never_stable(self):
        result = self.audit()
        self.assertFalse(result['program_binding_form_valid'])
        self.assertFalse(result['stable_maskco_core_online_gain'])

    def test_four_day_positive_never_stable(self):
        result = self.audit()
        self.assertTrue(result['candidate_gain_under_supplied_v1_protocol'])
        self.assertFalse(result['stable_gain_in_registered_setting'])

    def test_self_selected_program_binding_rejected(self):
        self.bundle['program_binding'] = self.write('binding.json', {'schema': 'maskco_program_binding_v1'})
        self.reject()

    def test_forged_program_hash_rejected(self):
        self.bundle['program_binding'] = self.write('binding.json', {
            'schema': 'maskco_program_binding_v1', 'program_sha256': 'd' * 64})
        anchor = self.root / 'trusted_program.json'
        anchor.write_text('{}', encoding='utf-8')
        self.write('bundle.json', self.bundle)
        with self.assertRaisesRegex(EvidenceError, 'PROGRAM hash mismatch'):
            audit_bundle(self.root / 'bundle.json', anchor)

    def test_forged_alpha_rejected(self):
        from evidence_gate import validate_program_binding
        anchor = self.write('trusted_program.json', {'program': 'trusted'})
        bound_protocol = copy.deepcopy(self.protocol)
        bound_protocol['units_per_run'] = 40
        self.bundle['program_binding'] = self.write('binding.json', {
            'schema': 'maskco_program_binding_v1', 'program_sha256': anchor['sha256'],
            'protocol_sha256': self.bundle['protocol']['sha256'], 'family_index': 1,
            'allocation_alpha': .05, 'bootstrap_endpoints': [.025, .975], 'bootstrap_repetitions': 2000})
        with self.assertRaisesRegex(EvidenceError, 'alpha spending mismatch'):
            validate_program_binding(self.root, self.bundle, bound_protocol, self.root / anchor['path'])

    def test_alpha_tail_insufficient_rejected(self):
        from evidence_gate import validate_program_binding
        anchor = self.write('trusted_program.json', {'program': 'trusted'})
        bound_protocol = copy.deepcopy(self.protocol)
        bound_protocol['units_per_run'] = 40
        alpha = .05 / 2 / 2 / (3 * 3)
        self.bundle['program_binding'] = self.write('binding.json', {
            'schema': 'maskco_program_binding_v1', 'program_sha256': anchor['sha256'],
            'protocol_sha256': self.bundle['protocol']['sha256'], 'family_index': 1,
            'allocation_alpha': alpha, 'bootstrap_endpoints': [alpha / 2, 1 - alpha / 2],
            'bootstrap_repetitions': 2000})
        with self.assertRaisesRegex(EvidenceError, 'tail Monte Carlo'):
            validate_program_binding(self.root, self.bundle, bound_protocol, self.root / anchor['path'])

    def test_duplicate_json_keys(self):
        (self.root / 'invalid.json').write_text('{"passed": false, "passed": true}', encoding='utf-8')
        with self.assertRaises(EvidenceError):
            load_json(self.root / 'invalid.json')


if __name__ == '__main__':
    unittest.main()
