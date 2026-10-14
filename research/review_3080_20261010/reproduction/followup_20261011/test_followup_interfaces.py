import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import support_replay_bridge as bridge

PROJECT = Path('D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
sys.path.insert(0, str(PROJECT / 'scripts/research_claim_gate'))
import program_support_current_context_candidate as context


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return {'path': path.name, 'sha256': bridge.sha(path)}


class ContextTests(unittest.TestCase):
    def test_unique_complete_day_is_not_only_path(self):
        self.assertEqual(context.current_document_context('x', '40唯一日配对')['current_assertion_hits'], [])
    def test_unique_rows_is_not_only_path(self):
        self.assertEqual(context.current_document_context('x', '32唯一行')['current_assertion_hits'], [])
    def test_only_untested_path_still_flagged(self):
        self.assertTrue(context.current_document_context('x', '唯一未测路径是训练。')['current_assertion_hits'])
    def test_equivalence_evidence_requirement_is_not_result(self):
        self.assertEqual(context.current_document_context('x', '| 等效 | 单独预声明界限与合适检验 |')['current_assertion_hits'], [])
    def test_equivalence_positive_result_still_flagged(self):
        self.assertTrue(context.current_document_context('x', '已经证明等效。')['current_assertion_hits'])
    def test_no_equivalence_test_implemented_is_negative(self):
        self.assertEqual(context.current_document_context('x', '没有实现等效检验。')['current_assertion_hits'], [])
    def test_absence_of_hits_not_scientific_pass(self):
        self.assertIs(context.current_document_context('x', '干净词境')['scientific_qualification_established'], False)
    def test_nonassertive_terms_still_reported(self):
        self.assertEqual(len(context.current_document_context('x', '32唯一行')['lexical_terms_with_nonassertive_context']), 1)
    def test_mixed_positive_and_negative_still_flagged(self):
        self.assertTrue(context.current_document_context('x', '没有实现等效检验；已经证明取货不可能。')['current_assertion_hits'])


class ReplayBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.project = self.root / 'proj'; self.project.mkdir()
        self.protocol = dump(self.project / 'protocol.json', {'synthetic_fixture': True})
        identities = {}
        for role in bridge.IDENTITY_ROLES:
            identities[role] = dump(self.project / (role + '.json'), {'synthetic_role': role})
        relative = bridge.VALIDATORS['delivery_event_contract_only']; target = self.project / relative; target.parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / relative, target)
        events = [
            {'time': 0., 'kind': 'load_start', 'vehicle': 0, 'orders': [1]},
            {'time': 0., 'kind': 'load_finish', 'vehicle': 0},
            {'time': 0., 'kind': 'plan', 'vehicle': 0, 'suffix': [1,0]},
            {'time': 0., 'kind': 'depart', 'vehicle': 0, 'target': 1},
            {'time': 1., 'kind': 'arrive', 'vehicle': 0, 'target': 1},
            {'time': 1., 'kind': 'service_start', 'vehicle': 0, 'order': 1},
            {'time': 1., 'kind': 'service_finish', 'vehicle': 0, 'order': 1},
            {'time': 1., 'kind': 'depart', 'vehicle': 0, 'target': 0},
            {'time': 2., 'kind': 'arrive', 'vehicle': 0, 'target': 0},
        ]
        self.raw = {'schema': 'support_delivery_trace_v1', 'day_id': 0, 'arm': 'synthetic',
                    'orders': {'1': {'quantity': 1., 'release': 0., 'ready': 0., 'tw_start': 0., 'tw_end': 2., 'service': 0., 'zone': 0}},
                    'coordinates': {'0': [0.,0.], '1': [1.,0.]}, 'events': events,
                    'contract': {'vehicles': 1, 'capacity': 2., 'horizon': 3., 'loading_duration': 0., 'speed': 1.}}
        identities['contract'] = dump(self.project / 'contract.json', {'delivery_event_contract': self.raw['contract']})
        identities['dataset'] = dump(self.project / 'dataset.json', {'days': {'0': {'orders': self.raw['orders'], 'coordinates': self.raw['coordinates']}}})
        rawref = dump(self.project / 'raw.json', self.raw)
        self.registry = {'schema': 'support_closure_registry_v1', 'data_seed': 20261023, 'protocol': self.protocol,
                         'identities': identities, 'validator': 'delivery_event_contract_only',
                         'validator_sources': [{'path': relative, 'sha256': bridge.sha(target)}],
                         'unit_arm_keys': [{'day_id': 0, 'arm': 'synthetic'}]}
        dump(self.root / 'registry.json', self.registry)
        ledger = {'schema': 'runtime_identity_ledger_v1', 'protocol_sha256': self.protocol['sha256'],
                  'identity_hashes': {k: v['sha256'] for k,v in identities.items()}}
        start, end = dump(self.project / 'start.json', ledger), dump(self.project / 'end.json', ledger)
        self.bundle = {'schema': 'support_closure_bundle_v1', 'registry_sha256': bridge.sha(self.root / 'registry.json'),
                       'protocol': self.protocol, 'identities': identities, 'source_start': start, 'source_end': end,
                       'traces': [{'day_id': 0, 'arm': 'synthetic', 'raw': rawref,
                                   'expected_replay': {'event_contract_pass': True, 'delivered': [1], 'distance_km': 2., 'quality_energy_certified': False, 'formal_online_ready': False}}]}
    def tearDown(self): self.temp.cleanup()
    def run_bridge(self):
        dump(self.root / 'bundle.json', self.bundle)
        return bridge.audit(self.project, self.root / 'bundle.json', self.root / 'registry.json')
    def rehash_raw(self):
        self.bundle['traces'][0]['raw'] = dump(self.project / 'raw.json', self.raw)
    def test_actual_event_validator_replays_but_never_scientific_certificate(self):
        result = self.run_bridge(); self.assertIs(result['actual_registered_validator_replay_pass'], True)
        self.assertEqual(result['replays'][0]['recomputed_result']['distance_km'], 2.)
        for field in ('full_from_zero_reproduction', 'formal_online_qualification_established', 'stable_maskco_core_online_gain', 'independent_semantic_implementation_established'):
            self.assertIs(result[field], False)
    def test_invalid_arrival_rejected_even_when_raw_rehashed(self):
        self.raw['events'][4]['time'] = .5; self.rehash_raw()
        with self.assertRaises(ValueError): self.run_bridge()
    def test_unreleased_cargo_rejected(self):
        self.raw['orders']['1']['release'] = .1; self.rehash_raw()
        with self.assertRaises(ValueError): self.run_bridge()
    def test_omitted_final_return_rejected(self):
        self.raw['events'].pop(); self.rehash_raw()
        with self.assertRaises(ValueError): self.run_bridge()
    def test_numeric_claim_tamper_rejected(self):
        self.bundle['traces'][0]['expected_replay']['distance_km'] = 1.
        with self.assertRaises(ValueError): self.run_bridge()
    def test_true_string_claim_rejected(self):
        self.bundle['traces'][0]['expected_replay']['event_contract_pass'] = 'true'
        with self.assertRaises(ValueError): self.run_bridge()
    def test_contract_tamper_cannot_rebind_with_raw_hash(self):
        self.raw['contract']['capacity'] = 200.; self.rehash_raw()
        with self.assertRaises(ValueError): self.run_bridge()
    def test_coordinate_tamper_cannot_rebind_with_raw_hash(self):
        self.raw['coordinates']['1'] = [2.,0.]; self.rehash_raw()
        with self.assertRaises(ValueError): self.run_bridge()
    def test_missing_raw_coverage_rejected(self):
        self.bundle['traces'] = []
        with self.assertRaises(ValueError): self.run_bridge()
    def test_missing_original_training_source_not_pretended_complete(self):
        del self.bundle['identities']['training_source']
        with self.assertRaises(ValueError): self.run_bridge()
    def test_start_end_runtime_identity_mismatch_rejected(self):
        ledger = bridge.load(self.project / 'end.json'); ledger['identity_hashes']['weights'] = '0'*64
        self.bundle['source_end'] = dump(self.project / 'end.json', ledger)
        with self.assertRaises(ValueError): self.run_bridge()
    def test_source_modified_rejected(self):
        (self.project / bridge.VALIDATORS['delivery_event_contract_only']).write_text('raise RuntimeError("must not execute")')
        with self.assertRaises(ValueError): self.run_bridge()
    def test_bundle_registry_selection_attack_rejected(self):
        self.bundle['registry_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.run_bridge()
    def test_path_traversal_rejected(self):
        self.bundle['traces'][0]['raw']['path'] = '../escape.json'
        with self.assertRaises(ValueError): self.run_bridge()
    def test_protected_final_seed_rejected(self):
        for seed in (20260927,20260928):
            self.registry['data_seed'] = seed; dump(self.root/'registry.json', self.registry)
            self.bundle['registry_sha256'] = bridge.sha(self.root/'registry.json')
            with self.assertRaises(ValueError): self.run_bridge()
    def test_unregistered_code_adapter_rejected(self):
        self.registry['validator'] = 'run_bundle_code'; dump(self.root/'registry.json',self.registry)
        self.bundle['registry_sha256'] = bridge.sha(self.root/'registry.json')
        with self.assertRaises(ValueError): self.run_bridge()
    def test_duplicate_nonfinite_json_rejected(self):
        for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": 1e309}'):
            path = self.root/'bad.json'; path.write_text(text)
            with self.assertRaises(ValueError): bridge.load(path)


if __name__ == '__main__': unittest.main()
