import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from independent_support_guard import (GuardError, bind_fields, design_subset,
                                       load, prose_review, raw_presence, sha)


def protocol():
    a = .05 / (1 * 2) / 2 / (3 * 2)
    return {'program_sha256': 'a' * 64, 'unit': 'complete_day',
            'kind': 'dynamic_online_confirmation',
            'confirmation_cohorts': [{'data_seed': seed, 'day_ids': list(range(40))}
                                     for seed in (123, 456, 789)],
            'confirmation_family_index': 1, 'registered_contrasts': 2,
            'contrast_cohort_two_sided_alpha': a,
            'bootstrap_quantiles': [a / 2, 1 - a / 2], 'bootstrap_repetitions': 30000}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.p = self.root / 'result.json'
        self.p.write_text(json.dumps({'gain': False, 'n': 32}), encoding='utf-8')
    def tearDown(self): self.temp.cleanup()
    def ref(self):
        return {'path': 'result.json', 'sha256': sha(self.p), 'field': ['gain'], 'expected': False}
    def test_partial_design_never_complete_or_stable(self):
        out = design_subset(protocol(), 'a' * 64)
        self.assertTrue(out['checked_design_subset_pass'])
        self.assertFalse(out['all_design_qualification_established'])
        self.assertFalse(out['stable_online_gain_supported'])
    def test_protected_seeds_including_malformed_day_cohorts(self):
        for seed in (20260927, 20260928):
            p = protocol(); p['confirmation_cohorts'][0] = {'data_seed': seed}
            self.assertIn('protected_pickup_final_seed_forbidden', design_subset(p, 'a' * 64)['missing_conditions'])
    def test_wrong_actual_program(self):
        self.assertFalse(design_subset(protocol(), 'b' * 64)['checked_design_subset_pass'])
    def test_reused_seed(self):
        p = protocol(); p['confirmation_cohorts'][1]['data_seed'] = 123
        self.assertFalse(design_subset(p, 'a' * 64)['checked_design_subset_pass'])
    def test_four_day_cannot_confirm(self):
        p = protocol(); p['confirmation_cohorts'][0]['day_ids'] = list(range(4))
        self.assertFalse(design_subset(p, 'a' * 64)['checked_design_subset_pass'])
    def test_bool_family_and_nan_alpha_fail(self):
        p = protocol(); p['confirmation_family_index'] = True
        self.assertFalse(design_subset(p, 'a' * 64)['checked_design_subset_pass'])
        p = protocol(); p['contrast_cohort_two_sided_alpha'] = math.nan
        self.assertFalse(design_subset(p, 'a' * 64)['checked_design_subset_pass'])
    def test_descriptive_ci_and_tail_count_fail(self):
        p = protocol(); p.update(bootstrap_quantiles=[.025, .975], bootstrap_repetitions=2000)
        self.assertFalse(design_subset(p, 'a' * 64)['checked_design_subset_pass'])
    def test_field_bind_never_scientific_proof(self):
        out = bind_fields(self.root, [self.ref()])
        self.assertTrue(out['reference_binding_pass'])
        self.assertFalse(out['scientific_qualification_established'])
    def test_sha_forgery(self):
        ref = self.ref(); ref['sha256'] = '0' * 64
        with self.assertRaises(GuardError): bind_fields(self.root, [ref])
    def test_true_string(self):
        ref = self.ref(); ref['expected'] = 'false'
        with self.assertRaises(GuardError): bind_fields(self.root, [ref])
    def test_no_reference_or_traversal(self):
        with self.assertRaises(GuardError): bind_fields(self.root, [])
        ref = self.ref(); ref['path'] = '../result.json'
        with self.assertRaises(GuardError): bind_fields(self.root, [ref])
    def test_empty_named_route_does_not_qualify(self):
        self.p.write_text('{}', encoding='utf-8')
        refs = [{'path': 'result.json', 'sha256': sha(self.p)}]
        with self.assertRaises(GuardError): raw_presence(self.root, {'source_files': refs, 'trajectory_files': refs})
    def test_hashbound_minimal_events_still_no_semantic_replay(self):
        self.p.write_text(json.dumps({'day_id': 0, 'events': [{'event_id': 1}]}), encoding='utf-8')
        refs = [{'path': 'result.json', 'sha256': sha(self.p)}]
        out = raw_presence(self.root, {'source_files': refs, 'trajectory_files': refs})
        self.assertTrue(out['provenance_manifest_pass']); self.assertFalse(out['raw_qualification_pass'])
    def test_raw_manifest_absence_fails(self):
        self.assertFalse(raw_presence(self.root, None)['raw_qualification_pass'])
    def test_negation_is_not_current_impossibility(self):
        out = prose_review('不能证明取货不可能。')
        self.assertEqual(out['cues'][0]['context'], 'negation_or_correction')
    def test_historical_claim_is_not_current_claim(self):
        out = prose_review('# 历史归档\n取货不可能。\n# 当前修正\n不能证明取货不可能。')
        self.assertEqual([cue['context'] for cue in out['cues']], ['historical', 'negation_or_correction'])
    def test_duplicate_nan_and_overflow_json_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
            self.p.write_text(text, encoding='utf-8')
            with self.assertRaises(GuardError): load(self.p)


if __name__ == '__main__': unittest.main()
