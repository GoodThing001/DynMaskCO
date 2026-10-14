"""Fresh adversarial cases for the current-source-derived v3 review copy."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path('D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem')
sys.path.insert(0, str(PROJECT / 'scripts/research_claim_gate'))
import program_support_audit_v3 as gate


def protocol():
    alpha = .05 / (1 * 2) / 2 / (2 * 3)
    return {'program_sha256': gate.PROGRAM_SHA, 'unit': 'complete_day',
            'kind': 'dynamic_online_confirmation',
            'confirmation_cohorts': [{'data_seed': seed, 'day_ids': list(range(40))}
                                    for seed in (20261020, 20261021, 20261022)],
            'confirmation_family_index': 1, 'registered_contrasts': 2,
            'contrast_cohort_two_sided_alpha': alpha,
            'bootstrap_quantiles': [alpha / 2, 1 - alpha / 2],
            'bootstrap_repetitions': 30000,
            **{key: True for key in ('candidate_frozen_before_confirmation',
                 'source_and_weight_ids_preregistered', 'complete_raw_trajectory_replay_required',
                 'qualified_strong_rolling_control_required', 'all_main_and_attribution_gates_required')}}


class FreshAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def check_never_stable(self, result):
        self.assertIs(result['stable_online_gain_supported'], False)
        self.assertIs(result['stable_maskco_attribution_supported'], False)
        self.assertIs(result['all_design_qualification_established'], False)
        self.assertIs(result['necessary_design_conditions_pass'], False)
    def test_valid_necessary_subset_never_full_qualification(self):
        result = gate.program_confirmation_conditions(protocol())
        self.assertIs(result['checked_design_subset_pass'], True)
        self.check_never_stable(result)
    def test_both_protected_seeds_rejected_even_if_days_malformed(self):
        for seed in (20260927, 20260928):
            for days in (list(range(40)), [], None):
                p = protocol(); p['confirmation_cohorts'][0].update(data_seed=seed, day_ids=days)
                result = gate.program_confirmation_conditions(p)
                self.assertIn('protected_pickup_final_seed_forbidden', result['missing_conditions'])
                self.assertIs(result['checked_design_subset_pass'], False)
                self.check_never_stable(result)
    def test_four_days_cannot_qualify(self):
        p = protocol(); p['confirmation_cohorts'][0]['day_ids'] = list(range(4))
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_forged_true_strings_fail(self):
        p = protocol(); p['candidate_frozen_before_confirmation'] = 'true'
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_wrong_program_hash_fail(self):
        p = protocol(); p['program_sha256'] = '0' * 64
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_unadjusted_95pct_fail(self):
        p = protocol(); p['contrast_cohort_two_sided_alpha'] = .05; p['bootstrap_quantiles'] = [.025, .975]
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_tail_2000_reps_insufficient(self):
        p = protocol(); p['bootstrap_repetitions'] = 2000
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_nonfinite_alpha_fail(self):
        for a in (float('nan'), float('inf'), True):
            p = protocol(); p['contrast_cohort_two_sided_alpha'] = a
            self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_duplicate_day_or_seed_fail(self):
        p = protocol(); p['confirmation_cohorts'][1]['day_ids'][0] = 1
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
        p = protocol(); p['confirmation_cohorts'][1]['data_seed'] = p['confirmation_cohorts'][0]['data_seed']
        self.assertFalse(gate.program_confirmation_conditions(p)['checked_design_subset_pass'])
    def test_empty_route_never_identity_or_replay_qualification(self):
        folder = self.root / 'panel'; folder.mkdir(); (folder / 'route.json').write_text('{}')
        inventory = gate.scan_directory(self.root, 'panel')
        r = gate.eligibility_dimension([inventory], [{'path': 'fake.py', 'usable': True}], [])
        self.assertEqual(r['verdict'], 'file_inventory_present_identity_and_replay_unverified')
        self.assertIs(r['pass'], False)
        self.assertIs(r['raw_trajectory_qualification_established'], False)
        self.assertIs(r['source_identity_qualification_established'], False)
    def test_wellformed_route_filename_still_not_semantic_qualification(self):
        folder = self.root / 'panel'; folder.mkdir()
        (folder / 'route.json').write_text(json.dumps({'day_id': 1, 'routes': [[0, 1, 0]], 'hard': True}))
        r = gate.eligibility_dimension([gate.scan_directory(self.root, 'panel')], [{'path': 'a.py', 'usable': True}], [])
        self.assertIs(r['pass'], False)
    def test_no_inventory_no_global_absence_claim(self):
        folder = self.root / 'panel'; folder.mkdir()
        r = gate.eligibility_dimension([gate.scan_directory(self.root, 'panel')], [], [])
        self.assertIs(r['pass'], False)
        self.assertIn('bounded', r['reason'])
    def test_chinese_negated_claim_not_scanned_positive(self):
        for text in ('不能证明取货不可能；未建立稳定收益。', '不应写 learning_side_closed；不得说唯一未测。', '不足以证明工时机制也不存在。'):
            self.assertEqual(gate.scan_universal_wording('current.md', text), [])
            self.assertEqual(gate.asserted_literal_lines(text, '不可能'), [])
    def test_explicit_current_negative_flags_not_positive_assertions(self):
        text = '> 稳定在线增益、取货不可能性、等价性：**False**。'
        self.assertEqual(gate.scan_universal_wording('current.md', text), [])
    def test_history_section_excluded_current_section_checked(self):
        text = '# 历史封存\n取货不可能，唯一未测。\n## 历史细节\n学习侧已经关闭。\n# 当前裁决\n不能证明不可能。\n唯一未测是训练。'
        hits = gate.scan_universal_wording('current.md', text)
        self.assertTrue(hits); self.assertEqual([h['line'] for h in hits], [7])
    def test_mixed_negated_and_positive_clauses_positive_retained(self):
        hits = gate.scan_universal_wording('current.md', '未建立稳定收益；取货不可能。')
        self.assertTrue(hits)
    def make_fixture(self, name='fixture', mutate=None):
        authentic = (PROJECT / 'results/research_20261010/PROGRAM.json').read_bytes()
        return gate.build_fixture(self.root, name, authentic, mutate)
    def test_negated_old_trigger_does_not_fire_in_end_to_end_audit(self):
        def mutate(files):
            files['results/research_20261010/PROGRAM_CLOSEOUT.md'] = '# 当前裁决\n不能证明取货不可能；不得写 learning_side_closed；不应说唯一未测。'
            return files
        result, code = gate.run_audit(self.make_fixture(mutate=mutate), self.root / 'out.json')
        self.assertEqual(code, 0); self.assertEqual(result['scientific_support']['claim_excesses'], [])
    def test_current_doc_negative_hash_bound_scope_only(self):
        project = self.make_fixture()
        p = project / 'current.md'; p.write_text('# 当前裁决\n不能证明取货不可能；未建立稳定收益。', encoding='utf-8')
        result, code = gate.run_audit(project, self.root / 'out.json', current_docs=['current.md'])
        self.assertEqual(code, 0)
        self.assertEqual(result['read_only_input_sha256']['current.md'], hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(result['current_documents_context_review'][0]['current_assertion_hits'], [])
        self.assertIs(result['current_documents_context_review'][0]['scientific_qualification_established'], False)
    def test_missing_current_doc_fail_closed(self):
        result, code = gate.run_audit(self.make_fixture(), self.root / 'out.json', current_docs=['missing.md'])
        self.assertEqual(code, 2)
        self.assertIn('current_document_missing_or_unreadable:missing.md', result['additional_findings'])
    def test_current_unsupported_claim_blocked(self):
        project = self.make_fixture(); (project / 'current.md').write_text('取货不可能。', encoding='utf-8')
        result, code = gate.run_audit(project, self.root / 'out.json', current_docs=['current.md'])
        self.assertEqual(code, 2)
    def test_actual_latest_source_derivation_bound(self):
        self.assertEqual(gate.BASE_SOURCE_SHA256, '2871e5b9971c79b3703252b33eb7ec567269786c0f0c1374accec7de9776cfc1')
        self.assertEqual(gate.SCHEMA, 'program-support-audit-v3')


if __name__ == '__main__': unittest.main()
