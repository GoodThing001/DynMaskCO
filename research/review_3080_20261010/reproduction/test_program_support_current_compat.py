import sys
from pathlib import Path
sys.path.insert(0, str(Path('D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_claim_gate')))
import copy
import math
import unittest
from program_support_audit_v3 import program_confirmation_conditions as assess, interval_support, dig, PROGRAM_SHA
from evidence_gate import EvidenceError


def necessary_design():
    alpha = .05 / (1 * 2) / 2 / (2 * 3)
    return {'program_sha256': PROGRAM_SHA, 'unit': 'complete_day', 'kind': 'dynamic_online_confirmation',
            'confirmation_cohorts': [{'data_seed': 100 + j, 'day_ids': list(range(j * 40, (j + 1) * 40))} for j in range(3)],
            'confirmation_family_index': 1, 'registered_contrasts': 2,
            'contrast_cohort_two_sided_alpha': alpha, 'bootstrap_quantiles': [alpha / 2, 1 - alpha / 2],
            'bootstrap_repetitions': 30000, 'candidate_frozen_before_confirmation': True,
            'source_and_weight_ids_preregistered': True, 'complete_raw_trajectory_replay_required': True,
            'qualified_strong_rolling_control_required': True, 'all_main_and_attribution_gates_required': True}


class SupportTests(unittest.TestCase):
    def test_necessary_design_is_not_scientific_proof(self):
        out = assess(necessary_design())
        self.assertTrue(out['checked_design_subset_pass'])
        self.assertFalse(out['stable_online_gain_supported'])
        self.assertFalse(out['stable_maskco_attribution_supported'])
    def test_snapshot_three32_is_not_program_confirmation(self):
        p = necessary_design(); p['unit'] = 'snapshot'
        for cohort in p['confirmation_cohorts']: cohort['day_ids'] = cohort['day_ids'][:32]
        out = assess(p); self.assertFalse(out['checked_design_subset_pass']); self.assertFalse(out['stable_online_gain_supported'])
    def test_forged_stable_flag_ignored(self):
        out = assess({'stable_gain_established': True, 'all_pass': True})
        self.assertFalse(out['stable_online_gain_supported'])
    def test_wrong_program_sha(self):
        p = necessary_design(); p['program_sha256'] = '0' * 64
        self.assertIn('no_exact_PROGRAM_binding', assess(p)['missing_conditions'])
    def test_family_index_boolean(self):
        p = necessary_design(); p['confirmation_family_index'] = True
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_alpha_nan(self):
        p = necessary_design(); p['contrast_cohort_two_sided_alpha'] = math.nan
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_descriptive95_and2000_never_qualify(self):
        p = necessary_design(); p.update(contrast_cohort_two_sided_alpha=.05, bootstrap_quantiles=[.025, .975], bootstrap_repetitions=2000)
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_alpha_not_adjusted_for_multiple_comparisons(self):
        p = necessary_design(); p['registered_contrasts'] = 3
        self.assertIn('family_alpha_spending_missing_or_wrong', assess(p)['missing_conditions'])
    def test_tail_MC_precision(self):
        p = necessary_design(); p['bootstrap_repetitions'] = 2000
        self.assertIn('bootstrap_tail_MC_precision_under_25_expected_draws', assess(p)['missing_conditions'])
    def test_duplicate_day(self):
        p = necessary_design(); p['confirmation_cohorts'][0]['day_ids'][0] = 1
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_local_day_ordinals_can_repeat_under_fresh_seeds(self):
        p = necessary_design()
        for cohort in p['confirmation_cohorts']: cohort['day_ids'] = list(range(40))
        self.assertTrue(assess(p)['checked_design_subset_pass'])
        self.assertFalse(assess(p)['stable_online_gain_supported'])
    def test_reused_cohort_seed(self):
        p = necessary_design(); p['confirmation_cohorts'][1]['data_seed'] = 100
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_unhashable_fake_seed(self):
        p = necessary_design(); p['confirmation_cohorts'][0]['data_seed'] = []
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_forged_required_literal(self):
        p = necessary_design(); p['candidate_frozen_before_confirmation'] = 'true'
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_four_days_not40(self):
        p = necessary_design()
        for cohort in p['confirmation_cohorts']: cohort['day_ids'] = cohort['day_ids'][:4]
        self.assertFalse(assess(p)['checked_design_subset_pass'])
    def test_negative_mean_cross_zero_is_not_confirmed_harm(self):
        out = interval_support(-.03, [-.067, .006], 'harmful')
        self.assertTrue(out['point_estimate_worse']); self.assertFalse(out['descriptive_interval_entirely_worse'])
        self.assertFalse(out['global_upper_bound_or_learning_exclusion'])
    def test_tiny_mean_can_have_positive_interval_despite_no_1percent_gate(self):
        out = interval_support(.005, [.001, .009], 'no_signal')
        self.assertFalse(out['point_estimate_worse']); self.assertFalse(out['equivalence_established'])
    def test_entirely_harmful_interval_not_family_discovery(self):
        out = interval_support(-.03, [-.06, -.01], 'harmful')
        self.assertTrue(out['descriptive_interval_entirely_worse']); self.assertFalse(out['family_adjusted_discovery'])
    def test_positive_sign_convention(self):
        self.assertTrue(interval_support(1.8, [1.6, 2.1], 'harmful_positive')['descriptive_interval_entirely_worse'])
    def test_nan_mean_rejected(self):
        with self.assertRaises(EvidenceError): interval_support(math.nan, [0, 1], 'harmful')
    def test_reverse_interval_rejected(self):
        with self.assertRaises(EvidenceError): interval_support(.1, [.2, -.2], 'harmful')
    def test_dotted_literal_key(self):
        self.assertEqual(dig({'dev': {'P@gw0.1': {'mean': .5}}}, 'dev.P@gw0.1.mean'), .5)
    def test_greedy_reference_cannot_prove_no_headroom(self):
        # Two feasible candidates are enough to refute the inference from a worse greedy candidate.
        baseline, greedy_full_info, feasible_unmeasured = 84, 77, 95
        self.assertLess(greedy_full_info, baseline); self.assertGreater(feasible_unmeasured, baseline)
    def test_average_cost_is_not_marginal_energy_lower_bound(self):
        # A nearby new order can share the existing trip, below prior average energy/order.
        prior_total_energy, prior_count, extra_energy = 100, 10, 1
        self.assertLess(extra_energy, prior_total_energy / prior_count)


if __name__ == '__main__': unittest.main()
