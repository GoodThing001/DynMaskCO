"""Semantic checks on intact author data, not implementation-mirroring mocks."""
import itertools
import math
import unittest

from net_benefit_model import RouteModel, config, load_case, structural_audit
from net_benefit_audit import audit


class NetBenefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = config()
        cls.ins = load_case('c105', cls.c)
        cls.routes = tuple((i,) for i in range(1, 101))
        cls.rho = cls.c['tomato']['rho_center']

    def checked(self, routes=None, rho=None, fleet=100):
        return audit(self.ins['nodes'], self.routes if routes is None else routes, self.c, self.rho if rho is None else rho, fleet)

    def test_accounting_and_optimizer_sign_agree_independently(self):
        r = self.checked()
        self.assertTrue(r['feasible_routing'])
        m = RouteModel(self.ins, self.c, self.rho, 100)
        self.assertAlmostEqual(-m.objective(self.routes), r['net_benefit_scenario_EUR'], places=7)
        self.assertAlmostEqual(r['sales_revenue_EUR']-r['purchase_EUR']-r['distance_cost_EUR']-r['refrigeration_cost_proxy_EUR'], r['net_benefit_scenario_EUR'])

    def test_waiting_service_return_and_no_fake_mass_loss(self):
        r = self.checked()
        t = r['tours'][0]
        node = self.ins['nodes'][1]
        outbound = math.hypot(node[1]-140, node[2]-175)
        expected = max(outbound, node[4])+90+outbound
        self.assertAlmostEqual(t['duration_min'], expected)
        self.assertGreater(t['waiting_min'], 0)
        self.assertEqual(t['trace'][0]['delivered_kg'], node[3])
        self.assertGreater(r['minimum_tomato_TA_retention'], 0)
        self.assertLess(r['minimum_tomato_TA_retention'], 1)

    def test_flat_price_control_and_scale_invariance(self):
        r = self.checked(rho=1)
        self.assertAlmostEqual(r['quality_discount_EUR'], 0, places=7)
        self.assertAlmostEqual(r['sales_revenue_EUR'], r['nominal_revenue_EUR'], places=7)
        self.assertAlmostEqual(r['normalized_net_benefit']*4.5, r['net_benefit_scenario_EUR'], places=7)

    def test_no_double_counting_quality_discount(self):
        r = self.checked()
        margin = r['nominal_revenue_EUR']-r['purchase_EUR']
        self.assertAlmostEqual(margin-r['quality_discount_EUR']-r['distance_cost_EUR']-r['refrigeration_cost_proxy_EUR'], r['net_benefit_scenario_EUR'], places=7)

    def test_duplicate_missing_invalid_and_fleet_rejected(self):
        self.assertIn('coverage_or_duplicate', self.checked(self.routes[1:])['errors'])
        self.assertIn('coverage_or_duplicate', self.checked(self.routes+((1,),))['errors'])
        self.assertIn('invalid_route_customer', self.checked(((101,),)+self.routes[1:])['errors'])
        self.assertIn('fleet_bound', self.checked(fleet=25)['errors'])

    def test_infeasible_route_rejected_by_both(self):
        routes = (tuple(range(1, 101)),)
        report = self.checked(routes)
        self.assertIn('capacity', report['errors'])
        self.assertIn('time_window', report['errors'])
        self.assertFalse(math.isfinite(RouteModel(self.ins, self.c, self.rho, 100).objective(routes)))

    def test_original_c201_has_fleet_independent_infeasibility_witnesses(self):
        report = structural_audit(load_case('c201', self.c), self.c)
        self.assertEqual([x['customer'] for x in report['impossible_customers']], [87, 90, 96])
        self.assertTrue(all(x['earliest_return_min'] > 1440 for x in report['impossible_customers']))

    def test_mullet_absolute_quality_is_counterfactual_not_price(self):
        r = self.checked()
        trace = r['tours'][0]['trace'][0]
        self.assertAlmostEqual(trace['mullet_counterfactual_sensory_score_approx'], 9-0.452*trace['delivery_completion_min']/1440)
        self.assertTrue(r['mullet_counterfactual_sensory_threshold_passed'])
        self.assertFalse(r['mullet_profit_computed'])
        self.assertFalse(r['tomato_acceptance_threshold_configured'])

    def test_profile_identity_survives_cross_sensitivity_evaluation(self):
        from run_net_benefit_scenario import record
        row = record(self.ins, self.c, self.routes, 100, 'test', self.rho, 20261020, {}, 'li_midpoint_ratio')
        self.assertEqual(row['optimization_setting'], 'li_midpoint_ratio')
        self.assertEqual(row['optimization_rho'], self.rho)
        self.assertEqual(row['seed'], 20261020)
        self.assertEqual(len(row['fixed_routes_sensitivity_not_reoptimization']), 4)

    def test_exact_restricted_neighborhood_agrees_with_independent_auditor(self):
        # All100 original customers remain. Exhaust only placement/order of
        # original IDs1..4 in two route slots; other96 singleton routes fixed.
        model = RouteModel(self.ins, self.c, self.rho, 100)
        best_fast, best_audit, valid = math.inf, math.inf, 0
        for order in itertools.permutations(range(1, 5)):
            for cut in range(5):
                slots = tuple(r for r in (order[:cut], order[cut:]) if r)
                routes = slots+self.routes[4:]
                fast = model.objective(routes)
                independent = self.checked(routes)
                self.assertEqual(math.isfinite(fast), independent['feasible_routing'])
                if independent['feasible_routing']:
                    valid += 1
                    best_fast = min(best_fast, fast)
                    best_audit = min(best_audit, -independent['net_benefit_scenario_EUR'])
        self.assertGreater(valid, 0)
        self.assertAlmostEqual(best_fast, best_audit, places=7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
