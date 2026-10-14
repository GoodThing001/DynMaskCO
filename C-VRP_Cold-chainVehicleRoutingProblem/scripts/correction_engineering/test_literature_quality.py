"""Closed-form, units, lifecycle and invalid-trajectory regressions."""
import math
import unittest
from literature_quality import load_profiles, rate_per_hour, evaluate_product, evaluate_cargo, electrical_energy_kwh

def segment(t, start=0., end=24.):
    return dict(start_h=start,end_h=end,product_temperature_c=t)

class LiteratureQuality(unittest.TestCase):
    def test_tomato_reference_and_first_order(self):
        p,_=load_profiles();k,_=rate_per_hour(p['cherry_tomato_TA'],18.)
        # Published ln(A)=4.65 expression; the old audit used rounded A=104.58.
        self.assertAlmostEqual(k,0.001854063323067746,places=14)
        q=evaluate_product('cherry_tomato_TA',1.,[segment(18)])
        self.assertAlmostEqual(q['final_value'],math.exp(-24*k),places=14)
        self.assertIsNone(q['sensory_below_threshold'])
    def test_mullet_units_and_zero_order(self):
        q=evaluate_product('chilled_mullet_overall_impression',9.,[segment(4)])
        self.assertAlmostEqual(q['loss'],.452,places=12)
        self.assertEqual(q['value_unit'],'sensory_score_points')
        self.assertIsNone(q['monetary_loss']);self.assertFalse(q['food_safety_certified'])
    def test_frozen_derivation_and_rejection(self):
        q=evaluate_product('frozen_seabass_overall_acceptability_scenario',9.,[segment(-5,end=133*24.)])
        self.assertAlmostEqual(q['final_value'],5.,places=12)
        self.assertIn('scenario-derived',q['rate_evidence'])
    def test_split_invariance(self):
        for name,initial,t in [('cherry_tomato_TA',1.,10),('chilled_mullet_overall_impression',9.,4)]:
            a=evaluate_product(name,initial,[segment(t)])
            b=evaluate_product(name,initial,[segment(t,0,11),segment(t,11,24)])
            self.assertAlmostEqual(a['final_value'],b['final_value'],places=14)
    def test_variable_temperature_integral(self):
        p,_=load_profiles();name='chilled_mullet_overall_impression'
        q=evaluate_product(name,9.,[segment(0,0,12),segment(10,12,24)])
        expected=9.-12*(rate_per_hour(p[name],0)[0]+rate_per_hour(p[name],10)[0])
        self.assertAlmostEqual(q['final_value'],expected,places=14)
        self.assertGreater(rate_per_hour(p[name],10)[0],rate_per_hour(p[name],0)[0])
    def test_extrapolation_is_explicit(self):
        name='frozen_seabass_overall_acceptability_scenario'
        with self.assertRaises(ValueError):evaluate_product(name,9.,[segment(-18)])
        q=evaluate_product(name,9.,[segment(-18)],allow_extrapolation=True)
        self.assertTrue(q['extrapolated']);self.assertFalse(q['business_calibrated'])
    def test_no_temperature_zone_inference(self):
        with self.assertRaises(ValueError):evaluate_product('frozen',9.,[segment(-12)])
    def test_invalid_input_and_gaps(self):
        name='cherry_tomato_TA'
        for rows in ([segment(float('nan'))],[segment(10,2,1)],[segment(10,0,1),segment(10,2,3)],
                     [segment(10,0,2),segment(10,1,3)],[segment(-273.15)]):
            with self.assertRaises(ValueError):evaluate_product(name,1.,rows)
        with self.assertRaises(ValueError):evaluate_product(name,0.,[segment(10)])
    def test_cargo_lifecycle_coverage(self):
        rows=[segment(4,2,5)]
        for direction in ('pickup','delivery'):
            q=evaluate_cargo('chilled_mullet_overall_impression',9.,rows,direction=direction,begin_h=2,end_h=5)
            self.assertAlmostEqual(q['loss'],3*.452/24,places=14)
        with self.assertRaises(ValueError):evaluate_cargo('chilled_mullet_overall_impression',9.,rows,direction='delivery',begin_h=1,end_h=5)
    def test_power_input_units_and_provenance(self):
        rows=[dict(start_h=0.,end_h=2.,electrical_input_kw=1.8,power_basis='simulation')]
        e=electrical_energy_kwh(rows)
        self.assertAlmostEqual(e['electrical_energy_kwh'],3.6)
        self.assertFalse(e['all_segments_labeled_measurement'])
        with self.assertRaises(ValueError):electrical_energy_kwh([dict(rows[0],electrical_input_kw=-1.)])
    def test_no_silent_sensory_clipping(self):
        q=evaluate_product('chilled_mullet_overall_impression',5.,[segment(4,end=300)])
        self.assertLess(q['final_value'],1.);self.assertTrue(q['outside_sensory_scale'])
        self.assertTrue(q['sensory_below_threshold'])

if __name__=='__main__':unittest.main(verbosity=2)
