"""Guard against the unit/product identity errors found in the evidence audit."""
import math
import unittest
from direct_value_evidence import calculations, electrical_energy_from_heat, fuel_from_heat, load, normalize_money, quoted_values, require_complete_routing_profile, source_quality_eligible


class EvidenceValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load()

    def test_carton_pallet_and_kg_quote_are_the_same_amount(self):
        r = quoted_values(self.c, 1600, 0.6)
        self.assertAlmostEqual(r['pallet_kg_derived'], 907.18474)
        self.assertAlmostEqual(r['high_quality_USD_per_kg']*r['pallet_kg_derived'], 80*20)
        self.assertEqual(r['low_quality_USD_per_pallet'], 960)

    def test_published_bounds_do_not_silently_expand(self):
        for price, discount in [(1600, 0.36666666666666664), (2001, 0.6), (1600, math.nan), (math.inf, 0.6)]:
            with self.assertRaises(ValueError):
                quoted_values(self.c, price, discount)

    def test_source_threshold_cannot_be_relabelled_as_TA_or_fish(self):
        unit = self.c['gaukler_2023_tomato']['quality_unit_id']
        self.assertTrue(source_quality_eligible(self.c, 55, unit))
        self.assertFalse(source_quality_eligible(self.c, 54.999, unit))
        for other in ['relative_TA', 'g_per_kg_TA', 'mullet_sensory_score']:
            with self.assertRaises(ValueError):
                source_quality_eligible(self.c, 55, other)

    def test_normalization_is_not_currency_exchange(self):
        self.assertEqual(normalize_money(960, 'USD', 1600, 'USD'), 0.6)
        with self.assertRaises(ValueError):
            normalize_money(960, 'USD', 4.5, 'EUR')

    def test_missing_COP_or_fuel_conversion_fails(self):
        self.assertAlmostEqual(electrical_energy_from_heat(1.75, 1.66), 1.0542168674698795)
        with self.assertRaises(ValueError):
            electrical_energy_from_heat(1.75, None)
        with self.assertRaises(ValueError):
            fuel_from_heat(1.75, 1.66, None, None)

    def test_thermal_tariff_identity_and_no_revenue_from_TA(self):
        r = calculations(self.c)
        self.assertAlmostEqual(r['v1_thermal_cost_unit_check']['EUR_per_thermal_kWh'], 3)
        self.assertAlmostEqual(r['v1_thermal_cost_unit_check']['EUR_per_vehicle_hour'], 5.184)
        self.assertFalse(r['new_profit_result'])
        self.assertFalse(r['quality_mapping_ready'])
        self.assertEqual(len(r['source_specific_value_quotes']), 9)
        with self.assertRaises(ValueError):
            require_complete_routing_profile(self.c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
