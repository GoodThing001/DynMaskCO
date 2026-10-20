"""Meaningful rejection tests use intact published instances, never generated customers."""
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import solomon_preliminary as sp
import quality_evidence_demo as qd

class PublishedInstanceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=next((sp.DATA/'instances').rglob('c101.txt'))
        cls.instance=sp.load(cls.path)
        result=sp.EXTENSION_ROOT/'results/static_delivery/classic_solomon_preliminary_v2_20261020/results.json'
        cls.rows=json.loads(result.read_text())['rows']

    def test_all_actual_solutions_independently_feasible(self):
        self.assertEqual(len(self.rows),12)
        for row in self.rows:
            path=next((sp.DATA/'instances').rglob(row['case']+'.txt'))
            ins=sp.load(path)
            r=sp.audit(ins,row['routes'])
            self.assertTrue(r['feasible'],(row['case'],row['method'],r['errors']))
            self.assertAlmostEqual(r['distance'],row['report']['distance'],places=9)

    def test_duplicate_and_missing_rejected(self):
        routes=[r[:] for r in self.rows[0]['routes']]
        routes[0][0]=routes[0][1]
        self.assertIn('coverage_or_duplicate',sp.audit(self.instance,routes)['errors'])

    def test_real_demand_overcapacity_rejected(self):
        report=sp.audit(self.instance,[list(range(1,101))])
        self.assertIn('capacity',report['errors'])

    def test_reversed_real_route_time_windows_rejected(self):
        routes=[list(reversed(r)) for r in self.rows[0]['routes']]
        self.assertIn('time_window',sp.audit(self.instance,routes)['errors'])

    def test_no_distance_gap_when_vehicle_count_differs(self):
        for row in self.rows:
            if row['report']['vehicles_used']!=row['reference']['vehicles']:
                self.assertIsNone(row['distance_gap_pct'])

    def test_primary_sha_lists_all_original_instances(self):
        manifest=json.loads((sp.DATA/'FILES_SHA256.json').read_text())
        self.assertEqual(len(manifest),56)
        self.assertEqual(len(list((sp.DATA/'instances').rglob('*.txt'))),56)

class QualityUnitsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((sp.EXTENSION_ROOT/'configs/static_delivery/product_evidence_v1.json').read_text())

    def test_fish_hour_conversion_and_no_invented_initial_score(self):
        rows=qd.calculate(self.config)
        row=next(r for r in rows if r['product'].startswith('chilled') and r['temperature_C']==4 and r['hours']==8)
        self.assertAlmostEqual(row['sensory_score_decrease'],0.452*8/24)
        self.assertIsNone(row['initial_score'])
        self.assertIsNone(row['sales_value'])

    def test_relative_tomato_is_not_mass_or_money(self):
        for r in qd.calculate(self.config):
            self.assertIsNone(r['mass_loss'])
            self.assertIsNone(r['sales_value'])
            if r['product'].startswith('cherry'):
                self.assertTrue(0<r['TA_remaining_ratio']<1)
                self.assertAlmostEqual(r['TA_remaining_ratio']+r['relative_TA_decrease'],1)

    def test_missing_economic_evidence_blocks_profit(self):
        with self.assertRaisesRegex(ValueError,'profit_not_configured'):
            qd.require_profit_inputs(self.config)

if __name__=='__main__': unittest.main(verbosity=2)
