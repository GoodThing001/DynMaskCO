import copy
import os
from pathlib import Path
import tempfile
import unittest
from known_exact import fixtures,validate,solve


class KnownExactTests(unittest.TestCase):
    def test_public_future_rejected(self):
        p=fixtures(710.6548152249378)[0]
        p['orders'][0]['reveal']=.1
        with self.assertRaises(ValueError):validate(p)

    def test_private_extra_rejected(self):
        p=fixtures(710.6548152249378)[0];p['future_day']=[]
        with self.assertRaises(ValueError):validate(p)

    def test_bound_and_exact_types(self):
        for value in [True,'1',float('nan')]:
            p=fixtures(710.6548152249378)[0];p['budget']=value
            with self.assertRaises(ValueError):validate(p)
        p=fixtures(710.6548152249378)[0];p['orders']*=2
        with self.assertRaises(ValueError):validate(p)

    def test_actual_contract_utility_witness_and_distance_zero_energy_change(self):
        scripts=Path(os.environ.get('MASKCO_SCRIPTS','D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts'))
        if not (scripts/'evaluation/scenario_saa.py').is_file():self.skipTest('actual extension source unavailable')
        with tempfile.TemporaryDirectory() as task_temp:
            a=solve(fixtures(710.6548152249378)[0],scripts,Path(task_temp)/'utility')
            b=solve(fixtures(710.6548152249378)[1],scripts,Path(task_temp)/'physics')
        self.assertIs(a['exact_within_declared_subproblem'],True)
        self.assertEqual(a['enumerated'],16) # sum(P(3,k)), including reject-all
        self.assertGreater(a['utility_gain'],0)
        self.assertIs(a['best_evaluation']['hard_feasible'],True)
        self.assertIs(a['independent_replay']['all_evaluations_equal'],True)
        self.assertEqual(b['enumerated'],2) # accepted set immutable; only two permutations
        x,y=[r['evaluation'] for r in b['all_candidates']]
        self.assertAlmostEqual(x['distance_km'],y['distance_km'],places=6)
        self.assertGreater(abs(x['energy_kwh']-y['energy_kwh']),1)
        self.assertNotAlmostEqual(x['tomato_mean_quality'],y['tomato_mean_quality'],places=8)
        self.assertIs(b['independent_replay']['all_evaluations_equal'],True)
        self.assertIs(a['global_optimality_claim'],False)
        rows={tuple(r['sequence']):r['evaluation'] for r in a['all_candidates']}
        # Actual shared vehicle cooling means marginal energy need not >= average.
        self.assertLess(rows[(1,2)]['energy_kwh']-rows[(1,)]['energy_kwh'],rows[(1,)]['energy_kwh'])


if __name__=='__main__': unittest.main(verbosity=2)
