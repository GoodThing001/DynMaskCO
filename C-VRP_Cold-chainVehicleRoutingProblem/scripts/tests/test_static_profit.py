"""Semantic tests on original 100-customer cases, not generated toy instances."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'static_profit'))
from profit_core import ProfitTask, VARIANTS, partial_batch, verify_research_freeze, EXTENSION_ROOT
from profit_search import remove_customers, repair, vns_search


class StaticProfitSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = ProfitTask('r101',fold=0,role='train')
        cls.routes = cls.task.initial()

    def test_scientific_freeze_is_intact(self):
        self.assertEqual(verify_research_freeze()['files'],72)

    def test_outer_test_and_role_misuse_are_rejected(self):
        for name,role in [('c105','train'),('r101','validation'),('r101','test')]:
            with self.assertRaises(ValueError):
                ProfitTask(name,fold=0,role=role)
        with self.assertRaises(ValueError):
            ProfitTask('r101',fold=0,role='train',value_multiplier=2)

    def test_full_multi_vehicle_solution_has_exact_coverage_and_money(self):
        score = self.task.evaluate(self.routes,audit=True)
        self.assertGreater(score['used_vehicles'],1)
        self.assertGreaterEqual(score['minimum_quality'],0.8-1e-7)
        nominal = self.task.model.price*sum(row[3] for row in self.task.ins['nodes'])
        self.assertAlmostEqual(nominal-score['net_benefit_U'],score['equivalent_loss_cost_U'])
        self.assertAlmostEqual(score['equivalent_loss_cost_U'],score['quality_loss_value_U']+
                               score['distance_km']+score['refrigeration_cost_U'])

    def test_missing_duplicate_and_noninteger_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.task.evaluate(self.routes[:-1])
        with self.assertRaises(ValueError):
            self.task.evaluate(self.routes+((1,),))
        bad = [[i] for i in range(1,101)]
        bad[0] = [1.0]
        with self.assertRaises(ValueError):
            self.task.evaluate(bad)

    def test_partial_context_does_not_leak_hidden_teacher_clocks(self):
        # The wide-window training instance admits both pair orientations; the
        # narrow r101 source need not have any such pair under 90-minute service.
        task = ProfitTask('r201',fold=0,role='train')
        pair = None
        for i in range(1,101):
            for j in range(i+1,101):
                if task.route((i,j)).feasible and task.route((j,i)).feasible:
                    pair = i,j
                    break
            if pair:
                break
        self.assertIsNotNone(pair)
        i,j = pair
        rest = tuple((node,) for node in range(1,101) if node not in pair)
        teacher_a,teacher_b = ((i,j),)+rest,((j,i),)+rest
        self.assertNotEqual(task.route((i,j)).stops[0]['customer'],task.route((j,i)).stops[0]['customer'])
        a,partial_a = partial_batch(task,teacher_a,pair,'quality_directed')
        b,partial_b = partial_batch(task,teacher_b,pair,'quality_directed')
        self.assertEqual(partial_a,partial_b)
        np.testing.assert_array_equal(a['features'],b['features'])
        np.testing.assert_array_equal(a['adjacency'],b['adjacency'])
        self.assertFalse(np.array_equal(a['targets'],b['targets']))

    def test_features_and_supervision_have_explicit_ablation_semantics(self):
        removed = tuple(range(1,11))
        batches = {v:partial_batch(self.task,self.routes,removed,v)[0] for v in VARIANTS}
        for variant,batch in batches.items():
            self.assertEqual(batch['features'].shape,(1,101,16))
            self.assertTrue(np.all(np.isfinite(batch['features'])))
            np.testing.assert_allclose(batch['targets'][0,1:].sum(axis=1),1)
            self.assertEqual(batch['row_weights'][0,0],0)
            self.assertTrue(np.all(np.diag(batch['targets'][0])==0))
        np.testing.assert_array_equal(batches['maskco_base']['features'][...,:11],
                                      batches['quality_directed']['features'][...,:11])
        self.assertTrue(np.all(batches['maskco_base']['features'][...,11:]==0))
        self.assertTrue(np.any(batches['quality_directed']['features'][...,11:]!=0))

    def test_regret_repair_preserves_all_original_orders_and_hard_constraints(self):
        removed = tuple(range(1,11))
        partial = remove_customers(self.routes,removed)
        result = repair(self.task,partial,removed)
        self.assertTrue(self.task.evaluate(result,audit=True)['independently_audited'])

    def test_quality_infeasibility_is_not_a_penalty(self):
        records = json.loads((EXTENSION_ROOT/'results/static_delivery/publication_protocol_v1_preflight_20261021/feasibility_witnesses.json').read_text(encoding='utf-8'))
        row = next(r for r in records if r['case']=='r102')
        task = ProfitTask('r102',fold=0,role='train')
        route = tuple(row['explicit_constraint_witness']['route'])
        self.assertTrue(task.model.route(route,enforce_quality=False).feasible)
        self.assertFalse(task.route(route).feasible)

    def test_vns_acceptance_cannot_worsen_total_net_benefit(self):
        task = ProfitTask('r101',fold=0,role='train')
        result = vns_search(task,seconds=2,iterations=30,seed=1000)
        self.assertGreaterEqual(result['final']['net_benefit_U'],result['initial']['net_benefit_U']-1e-7)
        self.assertTrue(result['final']['independently_audited'])

    def test_real_maskco_backbone_and_direction_module(self):
        from profit_neural import ProfitMaskCO,predict
        for variant in ('maskco_base','quality_directed'):
            model = ProfitMaskCO(variant,'smoke_cpu',1000)
            self.assertEqual(type(model.backbone).__module__,'models.CVRPModel')
            batch,_ = partial_batch(self.task,self.routes,tuple(range(1,21)),variant)
            values = predict(model,batch)
            self.assertEqual(values.shape,(101,101))
            error = float(np.max(np.abs(values-values.T)))
            if variant=='maskco_base':
                self.assertLess(error,1e-5)
            else:
                self.assertGreater(error,1e-4)


if __name__ == '__main__':
    unittest.main()
