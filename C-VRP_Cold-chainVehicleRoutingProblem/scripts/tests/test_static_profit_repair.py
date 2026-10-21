"""Vectorized insertion must retain exact candidate order and original routes."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'static_profit'))
from profit_core import ProfitTask
import profit_search as search


class RepairEquivalence(unittest.TestCase):
    def test_original_cases_have_identical_candidates_and_repairs(self):
        self.assertTrue(hasattr(search,'InsertionIndex'),'shared vectorized insertion context absent')
        for case in ('r101','r201','rc101'):
            task=ProfitTask(case,fold=0,role='train')
            routes=task.initial();removed=tuple(range(1,11))
            partial=search.remove_customers(routes,removed)
            for dtype in (np.float32,np.float64):
                for tie in (False,True):
                    heat=np.zeros((101,101),dtype) if tie else np.random.default_rng(1000).normal(size=(101,101)).astype(dtype)
                    index=search.InsertionIndex(task,partial,heat,8)
                    for customer in removed:
                        self.assertEqual(index.options(customer),search.insertion_options(task,partial,customer,heat,8))
                    expected=search.repair(task,partial,removed,heatmap=heat)
                    actual=search.repair(task,partial,removed,heatmap=heat,implementation='vectorized')
                    self.assertEqual(actual,expected)
                    self.assertTrue(task.evaluate(actual,audit=True)['independently_audited'])


if __name__=='__main__':unittest.main()
