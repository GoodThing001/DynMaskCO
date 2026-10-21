"""Teacher refinement preserves audited incumbents and training-role isolation."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'static_profit'))
from profit_core import ProfitTask
from profit_search import alns_search


class TeacherRefinement(unittest.TestCase):
    def test_zero_iteration_refinement_preserves_source_teacher(self):
        task=ProfitTask('r101',fold=0,role='train')
        routes=task.initial()
        result=alns_search(task,seconds=2,iterations=0,seed=1000,initial_routes=routes)
        self.assertEqual(tuple(map(tuple,result['routes'])),routes)
        self.assertTrue(result['final']['independently_audited'])

    def test_invalid_and_validation_warmstart_rejected(self):
        train=ProfitTask('r101',fold=0,role='train')
        with self.assertRaises(ValueError):
            alns_search(train,seconds=1,iterations=0,seed=1000,initial_routes=((1,),))
        validation=ProfitTask('c205',fold=0,role='validation')
        with self.assertRaises(ValueError):
            alns_search(validation,seconds=1,iterations=0,seed=1000,initial_routes=validation.initial())


if __name__=='__main__':
    unittest.main()
