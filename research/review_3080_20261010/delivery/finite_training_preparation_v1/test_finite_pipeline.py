import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from finite_common import schedule, classify_pair, verify_research_gate


class FiniteTests(unittest.TestCase):
    def test_128steps_exact_two_exposures_every_state(self):
        order=schedule(256,128,4,73001)
        self.assertEqual(len(order),128)
        self.assertTrue(all(len(b)==4 for b in order))
        flat=[i for b in order for i in b]
        self.assertTrue(all(flat.count(i)==2 for i in range(256)))
        self.assertEqual(order,schedule(256,128,4,73001))

    def test_boolean_and_unbalanced_schedule_rejected(self):
        with self.assertRaises(ValueError): schedule(True,128,4,1)
        with self.assertRaises(ValueError): schedule(256,127,4,1)

    def test_negative_retained_but_CE_target_better_incumbent(self):
        initial={'hard':True,'complete':True,'cost':10.}
        candidate=dict(initial,cost=12.)
        row=classify_pair(initial,candidate,[[2,1]],[[1,2]],'complete',False)
        self.assertEqual(row['preference_sign'],-1)
        self.assertEqual(row['category'],'negative')
        self.assertEqual(row['CE_target_routes'],[[1,2]])
        self.assertTrue(row['all_states_retained'])

    def test_positive_and_small_and_timeout_all_retained(self):
        initial={'hard':True,'complete':True,'cost':10.}
        positive=classify_pair(initial,dict(initial,cost=8.),[[2,1]],[[1,2]],'complete',False)
        small=classify_pair(initial,dict(initial,cost=10.-1e-9),[[2,1]],[[1,2]],'complete',False)
        late=classify_pair(initial,dict(initial,cost=8.),[[2,1]],[[1,2]],'complete',True)
        self.assertEqual(positive['CE_target_routes'],[[2,1]])
        self.assertEqual(small['category'],'small')
        self.assertEqual(small['CE_target_routes'],[[1,2]])
        self.assertEqual(late['category'],'invalid')
        self.assertEqual(late['preference_sign'],0)
        self.assertTrue(all(r['all_states_retained'] for r in (positive,small,late)))

    def test_true_initial_hard_failure_blocks_no_resample(self):
        with self.assertRaises(ValueError):
            classify_pair({'hard':False,'complete':False,'cost':None}, {}, [],[],'failed',False)

    def test_draft_cannot_be_research(self):
        with self.assertRaises(ValueError):
            verify_research_gate({'status':'DRAFT'}, {})


if __name__=='__main__': unittest.main()
