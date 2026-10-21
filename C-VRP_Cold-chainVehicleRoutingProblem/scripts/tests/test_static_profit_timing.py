"""Deadline tests use the original customer instance, with controlled clock delays."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'static_profit'))
from profit_core import ProfitTask
from profit_search import repair,remove_customers
from timed_search import SearchClock


class StrictTiming(unittest.TestCase):
    def test_independent_validation_finishing_late_is_excluded(self):
        task=ProfitTask('r101',fold=0,role='train');routes=task.initial()
        elapsed=[0.0];clock=SearchClock(task,1,now=lambda:elapsed[0])
        original=task.evaluate
        def delayed(r,*,audit=False):
            score=original(r,audit=audit)
            if audit:elapsed[0]=1.01
            return score
        task.evaluate=delayed
        self.assertFalse(clock.offer(routes,1))
        self.assertEqual(clock.events,[])
        self.assertEqual(clock.rejected_late,1)

    def test_anytime_uses_completed_audited_incumbent_and_exposes_overhead(self):
        task=ProfitTask('r101',fold=0,role='train');routes=task.initial()
        elapsed=[0.0];clock=SearchClock(task,60,now=lambda:elapsed[0])
        clock.fallback=routes
        elapsed[0]=1.01;self.assertTrue(clock.offer(routes,0))
        elapsed[0]=60.02
        result=clock.finish('clock_test',1)
        self.assertEqual(result['anytime']['1']['status'],'NO_VALID_INCUMBENT')
        self.assertEqual(result['anytime']['10']['status'],'VALID')
        self.assertEqual(result['anytime']['60']['status'],'VALID')
        self.assertAlmostEqual(result['timing']['search_overrun_s'],0.02)
        self.assertTrue(result['selected_incumbent_in_budget'])

    def test_initialization_timeout_is_labelled_fallback_not_budget_success(self):
        task=ProfitTask('r101',fold=0,role='train');routes=task.initial()
        elapsed=[0.0];clock=SearchClock(task,0.1,now=lambda:elapsed[0])
        clock.fallback=routes;elapsed[0]=0.2
        clock.offer(routes,0)
        result=clock.finish('slow_init',0)
        self.assertEqual(result['status'],'INITIALIZATION_TIMEOUT')
        self.assertFalse(result['selected_incumbent_in_budget'])
        self.assertTrue(result['final']['independently_audited'])

    def test_top8_controls_share_exact_position_count_and_no_fallback(self):
        import numpy as np
        task=ProfitTask('r201',fold=0,role='train');routes=task.initial();removed=tuple(range(1,11))
        counts=[]
        for heatmap in (np.zeros((101,101)),np.random.default_rng(1000).normal(size=(101,101))):
            current=ProfitTask('r201',fold=0,role='train');stats={}
            result=repair(current,remove_customers(routes,removed),removed,heatmap=heatmap,implementation='vectorized',fallback=False,stats=stats)
            self.assertTrue(current.evaluate(result,audit=True)['independently_audited'])
            self.assertNotIn('fallback_scans',stats)
            counts.append(stats['exact_positions'])
        self.assertEqual(counts,[440,440])


if __name__=='__main__':unittest.main()
