"""Portable engineering only: actual model activation requires isolated server QA."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from delivery_destroy_eq8 import (REGISTERED, verify_protocol, run_arm,
    select_guided, synthetic_data, DeliveryContract, validate_qa)


class FakeProvider:
    def __init__(self, invert=False):
        self.calls = 0
        self.invert = invert
    def features(self, data, capacity):
        return np.zeros((1, len(data['demands']), 1))
    def logits(self, features, adj):
        self.calls += 1
        values = np.arange(adj.size, dtype=float).reshape(adj.shape)
        return -values if self.invert else values


class Tests(unittest.TestCase):
    def test_protocol_fail_closed(self):
        p = dict(REGISTERED, stage='synthetic_engineering_qa')
        verify_protocol(p)
        for key, wrong in [('steps',True),('time_limit_s',10),('claims_online_gain',0),
                           ('dataset_seed',2026101002),('mask_fraction',.3)]:
            q = copy.deepcopy(p); q[key] = wrong
            with self.assertRaises(ValueError): verify_protocol(q)
        q = copy.deepcopy(p); q['instance_ids'][0] = 17000.0
        with self.assertRaises(ValueError): verify_protocol(q)

    def test_exact_guided_mask_and_signal(self):
        _, routes = synthetic_data()
        logits = np.arange(51*51, dtype=float).reshape((51,51))
        a = select_guided(logits,routes,51,25)
        b = select_guided(-logits,routes,51,25)
        self.assertEqual(len(a),25)
        self.assertNotEqual(a,b)
        logits[1,2] = float('nan')
        with self.assertRaises(ValueError): select_guided(logits,routes,51,25)

    def test_all_eight_complete_preserve_manifest_and_edges(self):
        data, initial = synthetic_data()
        for name in REGISTERED['arms']:
            provider = FakeProvider()
            arm = run_arm(data,initial,DeliveryContract(),name,provider,9001,0,time_limit_s=20.)
            self.assertEqual(provider.calls,8)
            self.assertTrue(arm['fixed_schedule_qualified'])
            self.assertEqual(len(arm['attempts']),8)
            for a in arm['attempts']:
                self.assertEqual(len(a['removed']),25)
                self.assertTrue(a['candidate_report']['hard'])
                self.assertEqual(len(a['candidate_report']['delivered']),50)
                self.assertEqual(a['guide_outputs_used'],name!='Pdiscard')
                self.assertTrue(a['retained_edges_preserved'])

    def test_discard_outputs_cannot_change_decisions(self):
        data, initial = synthetic_data()
        a = run_arm(data,initial,DeliveryContract(),'Pdiscard',FakeProvider(),9001,0,time_limit_s=20.)
        b = run_arm(data,initial,DeliveryContract(),'Pdiscard',FakeProvider(True),9001,0,time_limit_s=20.)
        self.assertEqual(a['routes'],b['routes'])
        for aa, bb in zip(a['attempts'],b['attempts']):
            self.assertEqual(aa['removed'],bb['removed'])
            self.assertEqual(aa['candidate_routes'],bb['candidate_routes'])
            self.assertEqual(aa['insertions'],bb['insertions'])
            self.assertNotEqual(aa['logits_sha256'],bb['logits_sha256'])

    def test_late_all_attempts_retained_and_fallback(self):
        data, initial = synthetic_data()
        class JumpProvider(FakeProvider):
            def features(self,*args):
                counter[0] = 20.
                return np.zeros((1,51,1))
        counter = [0.]
        arm = run_arm(data,initial,DeliveryContract(),'Pguided',JumpProvider(),9001,0,
                      clock=lambda:counter[0])
        self.assertTrue(arm['timeout'])
        self.assertFalse(arm['fixed_schedule_qualified'])
        self.assertEqual(arm['routes'],initial)
        self.assertEqual(len(arm['attempts']),8)
        self.assertEqual(arm['decode_calls'],0)

    def test_qa_another_panel_rejected(self):
        with self.assertRaises(ValueError):
            validate_qa({'stage':'synthetic_engineering_qa',
                'actual_engineering_qa_pass':True,'protocol':{'version':'D-EQ8-v1'}},{})


if __name__ == '__main__':
    unittest.main()
