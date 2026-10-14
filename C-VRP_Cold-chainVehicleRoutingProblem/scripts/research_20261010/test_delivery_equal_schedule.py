"""Engineering checks with explicit synthetic guidance, not real-model science."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from delivery_equal_schedule import (DeliveryContract, evaluate, digest,
    make_schedule, generate, run_arm, partial_adjacency, verify_protocol, validate_qa)


class SyntheticProvider:
    def __init__(self):
        self.features_calls = 0
        self.decode_calls = 0
        self.adjacencies = []

    def features(self, instance, capacity):
        self.features_calls += 1
        return np.asarray(instance['coords'])

    def logits(self, features, adjacency):
        self.decode_calls += 1
        self.adjacencies.append(adjacency.copy())
        return -np.linalg.norm(features[:,None]-features[None,:],axis=-1)


def tiny():
    data = {'coords': [[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,0.],[0.,2.],[2.,2.]],
            'demands':[0]+[1]*6,'temp_class':[0,0,1,2,0,1,2],
            'reveal_time':[0.]*7,'ready_time':[0.]*7,'tw_start':[0.]*7,
            'tw_end':[8.]*7,'service_time':[0.]+[.03]*6}
    return data, [[1,2,3],[4,5,6]]


class Tests(unittest.TestCase):
    def test_schedule_repeatability_and_not_route_dependent(self):
        a=make_schedule(51,8,2026101041,4)
        self.assertEqual(a,make_schedule(51,8,2026101041,4))
        self.assertTrue(all(len(mask)==15 and len(set(mask))==15 for mask in a))
        self.assertNotEqual(a,make_schedule(51,8,2026101041,5))

    def test_shortcut_not_retained(self):
        adj=partial_adjacency([[1,2,3]],{2},4)
        self.assertEqual(adj[1,3],0)
        self.assertEqual(adj[0,1],1)
        self.assertEqual(adj[3,0],1)

    def test_exact_eight_activated_and_edges_preserved(self):
        data,initial=tiny();provider=SyntheticProvider()
        result=run_arm(data,initial,DeliveryContract(),make_schedule(7,8,71,0),provider)
        self.assertEqual(provider.features_calls,1)
        self.assertEqual(provider.decode_calls,8)
        self.assertEqual(result['actual_attempts'],8)
        self.assertTrue(result['fixed_schedule_qualified'])
        self.assertTrue(all(r.get('retained_edges_preserved') for r in result['attempts']))
        self.assertTrue(evaluate(data,result['routes'],DeliveryContract())['hard'])
        self.assertTrue(all(r['cost_after']<=r['cost_before'] for r in result['attempts']))
        json.dumps(result,allow_nan=False)

    def test_nolearned_same_engine_no_inference(self):
        data,initial=tiny();schedule=make_schedule(7,8,71,0)
        p=run_arm(data,initial,DeliveryContract(),schedule,SyntheticProvider(),guide_weight=0.)
        h=run_arm(data,initial,DeliveryContract(),schedule)
        self.assertEqual(p['routes'],h['routes'])
        self.assertEqual(h['decode_calls'],0)
        self.assertFalse(h['learning_activated'])
        self.assertEqual([r['removed'] for r in h['attempts']],schedule)
        self.assertEqual([r['candidate_routes'] for r in p['attempts']],
                         [r['candidate_routes'] for r in h['attempts']])

    def test_feature_deadline_keeps_all_masks_and_fallback(self):
        class Clock:
            now=0.
            def __call__(self):return self.now
        clock=Clock()
        class Slow(SyntheticProvider):
            def features(self,*args):
                clock.now=11.
                return super().features(*args)
        data,initial=tiny();result=run_arm(data,initial,DeliveryContract(),
                                         make_schedule(7,8,71,0),Slow(),clock=clock)
        self.assertTrue(result['timeout'])
        self.assertFalse(result['fixed_schedule_qualified'])
        self.assertEqual(result['routes'],initial)
        self.assertEqual(len(result['attempts']),8)
        self.assertEqual(result['actual_attempts'],0)

    def test_decode_deadline_retains_partial_attempt_and_final_fallback(self):
        class Clock:
            now=0.
            def __call__(self):return self.now
        clock=Clock()
        class Slow(SyntheticProvider):
            def logits(self,*args):
                clock.now=11.
                return super().logits(*args)
        data,initial=tiny();result=run_arm(data,initial,DeliveryContract(),
                                         make_schedule(7,8,71,0),Slow(),clock=clock)
        self.assertEqual(result['decode_calls'],1)
        self.assertEqual(result['actual_attempts'],1)
        self.assertEqual(result['attempts'][0]['status'],'deadline')
        self.assertEqual(result['routes'],initial)
        self.assertTrue(result['timeout'])

    def test_invalid_common_route_is_not_screened(self):
        data,_=tiny()
        with self.assertRaises(ValueError):
            run_arm(data,[[1,2]],DeliveryContract(),make_schedule(7,8,71,0))

    def test_wrong_schedule_or_bad_guidance_rejected(self):
        data,initial=tiny()
        with self.assertRaises(ValueError):
            run_arm(data,initial,DeliveryContract(),[[1,1]]*8)
        class Bad(SyntheticProvider):
            def logits(self,*args):return np.full((7,7),np.nan)
        with self.assertRaises(ValueError):
            run_arm(data,initial,DeliveryContract(),make_schedule(7,8,71,0),Bad())

    def test_fresh_generator_preserves_all_unique_ids(self):
        data,routes=generate(2026101002,0)
        self.assertEqual(sorted(i for r in routes for i in r),list(range(1,51)))
        self.assertEqual(digest([data,routes]),digest(generate(2026101002,0)))
        self.assertTrue(evaluate(data,routes,DeliveryContract())['hard'])
        self.assertNotEqual(digest(data),digest(generate(2026101002,1)[0]))

    def test_actual_qa_failure_cannot_open_pilot(self):
        from copy import deepcopy
        data,initial=tiny();schedule=make_schedule(7,8,71,0)
        arms={k:run_arm(data,initial,DeliveryContract(),schedule,
                       None if k=='H' else SyntheticProvider()) for k in ('P','R','H')}
        identity={'source_manifest_sha256':'src','checkpoint_sha256':'cp',
                  'contract_sha256':'contract','model_parameters':{'P':'p','R':'r'}}
        qa={'stage':'synthetic_engineering_qa','actual_engineering_qa_pass':True,
            'rows':[{'instance_id':-1,'arms':arms}],'identity':identity}
        validate_qa(qa,identity)
        wrong=deepcopy(qa);wrong['rows'][0]['arms']['P']['decode_calls']=0
        with self.assertRaises(ValueError):validate_qa(wrong,identity)
        wrong=deepcopy(qa);wrong['identity']['source_manifest_sha256']='changed'
        with self.assertRaises(ValueError):validate_qa(wrong,identity)

    def test_protocol_boolean_seed_rejected(self):
        protocol={'version':'D-EQ8-v1','dataset_seed':2026101002,'mask_seed':2026101041,
          'arm_order_seed':2026101099,'bootstrap_seed':2026101077,'bootstrap_repetitions':2000,
          'random_model_seed':42,'steps':8,'mask_fraction':.3,'guide_weight':1.,'time_limit_s':10.,
          'instance_ids':list(range(32)),'stage':'synthetic_engineering_qa',
          'scope':'fixed_schedule_snapshot_attribution_no_online_gain'}
        verify_protocol(protocol)
        protocol['guide_weight']=True
        with self.assertRaises(ValueError):verify_protocol(protocol)


if __name__=='__main__':
    unittest.main(verbosity=2)
