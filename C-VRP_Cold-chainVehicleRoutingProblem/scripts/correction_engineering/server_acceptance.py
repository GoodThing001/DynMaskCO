"""Portable unittest regression for hosts without pytest; synthetic engineering only."""
from pathlib import Path
import sys, tempfile, unittest, json, copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import Journal, encode, decode, verify
from delivery_trace_contract import validate_trace
from qualification import assess, REQUIRED

def fixture():
    o={1:dict(quantity=2.,release=0.,ready=.2,tw_start=0.,tw_end=2.,service=.1,zone=1)}
    e=[dict(time=.2,kind='load_start',vehicle=0,orders=[1]),dict(time=.3,kind='load_finish',vehicle=0),
       dict(time=.3,kind='plan',vehicle=0,suffix=[1,0]),dict(time=.3,kind='depart',vehicle=0,target=1),
       dict(time=.4,kind='arrive',vehicle=0,target=1),dict(time=.4,kind='service_start',vehicle=0,order=1),
       dict(time=.5,kind='service_finish',vehicle=0,order=1),dict(time=.5,kind='depart',vehicle=0,target=0),
       dict(time=.6,kind='arrive',vehicle=0,target=0)]
    a=dict(vehicles=1,capacity=5.,horizon=2.,loading_duration=.1,speed=10.,coordinates={0:(0.,0.),1:(1.,0.)})
    return o,e,a

class Acceptance(unittest.TestCase):
    def test_journal(self):
        original={3:(np.array([float('inf'),2.],dtype=np.float32),{2,1},frozenset({4}))}
        restored=decode(encode(original))
        self.assertTrue(np.array_equal(restored[3][0],original[3][0]))
        self.assertEqual(restored[3][1:],original[3][1:])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'journal.jsonl'; j=Journal(p); j.write('day_end',day_id=0)
            self.assertFalse(verify(p,1)['complete']); j.write('run_end'); j.close()
            self.assertTrue(verify(p,1)['complete'])
            with self.assertRaises(FileExistsError):Journal(p)
            raw=p.read_bytes();p.write_bytes(raw.replace(b'day_end',b'day_bad'))
            with self.assertRaises(ValueError):verify(p)
            p.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):verify(p)
    def test_delivery(self):
        o,e,a=fixture(); r=validate_trace(o,e,**a)
        self.assertEqual(r['distance_km'],2.);self.assertFalse(r['formal_online_ready'])
    def test_reuse(self):
        o,e,a=fixture();o[2]={**o[1],'release':.8,'ready':.8};a['coordinates'][2]=(1.,0.)
        second=copy.deepcopy(e)
        for x in second:
            x['time']=round(x['time']+.6,6)
            if 'orders' in x:x['orders']=[2]
            if 'order' in x:x['order']=2
            if x.get('target')==1:x['target']=2
            if 'suffix' in x:x['suffix']=[2,0]
        self.assertEqual(validate_trace(o,e+second,**a)['delivered'],[1,2])
    def test_pickup_replay(self):
        scripts=Path(__file__).resolve().parents[1]
        for folder in ('evaluation','coldchain','simulation','models'):sys.path.insert(0,str(scripts/folder))
        from coldchain_evaluator_a1 import add_v2_initial_quality
        from scenario_saa import make_c0_contract_v2
        from pickup_trace_runner import run_observed,replay
        c=make_c0_contract_v2()
        ds={'coords':np.array([[[.5,.5],[.6,.5],[.4,.5]]],np.float32),'demands':np.array([[0.,2.,2.]],np.float32),
          'tw_start':np.zeros((1,3),np.float32),'tw_end':np.full((1,3),22.,np.float32),
          'service_time':np.array([[0.,.05,.05]],np.float32),'reveal_time':np.array([[1e6,.5,1.]],np.float32),
          'temp_class':np.array([[0,1,0]],np.int32)}
        ds=add_v2_initial_quality(ds,c)
        class EmptySampler:
            def sample(self,snapshot,rng,k):return [[] for _ in range(k)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pickup.jsonl'
            rows=run_observed(dataset=ds,contract=c,budget=1e9,capacity=50.,vehicles=2,cooling_share=2.,
                sampler=EmptySampler(),penalties={0:5.,1:10.,2:15.},k=2,time_limit=60.,arm_seed=17,output=p)
            self.assertEqual(len(rows),1);self.assertTrue(replay(p)['all_evaluations_equal'])

def invalid_test(kind):
    def test(self):
        o,e,a=fixture()
        if kind=='duplicate':e[0]['orders']=[1,1]
        elif kind=='unready':o[1]['ready']=.25
        elif kind=='capacity':a['capacity']=1.
        elif kind=='compartment':a['compartment_capacity']=(2.,1.,2.)
        elif kind=='early_arrival':e[4]['time']=.31
        elif kind=='busy_vehicle':e.insert(4,dict(time=.31,kind='load_start',vehicle=0,orders=[1]))
        elif kind=='frozen_leg':e.insert(4,dict(time=.31,kind='plan',vehicle=0,suffix=[0]))
        elif kind=='time_window':o[1]['tw_end']=.35
        elif kind=='missing_return':e=e[:-1]
        elif kind=='unknown_vehicle':e[0]['vehicle']=1
        else:e[0]['time']=float('nan')
        with self.assertRaises(ValueError):validate_trace(o,e,**a)
    return test
for k in ('duplicate','unready','capacity','compartment','early_arrival','busy_vehicle','frozen_leg',
          'time_window','missing_return','unknown_vehicle','nan'):
    setattr(Acceptance,'test_reject_'+k,invalid_test(k))
def readiness_test(scope):
    def test(self):
        self.assertTrue(assess(scope,{})['missing'])
        e={k:True for k in REQUIRED[scope]};self.assertTrue(assess(scope,e)['technical_prerequisites_met'])
        self.assertFalse(assess(scope,e)['scientific_execution_authorized'])
        e[REQUIRED[scope][0]]=1;self.assertTrue(assess(scope,e)['missing'])
    return test
for k in REQUIRED:setattr(Acceptance,'test_readiness_'+k,readiness_test(k))
if __name__=='__main__':unittest.main(verbosity=2)
