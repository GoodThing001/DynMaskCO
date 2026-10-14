from pathlib import Path
import copy
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_io import Journal, encode, decode, verify
from delivery_trace_contract import validate_trace
from qualification import assess,REQUIRED


def test_audit_lossless_and_immutable(tmp_path):
    value = {3: (np.array([float('inf'), 2.],dtype=np.float32), {2,1}, frozenset({4}))}
    restored = decode(encode(value))
    assert np.array_equal(restored[3][0],value[3][0])
    assert restored[3][1:] == value[3][1:]
    p = tmp_path/'events.jsonl'
    j=Journal(p)
    j.write('day_end',day_id=0)
    assert not verify(p,1)['complete']
    j.write('run_end')
    j.close()
    assert verify(p,1)['complete']
    with pytest.raises(FileExistsError):
        Journal(p)
    content=p.read_bytes()
    p.write_bytes(content.replace(b'day_end',b'day_bad'))
    with pytest.raises(ValueError,match='changed'):
        verify(p)
    p.write_bytes(content[:-1])
    with pytest.raises(ValueError,match='Truncated'):
        verify(p)


def delivery_case():
    orders={1:dict(quantity=2.,release=0.,ready=.2,tw_start=0.,tw_end=2.,service=.1,zone=1)}
    events=[dict(time=.2,kind='load_start',vehicle=0,orders=[1]),
            dict(time=.3,kind='load_finish',vehicle=0),
            dict(time=.3,kind='plan',vehicle=0,suffix=[1,0]),
            dict(time=.3,kind='depart',vehicle=0,target=1),
            dict(time=.4,kind='arrive',vehicle=0,target=1),
            dict(time=.4,kind='service_start',vehicle=0,order=1),
            dict(time=.5,kind='service_finish',vehicle=0,order=1),
            dict(time=.5,kind='depart',vehicle=0,target=0),
            dict(time=.6,kind='arrive',vehicle=0,target=0)]
    args=dict(vehicles=1,capacity=5.,horizon=2.,loading_duration=.1,speed=10.,coordinates={0:(0.,0.),1:(1.,0.)})
    return orders,events,args


def test_delivery_complete():
    o,e,a=delivery_case()
    result=validate_trace(o,e,**a)
    assert result['distance_km']==2.
    assert not result['formal_online_ready']


def test_delivery_reuses_same_vehicle_after_return():
    o,e,a=delivery_case()
    o[2]={**o[1],'release':.8,'ready':.8}
    a['coordinates'][2]=(1.,0.)
    second=[]
    for event in e:
        item={**event,'time':round(event['time']+.6,6)}
        if 'orders' in item:item['orders']=[2]
        if 'order' in item:item['order']=2
        if item.get('target')==1:item['target']=2
        if 'suffix' in item:item['suffix']=[2,0]
        second.append(item)
    assert validate_trace(o,e+second,**a)['delivered']==[1,2]


@pytest.mark.parametrize('failure', ['duplicate','unready','capacity','compartment','early_arrival',
    'busy_vehicle','frozen_leg','time_window','missing_day','unknown_vehicle','nan'])
def test_delivery_rejects_invalid(failure):
    o,e,a=delivery_case()
    if failure=='duplicate':e[0]['orders']=[1,1]
    elif failure=='unready':o[1]['ready']=.25
    elif failure=='capacity':a['capacity']=1.
    elif failure=='compartment':a['compartment_capacity']=(2.,1.,2.)
    elif failure=='early_arrival':e[4]['time']=.31
    elif failure=='busy_vehicle':e.insert(4,dict(time=.31,kind='load_start',vehicle=0,orders=[1]))
    elif failure=='frozen_leg':e.insert(4,dict(time=.31,kind='plan',vehicle=0,suffix=[0]))
    elif failure=='time_window':o[1]['tw_end']=.35
    elif failure=='missing_day':e=e[:-1]
    elif failure=='unknown_vehicle':e[0]['vehicle']=1
    else:e[0]['time']=float('nan')
    with pytest.raises(ValueError):validate_trace(o,e,**a)


def test_pickup_real_trace_replay(tmp_path):
    scripts=Path(__file__).resolve().parents[1]
    for folder in ('evaluation','coldchain','simulation','models'):
        sys.path.insert(0,str(scripts/folder))
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from scenario_saa import make_c0_contract_v2
    from pickup_trace_runner import run_observed,replay
    c=make_c0_contract_v2()
    ds={'coords':np.array([[[.5,.5],[.6,.5],[.4,.5]]],dtype=np.float32),
        'demands':np.array([[0.,2.,2.]],dtype=np.float32),
        'tw_start':np.zeros((1,3),dtype=np.float32),
        'tw_end':np.full((1,3),22.,dtype=np.float32),
        'service_time':np.array([[0.,.05,.05]],dtype=np.float32),
        'reveal_time':np.array([[1e6,.5,1.]],dtype=np.float32),
        'temp_class':np.array([[0,1,0]],dtype=np.int32)}
    ds=add_v2_initial_quality(ds,c)
    class EmptySampler:
        def sample(self,snapshot,rng,k):return [[] for _ in range(k)]
    p=tmp_path/'pickup.jsonl'
    rows=run_observed(dataset=ds,contract=c,budget=1e9,capacity=50.,vehicles=2,
        cooling_share=2.,sampler=EmptySampler(),penalties={0:5.,1:10.,2:15.},
        k=2,time_limit=60.,arm_seed=17,output=p)
    assert len(rows)==1
    assert verify(p,1)['complete']
    assert replay(p)['all_evaluations_equal']
    assert any(json.loads(line)['kind']=='decision' for line in p.read_text().splitlines())


@pytest.mark.parametrize('scope',list(REQUIRED))
def test_qualification_fails_closed(scope):
    assert assess(scope,{})['missing']
    result=assess(scope,{key:True for key in REQUIRED[scope]})
    assert result['technical_prerequisites_met']
    assert not result['scientific_execution_authorized']
    partial={key:True for key in REQUIRED[scope]}
    partial[REQUIRED[scope][0]]=1
    assert assess(scope,partial)['missing']
