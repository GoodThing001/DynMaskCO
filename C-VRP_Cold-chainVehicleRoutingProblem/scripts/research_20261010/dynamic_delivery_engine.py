"""Causal finite-fleet delivery simulator: engineering, not a policy or benchmark.

Quality uses explicitly supplied product temperatures and literature profiles.
No vehicle-air model, physical calibration, food-safety or monetary conversion.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'correction_engineering'))
from delivery_trace_contract import validate_trace
from literature_quality import evaluate_product, evaluate_cargo, load_profiles, electrical_energy_kwh


def require(test, reason):
    if not test:raise ValueError(reason)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DynamicDeliveryContract:
    version:str='dynamic-delivery-engineering-v1'
    vehicles:int=2
    capacity:float=12.
    compartment_capacity:tuple=(4.,4.,4.)
    horizon_h:float=8.
    speed_kmph:float=30.
    loading_h:float=.05
    loading_bays:int=1
    product_temperature_c:tuple=(18.,4.,-5.)
    electrical_input_kw:tuple=(.30,1.80,3.20)
    product_temperature_basis:str='explicit_isothermal_simulation_not_air_temperature'
    cooling_scope:str='occupied_compartments_only_empty_return_zero_refrigeration'
    ready_information:str='only_actual_ready_status_after_release'
    allow_extrapolation:bool=False

    def validate(self):
        require(type(self.vehicles) is int and self.vehicles>0,'finite fleet')
        require(type(self.loading_bays) is int and self.loading_bays>0,'loading bays')
        require(all(finite(x) and x>0 for x in (self.capacity,self.horizon_h,self.speed_kmph)), 'units')
        require(finite(self.loading_h) and self.loading_h>0,'nonzero loading')
        require(len(self.compartment_capacity)==3 and all(finite(x) and x>=0 for x in self.compartment_capacity)
                and sum(self.compartment_capacity)<=self.capacity+1e-9,'fixed compartments')
        require(len(self.product_temperature_c)==3 and all(finite(x) for x in self.product_temperature_c),'product temperatures')
        require(len(self.electrical_input_kw)==3 and all(finite(x) and x>=0 for x in self.electrical_input_kw),'electric input')
        require(type(self.allow_extrapolation) is bool,'extrapolation policy')


def validate_orders(orders,contract,coordinates):
    contract.validate()
    profiles,_=load_profiles()
    require(0 in coordinates,'depot')
    require(all(len(x)==2 and all(finite(v) for v in x) for x in coordinates.values()),'coordinates')
    for oid,o in orders.items():
        require(type(oid) is int and oid>0 and oid in coordinates,'order identity')
        require(all(finite(o[k]) for k in ('quantity','release','ready','tw_start','tw_end','service',
                                        'quality_origin_h','initial_quality','warehouse_product_temperature_c')),'order numerics')
        require(o['quantity']>0 and o['ready']>=o['release']>=o['quality_origin_h']>=0,'release/ready/origin')
        require(o['tw_end']>=o['tw_start']>=0 and o['service']>0,'service window')
        require(type(o['zone']) is int and o['zone'] in range(3),'zone')
        require(o['profile_id'] in profiles,'explicit commodity')
        # Early validation checks domain/rate semantics without generating research data.
        evaluate_product(o['profile_id'],o['initial_quality'],[{
            'start_h':o['quality_origin_h'],'end_h':o['quality_origin_h'],
            'product_temperature_c':o['warehouse_product_temperature_c']}],
            allow_extrapolation=contract.allow_extrapolation)


class DynamicDeliveryEngine:
    def __init__(self,orders,coordinates,contract=None,journal_path=None):
        self.contract=contract or DynamicDeliveryContract()
        validate_orders(orders,self.contract,coordinates)
        self._orders=deepcopy(orders);self._coordinates=deepcopy(coordinates)
        self.time=0.;self._queue=[];self._sequence=0;self.events=[];self.commands=[]
        self._released=set();self._ready=set();self._delivered=set()
        self._owner={i:'warehouse' for i in orders}
        self._fleet={v:{'node':0,'mode':'idle','cargo':set(),'load_ids':set(),
                       'suffix':[],'target':None,'until':0.,'trip':0,
                       'product_temperature_c':tuple(self.contract.product_temperature_c)}
                     for v in range(self.contract.vehicles)}
        self._quality_segments={i:[] for i in orders}
        self._power_segments={v:[] for v in self._fleet}
        self._load_records={};self._deliver_records={};self.distance_km=0.
        self._journal=None
        if journal_path is not None:
            path=Path(journal_path);path.parent.mkdir(parents=True,exist_ok=True)
            self._journal=path.open('x',encoding='utf-8')
        for oid,o in orders.items():
            self._schedule(o['release'],0,'release',None,oid)
            self._schedule(o['ready'],1,'ready',None,oid)
        self.advance(0.)

    def _schedule(self,t,priority,kind,vehicle,order=None):
        self._sequence+=1
        heapq.heappush(self._queue,(float(t),priority,self._sequence,kind,vehicle,order))

    def _event(self,kind,vehicle=None,**data):
        row={'time':self.time,'kind':kind,'vehicle':vehicle,**data}
        row['sequence']=len(self.events)
        row['previous_sha256']=self.events[-1]['sha256'] if self.events else None
        row['sha256']=sha(row);self.events.append(row)
        if self._journal is not None:
            self._journal.write(json.dumps(row,allow_nan=False)+'\n')
            self._journal.flush();os.fsync(self._journal.fileno())

    def close(self):
        if self._journal is not None:self._journal.close();self._journal=None

    def _integrate(self,end):
        if end<=self.time:return
        for oid,o in self._orders.items():
            if oid in self._delivered:continue
            begin=max(self.time,o['quality_origin_h'])
            if begin>=end:continue
            owner=self._owner[oid]
            if isinstance(owner,tuple) and owner[0]=='vehicle':
                temp=self._fleet[owner[1]]['product_temperature_c'][o['zone']]
                stage='cargo'
            else:
                temp=o['warehouse_product_temperature_c'];stage='warehouse_including_loading'
            self._quality_segments[oid].append({'start_h':begin,'end_h':end,
                      'product_temperature_c':temp,'stage':stage})
        for vid,v in self._fleet.items():
            zones={self._orders[i]['zone'] for i in v['cargo']}
            power=sum(self.contract.electrical_input_kw[z] for z in zones)
            self._power_segments[vid].append({'start_h':self.time,'end_h':end,
                'electrical_input_kw':power,'power_basis':'simulation'})

    def advance(self,to_h):
        require(finite(to_h) and self.time<=to_h<=self.contract.horizon_h,'monotone horizon')
        while self._queue and self._queue[0][0]<=to_h:
            t,_,_,kind,vid,oid=heapq.heappop(self._queue)
            self._integrate(t);self.time=t
            if kind=='release':
                self._released.add(oid);self._event(kind,order=oid)
            elif kind=='ready':
                self._ready.add(oid);self._event(kind,order=oid)
            else:
                v=self._fleet[vid]
                if kind=='load_finish':
                    require(v['mode']=='loading','load timer')
                    v.update(mode='idle',cargo=set(v['load_ids']),load_ids=set())
                    for i in v['cargo']:
                        self._owner[i]=('vehicle',vid);self._load_records[i]=(self.time,vid)
                    self._event(kind,vid,orders=sorted(v['cargo']))
                elif kind=='arrive':
                    require(v['mode']=='travel' and v['target']==oid,'travel timer')
                    v.update(mode='idle',node=oid)
                    self._event(kind,vid,target=oid)
                    if oid==0:
                        require(not v['cargo'],'cannot return with unserved cargo')
                        v.update(suffix=[],target=None)
                    else:
                        begin=max(self.time,self._orders[oid]['tw_start'])
                        require(begin<=self._orders[oid]['tw_end']+1e-9,'hard service-start window')
                        v['mode']='waiting';v['until']=begin
                        self._schedule(begin,3,'service_start',vid,oid)
                elif kind=='service_start':
                    require(v['mode']=='waiting' and v['target']==oid,'service commitment')
                    v.update(mode='service',until=self.time+self._orders[oid]['service'])
                    self._event(kind,vid,order=oid)
                    self._schedule(v['until'],4,'service_finish',vid,oid)
                elif kind=='service_finish':
                    require(v['mode']=='service' and v['target']==oid and v['suffix'][0]==oid,'unload commitment')
                    o=self._orders[oid]
                    # Domain failures are retained; no silent clipping or inferred price.
                    quality=evaluate_product(o['profile_id'],o['initial_quality'],self._quality_segments[oid],
                                             allow_extrapolation=self.contract.allow_extrapolation)
                    self._event(kind,vid,order=oid,quality=quality)
                    v['cargo'].remove(oid);v['suffix']=v['suffix'][1:]
                    v.update(mode='idle',target=None)
                    self._owner[oid]='delivered';self._delivered.add(oid)
                    self._deliver_records[oid]=(self.time,vid)
                else:raise RuntimeError('internal event kind')
        self._integrate(to_h);self.time=float(to_h)

    def observation(self):
        """No future identities/cardinality/coords/readiness times enter policy view."""
        orders={i:{**{k:deepcopy(o[k]) for k in ('quantity','zone','tw_start','tw_end','service',
                 'profile_id','quality_origin_h','initial_quality')},
                 'coords':deepcopy(self._coordinates[i]),'ready':i in self._ready,
                 'owner':deepcopy(self._owner[i])} for i,o in self._orders.items() if i in self._released}
        fleet={vid:{k:sorted(value) if isinstance(value,set) else deepcopy(value)
                    for k,value in v.items()} for vid,v in self._fleet.items()}
        return {'time_h':self.time,'orders':orders,'fleet':fleet,'depot':deepcopy(self._coordinates[0]),
                'engineering_only':True}

    def load(self,vehicle,order_ids):
        require(type(vehicle) is int and vehicle in self._fleet,'vehicle identity')
        v=self._fleet[vehicle];ids=list(order_ids)
        require(ids and len(ids)==len(set(ids)) and all(type(i) is int for i in ids),'unique cargo')
        require(v['mode']=='idle' and v['node']==0 and not v['cargo'],'reload only after depot return')
        require(sum(f['mode']=='loading' for f in self._fleet.values())<self.contract.loading_bays,'loading bay occupied')
        require(all(i in self._released and i in self._ready for i in ids),'release and ready gating')
        require(all(self._owner[i]=='warehouse' for i in ids),'exclusive reservation')
        require(sum(self._orders[i]['quantity'] for i in ids)<=self.contract.capacity+1e-9,'total capacity')
        require(all(sum(self._orders[i]['quantity'] for i in ids if self._orders[i]['zone']==z)
                    <=self.contract.compartment_capacity[z]+1e-9 for z in range(3)),'compartment capacity')
        require(self.time+self.contract.loading_h<=self.contract.horizon_h,'load exceeds horizon')
        v.update(mode='loading',load_ids=set(ids),until=self.time+self.contract.loading_h,trip=v['trip']+1)
        for oid in ids:self._owner[oid]=('loading',vehicle)
        self._event('load_start',vehicle,orders=ids,trip=v['trip'])
        self._schedule(v['until'],2,'load_finish',vehicle)

    def plan(self,vehicle,suffix):
        require(type(vehicle) is int and vehicle in self._fleet,'vehicle identity')
        v=self._fleet[vehicle];route=list(suffix)
        require(v['mode']!='loading','cannot plan unfinished load')
        require(route and route[-1]==0 and 0 not in route[:-1],'depot-ending suffix')
        require(all(type(i) is int for i in route) and len(route[:-1])==len(set(route[:-1])),'unique route')
        require(set(route[:-1])==v['cargo'],'cannot insert warehouse cargo into in-transit vehicle')
        if v['mode'] in ('travel','waiting','service'):
            require(route[0]==v['target'],'committed leg/current service immutable')
        if v['mode']=='travel' and v['target']==0:require(route==[0],'committed return immutable')
        v['suffix']=route;self._event('plan',vehicle,suffix=route)

    def depart(self,vehicle):
        require(type(vehicle) is int and vehicle in self._fleet,'vehicle identity')
        v=self._fleet[vehicle]
        require(v['mode']=='idle' and bool(v['suffix']),'depart only idle planned vehicle')
        target=v['suffix'][0]
        require(target==0 or self._owner.get(target)==('vehicle',vehicle),'loaded delivery only')
        require(target!=v['node'],'same-location departure not a service substitute')
        a,b=self._coordinates[v['node']],self._coordinates[target]
        distance=math.hypot(a[0]-b[0],a[1]-b[1]);arrival=self.time+distance/self.contract.speed_kmph
        require(arrival<=self.contract.horizon_h,'arrival exceeds horizon')
        v.update(mode='travel',target=target,until=arrival);self.distance_km+=distance
        self._event('depart',vehicle,target=target,origin=v['node'],arrival_h=arrival,distance_km=distance)
        self._schedule(arrival,2,'arrive',vehicle,target)

    def set_product_temperature(self,vehicle,temperatures):
        """Explicit external simulation scenario operation, never policy action."""
        require(type(vehicle) is int and vehicle in self._fleet,'vehicle identity')
        require(len(temperatures)==3 and all(finite(x) for x in temperatures),'product temperatures')
        self._fleet[vehicle]['product_temperature_c']=tuple(temperatures)
        self._event('product_temperature',vehicle,values=list(temperatures),basis='exogenous_simulation')

    def apply_action(self,action):
        """Policy surface: cannot set temperatures or read the private future stream."""
        require(type(action) is dict and action.get('kind') in ('load','plan','depart'),'legal policy action')
        if action['kind']=='load':self.load(action['vehicle'],action['orders'])
        elif action['kind']=='plan':self.plan(action['vehicle'],action['suffix'])
        else:self.depart(action['vehicle'])
        return self.observation()

    def result(self):
        operating=validate_trace(self._orders,[{k:v for k,v in e.items() if k not in ('sequence','previous_sha256','sha256')}
             for e in self.events if e['kind'] not in ('release','ready','product_temperature')],
             vehicles=self.contract.vehicles,capacity=self.contract.capacity,horizon=self.contract.horizon_h,
             loading_duration=self.contract.loading_h,speed=self.contract.speed_kmph,
             coordinates=self._coordinates,compartment_capacity=self.contract.compartment_capacity)
        quality={}
        for oid,o in self._orders.items():
            load,vid=self._load_records[oid];end,dvid=self._deliver_records[oid]
            require(vid==dvid,'no intervehicle transfer')
            warehouse=[s for s in self._quality_segments[oid] if s['stage']=='warehouse_including_loading']
            cargo=[s for s in self._quality_segments[oid] if s['stage']=='cargo']
            before=evaluate_product(o['profile_id'],o['initial_quality'],warehouse,
                                     allow_extrapolation=self.contract.allow_extrapolation)
            transported=evaluate_cargo(o['profile_id'],before['final_value'],cargo,direction='delivery',
                    begin_h=load,end_h=end,allow_extrapolation=self.contract.allow_extrapolation)
            full=evaluate_product(o['profile_id'],o['initial_quality'],self._quality_segments[oid],
                                  allow_extrapolation=self.contract.allow_extrapolation)
            require(abs(full['final_value']-transported['final_value'])<1e-10,'continuous warehouse quality')
            quality[str(oid)]={'warehouse':before,'cargo':transported,'full':full}
        energy={str(v):electrical_energy_kwh(s) for v,s in self._power_segments.items()}
        return {'engineering_only':True,'operating':operating,'quality':quality,'energy':energy,
                'distance_km':self.distance_km,'events':deepcopy(self.events),
                'event_log_sha256':self.events[-1]['sha256'] if self.events else None,
                'contract':asdict(self.contract),'profile_sha256':load_profiles()[1],
                'cargo_quality_certified_under_declared_simulation':True,
                'air_product_thermal_dynamics_implemented':False,'warehouse_energy_included':False,
                'trajectory_objective_registered':False,'stable_maskco_gain_established':False,
                'formal_online_ready':False}


def independent_replay(orders,coordinates,contract,result):
    """Recompute from events, never use simulator internal state/quality segments."""
    contract.validate();validate_orders(orders,contract,coordinates)
    require(sha(result['contract'])==sha(asdict(contract)),'actual contract replay identity')
    events=result['events'];last=0.;previous=None
    for seq,e in enumerate(events):
        require(e['sequence']==seq and e['previous_sha256']==previous,'event chain order')
        core={k:v for k,v in e.items() if k!='sha256'}
        require(e['sha256']==sha(core),'event chain hash')
        require(last<=e['time']<=contract.horizon_h,'event time');last=e['time'];previous=e['sha256']
    require(previous==result['event_log_sha256'],'terminal event chain')
    vehicle_events=[e for e in events if e['kind'] not in ('release','ready','product_temperature')]
    operating=validate_trace(orders,vehicle_events,vehicles=contract.vehicles,capacity=contract.capacity,
        horizon=contract.horizon_h,loading_duration=contract.loading_h,speed=contract.speed_kmph,
        coordinates=coordinates,compartment_capacity=contract.compartment_capacity)
    require(abs(operating['distance_km']-result['distance_km'])<1e-10,'distance replay')
    # Stronger commitment/bay rules than the historical independent validator.
    in_loading=set();committed={};loaded={};delivered={};timers={}
    temps=[]
    for e in events:
        t,k,v=e['time'],e['kind'],e['vehicle']
        if k=='load_start':
            require(len(in_loading)<contract.loading_bays,'bay replay');in_loading.add(v)
            timers[v]=('load_finish',t+contract.loading_h)
        elif k=='load_finish':
            require(timers.get(v,('missing',None))[0]=='load_finish' and
                    math.isclose(t,timers[v][1],abs_tol=1e-10,rel_tol=0.),'exact load timer replay')
            in_loading.remove(v)
            for oid in e['orders']:require(oid not in loaded,'duplicate load replay');loaded[oid]=(t,v)
        elif k=='depart':
            committed[v]=e['target']
            a,b=coordinates[e['origin']],coordinates[e['target']]
            distance=math.hypot(a[0]-b[0],a[1]-b[1])
            require(math.isclose(e['distance_km'],distance,abs_tol=1e-10,rel_tol=0.),'leg distance replay')
            timers[v]=('arrive',t+distance/contract.speed_kmph)
        elif k=='arrive':
            require(timers.get(v,('missing',None))[0]=='arrive' and
                    math.isclose(t,timers[v][1],abs_tol=1e-10,rel_tol=0.),'exact travel timer replay')
            if e['target']==0:committed.pop(v,None)
            else:timers[v]=('service_start',max(t,orders[e['target']]['tw_start']))
        elif k=='plan' and v in committed:require(e['suffix'][0]==committed[v],'committed stop replay')
        elif k=='service_start':
            require(timers.get(v,('missing',None))[0]=='service_start' and
                    math.isclose(t,timers[v][1],abs_tol=1e-10,rel_tol=0.),'exact waiting timer replay')
            timers[v]=('service_finish',t+orders[e['order']]['service'])
        elif k=='service_finish':
            require(committed.get(v)==e['order'],'service commitment replay')
            require(timers.get(v,('missing',None))[0]=='service_finish' and
                    math.isclose(t,timers[v][1],abs_tol=1e-10,rel_tol=0.),'exact service timer replay')
            delivered[e['order']]=(t,v);committed.pop(v,None)
        elif k=='product_temperature':
            require(e.get('basis')=='exogenous_simulation','temperature provenance replay')
            temps.append((t,v,tuple(e['values'])))
    require(set(loaded)==set(orders)==set(delivered),'complete lifecycle replay')
    quality={};power={v:0. for v in range(contract.vehicles)}
    finish=max([last]+[s['cargo']['end_h'] for s in result['quality'].values()])
    breakpoints=sorted(set([0.,finish]+[e['time'] for e in events]))
    for oid,o in orders.items():
        start,vid=loaded[oid];end,dvid=delivered[oid]
        require(vid==dvid,'vehicle identity replay')
        segments=[]
        points=sorted(set([o['quality_origin_h'],start,end]+[x for x in breakpoints if o['quality_origin_h']<x<end]))
        for a,b in zip(points,points[1:]):
            if a>=end:continue
            if a<start:temp=o['warehouse_product_temperature_c'];stage='warehouse_including_loading'
            else:
                ts=tuple(contract.product_temperature_c)
                for when,v,values in temps:
                    if when<=a and v==vid:ts=values
                temp=ts[o['zone']];stage='cargo'
            segments.append({'start_h':a,'end_h':b,'product_temperature_c':temp,'stage':stage})
        full=evaluate_product(o['profile_id'],o['initial_quality'],segments,
                              allow_extrapolation=contract.allow_extrapolation)
        observed=result['quality'][str(oid)]['full']
        require(abs(full['final_value']-observed['final_value'])<1e-10 and
                full['extrapolated']==observed['extrapolated'],'product quality replay')
        quality[str(oid)]={'final_value':full['final_value'],'extrapolated':full['extrapolated']}
    for a,b in zip(breakpoints,breakpoints[1:]):
        for vid in power:
            zones={orders[oid]['zone'] for oid,(load,v) in loaded.items()
                   if v==vid and load<=a<delivered[oid][0]}
            power[vid]+=(b-a)*sum(contract.electrical_input_kw[z] for z in zones)
    for vid,total in power.items():
        require(abs(total-result['energy'][str(vid)]['electrical_energy_kwh'])<1e-10,'input energy replay')
    require(result['profile_sha256']==load_profiles()[1],'actual commodity profile identity')
    return {'independent_event_replay_pass':True,'quality':quality,'electrical_energy_kwh':power,
            'distance_km':operating['distance_km'],'stable_maskco_gain_established':False,
            'business_calibrated':False,'food_safety_certified':False}
