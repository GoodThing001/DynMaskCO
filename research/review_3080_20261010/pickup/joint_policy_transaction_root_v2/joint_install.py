"""Guarded single-pointer admission/plan publication on the actual C0 interface.

The input is an already observed public packet. This component does not qualify
live public extraction, formal learned controls, or whole-day hard deadlines.
"""
from dataclasses import dataclass
import hashlib,json,math,time

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class PolicyState:
    budget: float
    mode: str
    accepted: set
    rejected: set
    plan: dict
    status_inst: int

def compile_state(state):
    if state.get('version')!='c0-reserve-v1':raise ValueError('Actual C0 state version required')
    if type(state['budget']) not in (int,float) or not math.isfinite(state['budget']):raise ValueError('Finite actual budget')
    if type(state['mode']) is not str or type(state['status_inst']) is not int:raise ValueError('Mode/status identity')
    accepted,rejected=set(state['accepted']),set(state['rejected'])
    if accepted&rejected or any(type(o) is not int for o in accepted|rejected):raise ValueError('Valid disjoint admission IDs')
    plan={int(v):list(r) for v,r in state['plan'].items()}
    if any(type(o) is not int for r in plan.values() for o in r):raise ValueError('Integer actual routes')
    return PolicyState(float(state['budget']),state['mode'],accepted,rejected,plan,state['status_inst'])

def state_dict(state):
    return {'version':'c0-reserve-v1','budget':state.budget,'mode':state.mode,
        'accepted':sorted(state.accepted),'rejected':sorted(state.rejected),
        'plan':{str(k):list(v) for k,v in state.plan.items()},'status_inst':state.status_inst}

def transactional_class(base):
    class TransactionalC0(base):
        def __init__(self,*args,**kwargs):
            self._policy=PolicyState(0.,'myopic',set(),set(),{},-1)
            super().__init__(*args,**kwargs)
        def _replace(self,field,value):
            from dataclasses import replace
            self._policy=replace(self._policy,**{field:value})
        budget=property(lambda s:s._policy.budget,lambda s,x:s._replace('budget',x))
        mode=property(lambda s:s._policy.mode,lambda s,x:s._replace('mode',x))
        _accepted=property(lambda s:s._policy.accepted,lambda s,x:s._replace('accepted',x))
        _rejected=property(lambda s:s._policy.rejected,lambda s,x:s._replace('rejected',x))
        _plan=property(lambda s:s._policy.plan,lambda s,x:s._replace('plan',x))
        _status_inst=property(lambda s:s._policy.status_inst,lambda s,x:s._replace('status_inst',x))
        def export_state(self):return state_dict(self._policy)
        def restore_state(self,state):self._policy=compile_state(state)
        def _reset_if_new(self,i):
            if i!=self._status_inst:self._policy=PolicyState(self.budget,self.mode,set(),set(),{},int(i))
    return TransactionalC0

class JointPolicyOwner:
    def __init__(self,replanner,fleet,clock,event):
        self.rp=replanner;self.fleet=fleet;self.clock=clock;self.event=event;self.generation=0;self.active=None
    def signature(self):
        from atomic_install import AtomicPlanOwner
        return AtomicPlanOwner(self.rp,self.fleet,self.clock,self.event).signature()
    def begin(self,request_id,order,deadline,packet_digest):
        if type(order) is not int or type(deadline) not in (int,float) or not math.isfinite(deadline):raise ValueError('Exact request order/deadline')
        if self.active is not None:raise RuntimeError('Concurrent admission transaction')
        old=self.rp._policy
        if order in old.accepted:raise ValueError('Existing commitment cannot be reopened')
        token={'request_id':request_id,'order':order,'deadline':deadline,'packet_digest':packet_digest,'old':old,
            'signature':self.signature(),'generation':self.generation}
        self.active=token;return token
    def cancel(self,token):
        if self.active is token:self.active=None
    def commit(self,token,response):
        if self.active is not token or token['generation']!=self.generation:return False,'stale'
        old=self.rp._policy
        try:
            if time.perf_counter()>=token['deadline']:return False,'expired'
            if old is not token['old'] or self.signature()!=token['signature']:return False,'changed_live_state'
            if response.get('request_id')!=token['request_id'] or response.get('packet_digest')!=token['packet_digest'] or response.get('deadline')!=token['deadline']:
                return False,'request_identity'
            if response.get('certified') is not True or response.get('source_qualified') is not True:return False,'unqualified'
            prepared=response.get('compiled_state')
            if type(prepared) is not PolicyState or response.get('compiled_state_sha256')!=digest(state_dict(prepared)):return False,'prepared_identity'
            order=token['order']
            if prepared.budget!=old.budget or prepared.mode!=old.mode or prepared.status_inst!=old.status_inst:return False,'fixed_state_changed'
            if prepared.accepted-old.accepted not in (set(),{order}) or old.accepted-prepared.accepted:return False,'prior_acceptance_changed'
            if prepared.rejected-{order}!=old.rejected-{order}:return False,'other_rejections_changed'
            if prepared.accepted&prepared.rejected or order not in prepared.accepted|prepared.rejected:return False,'undecided_or_overlap'
            if time.perf_counter()>=token['deadline']:return False,'precommit_expired'
            # Actual C0 getters now observe the entire prepared admission+plan state.
            self.rp._policy=prepared
            if time.perf_counter()>=token['deadline']:
                self.rp._policy=old;return False,'late_rolled_back'
            self.generation+=1;return True,'published'
        finally:
            if self.active is token:self.active=None
