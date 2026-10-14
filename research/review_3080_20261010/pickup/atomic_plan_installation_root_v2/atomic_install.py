"""Candidate-only preparation and guarded live plan pointer publication.

This does not cover public export, admission, full-day policy or all five arms.
Full C0 certification belongs in the interruptible preparation worker. The
parent receives a prepared candidate and only publishes the plan pointer.
"""
import hashlib,json,math,time

def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class AtomicPlanOwner:
    def __init__(self,replanner,fleet,clock,event):
        self.rp=replanner;self.fleet=fleet;self.clock=float(clock);self.event=int(event)
        self.generation=0;self.active=None

    def signature(self):
        # Bind mutable admissions and live physical fields, not merely object IDs.
        return (self.clock,self.event,self.rp._status_inst,self.rp.budget,
                tuple(sorted(self.rp._accepted)),tuple(sorted(self.rp._rejected)),
                tuple((k,tuple(v)) for k,v in sorted(self.rp._plan.items())),
                tuple((v.vehicle_id,v.status,v.current_node,v.committed_next,v.committed_arrive,
                       v.committed_finish,v.ready_time,v.return_finish,v.current_load,v.dispatch_time,
                       tuple(v.mutable_suffix),tuple(v.served_route),v.needs_replan,v.replan_reason,
                       v.coldchain_updated_time,v.coldchain_state,v.committed_coldchain_before,v.coldchain_accounting_anchor)
                      for v in self.fleet))

    def begin(self,request_id,deadline,packet_digest):
        if isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not math.isfinite(deadline):
            raise ValueError('Finite actual shared deadline required')
        if self.active is not None:raise RuntimeError('Only one live plan transaction')
        token={'request_id':request_id,'deadline':float(deadline),'packet_digest':packet_digest,
               'generation':self.generation,'old_plan':self.rp._plan,'signature':self.signature()}
        self.active=token;return token

    def cancel(self,token):
        if self.active is token:self.active=None

    def commit(self,token,prepared):
        started=time.perf_counter();old=self.rp._plan
        if self.active is not token or token['generation']!=self.generation:return False,'stale_token'
        try:
            if time.perf_counter()>=token['deadline']:return False,'expired'
            if self.rp._plan is not token['old_plan'] or self.signature()!=token['signature']:
                return False,'live_state_changed'
            if prepared.get('request_id')!=token['request_id'] or prepared.get('packet_digest')!=token['packet_digest']:
                return False,'request_identity'
            if prepared.get('certified') is not True or prepared.get('source_qualified') is not True:
                return False,'worker_not_qualified'
            if prepared.get('deadline')!=token['deadline']:return False,'deadline_identity'
            plan=prepared.get('compiled_plan')
            if type(plan) is not dict or any(type(v) is not int or type(r) is not list or
                    any(type(o) is not int for o in r) for v,r in plan.items()):
                return False,'compiled_plan_format'
            if prepared.get('compiled_plan_sha256')!=plan_hash(plan):return False,'compiled_plan_identity'
            # All loops/physical work occurred before publication. The actual
            # replanner consumes lists, so this is its real plan object, not a mock.
            if time.perf_counter()>=token['deadline']:return False,'precommit_expired'
            self.rp._plan=plan
            completed=time.perf_counter()
            if completed>=token['deadline']:
                self.rp._plan=old;return False,'late_publication_rolled_back'
            self.generation+=1
            return True,'published'
        finally:
            if self.active is token:self.active=None

