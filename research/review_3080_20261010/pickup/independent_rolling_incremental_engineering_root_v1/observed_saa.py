"""New candidate observations only; canonical simulator is not edited.

incr_eval changes summary/ranking calculations: equivalence is a gate to test,
not an assumed property. All action/terminal observations count in the request.
"""
import copy,json,time
from pathlib import Path
from energy_complete_saa import EnergyCompleteSAA
HERE=Path(__file__).resolve().parent
MAX_BYTES=2*1024*1024
def save_payload(name,value):
    from audit_io import encode
    data=json.dumps(encode(value),separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(data)>MAX_BYTES:raise OverflowError('bounded evidence payload')
    final=HERE/name;part=HERE/(name+'.part')
    if final.exists():raise FileExistsError('no output overwrite')
    with part.open('xb') as f:f.write(data)
    part.rename(final)
def plans(states):
    return {str(v):dict(route=[int(o) for o in s['route']],cur=int(s['cur']),cur_time=float(s['cur_time']).hex(),load=float(s['load']).hex(),frozen=int(s.get('frozen',0))) for v,s in states.items()}
class ObservedSAA(EnergyCompleteSAA):
    def _sim_scenario(self,*args,**kwargs):
        self.events=[];self.terminal=[];self.started=time.perf_counter()
        value=super()._sim_scenario(*args,**kwargs);self.completed=time.perf_counter();return value
    def _insert_u(self,space,env,o,states,deadline):
        before=plans(states);ok=super()._insert_u(space,env,o,states,deadline)
        self.events.append(dict(kind='insert',order=int(o),before=before,accepted=bool(ok),after=plans(states)))
        return ok
    def _local_search_u(self,space,env,states,deadline,max_passes=2):
        before=plans(states);result=super()._local_search_u(space,env,states,deadline,max_passes)
        self.events.append(dict(kind='local_search',before=before,after=plans(states),max_passes=max_passes));return result
    def _complete_u(self,space,env,inst_idx,clock,states,wait_until):
        before=plans(states);out=super()._complete_u(space,env,inst_idx,clock,states,wait_until)
        self.terminal.append(dict(states=before,wait_until=wait_until,energy_hex=None if out[0] is None else float(out[0]).hex(),distance_hex=None if out[1] is None else float(out[1]).hex()));return out
    def publish(self,decision,index,value):
        save_payload('ROLLOUT_%s_%02d.json'%(decision,index),dict(decision=decision,index=index,corrected_utility_hex=float(value).hex(),events=self.events,terminal=self.terminal,started=self.started,completed=self.completed,elapsed=self.completed-self.started,incr_eval=self.incr_eval))
