"""Only original LS insertion base-distance evaluation is reused per vehicle."""
from energy_complete_saa import EnergyCompleteSAA
class LazyBaseDistanceSAA(EnergyCompleteSAA):
    def _insert_u(self,space,env,o,states,deadline):
        if self.incr_eval:raise ValueError('Exact original ranking mode required')
        if self.shadow_mode!='ls':return super()._insert_u(space,env,o,states,deadline)
        if space.D.flags.writeable:raise ValueError('Immutable owned distance matrix required')
        best=None
        for vid,s in states.items():
            route=s['route'];have_base=False;base=None
            for pos in range(s.get('frozen',0),len(route)+1):
                trial=route[:pos]+[o]+route[pos:]
                if self._route_feasible_u(space,env,s['cur'],s['cur_time'],s['load'],trial,deadline):
                    # Preserve first trial THEN base call, original dtype and sum.
                    trial_cost=self._route_dist_u(space,s['cur'],trial)
                    if not have_base:
                        base=self._route_dist_u(space,s['cur'],route);have_base=True
                        self.base_computations+=1
                    else:self.base_reuses+=1
                    incr=trial_cost-base
                    self.score_log.append(dict(order=int(o),vehicle=int(vid),position=int(pos),trial_hex=float(trial_cost).hex(),base_hex=float(base).hex(),score_hex=float(incr).hex(),score_type=type(incr).__module__+'.'+type(incr).__qualname__))
                    if best is None or incr<best[0]:best=(incr,vid,pos)
        if best is None:return False
        _,vid,pos=best;r=states[vid]['route'];states[vid]['route']=r[:pos]+[o]+r[pos:]
        return True
