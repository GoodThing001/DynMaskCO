"""Exact original ordered distance summation cache, one bound immutable space."""
from collections import OrderedDict
import hashlib
import numpy as np
from energy_complete_saa import EnergyCompleteSAA
class ExactDistanceCache(EnergyCompleteSAA):
    CACHE_CAPACITY=256
    def bind_distance_space(self,space):
        if self.incr_eval:raise ValueError('Original exact-ranking mode required')
        if getattr(self,'_distance_open',False):raise ValueError('Cannot overwrite live cache scope')
        matrix=np.array(space.D,copy=True,order='C');matrix.flags.writeable=False
        space.D=matrix;self._distance_space=space;self._distance_matrix=matrix
        self._distance_sha=hashlib.sha256(matrix.tobytes()).hexdigest()
        self._distance_cache=OrderedDict();self._distance_open=True
        self.distance_calls=0;self.distance_hits=0;self.distance_misses=0;self.distance_evictions=0
    def verify_distance_space(self,space):
        if not getattr(self,'_distance_open',False):raise ValueError('Expired cache scope')
        if space is not self._distance_space or space.D is not self._distance_matrix:raise ValueError('Wrong space or matrix object')
        if space.D.flags.writeable or hashlib.sha256(space.D.tobytes()).hexdigest()!=self._distance_sha:raise ValueError('Distance matrix changed')
        return True
    def close_distance_space(self,space):
        self.verify_distance_space(space);self._distance_open=False;self._distance_cache.clear()
    def _route_dist_u(self,space,cur,route):
        if not getattr(self,'_distance_open',False):raise ValueError('Expired cache scope')
        if space is not self._distance_space or space.D is not self._distance_matrix or space.D.flags.writeable:raise ValueError('Wrong or mutable distance space')
        key=(int(cur),tuple(int(x) for x in route));self.distance_calls+=1
        if key in self._distance_cache:
            self.distance_hits+=1;value=self._distance_cache.pop(key);self._distance_cache[key]=value
        else:
            self.distance_misses+=1;value=super()._route_dist_u(space,cur,route)
            if len(self._distance_cache)>=self.CACHE_CAPACITY:self._distance_cache.popitem(last=False);self.distance_evictions+=1
            self._distance_cache[key]=value
        return value
