"""Encode static customer features once and decode each current masked adjacency."""
import time
import numpy as np

class StaticEncoderHeat:
    def __init__(self,policy):
        self.policy=policy
        self.inputs=None
        self.cache={}
        self.forward_calls=0
        self.forward_seconds=0.

    def heatmap(self,task,routes,removed,variant,step):
        from base_inputs import BaseInferenceInputs
        from encoded_mask import split_batch,log_heat
        start=time.perf_counter()
        if self.inputs is None:self.inputs=BaseInferenceInputs(task)
        batch=self.inputs.build(routes,removed)
        if not self.cache:
            batch["features"][0,1:,10]=.60
        result=log_heat(split_batch(self.policy,batch,"instance",self.cache))
        self.forward_calls+=1
        self.forward_seconds+=time.perf_counter()-start
        return result
