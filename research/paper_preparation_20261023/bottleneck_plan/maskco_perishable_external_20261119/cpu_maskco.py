"""Single-core FP32 evaluation of the frozen symmetric MaskCO backbone."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.special import expit


def array(parameter):
    return np.ascontiguousarray(np.asarray(parameter.value),dtype=np.float32)


def dense(layer):
    return array(layer.kernel),array(layer.bias)


def project(x,parameters):
    w,b=parameters
    return x@w+b


def standardize(x,scale):
    avg=x.mean(axis=1,keepdims=True)
    var=np.maximum((x*x).mean(axis=1,keepdims=True)-avg*avg,0.)
    return ((x-avg)/np.sqrt(var+np.float32(1e-6)))*scale


@dataclass
class Layer:
    norm1:np.ndarray
    norm2:np.ndarray
    qkv:np.ndarray
    output:np.ndarray
    gates:tuple
    down:tuple
    heads:int

    @classmethod
    def load(cls,layer):
        att,ff=layer.attn,layer.ffn
        if att.normalize_qk or not att.qkv_packed:raise ValueError("Requires the fixed packed MaskCO attention")
        fused=(np.concatenate((array(ff.w),array(ff.v)),axis=1),
               np.concatenate((array(ff.w_bias),array(ff.v_bias))))
        return cls(array(layer.ln1.scale),array(layer.ln2.scale),array(att.qkv_proj_params),
                   array(att.out_proj_params),fused,(array(ff.w2),array(ff.w2_bias)),att.num_heads)

    def apply(self,x,bias):
        n,dim=x.shape
        qkv=(standardize(x,self.norm1)@self.qkv).reshape(n,3,self.heads,dim//self.heads)
        q,k,v=qkv.transpose(1,2,0,3)
        scores=(q@k.transpose(0,2,1))*np.float32((dim//self.heads)**-.5/30.)
        np.tanh(scores,out=scores);scores*=30.
        if bias is not None:scores+=bias
        scores-=scores.max(axis=2,keepdims=True)
        np.exp(scores,out=scores);scores/=scores.sum(axis=2,keepdims=True)
        x=x+(scores@v).transpose(1,0,2).reshape(n,dim)@self.output
        a,b=np.split(project(standardize(x,self.norm2),self.gates),2,axis=1)
        return x+project(a*expit(a)*b,self.down)


class CpuMaskCO:
    def __init__(self,model):
        net=model.backbone
        if (model.config["variant"]!="maskco_base" or net.softcap!=30. or net.sm_scale is not None
                or net.final_softcap is not None or not net.apply_final_norm):
            raise ValueError("Unsupported frozen model")
        self.first=dense(net.init_proj);self.depot=array(net.depot_bias)
        self.encoder=[Layer.load(v) for v in net.encoder.layers]
        self.decoder=[Layer.load(v) for v in net.decoder.layers]
        self.mid=dense(net.mid_proj)
        timestep=net.timestep_embedder
        k=timestep.frequency_embed_dim//2
        self.frequency=np.exp(np.float32(-np.log(10000))*np.arange(k,dtype=np.float32)/k)
        self.time_scale=np.float32(timestep.timestep_scale)
        self.time_first=dense(timestep.mlp.layers[0]);self.time_last=dense(timestep.mlp.layers[2])
        self.out_first=dense(net.final_proj.inner.layers[0]);self.out_last=dense(net.final_proj.inner.layers[2])
        self.final_scale=array(net.final_norm.scale);self.logit_scale=np.float32(net.final_logit_scale)

    def logits(self,batch):
        from experiment_scope import require_operation
        require_operation("infer")
        features=np.asarray(batch["features"][0],dtype=np.float32)
        bias=np.asarray(batch["adjacency"][0],dtype=np.float32)[None,:,:]
        x=project(features,self.first);x[0]+=self.depot
        for layer in self.encoder:x=layer.apply(x,None)
        args=np.float32(batch["timestep"][0])*self.time_scale*self.frequency
        t=np.concatenate((np.cos(args),np.sin(args)))
        t=project(t,self.time_first)
        x=project(x,self.mid)+project(t*expit(t),self.time_last)
        for layer in self.decoder:x=layer.apply(x,bias)
        x=project(x,self.out_first);x=project(x*expit(x),self.out_last)
        x=x/np.sqrt((x*x).mean(axis=1,keepdims=True)+np.float32(1e-6))*self.final_scale
        scores=(x@x.T)*self.logit_scale
        np.fill_diagonal(scores,np.float32(.5)*np.finfo(np.float32).min)
        return scores


class CpuHeat:
    def __init__(self,weights):
        self.weights=weights
        self.inputs=None
        self.forward_seconds=0.
        self.forward_calls=0

    def heatmap(self,task,routes,removed,variant,step):
        from base_inputs import BaseInferenceInputs
        begin=time.perf_counter()
        if self.inputs is None:self.inputs=BaseInferenceInputs(task)
        logits=self.weights.logits(self.inputs.build(routes,removed))
        shifted=logits-logits.max(axis=1,keepdims=True)
        result=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
        self.forward_calls+=1
        self.forward_seconds+=time.perf_counter()-begin
        return result


def load_pair(study,access,core,source,locked):
    from unittest.mock import patch
    import profit_neural
    factory=profit_neural.CompiledPredictor
    def both(model):
        pair=factory(model)
        pair.blas=CpuMaskCO(model)
        return pair
    with patch.object(profit_neural,"CompiledPredictor",both):
        return study.load("SLOT",access,core,source,locked,"c205")[0]
