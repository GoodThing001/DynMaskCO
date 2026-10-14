"""New timed integration engine; completed evidence names are atomic.

This is a new timed integration component, not a trained or dynamic policy.
No immutable first-step scores are reused. The frozen old chain stays unchanged.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,gc,hashlib,io,json,sys,time
from pathlib import Path

class InitialEngine:
    def __init__(self, base, registration, arm):
        import psutil
        from action_head_runtime_v1 import sha,digest,closure,delta
        from timed_evidence_io_v1 import save_npz
        self.save_npz=save_npz
        self.base=Path(base);self.reg=registration;self.arm=arm
        self.sha,self.digest,self.closure,self.delta=sha,digest,closure,delta
        psutil.Process().cpu_affinity([15])
        if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Unchanged8GiB barrier')
        old=Path(registration['old_remote_root']);up=old/'upstream';ext=old/'extension'
        for p in (up,old/'actual_checkpoint_engineering_v1',self.base,ext/'scripts/delivery',ext/'scripts/coldchain'):
            sys.path.insert(0,str(p))
        import numpy as np,jax,jax.numpy as jnp
        from flax import nnx
        from inspect_checkpoint import load_trusted
        from features_objective import make_features,extend_projection,UNITS
        from modules.functional import coord_normalize
        from delivery_contract import DeliveryContract
        from delivery_evaluator import evaluate
        from delivery_actual_action_head_v1 import construct_head,anchor_weights
        from delivery_actual_score_chain_v1 import enumerate_step,binding_identity,consume_step
        self.np,self.jax,self.jnp,self.nnx=np,jax,jnp,nnx
        self.make_features,self.units,self.normalize=make_features,UNITS,coord_normalize
        self.evaluate,self.anchor,self.enumerate,self.binding,self.consume=evaluate,anchor_weights,enumerate_step,binding_identity,consume_step
        self.contract=DeliveryContract()
        if self.contract.identity()!=registration['contract_hash']:raise ValueError('Original contract identity')
        checkpoint=Path(registration['checkpoint'])
        if sha(checkpoint)!=registration['checkpoint_sha256']:raise ValueError('Original checkpoint identity')
        saved,_=load_trusted(checkpoint,up)
        cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10;cfg.rngs=888
        if arm=='pretrained_initial':params=copy.deepcopy(saved['params'])
        elif arm=='random888_initial':
            cfg3=copy.deepcopy(cfg);cfg3.encoder_input_dim=3;fresh=cfg3.construct_model()
            params=copy.deepcopy(nnx.state(fresh,nnx.Param));del fresh
        else:raise ValueError('Fixed two initial engineering arms only')
        params['init_proj']['kernel'].value=jnp.asarray(extend_projection(params['init_proj']['kernel'].value))
        self.model=cfg.construct_model();nnx.update(self.model,params);self.head=construct_head()
        del params,saved;gc.collect()
        self.parameter_identity=self.inventory(self.model);self.head_identity=self.inventory(self.head)
        self.source_identity=digest(dict(arm=arm,backbone=self.parameter_identity,head=self.head_identity,
            registration_sha256=sha(self.base/'EXECUTION_REGISTRATION.json'),optimizer_steps=0))
        self.contract.physics();jax.devices();list(psutil.process_iter(['pid']))
        b=io.BytesIO();np.savez(b,x=np.zeros(1,np.float32));b.seek(0);np.load(b).close()
        # Library/serialization setup only. No encode, decode, head or C0 evaluation warmup.
        self.extras=[self.base/n for n in registration['package_files']]+[self.base/'EXECUTION_REGISTRATION.json',checkpoint]
        self.trees=[up,ext/'scripts/delivery',ext/'scripts/coldchain']
        self.start=closure(self.extras,self.trees)

    def inventory(self, model):
        np=self.np
        return {str(k):dict(shape=list(np.asarray(v.value).shape),dtype=str(np.asarray(v.value).dtype),
            sha256=hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()) for k,v in self.nnx.state(model,self.nnx.Param).flat_state()}

    def request(self, raw, directory, stage, cancelled, trace):
        """All spans and mandatory source/parameter gates stay inside request."""
        np,jax,jnp=self.np,self.jax,self.jnp
        from timed_evidence_io_v1 import save
        directory=Path(directory);directory.mkdir(exist_ok=False)
        m=trace.measure
        stage(1,'input_decode')
        def decode_public():
            row=json.loads(raw.decode('utf-8'))
            if self.digest(row)!=self.reg['input_digest']:raise ValueError('Fixed public input identity')
            if row['key']!=[12,3] or len(row['remaining'])!=25:raise ValueError('Fixed first engineering input')
            return row
        row=m(1,0,decode_public)
        def project_public():
            coordinate=np.asarray(self.normalize(jnp.asarray(row['data']['coords'],jnp.float32)[None]))[0]
            return self.make_features(row['data'],self.contract,coordinate,self.units,self.contract.identity())
        features=m(2,0,project_public)
        m(8,0,self.save_npz,directory/'FEATURES.npz',features=features)
        stage(2,'public_features_complete')
        incumbent=m(3,0,self.evaluate,row['data'],row['initial'],self.contract,wave_h=0.)
        m(10,0,save,directory,'INCUMBENT_C0.json',incumbent)
        if not incumbent['hard']:raise ValueError('Incumbent complete C0 barrier')
        def encode_synchronized():
            encoded=self.model.encode(jnp.asarray(features,jnp.float32),attn_options={})
            jax.block_until_ready(encoded)
            return encoded
        encoded=m(4,0,encode_synchronized)
        m(8,0,self.save_npz,directory/'ENCODED.npz',encoded=np.asarray(encoded),features=features)
        stage(3,'encoder_complete')
        routes=copy.deepcopy(row['before']);remaining=row['remaining'][:];steps=[]
        for step in range(25):
            if cancelled():raise TimeoutError('Cancelled before next action')
            actions,groups=m(5,step,self.enumerate,row['data'],routes,remaining,row['partial'],self.contract)
            binding=m(6,step,self.anchor,actions,routes,remaining,len(row['data']['demands']))
            def score_synchronized():
                # Additional diagnostic barriers change synchronization scheduling,
                # not the original arithmetic expressions or input/action ordering.
                def decode_sync():
                    latent=self.model.decode(encoded,jnp.asarray(binding['timestep']),jnp.asarray(binding['context']),attn_options={},target='embed')
                    jax.block_until_ready(latent)
                    return latent
                latent=m(20,step,decode_sync)
                if self.model.apply_final_norm:
                    def norm_sync():
                        normalized=self.model.final_norm(latent)
                        jax.block_until_ready(normalized)
                        return normalized
                    latent=m(21,step,norm_sync)
                def head_sync():
                    scores=self.head(latent,binding)
                    jax.block_until_ready(scores)
                    return scores
                scores=m(22,step,head_sync)
                return m(23,step,lambda:(np.asarray(scores),np.asarray(latent)))
            scores,latent=m(7,step,score_synchronized)
            m(8,step,self.save_npz,directory/('%02d.npz'%step),latent=latent,scores=scores,features=features,**binding)
            if step==0:stage(4,'first_actual_score_complete')
            def bind_consume():
                packet=dict(scores=scores,source='actual_initial_model_untrained',learned_output_qualified=False,
                    binding_digest=self.binding(row['data'],row['key'],step,routes,remaining,actions),source_identity=self.source_identity)
                return self.consume(row['data'],row['key'],step,routes,remaining,actions,groups,packet,self.source_identity)
            transition=m(9,step,bind_consume)
            m(10,step,save,directory,'%02d.json'%step,transition)
            transition_sha=m(11,step,self.sha,directory/('%02d.json'%step))
            steps.append(dict(file='%02d.json'%step,sha256=transition_sha))
            routes,remaining=transition['after'],transition['next_remaining']
            if transition['chosen'] is None:break
        m(12,0,save,directory,'CHAIN_STOP.json',dict(routes=routes,remaining=remaining,steps=steps,
            status='complete' if not remaining else 'no_feasible_action'))
        stage(5,'action_chain_stopped')
        def certify_select():
            candidate=self.evaluate(row['data'],routes,self.contract,wave_h=0.) if not remaining else None
            choose=bool(candidate and candidate['hard'] and candidate['cost']<incumbent['cost'])
            return candidate,choose
        candidate,choose=m(13,0,certify_select)
        m(10,25,save,directory,'C0_SELECTION.json',dict(incumbent=incumbent,candidate=candidate,
            candidate_selected=choose,routes=routes,remaining=remaining))
        stage(6,'complete_C0_and_selection_complete')
        end=m(14,0,self.closure,self.extras,self.trees)
        # Both inventories are evaluated even if the first differs.
        def parameter_gate():
            model_identity=self.inventory(self.model);head_identity=self.inventory(self.head)
            return model_identity==self.parameter_identity and head_identity==self.head_identity
        unchanged=m(15,0,parameter_gate)
        def source_report_gate():
            changes=self.delta(self.start,end)
            qualified=unchanged and not changes['changed'] and not changes['new_files']
            save(directory,'SOURCE_END.json',dict(**end,**changes))
            report=dict(arm=self.arm,key=row['key'],steps=steps,status='complete' if not remaining else 'no_feasible_action',
                routes=routes,remaining=remaining,incumbent=incumbent,candidate=candidate,candidate_selected=choose,
                selected_routes=routes if choose and qualified else row['initial'],parameters_unchanged=unchanged,
                source_qualified=qualified,source_identity=self.source_identity,optimizer_steps=0,learned_output_qualified=False)
            save(directory,'REPORT_BEFORE_PUBLICATION.json',report)
            if not qualified:raise ValueError('Actual loaded source/parameter qualification barrier')
            return report
        return m(16,0,source_report_gate)
