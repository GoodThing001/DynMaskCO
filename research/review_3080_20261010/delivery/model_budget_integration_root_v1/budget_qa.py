"""Fixed observed snapshots, real 10D models, shared hard10s parent deadline."""
import argparse,copy,hashlib,json,os,pickle,queue,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def bootstrap(root):
    for sub in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(root/'scripts'/sub))
    sys.path.insert(0,str(HERE/'features_objective_qa_v1'))
    sys.path.insert(0,str(HERE/'actual_checkpoint_engineering_v1'))
class Provider:
    def __init__(self,a,reg,mode):
        import numpy as np
        from inspect_checkpoint import load_trusted
        saved,original_sha=load_trusted(a.checkpoint,a.upstream)
        from flax import nnx
        import jax.numpy as jnp
        from features_objective import extend_projection
        cfg=copy.deepcopy(saved['model_config']);cfg.dtype='float32';cfg.encoder_input_dim=10
        if mode=='random':cfg.rngs=42
        model=cfg.construct_model();graph=nnx.graphdef(model)
        if mode in ('reconstruction_control','paired_objective'):
            ref=reg['engineering_weights'][mode]
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Actual engineering saved weight changed')
            with Path(ref['path']).open('rb') as f:actual=pickle.load(f)
            if actual['fixed_step']!=2 or actual['research_model_eligible'] is not False:raise ValueError('Expected engineering fixed2 weights')
            self.model=nnx.merge(graph,actual['params'])
        elif mode=='random':self.model=model
        else:
            params=copy.deepcopy(saved['params'])
            params['init_proj']['kernel'].value=jnp.asarray(extend_projection(params['init_proj']['kernel'].value))
            self.model=nnx.merge(graph,params)
        self.encode=nnx.jit(lambda m,x:m.encode(x));self.decode=nnx.jit(lambda m,h,t,c:m.decode(h,t,c,target='logit'))
        self.identity={'original_checkpoint_sha256':original_sha,'mode':mode,'actual_dtype':'float32','encoder_input_dim':10,
                       'complete_encoder_decoder':True,'research_model':False,'random_seed':42 if mode=='random' else None}
    def features(self,data,capacity):
        import numpy as np,jax.numpy as jnp
        from modules.functional import coord_normalize
        from features_objective import make_features,UNITS
        from delivery_contract import DeliveryContract
        c=DeliveryContract();x=np.asarray(coord_normalize(jnp.asarray(data['coords'],jnp.float32)[None]))[0]
        return self.encode(self.model,jnp.asarray(make_features(data,c,x,UNITS,c.identity()),jnp.float32))
    def logits(self,features,adj):
        import numpy as np,jax.numpy as jnp
        t=float(adj[1:].sum())/100.
        return np.asarray(self.decode(self.model,features,jnp.asarray([t],jnp.float32),jnp.asarray(adj[None],jnp.float32)))[0]
def worker(a,reg):
    os.environ['JAX_PLATFORMS']='cpu'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    import numpy as np,psutil
    psutil.Process().cpu_affinity([a.cpu]);bootstrap(a.root)
    from delivery_contract import DeliveryContract
    from delivery_equal_schedule import run_arm
    from delivery_baselines import improve_ortools
    contract=DeliveryContract();provider=None
    if a.arm not in ('nonlearning','gls'):
        provider=Provider(a,reg,'pretrained' if a.arm=='discard' else a.arm)
        # Only fixed synthetic51 nodes; no teacher routes/results used in warmup.
        n=51;warm={'coords':np.c_[np.linspace(0,1,n),np.zeros(n)].tolist(),'demands':[0]+[1]*50,
            'tw_start':[0.]*n,'tw_end':[8.]*n,'service_time':[0.]+[.01]*50,
            'reveal_time':[0.]*n,'ready_time':[0.]*n,'temp_class':[0]+[i%3 for i in range(50)]}
        h=provider.features(warm,50);provider.logits(h,np.zeros((n,n),np.float32))
    elif a.arm=='gls':
        from ortools.constraint_solver import pywrapcp
    print(json.dumps({'kind':'ready','pid':os.getpid(),'mode':a.arm,'contract_hash':contract.identity(),
                      'provider':provider.identity if provider else None}),flush=True)
    for line in sys.stdin:
        req=json.loads(line);deadline=req['deadline'];start=time.perf_counter()
        if req.get('sleep_qa'):time.sleep(1.)
        remaining=deadline-time.perf_counter()
        if remaining<=0:out={'timeout':True,'status':'deadline_before_start'}
        elif a.arm=='gls':out=improve_ortools(req['data'],req['initial_routes'],contract,time_limit_s=remaining)
        else:out=run_arm(req['data'],req['initial_routes'],contract,req['schedule'],provider,
                        time_limit_s=remaining,guide_weight=0. if a.arm in ('discard','nonlearning') else 1.)
        out.update(worker_start=start,worker_end=time.perf_counter(),request_deadline=deadline,arm=a.arm)
        print(json.dumps({'kind':'response','out':out},allow_nan=False),flush=True)
def main(a,reg):
    import psutil
    bootstrap(a.root)
    if sha(a.registration)!=a.expected_registration_sha or sha(Path(__file__))!=reg['program_sha256']:raise ValueError('Frozen source or registration changed')
    if sha(a.teacher_archive)!=reg['teacher_archive_sha256']:raise ValueError('Teacher input bytes changed')
    for family,ref in reg['source_sha256'].items():
        base,rel=family.split('/',1);p=(a.upstream if base=='upstream' else a.root)/rel
        if sha(p)!=ref:raise ValueError('Frozen source changed '+family)
    rows=[json.loads(s) for s in a.teacher_archive.read_text().splitlines()]
    fixtures=[next(r for r in rows if r['instance_id']==i and r['mask_ordinal']==0) for i in [900000,900001]]
    from delivery_equal_schedule import make_schedule
    a.out.mkdir(exist_ok=False);records=[]
    for arm in reg['arms']:
        argv=[sys.executable,'-B',str(Path(__file__)),'--worker','--arm',arm,'--root',str(a.root),
              '--upstream',str(a.upstream),'--checkpoint',str(a.checkpoint),'--registration',str(a.registration),'--cpu',str(a.cpu)]
        q=queue.Queue();err=[];p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
        def read(stream,q):
            for line in stream:q.put(json.loads(line))
            q.put({'kind':'closed'})
        def errors(stream,arr):arr.extend(stream.readlines())
        threading.Thread(target=read,args=(p.stdout,q),daemon=True).start();threading.Thread(target=errors,args=(p.stderr,err),daemon=True).start()
        try:
            ready=q.get(timeout=120)
            if ready.get('kind')!='ready':raise RuntimeError('Worker failed before ready '+str(err))
            for fixture in fixtures:
                # Start before schedule, data serialization, transmission or I/O.
                t0=time.perf_counter();deadline=t0+10.
                request={'data':fixture['data'],'initial_routes':fixture['incumbent_routes'],
                         'schedule':make_schedule(51,8,202610110072,fixture['instance_id'],fraction=.5),'deadline':deadline}
                p.stdin.write(json.dumps(request)+'\n');p.stdin.flush()
                try:response=q.get(timeout=max(0.,deadline-time.perf_counter()))
                except queue.Empty:
                    p.kill();p.wait(timeout=5);response={'kind':'timeout'}
                submitted=time.perf_counter();late=submitted>deadline or response.get('kind')!='response'
                record={'arm':arm,'instance_id':fixture['instance_id'],'started':t0,'deadline':deadline,'submitted':submitted,
                        'wall_s':submitted-t0,'timeout':late,'own_incumbent_fallback':late,'worker_pid':p.pid,'ready':ready,
                        'actual_result':response.get('out'),'fallback_routes':fixture['incumbent_routes'] if late else None}
                records.append(record);write(a.out/(arm+'_'+str(fixture['instance_id'])+'.json'),record)
                if p.poll() is not None:break
        finally:
            if p.poll() is None:p.kill();p.wait(timeout=5)
            write(a.out/(arm+'_worker_end.json'),{'pid':p.pid,'reaped':p.poll() is not None,'stderr':err})
    end={family:sha((a.upstream if family.split('/',1)[0]=='upstream' else a.root)/family.split('/',1)[1]) for family in reg['source_sha256']}
    write(a.out/'RESULT.json',{'registration_sha256':sha(a.registration),'records':records,'source_end':end,
        'source_unchanged':end==reg['source_sha256'],'engineering_only':True,'snapshot_only':True,
        'all_fixed_attempts_complete_within_total10s':len(records)==len(reg['arms'])*2 and all(not r['timeout'] for r in records),
        'formal_training_eligible':False,'stable_online_gain':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--arm')
    for key in ('root','upstream','checkpoint','registration','teacher-archive','out'):ap.add_argument('--'+key,type=Path)
    ap.add_argument('--cpu',type=int,default=15);ap.add_argument('--expected-registration-sha');args=ap.parse_args()
    registration=json.loads(args.registration.read_text())
    if args.worker:worker(args,registration)
    else:main(args,registration)
