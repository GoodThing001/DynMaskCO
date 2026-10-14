"""Isolated, finite engineering revalidation; no original experiment reruns."""
from pathlib import Path
import os, sys, json, hashlib, shutil, subprocess, datetime, traceback

ROOT=Path('/home/hzeng/project/MASKCO-Main')
RUN=Path(__file__).resolve().parent
PY=sys.executable
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (RUN/name).open('x') as f: json.dump(obj,f,indent=2)

def main():
    protocol=json.loads((RUN/'protocol.json').read_text())
    os.sched_setaffinity(0,{54})
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    workspace=RUN/'workspace'
    workspace.mkdir()
    (workspace/'MASKCO_code').symlink_to(ROOT/'MASKCO_code',target_is_directory=True)
    recovered=RUN/'recovered'
    recovered.mkdir()
    (RUN/'MASKCO_code').symlink_to(ROOT/'MASKCO_code',target_is_directory=True)
    old=ROOT/'B_REPRO_CLEAN'
    manifest=json.loads((RUN/'recovery_manifest.json').read_text())
    start={}
    for row in manifest['files']:
        rel=Path(row['relative'])
        if '..' in rel.parts or rel.is_absolute(): raise ValueError('Unsafe recovery path')
        src=old/rel
        if sha(src)!=row['sha256']: raise ValueError('Recovered source changed: '+str(rel))
        start[str(src)]=row['sha256']
        dst=recovered/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
    inputs={
        'cal':old/'data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz',
        'train':old/'data/m0_scale/dcc_50_r1_edod05_train_teacher.npz',
        'model':old/'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt',
        'profile':old/'results/o0cc/scale_v2/objective_profile.json',
        'cvrp':ROOT/'MASKCO_code/ckpts/cvrp100.ckpt'}
    for p in inputs.values(): start[str(p)]=sha(p)
    code=RUN/'engineering_code'
    ext=workspace/'C-VRP_Cold-chainVehicleRoutingProblem'
    ext.mkdir()
    shutil.move(str(code),str(ext/'scripts'))
    for p in (ext/'scripts').rglob('*.py'):start[str(p)]=sha(p)
    save('SOURCE_START.json',start)
    env=os.environ.copy()
    env['PYTHONPATH']=str(recovered/'scripts')
    common=['--objective-profile',str(inputs['profile'])]
    stages=[('engineering',[PY,str(ext/'scripts/correction_engineering/server_acceptance.py')]),
      ('fixed_state',[PY,str(recovered/'negative_result_paper/release/scripts/run_fixed_state_quality.py'),
       '--data',str(inputs['cal']),'--cvrp-ckpt',str(inputs['cvrp']),
       '--model-ckpt',str(inputs['model']),*common,'--split','cal','--max-instances','1',
       '--n-samples','1','--out',str(RUN/'fixed_state')]),
      ('headroom',[PY,str(recovered/'negative_result_paper/release/scripts/run_headroom_census.py'),
       '--data',str(inputs['train']),*common,'--split','train','--max-instances','1',
       '--out',str(RUN/'headroom')]),
      ('figures',[PY,str(recovered/'negative_result_paper/release/scripts/make_figures.py')])]
    if 'stage_names' in protocol:
        stages=[(n,a) for n,a in stages if n in protocol['stage_names']]
    reports=[]
    for name,args in stages:
        began=datetime.datetime.now().isoformat()
        with (RUN/(name+'.log')).open('xb') as log:
            try:
                p=subprocess.run(args,cwd=recovered,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 timeout=protocol['per_stage_timeout_seconds'])
                rc=p.returncode
            except subprocess.TimeoutExpired: rc=124
        item={'stage':name,'started':began,'ended':datetime.datetime.now().isoformat(),'exit':rc}
        reports.append(item);save(name+'.exit.json',item)
        # Independent engineering failures do not erase the other recovery checks.
    end={p:sha(p) for p in start}
    save('SOURCE_END.json',end)
    save('QA_RESULT.json',{'stages':reports,'source_and_inputs_unchanged':start==end,
       'all_stage_exits_zero':all(x['exit']==0 for x in reports),
       'original_training_hash_match':sha(old/'scripts/models/mpre_trained.py')==protocol['expected_training_sha256'],
       'original_full_reproduction_qualified':False,'online_gain_established':False,
       'qualification':'finite engineering runtime smoke; not original full training/online reproduction'})
    # Shell completion must never hide a failed child-stage qualification.
    if start!=end or any(x['exit']!=0 for x in reports): raise SystemExit(1)

if __name__=='__main__':
    try: main()
    except Exception:
        save('pipeline_failure.json',{'error':traceback.format_exc()})
        raise
