"""Read-only final I3 download and independent actual end identity audit.

Uses tools/_srv.py credential handling. No remote writes, policy execution, model
inference, final seed generation or experimental restarts.
"""
from pathlib import Path
import base64, datetime, hashlib, json, math, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srv import run

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "C-VRP_Cold-chainVehicleRoutingProblem"
OUT = EXT / "results/i3_online_recovery_1009_v1"
REMOTE = "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem"
SNIPPET = r'''
import base64, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
root=Path.cwd()
out=root/'results/i3_online_recovery_1009_v1'
cfg=json.loads((out/'online_config_20260926.json').read_text())
gate=json.loads((out/'online_20260926/gate.json').read_text())
ident=gate['results']['20260926']['_identity']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];mismatches=[]
for line in (out/'SOURCE_MANIFEST.sha256').read_text().splitlines():
    expected,raw=line.split(None,1);p=Path(raw.lstrip('*'))
    if not p.is_absolute():p=root/p
    actual=sha(p) if p.is_file() else None
    row={'path':str(p),'expected':expected,'actual':actual}
    checks.append(row)
    if actual!=expected:mismatches.append(row)
source_actual={}
for rel,expected in ident['source_sha256'].items():
    p=Path(cfg['frozen_extension'])/rel
    actual=sha(p)
    source_actual[rel]=actual
    if actual!=expected:mismatches.append({'gate_source':rel,'expected':expected,'actual':actual})
weights={}
for name,spec in cfg['models'].items():
    weights[name]={'model_sha256':sha(spec['path']),'card_sha256':sha(spec['card'])}
    if weights[name]['model_sha256']!=spec['sha256'] or weights[name]['card_sha256']!=spec['card_sha256']:
        mismatches.append({'weight_or_card':name})
cvrp=root.parent/'MASKCO_code/ckpts/cvrp100.ckpt'
actual_cvrp=sha(cvrp)
if actual_cvrp!=cfg['cvrp_sha256']:mismatches.append({'cvrp_mismatch':actual_cvrp})
# A new independent read-only data regeneration. Does not run policy actions.
scripts=Path(cfg['frozen_extension'])/'scripts'
for folder in ('evaluation','simulation','coldchain'):
    sys.path.insert(0,str(scripts/folder))
os.sched_setaffinity(0,{52})
from run_exp_reserve import generate_dataset
from coldchain_evaluator_a1 import add_v2_initial_quality
from scenario_saa import make_c0_contract_v2
import numpy as np
contract=make_c0_contract_v2()
def hashes(ds):
    h=hashlib.sha256();meta={}
    for k in sorted(ds):
        a=np.asarray(ds[k]);h.update(k.encode('utf-8'));h.update(a.tobytes())
        meta[k]={'shape':list(a.shape),'dtype':str(a.dtype)}
    return {'data_sha256':h.hexdigest(),'data_meta_sha256':hashlib.sha256(
        json.dumps(meta,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'meta':meta}
development=add_v2_initial_quality(generate_dataset(40,200,20260926),contract)
history=generate_dataset(200,200,20260925)
dev_hash=hashes(development);hist_hash=hashes(history)
reconstructed={'data_sha256':dev_hash['data_sha256'],'data_meta_sha256':dev_hash['data_meta_sha256'],
               'contract_sha256':contract.contract_hash}
for k,v in reconstructed.items():
    if v!=cfg['expected_identity'][k] or v!=ident[k]:mismatches.append({'regenerated_identity':k,'actual':v})
for k in ('data_sha256','data_meta_sha256'):
    expected=cfg['history_pool_lineage']['history_'+k]
    if hist_hash[k]!=expected:mismatches.append({'regenerated_history':k,'actual':hist_hash[k]})
expected_source=cfg['expected_identity']['shared_source_sha256']
for rel,expected in expected_source.items():
    if sha(Path(cfg['frozen_extension'])/rel)!=expected:mismatches.append({'shared_source':rel})
for k,v in cfg['expected_identity'].items():
    if ident.get(k)!=v:mismatches.append({'expected_identity':k})
for k,v in cfg['online_parameters'].items():
    if gate['config'].get(k)!=v:mismatches.append({'parameter':k})
files={}
for p in sorted(out.rglob('*')):
    if not p.is_file() or 'src' in p.relative_to(out).parts:continue
    if p.suffix not in ('.json','.sha256','.exit','.log','.py','.sh'):continue
    data=p.read_bytes()
    files[str(p.relative_to(root))]={'sha256':hashlib.sha256(data).hexdigest(),
        'bytes':len(data),'base64':base64.b64encode(data).decode()}
ps=subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True)
owned=[line for line in ps.splitlines() if str(out/'pipeline.py') in line and
       ('timeout ' in line or line.strip().split(None,2)[-1].startswith('/home/hzeng/miniconda3/envs/MASKCO_env/bin/python'))]
answer={'checked_at_server':str(datetime.datetime.now()),'qualified':not mismatches,
 'actual_manifest_entries':len(checks),'actual_manifest_checks':checks,
 'gate_source_actual':source_actual,'actual_models':weights,'actual_cvrp_sha256':actual_cvrp,
 'actual_protocol_sha256':sha(root/'configs/i3_20261008_v2/protocol.json'),
 'actual_online_config_sha256':sha(out/'online_config_20260926.json'),
 'development_regeneration':dev_hash,'history_regeneration':hist_hash,
 'history_regeneration_scope':'raw historical dataset, as used by build_history and verified_pool_days; development data separately adds v2 initial_quality',
 'actual_contract_sha256':contract.contract_hash,'actual_contract':__import__('dataclasses').asdict(contract),
 'mismatches':mismatches,'owned_pipeline_processes':owned,
 'first_gate_sha256':sha(out/'online_20260926/gate.json'),
 'failure_paths':[str(p.relative_to(out)) for p in out.rglob('*failure*.json')],
 'additional_development_directories_present':[s for s in (20260930,20261001) if (out/('online_'+str(s))).exists()],
 'final_seed_directories_present':[s for s in (20260927,20260928) if (out/('online_'+str(s))).exists()],
 'new_policy_or_scientific_runs':False,'remote_writes':False,'files':files}
print(json.dumps(answer,ensure_ascii=False))
'''
cmd = "cd " + REMOTE + " && env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/hzeng/miniconda3/envs/MASKCO_env/bin/python - <<'PY'\n" + SNIPPET + "\nPY"
raw=run(cmd,timeout=180)
start=raw.find('{"checked_at_server"')
if start<0:raise RuntimeError("Remote audit did not return JSON: "+raw[-2500:])
packet, end=json.JSONDecoder().raw_decode(raw[start:])
downloads=[]
for rel,spec in packet.pop('files').items():
    p=(EXT/rel).resolve()
    if not p.is_relative_to(OUT.resolve()):raise RuntimeError('Download escaped result directory')
    data=base64.b64decode(spec['base64'])
    if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise RuntimeError('Transfer hash mismatch')
    if p.exists() and p.read_bytes()!=data:
        # Prior snapshots of mutable state remain in heartbeat archives.
        if p.name not in ('pipeline_state.json','pipeline.log','run.log'):
            raise RuntimeError('Refusing to replace differing existing sealed artifact: '+str(p))
        snapshot=OUT/'prior_download_snapshots'/p.name
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(p.read_bytes())
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    downloads.append({'path':rel,'sha256':spec['sha256'],'bytes':len(data)})
packet['downloaded_artifacts']=downloads
def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    if isinstance(value, dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, list):return [finite_json(v) for v in value]
    return value
packet=finite_json(packet)
packet['trailing_remote_diagnostics']=raw[start+end:].strip()[-1000:]
target=OUT/'INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009_v2.json'
if target.exists():raise RuntimeError('Independent end audit already exists')
target.write_text(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:packet[k] for k in ('checked_at_server','qualified','actual_manifest_entries','mismatches',
 'owned_pipeline_processes','additional_development_directories_present','final_seed_directories_present')},
 ensure_ascii=False))
print(json.dumps({'downloaded':len(downloads),'bytes':sum(d['bytes'] for d in downloads),'report':str(target)}))
