"""Independent local raw-day arithmetic/registered bootstrap audit, no model runs."""
from pathlib import Path
import datetime, hashlib, json, math
import numpy as np

EXT=Path(__file__).resolve().parents[1]
OUT=EXT/'results/i3_online_recovery_1009_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,label):
    if not v:raise RuntimeError(label)
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
gate=read(OUT/'online_20260926/gate.json')
reg=read(OUT/'protocol.json')
cfg=read(OUT/'online_config_20260926.json')
actual=read(OUT/'INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009_v2.json')
server=read(OUT/'online_20260926/INDEPENDENT_ADJUDICATION.json')
require(actual['qualified'] and not actual['mismatches'],'actual identity unqualified')
require(actual['actual_manifest_entries']==348,'actual source count')
require(actual['first_gate_sha256']==sha(OUT/'online_20260926/gate.json'),'actual gate bytes')
require(actual['actual_protocol_sha256']==sha(OUT/'protocol.json')==
 '4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c','protocol SHA')
require(actual['actual_online_config_sha256']==sha(OUT/'online_config_20260926.json')==
 gate['model_adapter']['config_sha256'],'config SHA')
require(gate['seeds']['ran']==[20260926] and gate['config']['final_test'] is False,'only first dev')
require(not actual['additional_development_directories_present'] and
 not actual['final_seed_directories_present'] and not actual['owned_pipeline_processes'] and
 not actual['failure_paths'],'closed process and no forbidden branches')
require(cfg['online_parameters']==reg['online_parameters'],'registered online settings')
ident=gate['results']['20260926']['_identity']
require(ident['source_sha256']==ident['source_end_sha256']==actual['gate_source_actual'],'actual source start/end')
require(all(ident[k]==v for k,v in cfg['expected_identity'].items()),'expected full identity')
require(all(gate['config'].get(k)==v for k,v in reg['online_parameters'].items()),'actual parameter settings')
require(gate['config']['only'] is None and gate['config']['baseline_from'] is None and
 gate['config']['allow_code_mismatch'] is False,'fresh matched controls')
require(cfg['history_pool_lineage']==gate['model_adapter']['history_pool_lineage'],'pool')
require(all(actual['actual_models'][a]['model_sha256']==s['sha256'] and
 actual['actual_models'][a]['card_sha256']==s['card_sha256'] for a,s in cfg['models'].items()),'actual weights/card')
require('RUN_EXIT=0' in (OUT/'online_20260926/run.exit').read_text().splitlines(),'run exit')
require(set((OUT/'pipeline.exit').read_text().splitlines())=={'RUN_EXIT=0','SEAL_EXIT=0','DONE=1'},'pipeline exits')
teacher=read(EXT/'results/i3_registered_1008_v5/LOCAL_128_STAGE_ACCEPTANCE_20261009.json')
training=read(EXT/'results/i3_registered_1008_v5/LOCAL_FIXED_SCHEDULE_AUDIT_20261009.json')
models=read(OUT/'INDEPENDENT_FIXED_MODEL_ACCEPTANCE.json')
require(teacher['qualification_verified'] and teacher['states']==128 and
 training['qualified_fixed_schedule'] and models['qualified'],'preceding stage qualifications')
require(all(models['models'][a]['model_sha256']==actual['actual_models'][a]['model_sha256']
 for a in cfg['models']),'same accepted final models')
tier=gate['results']['20260926']['p_c=(5,10,15)']
rows=tier['per_day']
require(set(rows)=={'maskco','cond_hist','warmstart','reconstruction_control'},'four arms')
stats={}; differences={}; means={}; totals=[]
fields=['utility','revenue','fuel_cost','reject_loss','n_served','n_rejected','energy_kwh',
 'distance_km','timeouts','tomato_mean_quality','n_fish_unsalable','elapsed_s']
for name,rr in rows.items():
    require(len(rr)==40 and all(type(r['i']) is int for r in rr) and
            sorted(r['i'] for r in rr)==list(range(40)),name+' unique 40 days')
    rr.sort(key=lambda r:r['i'])
    require(all(math.isfinite(float(r[k])) for r in rr for k in fields),name+' finite days')
    require(all(r['hard_feasible'] is True and not r['failures'] and
                r['budget_violated'] is False and r['energy_kwh']<=gate['budget']['B']+1e-5 for r in rr),name+' hard/budget')
    residual=max(abs(r['utility']-(r['revenue']-r['fuel_cost']-r['reject_loss'])) for r in rr)
    require(residual<1e-7,name+' utility decomposition')
    require(all(abs(r['fuel_cost']-.1*r['distance_km'])<1e-7 for r in rr),name+' fuel-distance')
    means[name]={k:float(np.mean([r[k] for r in rr])) for k in fields}
    stats[name]={'days':40,'finite_days':40,'hard_rate':1.,
        'means':means[name],'total_timeouts':sum(r['timeouts'] for r in rr),
        'days_with_timeout':sum(r['timeouts']>0 for r in rr),
        'utility_arithmetic_max_residual':residual,
        'service_rates':{c:float(np.mean([r['service_rate'][c] for r in rr])) for c in ('0','1','2')},
        'salable_rates':{c:float(np.mean([r['salable_rate'][c] for r in rr if r['salable_rate'][c] is not None]))
                         for c in ('1','2')}}
    require(abs(means[name]['utility']-tier['arms'][name]['utility_all_days']['mean'])<1e-9,'arm summary')
    totals.append([r['n_served']+r['n_rejected'] for r in rr])
require(all(t==totals[0] for t in totals),'same actual daily order counts')
for control in ('cond_hist','reconstruction_control','warmstart'):
    d=np.array([a['utility']-b['utility'] for a,b in zip(rows['maskco'],rows[control])])
    rng=np.random.default_rng(20260926+777)
    boot=d[rng.integers(0,40,size=(2000,40))].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    value={'mean':float(d.mean()),'ci_lo':float(lo),'ci_hi':float(hi),'paired_days':40,
        'wins':int((d>0).sum()),'losses':int((d<0).sum()),'ties':int((d==0).sum()),
        'decomposition':{k:means['maskco'][k]-means[control][k] for k in
         ('revenue','fuel_cost','reject_loss','n_served','n_rejected','timeouts','energy_kwh')},
        'per_day':[{'day':i,'utility_difference':float(v)} for i,v in enumerate(d)]}
    for k in ('mean','ci_lo','ci_hi'):
        require(abs(value[k]-server['by_seed']['20260926']['comparisons'][control][k])<1e-9,'independent CI matches '+control)
    de=value['decomposition']
    require(abs(value['mean']-(de['revenue']-de['fuel_cost']-de['reject_loss']))<1e-7,'paired decomposition')
    differences[control]=value
primary=differences['cond_hist'];attribution=differences['reconstruction_control']
report={'checked_at_local':str(datetime.datetime.now()),'formal_adjudicable':True,
 'three_stage_qualifications':{'teacher_128':True,'fixed_training_pair_192':True,'first_online_40x4':True},
 'protocol_sha256':sha(OUT/'protocol.json'),'gate_sha256':sha(OUT/'online_20260926/gate.json'),
 'config_sha256':sha(OUT/'online_config_20260926.json'),
 'actual_identity_report_sha256':sha(OUT/'INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009_v2.json'),
 'raw_day_count':160,'all_days_and_timeouts_retained':True,'by_arm':stats,
 'paired_comparisons':differences,'bootstrap':{'samples':2000,'seed':20260926+777,'unit':'paired_day','quantiles':[.025,.975]},
 'registered_primary_passed':bool(primary['mean']>=20 and primary['ci_lo']>0),
 'decision_objective_attribution_passed':bool(attribution['mean']>0 and attribution['ci_lo']>0),
 'core_maskco_single_step_online_gain_established':False,
 'equal_budget_iteration_gain_established':False,'valid_strong_dynamic_comparison_complete':False,
 'additional_seeds_run':False,'final_reserved_seeds_run':False,'numerical_retry_performed':False,
 'limits':['Hard flags from sealed runtime trace evaluator; no new independent online raw-route replay because gate records daily summaries.',
 'No per-decision margin, actor sample or timeout-cause logs in final online outputs; causal mechanism remains unresolved.',
 'This is a single registered development seed under C0 pilot parameters; no claim all MaskCO methods fail or statistical equivalence.',
 'Teacher objective holdout historically seen by warmstart and history pool; teacher continuation unlimited vs online 10s.',
 'Initial flawed identity audit preserved; corrected audit regenerates raw historical pool, quality-augmented development data separately.'],
 'remote_document_uploads':False}
require(not report['registered_primary_passed'],'unexpected pass')
write(OUT/'LOCAL_ONLINE_FINAL_ADJUDICATION_20261009.json',report)
print(json.dumps({'formal_adjudicable':True,'primary':{k:primary[k] for k in ('mean','ci_lo','ci_hi','wins','losses')},
 'attribution':{k:attribution[k] for k in ('mean','ci_lo','ci_hi')},
 'arm_means':means,'decompositions':{c:v['decomposition'] for c,v in differences.items()},
 'timeout_totals':{a:s['total_timeouts'] for a,s in stats.items()}},ensure_ascii=False,indent=2))

