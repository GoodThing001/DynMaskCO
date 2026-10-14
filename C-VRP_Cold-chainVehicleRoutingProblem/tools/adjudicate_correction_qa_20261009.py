"""Independent arithmetic and qualification adjudication of downloaded QA originals."""
from pathlib import Path
import json,math,datetime
E=Path(__file__).resolve().parents[1];A=E/'results/three_line_corrections_20261009'
old=E/'03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/repro_work'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
v2=A/'server_qa_v2';v3=A/'server_qa_v3';protocol=read(A/'SERVER_QA_PROTOCOL_v3.json')
actual=read(v3/'fixed_state/per_state.json');baseline=read(old/'fsq_smoke/per_state.json')
head=read(v2/'headroom/per_state.json');old_head=read(old/'hc_smoke/per_state.json')
checks=[]
def check(name,ok,detail=None):checks.append({'name':name,'pass':bool(ok),'detail':detail})
check('two_unique_fixed_states',len(actual)==2 and len({(x['inst'],x['event']) for x in actual})==2)
check('two_unique_headroom_states',len(head)==2 and len({(x['inst'],x['event']) for x in head})==2)
for i,(new,original) in enumerate(zip(actual,baseline)):
 check('fixed_identity_'+str(i),(new['inst'],new['event'])==(original['inst'],original['event']))
 check('fixed_J0_'+str(i),math.isclose(new['J0'],original['J0'],abs_tol=protocol['numerical_tolerance']))
 for arm in ('R','Mpre','Mtrained'):
  a=new[arm]['det'];b=original[arm]['det']
  for key in ('J','D','Q','E'):
   check('fixed_%d_%s_%s'%(i,arm,key),math.isclose(a[key],b[key],rel_tol=0.,abs_tol=protocol['numerical_tolerance']),[a[key],b[key]])
  check('fixed_%d_%s_plan_and_certificate'%(i,arm),a['hash']==b['hash'] and a['cert_ok'] and not a['fail'])
  if arm!='R':check('fixed_%d_%s_one_sample'%(i,arm),len(new[arm]['samples'])==1)
for i,(new,original) in enumerate(zip(head,old_head)):
 for key in ('J0','J_dist_greedy','J_exhaustive','n_first_step_actions','n_dev_tried','n_dev_improved'):
  check('headroom_%d_%s'%(i,key),math.isclose(new[key],original[key],rel_tol=0.,abs_tol=1e-8),[new[key],original[key]])
for folder,names in ((v2,('engineering','headroom','figures')),(v3,('fixed_state',))):
 for name in names:check(folder.name+'_'+name+'_exit_zero',read(folder/(name+'.exit.json'))['exit']==0)
 identity=read(folder/'INDEPENDENT_ACTUAL_END_IDENTITY.json')
 check(folder.name+'_independent_identity',not identity['mismatch'] and not identity['upload_mismatch'])
 check(folder.name+'_source_start_equals_end',read(folder/'SOURCE_START.json')==read(folder/'SOURCE_END.json'))
check('original_636_protected_unchanged',read(A/'CREDENTIAL_AUTHORIZED_PROTECTED_ACCEPTANCE.json')['static_qualification_pass'])
check('archived_14_json_byte_equal',all(x['byte_equal'] for x in read(A/'NEGATIVE_RECOVERY_COMPARISON.json')['formal_json_comparisons']))
fig=read(E/'03_负结果线/图表工作稿/2026-10-09/FIGURE_PROVENANCE.json')
check('current_three_figures_rendered',fig['render_exit']==0 and fig['source_unchanged'] and len(fig['figures'])==3)
result={'checked_at':datetime.datetime.now().isoformat(),'checks':checks,'all_finite_engineering_checks_pass':all(x['pass'] for x in checks),
 'original_training_source_qualified':False,'original_full_reproduction_qualified':False,
 'three_direction_core_online_gain_established':False,
 'scope':'18 synthetic tests, 2 fixed decision states, 2 headroom states, figure regeneration and byte/hash identity',
 'v2_failed_stage_retained':'fixed_state exit 124 at 300 seconds; first state completed; full pair rerun in separate v3 at 900 seconds',
 'v2_shell_exit_zero_is_not_stage_pass':True,
 'remaining':['original training SHA and s43/s44 training lineage','historical full closed-loop event logs',
              'business physical calibration data','new method and full dynamic strong control research are not established by this QA']}
with (A/'INDEPENDENT_SERVER_CORRECTION_ADJUDICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps({'checks':len(checks),'passed':sum(x['pass'] for x in checks),'all_pass':result['all_finite_engineering_checks_pass']}))
if not result['all_finite_engineering_checks_pass']:raise SystemExit(1)
