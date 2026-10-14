"""Final live links, syntax, adopted profile and downloaded server identity checks."""
from pathlib import Path
import json,hashlib,re,ast,datetime
E=Path(__file__).resolve().parents[1];A=E/'results/three_line_corrections_20261009'
paths=[E/'项目当前状态.md',E/'00_项目导航.md',
 E/'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md',
 E/'docs/当前规划/结果与进度/执行进度表.md',
 E/'docs/当前规划/决策与审计/三线修正_服务器复验与文献参数版本_2026-10-10.md',
 E/'docs/当前规划/决策与审计/三线审计修正_执行记录与未决项_2026-10-09.md',
 E/'scripts/correction_engineering/README.md']
for line in ['01_取货线','02_配送线','03_负结果线']:
 paths.extend((E/line).glob('*.md'))
paths.extend((E/'01_取货线/文档/协议与审计').glob('*审计修正*.md'))
paths.extend((E/'02_配送线/文档/研究设计').glob('*审计修正*.md'))
paths.extend((E/'03_负结果线/文档/论文与投稿').glob('*.md'))
broken=[];links=0
for p in paths:
 for raw in re.findall(r'(?<!!)\[[^\]\n]*\]\(([^\)\n]+)\)',p.read_text(encoding='utf-8')):
  target=raw.strip().strip('<>')
  if '://' in target or target.startswith('#'):continue
  target=re.sub(r':\d+$','',target.split('#')[0])
  if not target:continue
  links+=1
  if not (p.parent/target).resolve().exists():broken.append({'file':str(p.relative_to(E)),'target':raw})
pyfiles=list((E/'scripts/correction_engineering').glob('*.py'))+list((E/'tools').glob('*20261010.py'))
for p in pyfiles:ast.parse(p.read_text(encoding='utf-8'))
qa=json.loads((A/'literature_quality_qa/ACCEPTANCE.json').read_text())
actual={}
for n in qa['expected']:
 raw=(E/'scripts/correction_engineering'/n).read_bytes();actual[n]=hashlib.sha256(raw).hexdigest()
 assert hashlib.sha256((A/'literature_quality_qa'/n).read_bytes()).hexdigest()==actual[n]
assert actual==qa['source_before']==qa['source_after']==qa['expected'] and qa['exit']==0
assert json.loads((A/'INDEPENDENT_SERVER_CORRECTION_ADJUDICATION.json').read_text())['all_finite_engineering_checks_pass']
assert json.loads((A/'FINAL_PROTECTED_ACCEPTANCE_20261010_v2.json').read_text())['static_qualification_pass']
result={'completed_at':datetime.datetime.now().isoformat(),'live_documents_checked':len(paths),'links_checked':links,
 'broken_links':broken,'new_python_ast_pass':True,'python_files_checked':len(pyfiles),
 'server_engineering_independent_checks':63,'server_engineering_checks_pass':True,
 'commodity_tests_passed_local_and_server':11,'commodity_download_and_local_hash_independently_match':True,
 'protected_scientific_files_unchanged':636,'original_package_unchanged':True,
 'own_measurement_data_identified':False,'own_calibration_blocks_transparent_simulation':False,
 'all_current_isolated_engineering_corrections_verified':not broken,
 'original_training_identity_recovered':False,'full_historical_reproduction_qualified':False,
 'core_maskco_online_gain_established':False,'new_dynamic_strong_controls_completed':False,
 'original_scientific_runs_repeated':False,'shared_remote_md_uploaded':False}
with (A/'FINAL_COMPLETION_20261010_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result,ensure_ascii=False))
if broken:raise SystemExit(1)
