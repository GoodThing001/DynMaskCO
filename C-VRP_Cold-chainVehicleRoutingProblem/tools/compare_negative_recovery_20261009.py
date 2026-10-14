"""Recovered runtime files do not automatically prove historical training identity."""
from pathlib import Path
import json,hashlib
E=Path(__file__).resolve().parents[1]
R=E/'03_负结果线/恢复证据/B_REPRO_CLEAN_20261009'
candidate=E/'03_负结果线/发布候选/2026-10-09'
inventory=json.loads((candidate/'dependency_inventory.json').read_text(encoding='utf-8'))
comparisons=[]
for row in inventory['files']:
 rel=row['path'].split('C-VRP_Cold-chainVehicleRoutingProblem/',1)[1]
 p=R/rel
 sha=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 comparisons.append({'path':rel,'recovered':sha is not None,'recovered_sha256':sha,
  'matches_static_inventory':sha==row['sha256'],'registered_sha256':row['registered_sha256'],
  'matches_registered_sha256':sha==row['registered_sha256'] if row['registered_sha256'] else None,
  'original_identity_evidence':row['identity_evidence']})
formal=[]
for p in (candidate/'snapshots/results').rglob('*.json'):
 rel=p.relative_to(candidate/'snapshots')
 q=R/'negative_result_paper/release'/rel
 formal.append({'path':str(rel),'recovered':q.is_file(),
  'byte_equal':q.is_file() and p.read_bytes()==q.read_bytes()})
report={'scope':'newly recovered runtime closure compared to prior static inventory; no retrospective identity inference',
 'dependency_comparisons':comparisons,'formal_json_comparisons':formal,
 'archived_logs_recovered':['repro_check.txt','repro_check_figs.txt'],
 'original_training_sha256':'e88c7d0d0d21c73b6335880a20111ee4d02b0a6693071872c74d217d34beb300',
 'actual_training_module_sha256':hashlib.sha256((R/'scripts/models/mpre_trained.py').read_bytes()).hexdigest(),
 'full_historical_training_reproduction_qualified':False}
with (E/'results/three_line_corrections_20261009/NEGATIVE_RECOVERY_COMPARISON.json').open('x',encoding='utf-8') as f:
 json.dump(report,f,indent=2,ensure_ascii=False)
print(json.dumps({'dependencies':len(comparisons),'recovered':sum(x['recovered'] for x in comparisons),
 'static_matches':sum(x['matches_static_inventory'] for x in comparisons),
 'formal_json':len(formal),'formal_json_equal':sum(x['byte_equal'] for x in formal)}))
