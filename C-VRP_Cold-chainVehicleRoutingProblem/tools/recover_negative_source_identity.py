"""Bounded read-only search for the lost historical module; never substitute current code."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import os

E=Path(__file__).resolve().parents[1]
ROOT=E.parent
TARGET='e88c7d0d0d21c73b6335880a20111ee4d02b0a6693071872c74d217d34beb300'
hits=[]
checked=[]
errors=[]

def inspect(label,data):
    digest=hashlib.sha256(data).hexdigest()
    checked.append({'path':label,'sha256':digest})
    if digest==TARGET:
        hits.append({'path':label,'sha256':digest})
        dest=E/'03_负结果线/核查工件/recovered_mpre_trained_e88c7d0d.py'
        if not dest.exists():
            with dest.open('xb') as f:f.write(data)

for path in ROOT.rglob('mpre_trained.py'):
    if '.git' not in path.parts:
        inspect(str(path),path.read_bytes())
for path in E.rglob('*.zip'):
    try:
        readable = '\\\\?\\' + str(path.resolve()) if os.name == 'nt' else str(path)
        with zipfile.ZipFile(readable) as z:
            for item in z.infolist():
                if item.filename.replace('\\','/').endswith('/mpre_trained.py') or item.filename=='mpre_trained.py':
                    inspect(str(path)+'::'+item.filename,z.read(item))
    except (OSError,zipfile.BadZipFile) as exc:
        errors.append({'path':str(path),'error':repr(exc)})
rel='C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre_trained.py'
git=['git','-c','safe.directory='+str(ROOT)]
proc=subprocess.run(git+['log','--all','--format=%H','--',rel],cwd=ROOT,capture_output=True,text=True)
if proc.returncode==0:
    for commit in proc.stdout.splitlines():
        p=subprocess.run(git+['show',commit+':'+rel],cwd=ROOT,capture_output=True)
        if p.returncode==0:inspect('git:'+commit+':'+rel,p.stdout)
else:
    errors.append({'path':'git_history','error':proc.stderr.strip()})
report={'target_sha256':TARGET,'searched_scope':'local repository filename matches, extension ZIP members and reachable git history',
        'files_checked':checked,'exact_hits':hits,'errors':errors,
        'historical_module_recovered':bool(hits),'historical_full_reproduction_ready':False,
        'remote_search_performed':False,'current_code_substituted':False}
out=E/'results/three_line_corrections_20261009/NEGATIVE_SOURCE_RECOVERY_SEARCH_v2.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'checked':len(checked),'exact_hits':len(hits),'errors':errors,'archive':str(out)},ensure_ascii=False))
