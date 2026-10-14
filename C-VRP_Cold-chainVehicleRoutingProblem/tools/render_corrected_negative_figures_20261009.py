"""Generate the current figure working artifacts from immutable archived JSON."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,os
E=Path(__file__).resolve().parents[1]
old=E/'negative_result_paper/release'
dst=E/'03_负结果线/图表工作稿/2026-10-09'
if dst.exists():raise ValueError('Figure workspace exists; preserve prior artifact')
dst.mkdir(parents=True)
manifest=[]
sources=[old/'scripts/make_figures.py',*sorted((old/'results').rglob('*.json'))]
for p in sources:
 rel=p.relative_to(old);q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('xb') as f:f.write(p.read_bytes())
 manifest.append({'source':str(p.relative_to(E)),'copy':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['MPLBACKEND']='Agg'
with (dst/'render.log').open('xb') as log:
 p=subprocess.run([sys.executable,str(dst/'scripts/make_figures.py')],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=60)
manifest_ok=all(hashlib.sha256((E/x['source']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
with (dst/'FIGURE_PROVENANCE.json').open('x',encoding='utf-8') as f:
 json.dump({'source_files':manifest,'source_unchanged':manifest_ok,'render_exit':p.returncode,
  'figures':[{'name':x.name,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sorted((dst/'figures').glob('*.png'))],
  'scope':'figure artifacts and archived-JSON bootstrap; no new strategy experiment or original rerun'},f,indent=2)
print(json.dumps({'render_exit':p.returncode,'source_unchanged':manifest_ok,'destination':str(dst)},ensure_ascii=False))
raise SystemExit(p.returncode)
