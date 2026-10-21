"""Public direct downloads after GitHub REST rate limit; no API/auth workaround."""
import concurrent.futures
import json
import subprocess
from pathlib import Path
from acquire_static_evidence_20261020 import BASE, DATA, COMPARE, get, save, extract, sha, download

REPOS = [
 ('ALNS','N-Wouda/ALNS',None),
 ('PyVRP','PyVRP/PyVRP','4b2aa285f34efb8e2ff1995a87b834414c1c5c17'),
 ('HGS-CVRP','vidalt/HGS-CVRP',None),
 ('RouteFinder','ai4co/routefinder','cc3ab078650e8db62c0562a60f421defe38d5ae3'),
 ('DeepACO','henry-yeh/DeepACO','9a756a3fe6cf9627ecc166110a02fdbb61f2a0d7'),
 ('OR-Tools','google/or-tools','refs/tags/v9.14'),
 ('VRPTWQ_Solomon','SelinHulagu/vrptwq-solomon-instances',None),
]

def repo_task(item):
    name, repo, commit = item
    if commit is None or commit.startswith('refs/'):
        p = subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git',commit or 'HEAD'],capture_output=True,text=True,timeout=50,check=True)
        commit = p.stdout.split()[0]
    url=f'https://codeload.github.com/{repo}/zip/{commit}'
    body=get(url).content
    is_data=name=='VRPTWQ_Solomon'
    root=(DATA/name) if is_data else (COMPARE/'upstream'/name)
    archive=(DATA/'archives' if is_data else COMPARE/'archives')/f'{name}-{commit[:12]}.zip'
    save(archive,body)
    files=extract(body,root,strip=True)
    if is_data:
        for f in root.glob('*.zip'):
            extract(f.read_bytes(),root/'instances')
    licenses=[r['path'] for r in files if Path(r['path']).name.lower().startswith(('license','copying'))]
    rec={'name':name,'repo':repo,'commit':commit,'url':url,'archive_sha256':sha(body),
         'archive_bytes':len(body),'files':files,'license_files':licenses,'source_executed':False,
         'status':'DOWNLOADED','pin_method':'known historical pin or public git ls-remote; REST unavailable'}
    save(root/'SNAPSHOT_PROVENANCE.json',json.dumps(rec,ensure_ascii=False,indent=2).encode())
    return rec

def main():
    jobs=[(repo_task,(r,)) for r in REPOS]
    jobs += [(download,(r,)) for r in [
        ('solomon_zip','https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip',DATA/'archives/solomon-100.zip'),
        ('tomato_hedonic_nanna','https://ageconsearch.umn.edu/nanna/record/29990/files/17020139.pdf?registerDownload=1&version=1&withMetadata=0&withWatermark=0',BASE/'sources/tomato_hedonic_1985.pdf'),
        ('tomato_fulltext','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9818793/fullTextXML',BASE/'sources/tomato_foods12010006.xml'),
        ('mullet_fulltext','https://www.mdpi.com/2410-3888/9/12/480/xml',BASE/'sources/mullet_fishes9120480.xml'),
    ]]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(fn,*args):str(args[0]) for fn,args in jobs}
        for f in concurrent.futures.as_completed(fs):
            try: row=f.result()
            except Exception as e: row={'task':fs[f],'status':'FAILED','error_type':type(e).__name__,'error':str(e)[:200]}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='files'},ensure_ascii=True),flush=True)
    path=BASE/'acquisition_fallback_manifest.json'
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    sol=DATA/'archives/solomon-100.zip'
    if sol.exists():
        files=extract(sol.read_bytes(),DATA/'solomon_100/instances')
        save(DATA/'solomon_100/FILES_SHA256.json',json.dumps(files,indent=2).encode())

if __name__=='__main__': main()
