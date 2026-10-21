"""Download named public evidence; never execute downloaded source or overwrite history."""
from __future__ import annotations
import concurrent.futures
import hashlib
import io
import json
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT

BASE = EXTENSION_ROOT / 'docs/static_delivery_research_20261020'
DATA = EXTENSION_ROOT / 'data/classic_static_sources_20261020'
COMPARE = EXTENSION_ROOT / 'comparisons/static_coldchain'
ASOF = '2026-10-20T15:59:59Z'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != body:
        raise ValueError(f'refuse overwrite: {path}')
    path.write_bytes(body)

def get(url):
    r = requests.get(url, timeout=(15, 75), headers={'User-Agent': 'StaticColdChainResearch/1.0'})
    r.raise_for_status()
    return r

def extract(body, root, strip=False):
    rows = []
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            p = Path(member.filename)
            if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('unsafe archive member')
            parts = p.parts[1:] if strip else p.parts
            if not parts:
                continue
            target = root.joinpath(*parts)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('archive escapes root')
            b = z.read(member)
            save(target, b)
            rows.append({'path': '/'.join(parts), 'size': len(b), 'sha256': sha(b)})
    return rows

def repository(name, repo, commit=None):
    meta = get(f'https://api.github.com/repos/{repo}').json()
    if commit is None:
        listing = get(f'https://api.github.com/repos/{repo}/commits?until={ASOF}&per_page=1').json()
        if not listing:
            return {'name': name, 'repo': repo, 'status': 'NO_COMMIT_BY_RESEARCH_DATE'}
        commit = listing[0]['sha']
    detail = get(f'https://api.github.com/repos/{repo}/commits/{commit}').json()
    commit = detail['sha']
    if detail['commit']['committer']['date'] > ASOF:
        raise ValueError('commit newer than research date')
    url = f'https://codeload.github.com/{repo}/zip/{commit}'
    b = get(url).content
    is_data = name == 'VRPTWQ_Solomon'
    root = DATA / name if is_data else COMPARE / 'upstream' / name
    archive = DATA / 'archives' / f'{name}-{commit[:12]}.zip' if is_data else COMPARE / 'archives' / f'{name}-{commit[:12]}.zip'
    save(archive, b)
    rows = extract(b, root, strip=True)
    if is_data:
        for child in root.glob('*.zip'):
            extract(child.read_bytes(), root / 'instances')
    rec = {'name': name, 'repo': repo, 'commit': commit,
           'commit_date': detail['commit']['committer']['date'], 'url': url,
           'archive_sha256': sha(b), 'archive_bytes': len(b), 'status': 'DOWNLOADED',
           'license_api': (meta.get('license') or {}).get('spdx_id'),
           'files': rows, 'source_executed': False}
    save(root / 'SNAPSHOT_PROVENANCE.json', json.dumps(rec, ensure_ascii=False, indent=2).encode())
    return rec

DOWNLOADS = [
 ('solomon_page', 'https://www.sintef.no/projectweb/top/vrptw/100-customers/', DATA/'solomon_100/source_page.html'),
 ('tomato_kinetics', 'https://mdpi-res.com/d_attachment/foods/foods-12-00006/article_deploy/foods-12-00006.pdf', BASE/'sources/tomato_foods12010006.pdf'),
 ('mullet_kinetics', 'https://mdpi-res.com/d_attachment/fishes/fishes-09-00480/article_deploy/fishes-09-00480.pdf', BASE/'sources/mullet_fishes9120480.pdf'),
 ('tomato_hedonic', 'https://ageconsearch.umn.edu/record/29990/files/17020139.pdf', BASE/'sources/tomato_hedonic_1985.pdf'),
 ('fish_hedonic', 'https://www.journals.uchicago.edu/doi/pdf/10.1086/727493', BASE/'sources/fish_hedonic_2024.pdf'),
 ('truck_experiment', 'https://repository.up.ac.za/server/api/core/bitstreams/4b7f075c-1d66-4a7c-99b0-071c74fa2d61/content', BASE/'sources/truck_hefat2014.pdf'),
 ('quality_vrptw', 'https://pubsonline.informs.org/doi/pdf/10.1287/trsc.2025.0262', BASE/'sources/vrptwq_2026.pdf'),
 ('quality_driven', 'https://transsonic.nl/documents/Stellingwerf_EtAl_IJPE_2021_The_Quality_Driven_Vehicle_Routing_Problem.pdf', BASE/'sources/quality_driven_2021.pdf'),
]

def download(item):
    name, url, path = item
    r = get(url)
    b = r.content
    if path.suffix == '.pdf' and not b.startswith(b'%PDF'):
        raise ValueError('response is not PDF')
    save(path, b)
    return {'name': name, 'url': url, 'final_url': r.url, 'status': 'DOWNLOADED', 'sha256': sha(b), 'bytes': len(b), 'path': str(path.relative_to(EXTENSION_ROOT))}

def main():
    tasks = [(repository, ('ALNS', 'N-Wouda/ALNS')),
             (repository, ('PyVRP', 'PyVRP/PyVRP', '4b2aa285f34efb8e2ff1995a87b834414c1c5c17')),
             (repository, ('HGS-CVRP', 'vidalt/HGS-CVRP')),
             (repository, ('RouteFinder', 'ai4co/routefinder', 'cc3ab078650e8db62c0562a60f421defe38d5ae3')),
             (repository, ('DeepACO', 'henry-yeh/DeepACO', '9a756a3fe6cf9627ecc166110a02fdbb61f2a0d7')),
             (repository, ('OR-Tools', 'google/or-tools', 'v9.14')),
             (repository, ('VRPTWQ_Solomon', 'SelinHulagu/vrptwq-solomon-instances'))]
    tasks += [(download, (item,)) for item in DOWNLOADS]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn, *args): (fn, args) for fn, args in tasks}
        for future in concurrent.futures.as_completed(futures):
            fn, args = futures[future]
            try:
                r = future.result()
            except Exception as e:
                r = {'task': str(args[0]), 'status': 'FAILED', 'error_type': type(e).__name__, 'error': str(e)[:240]}
            results.append(r)
            print(json.dumps({k:v for k,v in r.items() if k != 'files'}, ensure_ascii=True), flush=True)
    out = BASE/'acquisition_manifest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'research_date':'2026-10-20', 'acquired_at_utc': datetime.now(timezone.utc).isoformat(), 'entries':results}, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
