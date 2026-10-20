"""Named public originals for the parameter-origin follow-up; no data generation."""
import concurrent.futures
import json
from pathlib import Path
from acquire_static_evidence_20261020 import download, BASE

OUT=BASE/'parameter_origins_followup'
JOBS=[
    ('author_thesis_2018','https://www.biblio.univ-evry.fr/theses/2018/2018SACLE036.pdf',OUT/'sources/li_author_thesis_2018.pdf'),
    ('crowd_shipping_2025','https://mdpi-res.com/d_attachment/symmetry/symmetry-17-01609/article_deploy/symmetry-17-01609.pdf',OUT/'sources/symmetry17101609.pdf'),
    ('mullet_market_2022','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9474837/fullTextXML',OUT/'sources/mullet_market_2022.xml'),
]
if __name__=='__main__':
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(download,j):j[0] for j in JOBS}
        for f in concurrent.futures.as_completed(pending):
            try: r=f.result()
            except Exception as e: r={'name':pending[f],'status':'FAILED','error_type':type(e).__name__,'error':str(e)[:200]}
            rows.append(r); print(json.dumps(r,ensure_ascii=True),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'acquisition_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
