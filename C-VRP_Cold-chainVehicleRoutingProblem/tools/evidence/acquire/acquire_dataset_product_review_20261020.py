"""Acquire public primary papers for dataset/product evidence; no customer generation."""
import concurrent.futures
import json
from acquire_static_evidence_20261020 import BASE, download

OUT = BASE/'dataset_product_review'
JOBS = [
    ('customer_centric_2021', 'https://mdpi-res.com/d_attachment/electronics/electronics-10-02018/article_deploy/electronics-10-02018.pdf', OUT/'sources/customer_centric_2021.pdf'),
    ('temperature_shelf_life_2020', 'https://mdpi-res.com/d_attachment/sustainability/sustainability-12-06668/article_deploy/sustainability-12-06668.pdf', OUT/'sources/temperature_shelf_life_2020.pdf'),
    ('freshness_investment_2021', 'https://mdpi-res.com/d_attachment/sustainability/sustainability-13-08110/article_deploy/sustainability-13-08110.pdf', OUT/'sources/freshness_investment_2021.pdf'),
    ('plos_milk_2020', 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0235950&type=manuscript', OUT/'sources/plos_milk_2020.xml'),
    ('amorim_author_thesis', 'https://paginas.fe.up.pt/~balobo/PdChain/Papers/AThesis.pdf', OUT/'sources/amorim_thesis.pdf'),
]

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(download, job): job[0] for job in JOBS}
        for future in concurrent.futures.as_completed(jobs):
            try:
                row = future.result()
            except Exception as err:
                row = {'name': jobs[future], 'status': 'FAILED', 'error_type': type(err).__name__, 'error': str(err)[:240]}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    (OUT/'acquisition_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
