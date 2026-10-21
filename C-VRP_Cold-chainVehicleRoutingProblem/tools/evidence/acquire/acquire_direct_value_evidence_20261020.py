"""Acquire named public primary documents for value-model evidence upgrade."""
import concurrent.futures
import json
from acquire_static_evidence_20261020 import BASE, download

OUT = BASE/'direct_value_evidence'
JOBS = [
    ('gaukler_tomato_value_2023', 'https://pure.eur.nl/files/155582174/1-s2.0-S037722172300200X-main.pdf', OUT/'sources/gaukler_tomato_2023.pdf'),
    ('south_india_grade_value_2022', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9170466/fullTextXML', OUT/'sources/south_india_tomato_2022.xml'),
    ('tomato_inventory_pricing_2023', 'https://www.rairo-ro.org/10.1051/ro/2023122/pdf', OUT/'sources/rairo2023122.pdf'),
]


if __name__ == '__main__':
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(download, j): j[0] for j in JOBS}
        for future in concurrent.futures.as_completed(pending):
            try:
                row = future.result()
            except Exception as err:
                row = {'name': pending[future], 'status': 'FAILED', 'error_type': type(err).__name__, 'error': str(err)[:240]}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'acquisition_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
