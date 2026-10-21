import concurrent.futures
import json
from acquire_static_evidence_fallback_20261020 import repo_task
from acquire_static_evidence_20261020 import download, BASE

jobs = [
    (repo_task, ('CaR', 'jieyibi/CaR-constraint', None)),
    (download, ('tomato_value_2023', 'https://pure.eur.nl/files/155582174/1-s2.0-S037722172300200X-main.pdf', BASE/'sources/tomato_value_2023.pdf')),
    (download, ('tomato_firmness_poster', 'https://repositorio.ucp.pt/bitstreams/bc493d39-f2f4-41a1-9942-d4217a123e46/download', BASE/'sources/tomato_firmness_poster.pdf')),
]
rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    pending = {pool.submit(fn, args): str(args[0]) for fn, args in jobs}
    for future in concurrent.futures.as_completed(pending):
        try:
            row = future.result()
        except Exception as e:
            row = {'task': pending[future], 'status': 'FAILED', 'error_type': type(e).__name__, 'error': str(e)[:200]}
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'files'}), flush=True)
(BASE/'acquisition_supplement_manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
