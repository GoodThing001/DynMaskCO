"""Acquire primary model papers and university journal-list evidence for a paper protocol."""
import concurrent.futures
import json
from pathlib import Path
from acquire_static_evidence_20261020 import EXTENSION_ROOT, download

OUT=EXTENSION_ROOT/'docs/static_delivery_research_20261021'
JOBS=[
 ('precooling_EJOR_2025','https://edepot.wur.nl/675716',OUT/'sources/precooling_EJOR_2025.pdf'),
 ('joint_distribution_RCR_2020','https://pureportal.coventry.ac.uk/files/28060196/Binder6.pdf',OUT/'sources/joint_distribution_RCR_2020.pdf'),
 ('CAS2025_Q1_SCNU','https://statics.scnu.edu.cn/pics/lib/2025/1031/1761900158678161.xls',OUT/'sources/CAS2025_Q1_SCNU.xls'),
]

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  pending={pool.submit(download,j):j[0] for j in JOBS}
  for f in concurrent.futures.as_completed(pending):
   try: row=f.result()
   except Exception as err: row={'name':pending[f],'status':'FAILED','error_type':type(err).__name__,'error':str(err)[:220]}
   rows.append(row); print(json.dumps(row,ensure_ascii=True),flush=True)
 (OUT/'acquisition_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
