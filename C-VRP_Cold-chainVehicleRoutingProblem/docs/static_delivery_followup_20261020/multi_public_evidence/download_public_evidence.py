"""Download only named public research evidence; never run source generators."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
SOURCES=[
 ('chen_2023_mixed_cold_chain.pdf','https://pure.tue.nl/ws/portalfiles/portal/309283747/1-s2.0-S0957417423014811-main.pdf'),
 ('thammasat_2024_opensolver.pdf','https://ethesisarchive.library.tu.ac.th/thesis/2024/TU_2024_6722040620_20924_30485.pdf'),
 ('vidal_split_1508.02759.pdf','https://arxiv.org/pdf/1508.02759'),
 ('vrprep_christofides_1979_set_m.zip','https://www.vrp-rep.org/datasets/download/christofides-et-al-1979-set-m.zip'),
 ('escobar_crossref.json','https://api.crossref.org/works/10.5267/j.ijiec.2021.6.001'),
 ('escobar_2022_rcvrp.pdf','https://www.growingscience.com/ijiec/Vol13/IJIEC_2021_16.pdf'),
]
def fetch(item):
 name,url=item
 result={'file':name,'url':url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
 try:
  response=requests.get(url,timeout=30)
  result.update(status=response.status_code,final_url=response.url,content_type=response.headers.get('Content-Type'))
  response.raise_for_status()
  raw=response.content
  assert len(raw)<25_000_000
  if name.endswith('.pdf'): assert raw.startswith(b'%PDF')
  if name.endswith('.zip'): assert raw.startswith(b'PK')
  if name.endswith('.json'): json.loads(raw)
  path=HERE/name
  if path.exists(): assert path.read_bytes()==raw
  else: path.write_bytes(raw)
  result.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),downloaded=True)
 except Exception as error:
  result.update(downloaded=False,error_type=type(error).__name__)
 return result
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=5) as pool:
  results=list(pool.map(fetch,SOURCES))
 (HERE/'download_manifest.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(results,indent=2))
