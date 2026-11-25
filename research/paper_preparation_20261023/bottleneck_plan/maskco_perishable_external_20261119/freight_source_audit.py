"""Inventory author-supplied freight instances and count deliveries at the largest original depot."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import json
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
URL="https://codeload.github.com/siemterbraake/vrppdl-algorithms-instances/zip/refs/heads/main"

def main():
    req=urllib.request.Request(URL,headers={"User-Agent":"Academic dataset inspection"})
    with urllib.request.urlopen(req,timeout=60) as res:
        archive=zipfile.ZipFile(BytesIO(res.read()))
    paths=[p for p in archive.namelist() if "/instances/large/" in p and not p.endswith(("/","json"))]
    if not paths:raise ValueError("Large source instance directory absent")
    print(len(paths),paths[:3])
    text=archive.read(paths[0]).decode("utf-8-sig").splitlines()
    print("\n".join(text[:10]))
    headers=[(i,l) for i,l in enumerate(text) if l.startswith("[")]
    print([(h,text[i+1]) for i,h in headers])
    receipt={"source":URL,"archive_commit":archive.comment.decode(),"files":len(paths),"instance_files":[p.split("/instances/")[1] for p in sorted(paths)],
             "first_header":[h for i,h in headers]}
    raw=HERE/"source_external"/"vrppdl_large"
    raw.mkdir(exist_ok=True)
    for path in paths:(raw/Path(path).name).write_bytes(archive.read(path))
    (HERE/"freight_source_inventory.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")
if __name__=="__main__":main()
