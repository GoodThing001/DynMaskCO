"""Download the official benchmark definitions required by the registered input audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import argparse
import json
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HERE = Path(__file__).resolve().parent
XTW = (716, 733, 749, 766, 783, 801, 819, 837, 856, 876, 895, 916, 936, 957, 979, 1001)
_LOCAL = threading.local()


def session():
    if not hasattr(_LOCAL, "http"):
        _LOCAL.http = requests.Session()
        _LOCAL.http.mount("https://", HTTPAdapter(max_retries=Retry(total=3,backoff_factor=.3,status_forcelist=(429,500,502,503,504))))
    return _LOCAL.http


def official_sources():
    for index, n in enumerate(XTW, start=820):
        yield f"XTW-n{n}.vrp", f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{index}"
    for n in (400, 600, 800, 1000):
        yield f"homberger{n}.zip", f"https://www.sintef.no/globalassets/project/top/vrptw/homberger/{n}/homberger_{n}_customer_instances.zip"


def fetch(item):
    name, url = item
    out = HERE / "raw" / name
    if out.exists():
        return
    response = session().get(url, timeout=(15, 60))
    response.raise_for_status()
    data = response.content
    if name.endswith(".zip"):
        with ZipFile(BytesIO(data)) as archive:
            if sum(s.lower().endswith(".txt") for s in archive.namelist()) != 60:
                raise ValueError(f"Unexpected official inventory: {name}")
    elif name[:-4].encode("ascii") not in data.splitlines()[0]:
        raise ValueError(f"Unexpected XTW source: {name}, {data[:100]!r}")
    out.write_bytes(data)
    print(name, len(data), flush=True)


def fetch_gh_cvrplib():
    base = HERE.parents[1]/"maskco_perishable_external_20261119"
    known = {}
    for folder in (base.parent/"maskco_heatmap_calibration_20261119/validation_sources",
                   base/"source_external",base/"reserved_validation/raw",base/"future_validation/raw"):
        for f in folder.glob("*.TXT"):
            known[f.stem.upper()] = f
    raw = HERE/"raw"
    folder = raw/"gh_members"
    folder.mkdir(exist_ok=True)
    sources = {}

    def one(pair):
        title, url = pair
        if title in known:
            return known[title]
        f = folder/(title+".TXT")
        if not f.is_file():
            response = session().get(url,timeout=(15,60))
            response.raise_for_status()
            data = response.content
            if not data.split() or data.split()[0].decode("ascii").upper()!=title:
                raise ValueError(title)
            f.write_bytes(data)
        return f

    for n in (400,600,800,1000):
        members = {}
        for k, family in enumerate(("C1","C2","R1","R2","RC1","RC2")):
            for i in range(1,11):
                title = f"{family}_{n//100}_{i}"
                ident = 495+(n-400)//200*60+10*k+i
                members[title] = f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{ident}"
        with ThreadPoolExecutor(max_workers=6) as pool:
            files = list(pool.map(one,members.items()))
        with ZipFile(raw/f"homberger{n}.zip","w",compression=8) as z:
            for (title,url),f in zip(members.items(),files):
                z.write(f,title+".TXT")
                sources[title]=url
        print("CVRPLIB",n,len(members),flush=True)
    (raw/"gh_source_urls.json").write_text(json.dumps(sources,indent=2),encoding="utf8")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--gh-cvrplib",action="store_true")
    mode=args.parse_args()
    (HERE / "raw").mkdir(exist_ok=True)
    if mode.gh_cvrplib:
        fetch_gh_cvrplib()
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, official_sources()))
