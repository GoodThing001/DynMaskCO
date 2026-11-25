"""Download the versioned author archive for the perishable-goods cross-dock study."""
from pathlib import Path
from zipfile import ZipFile
import requests

HERE=Path(__file__).resolve().parent
URL="https://data.mendeley.com/public-api/zip/93w8ynrk59/download/2"
if __name__=="__main__":
    path=HERE/"source_perishable"/"93w8ynrk59-v2.zip"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        reply=requests.get(URL,timeout=60)
        reply.raise_for_status()
        from io import BytesIO
        with ZipFile(BytesIO(reply.content)) as z:
            if not any(s.lower().endswith("_tw.txt") for s in z.namelist()):
                raise ValueError("Missing time windows in perishable benchmark")
        path.write_bytes(reply.content)
    with ZipFile(path) as z:
        names=sorted(s for s in z.namelist() if "/Instances_static/Data100_" in s and not s.endswith("_tw.txt"))
        print(len(names),"static 100-request instances")
        print([(Path(s).name,z.read(s).splitlines()[0]) for s in names])
