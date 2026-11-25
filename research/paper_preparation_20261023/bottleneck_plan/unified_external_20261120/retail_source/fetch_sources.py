"""Retrieve public supermarket benchmark inputs from the authors' pinned repository."""
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import requests


HERE = Path(__file__).resolve().parent
REV = "2c3a976f2212923db31ac86aea11483930c64b14"
BASE = f"https://raw.githubusercontent.com/unizg-fer-lares/RWVRP_benchmark/{REV}/"
CITIES = {"Zagreb":"ZG170","Brasilia":"BR100","Greater Copenhagen":"CP150","Ruhr Area":"RU120",
          "Singapore":"SG200","Melbourne":"ME340","Greater Cairo":"CA240","SF Bay Area":"SF300"}


def retrieve(name):
    path = HERE/"raw"/name
    if path.is_file():
        return
    response = requests.get(BASE+quote(name),timeout=30)
    response.raise_for_status()
    if not response.content or (name.endswith(".json") and not isinstance(response.json(),list)):
        raise ValueError(name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(response.content)


if __name__ == "__main__":
    names = ["README.md","src/validate_solution.py","src/locations_generator_osm.py","src/config.py"]
    names += [f"data/{city}/{name}.json" for city in CITIES for name in ("locations","vehicles")]
    with ThreadPoolExecutor(4) as workers:
        list(workers.map(retrieve,names))
    print("Retrieved",len(names),"public source files for",len(CITIES),"cities")
