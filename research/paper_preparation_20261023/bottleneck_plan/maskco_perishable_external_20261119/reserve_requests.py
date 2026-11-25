"""Identify the untouched demand records in the published 200-request parent files."""
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
def main():
    base="Dynamic PDPCDPG with Rolling-Horizon/Instances_static/"
    with ZipFile(HERE/"source_perishable/93w8ynrk59-v2.zip") as archive:
        def body(n,letter):
            ls=archive.read(f"{base}Data{n}_{letter}.txt").decode().splitlines()[2:]
            return [tuple(int(t) for t in s.split()) for s in ls if s.strip()]
        seen={v[1:3] for letter in "abcde" for v in body(100,letter)}
        reserved={}
        for letter in "abcde":
            used=Counter(r[1:] for r in body(100,letter))
            parent=body(200,letter)
            fresh=[]
            for row in parent:
                if used[row[1:]]:
                    used[row[1:]]-=1
                else:
                    fresh.append(row)
            if any(used.values()) or len(fresh)!=100:
                raise ValueError("Parent is not a 100-request extension")
            reserved[letter]=dict(source=f"Data200_{letter}",row_ids=[r[0] for r in fresh],
                                  new_delivery_locations_among_100_requests=sum(r[1:3] not in seen for r in fresh))
    out=dict(unit="original request, destination, pickup location and pallet count",
             data_doi="10.17632/93w8ynrk59.2",scope="reserved for prospective testing of a subsequent development-qualified method",
             complete_parent_difference=reserved)
    (HERE/"reserved_requests.json").write_text(json.dumps(out,indent=2),encoding="utf8")
    print({l:r["new_delivery_locations_among_100_requests"] for l,r in reserved.items()})
if __name__=="__main__":main()
