"""Transfer the dispatch metadata only; reuse the existing reference inputs and assets in place."""
from pathlib import Path
import json
import shlex
import sys
import time
import tarfile
import ast

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"maskco_masked_reconstruction_20261118/run01"))
from host_access import connect,PROJECT,PYTHON,transport,remote_python
CAMPAIGN=PROJECT+"/_maskco_unified_external_20261120"
TARGET=CAMPAIGN+"/benchmark01"

def probe(client):
    code='''import json,os,shutil,psutil
from pathlib import Path
root=Path("/home/hzeng/project/MASKCO-Main")
load=psutil.cpu_percent(interval=1,percpu=True)
groups={}
for i in os.sched_getaffinity(0):
 p=Path(f"/sys/devices/system/cpu/cpu{i}/topology");k=tuple((p/f).read_text().strip() for f in ("physical_package_id","core_id"));groups.setdefault(k,[]).append(i)
print(json.dumps(dict(physical=len(groups),eligible=sum(max(load[i] for i in g)<40 for g in groups.values()),ram_gib=psutil.virtual_memory().available/2**30,disk_gib=shutil.disk_usage(root).free/2**30,prior_recent=json.loads((root/"_maskco_recent5_20261120/benchmark01/status.json").read_text())["cells"],prior_fleet_closed=json.loads((root/"_maskco_perishable_external_20261119/fleetscreen01/status.json").read_text())["end"] is not None)))'''
    info=json.loads(remote_python(client,code))
    if info["eligible"]<1 or info["ram_gib"]<5 or info["disk_gib"]<1 or not info["prior_fleet_closed"] or info["prior_recent"]!=540:
        raise ValueError(info)
    print(json.dumps(info),flush=True)
    (HERE/"resources.json").write_text(json.dumps(info,indent=2),encoding="utf8")
    return info

def main(action):
    if action not in ("probe","submit","status","follow","recover"):raise ValueError(action)
    with connect() as c,c.open_sftp() as s:
        if action=="probe":probe(c);return
        if action=="submit":
            acceptance=json.loads((HERE/"qualification.json").read_text(encoding="utf8"))
            if acceptance["status"]!="PASS" or acceptance["endpoints"]!=1440:raise ValueError(acceptance)
            probe(c)
            try:s.mkdir(CAMPAIGN)
            except OSError:s.stat(CAMPAIGN)
            s.mkdir(TARGET)
            s.put(str(HERE/"benchmark01.tar.gz"),TARGET+"/dispatch.tar.gz")
            command="cd "+TARGET+" && "+PYTHON+" -B unified_runner.py > controller.log 2>&1"
            print(transport.remote_exec(c,"cd "+TARGET+" && tar -xzf dispatch.tar.gz && "+PYTHON+
                  " -B stage_existing.py && tmux new-session -d -s maskco_unified_benchmark01 "+shlex.quote(command)),flush=True)
            print("submitted benchmark01",flush=True)
            return
        while True:
            with s.open(TARGET+"/status.json") as f:state=json.load(f)
            print(json.dumps({k:state[k] for k in ("complete","cells","total_cells","failed","end")}),flush=True)
            if action!="follow" or state["end"]:break
            time.sleep(45)
        if action in ("follow","recover"):
            if not state["end"]:raise RuntimeError("The original controller is still running")
            out=HERE/"results/benchmark01";out.mkdir(parents=True,exist_ok=True)
            transport.remote_exec(c,"cd "+TARGET+" && tar -czf receipts.tar.gz out status.json controller.log")
            s.get(TARGET+"/receipts.tar.gz",str(out/"receipts.tar.gz"))
            with tarfile.open(out/"receipts.tar.gz") as z:
                for m in z:
                    p=(out/m.name).resolve()
                    if not p.is_relative_to(out.resolve()):raise ValueError(m.name)
                    if m.isdir():p.mkdir(parents=True,exist_ok=True)
                    elif m.isfile():p.write_bytes(z.extractfile(m).read())
if __name__=="__main__":main(*sys.argv[1:])

