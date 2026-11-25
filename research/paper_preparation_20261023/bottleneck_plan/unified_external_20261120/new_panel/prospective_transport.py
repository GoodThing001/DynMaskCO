"""Submit once and recover the same frozen prospective comparison on the reference host."""
import json
import inspect
from pathlib import Path
import shlex
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(STUDY))
from unified_transport import PROJECT, PYTHON, CAMPAIGN, connect, remote_python, transport
from unified_runner import available

TARGET=CAMPAIGN+"/prospective01"
RESULTS=STUDY/"results"/"prospective01"


def resources(client):
    code=f"""import json, os, psutil
from pathlib import Path
{inspect.getsource(available)}
p=Path({CAMPAIGN!r})/'benchmark01'
state=json.loads((p/'status.json').read_text())
excluded=set(state['cores']) if state['end'] is None else set()
cores,n,ram=available(200,excluded)
print(json.dumps(dict(physical=n,peer_cores=list(sorted(excluded)),eligible=cores,cap=max(0,min(18,len(cores)-4)),ram_gib=ram,prior_cells=state['cells'],target_exists=Path({TARGET!r}).exists())))"""
    return json.loads(remote_python(client,code))


def main(action):
    if action not in {"probe","submit","status","follow","recover"}:raise ValueError(action)
    with connect() as c,c.open_sftp() as s:
        if action in ("probe","submit"):
            info=resources(c)
            print(json.dumps(info),flush=True)
            (HERE/"prospective_host.json").write_text(json.dumps(info,indent=2),encoding="utf8")
            if action=="probe":return
            qual=json.loads((HERE/"prospective_preflight.json").read_text(encoding="utf8"))
            if qual["status"]!="PASS" or qual["endpoints"]!=1152:raise ValueError(qual)
            if info["cap"]<1 or info["target_exists"]:raise RuntimeError("Unavailable new phase or pre-existing dispatch")
            s.mkdir(TARGET)
            s.put(str(HERE/"prospective01.tar.gz"),TARGET+"/dispatch.tar.gz")
            run="cd "+TARGET+" && "+PYTHON+" -B unified_runner.py > controller.log 2>&1"
            command="cd "+TARGET+" && tar -xzf dispatch.tar.gz && "+PYTHON+" -B stage_prospective.py && tmux new-session -d -s maskco_unified_prospective01 "+shlex.quote(run)
            print(transport.remote_exec(c,command),flush=True)
            print("submitted prospective01",flush=True)
            return
        while True:
            with s.open(TARGET+"/status.json") as f:state=json.load(f)
            print(json.dumps({k:state[k] for k in ("cores","cells","complete","total_cells","failed","end")}),flush=True)
            if action!="follow" or state["end"]:break
            time.sleep(45)
        if action in ("follow","recover"):
            if not state["end"]:raise RuntimeError("Only recover the completed original phase")
            RESULTS.mkdir(parents=True,exist_ok=True)
            transport.remote_exec(c,"cd "+TARGET+" && tar -czf receipts.tar.gz out status.json controller.log resource_admission.json")
            s.get(TARGET+"/receipts.tar.gz",str(RESULTS/"receipts.tar.gz"))
            with tarfile.open(RESULTS/"receipts.tar.gz") as z:
                for m in z:
                    p=(RESULTS/m.name).resolve()
                    if not p.is_relative_to(RESULTS.resolve()):raise ValueError(m.name)
                    if m.isdir():p.mkdir(parents=True,exist_ok=True)
                    elif m.isfile():p.write_bytes(z.extractfile(m).read())


if __name__=="__main__":main(*sys.argv[1:])
