import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    cfg=json.loads((HERE/"opprobe01.json").read_text(encoding="utf8"))
    cfg.update(worker="view_frame_probe.py",purpose="development input-axis reconstruction",block_timeout_s=480)
    cfg["files"]+=["view_frame_probe.py"]
    blocks=cfg["blocks"]
    for name,items,n in (("viewqual02",blocks[:1],1),("viewprobe01",blocks,2)):
        (HERE/(name+".json")).write_text(json.dumps(dict(cfg,blocks=items,groups_per_state=n),indent=2),encoding="utf8")
if __name__=="__main__":main()
