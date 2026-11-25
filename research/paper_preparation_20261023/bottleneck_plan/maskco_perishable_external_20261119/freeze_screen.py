"""Bind the full development run to the exact programs and parameters used in screening."""
import json
from pathlib import Path
import sys
import tarfile
HERE=Path(__file__).resolve().parent
def bind(screen,full):
    manifest=HERE/(full+".json")
    cfg=json.loads(manifest.read_text(encoding="utf8"))
    folder=HERE/"development_lock"/screen
    folder.mkdir(parents=True,exist_ok=True)
    with tarfile.open(HERE/(screen+"_inputs.tar.gz")) as arc:
        for name in cfg["files"]:
            path=folder/name
            path.write_bytes(arc.extractfile(name).read())
    cfg["file_sources"]={name:(folder/name).relative_to(HERE).as_posix() for name in cfg["files"]}
    manifest.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf8")
if __name__=="__main__":bind(*sys.argv[1:])
