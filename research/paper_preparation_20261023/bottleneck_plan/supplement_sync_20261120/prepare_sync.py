"""Package the four-method paired replication using the existing audited panel."""
import ast
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent / "maskco_perishable_external_20261119"
RECENT = HERE.parent / "recent5_20261120"
METHODS = ("ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS", "PyVRP")
SEEDS = (33001, 33002, 33003)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def member(archive, name, data):
    raw = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    archive.addfile(info, io.BytesIO(raw))


def prepare():
    meta = read(RECENT / "metadata.json")
    old = read(RECENT / "matrix.json")
    audit = read(RECENT / "results/benchmark01/independent_audit.json")
    if audit["status"] != "PASS" or audit["endpoints"] != 540:
        raise ValueError("The existing panel must have its completed independent audit")
    cases = set(meta["cases"])
    expected = {(c, s) for c in cases for s in SEEDS}
    if (len(cases), len(meta["primary"]), len(meta["low_overlap"])) != (30, 27, 10):
        raise ValueError("Panel size differs from the existing comparison")
    if len(old["blocks"]) != 90 or {(b["case"], b["seed"]) for b in old["blocks"]} != expected:
        raise ValueError("Case/seed roster differs from the audited panel")
    blocks = [dict(case=b["case"], seed=b["seed"],
                   methods=list(METHODS[i % 4:] + METHODS[:i % 4]))
              for i, b in enumerate(old["blocks"])]
    positions = {m: [sum(b["methods"][p] == m for b in blocks) for p in range(4)]
                 for m in METHODS}
    if any(max(counts) - min(counts) > 1 for counts in positions.values()):
        raise ValueError("Method positions are not balanced")
    sources = {key: {} for key in ("cases", "derived_instances", "sources")}
    admission = dict(status="PASS", cases={})
    files = {}
    for folder in (EXTERNAL / "reserved_validation", EXTERNAL / "xtw_low_overlap"):
        catalogue, admitted = read(folder / "heldout_sources.json"), read(folder / "heldout_admission.json")
        if admitted["status"] != "PASS":
            raise ValueError("An existing case panel lacks admission")
        for case in sorted(set(catalogue["cases"]) & cases):
            if case in sources["cases"] or catalogue["cases"][case] != meta["cases"][case]:
                raise ValueError(f"Case metadata differs: {case}")
            for key in sources:
                sources[key][case] = catalogue[key][case]
            admission["cases"][case] = admitted["cases"][case]
            name = catalogue["derived_instances"][case]["file"]
            if name in files:
                raise ValueError(f"Two instances share an archive path: {name}")
            files[name] = folder / name
    if set(sources["cases"]) != cases:
        raise ValueError("The archive is missing registered cases")
    solver = EXTERNAL / "reserved_validation/solver"
    files.update({name: solver / name for name in (
        "pool_search.py", "masked_repair.cpp", "split_order.cpp", "block_split.py",
        "calibration.py", "calibrated.json")})
    files.update({"external_common.py": EXTERNAL / "independent.py",
                  "locked_method.json": EXTERNAL / "reserved_validation/locked_method.json",
                  "recent_worker.py": RECENT / "recent_worker.py",
                  "recent_or.py": RECENT / "recent_or.py",
                  "sync_worker.py": HERE / "sync_worker.py"})
    phase = dict(name="sync01", purpose="paired replication of the frozen cold-chain methods",
                 seconds=60, training_only=False, methods=list(METHODS), seeds=list(SEEDS),
                 blocks=blocks, total_cells=360, position_counts=positions,
                 online_core_seconds=21600, worker="sync_worker.py",
                 checkpoint="repeat02 f0_r0 maskco_base 1440")
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if name.endswith(".py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    write(HERE / "phase.json", phase)
    write(HERE / "metadata.json", meta)
    write(HERE / "heldout_sources.json", sources)
    write(HERE / "heldout_admission.json", admission)
    with tarfile.open(HERE / "sync01.tar.gz", "w:gz", compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name)
        for name, value in (("phase.json", phase), ("metadata.json", meta),
                            ("heldout_sources.json", sources), ("heldout_admission.json", admission)):
            member(archive, name, value)
    print(json.dumps(dict(package=str(HERE / "sync01.tar.gz"), cases=30,
                          blocks=90, cells=360, position_counts=positions)))


if __name__ == "__main__":
    prepare()
