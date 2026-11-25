"""Preview dispatch, or freeze the validation package after both development passes."""
import argparse
import json
from pathlib import Path
import tarfile

from future_dispatch import FLEET, RECENT
from future_matrix import build

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
STUDIES = DEV.parent
RECENT_STUDY = STUDIES / "recent5_20261120"
PHASE = "fleetfuture01"
STAGES = {"fleetscreen01": (132, "RUN_FULL_DEVELOPMENT"), "fleetdev01": (486, "LOCK_FOR_VALIDATION")}
FLEET_FILES = ("runner.py", "external_common.py", "locked_method.json", "pool_search.py",
               "masked_repair.cpp", "split_order.cpp", "block_split.py", "calibration.py", "calibrated.json",
               "fleet_timed_worker.py", "fleet_timed_pool.py", "fleet_pricing.py")
RECENT_FILES = ("recent_worker.py", "recent_or.py", "neural_methods.py", "neural_mask_oracle.py", "other_methods.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interpreters(methods):
    base = read(STUDIES / "maskco_cross_dataset_20261119/matrix.json")["bindings"]
    target = {}
    for method in methods:
        if method in FLEET | {"ORDER_SPLIT", "ORDER_SPLIT_D", "PyVRP"}:
            target[method] = base["SLOT"]["python"]
        elif method in RECENT:
            target[method] = "/home/hzeng/envs/cc_compare/bin/python"
        else:
            target[method] = base[method]["python"]
    return target


def gates():
    passed = {}
    for phase, (count, decision) in STAGES.items():
        folder = DEV / "results" / phase
        result, audit = read(folder / "analysis.json"), read(folder / "independent_audit.json")
        if not (result["phase"] == phase and result["status"] == "COMPLETE" and result["passed"] is True
                and result["decision"] == decision and result["audit"] == audit
                and audit["status"] == "PASS" and audit["endpoints"] == count):
            raise ValueError(f"Required development pass missing for {phase}")
        passed[phase] = {"decision": decision, "audited_endpoints": count}
    return passed


def retain(path, data):
    if path.is_file():
        if path.read_bytes() != data:
            raise ValueError(f"Existing frozen file differs: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def freeze(schedule, runtime):
    evidence = gates()
    receipt = read(RECENT_STUDY / "results/benchmark01/independent_audit.json")
    if receipt["status"] != "PASS" or receipt["endpoints"] != 540:
        raise ValueError("Recent comparator batch incomplete")
    sources = {}
    with tarfile.open(DEV / "fleetdev01_inputs.tar.gz") as development:
        for name in FLEET_FILES:
            sources[name] = development.extractfile(name).read()
    with tarfile.open(RECENT_STUDY / "benchmark01.tar.gz") as recent:
        for name in RECENT_FILES:
            sources[name] = recent.extractfile(name).read()
        for shared in ("external_common.py", "pool_search.py", "block_split.py", "masked_repair.cpp",
                       "split_order.cpp", "calibration.py", "calibrated.json"):
            if sources[shared] != recent.extractfile(shared).read():
                raise ValueError(f"The two tested paths differ in shared dependency {shared}")
    original = json.loads(sources["locked_method.json"])
    if original["candidate"] != "ORDER_SPLIT" or original["matched"] != "ORDER_SPLIT_D":
        raise ValueError("The registered previous-method comparator does not match the recovered source")
    sources["future_dispatch.py"] = (HERE / "future_dispatch.py").read_bytes()
    sources["future_lock.json"] = encoded(dict(decision="LOCK_FOR_VALIDATION", arms=sorted(FLEET),
                                            development=evidence, contemporary_reference="benchmark01"))
    frozen = HERE / "solver_lock"
    frozen.mkdir(exist_ok=True)
    for name, contents in sources.items():
        retain(frozen / name, contents)
    catalog = read(HERE / "heldout_sources.json")
    plan = dict(schedule, worker="future_dispatch.py", training_only=False, block_timeout_s=4200,
                interpreters=runtime, file_sources={n: "solver_lock/" + n for n in sources},
                files=[*sources, "heldout_sources.json", "heldout_admission.json",
                       *(x["file"] for x in catalog["derived_instances"].values())])
    retain(HERE / (PHASE + ".json"), encoded(plan))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true", help="assemble from the scored development archive after both passes")
    args = cli.parse_args()
    schedule = build()
    runtime = interpreters(schedule["methods"])
    if args.freeze:
        freeze(schedule, runtime)
    print(json.dumps(dict(action="FROZEN" if args.freeze else "PREVIEW", methods=len(runtime),
                          blocks=len(schedule["blocks"]), endpoints=sum(len(r["methods"]) for r in schedule["blocks"]),
                          interpreters=runtime), ensure_ascii=False))


if __name__ == "__main__":
    main()
