"""Read-only validation of the B paper's static candidate; never imports project code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Unsafe relative artifact path: " + relative)
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its declared root")
    return path


def validate(candidate: Path) -> dict:
    candidate = candidate.resolve()
    manifest = json.loads((candidate / "candidate_manifest.json").read_text(encoding="utf-8"))
    repository = Path(manifest["repository_root"]).resolve()
    package = bounded(repository, manifest["original_package_relative"])
    checks = []

    for item in manifest["copied_files"]:
        destination = bounded(candidate, item["destination"])
        source = bounded(package, item["source_package_relative"])
        checks.append({"kind": "copied_artifact", "path": item["destination"],
                       "match": sha(destination) == item["sha256"] == sha(source)})
        if destination.suffix == ".py":
            ast.parse(destination.read_text(encoding="utf-8-sig"), filename=str(destination))

    for item in manifest["assets_read_only"]:
        path = bounded(repository, item["repository_relative"])
        checks.append({"kind": "original_asset_sha", "path": item["repository_relative"],
                       "match": path.is_file() and sha(path) == item["sha256"]})

    baseline = manifest["original_package_baseline"]
    expected = {item["path"]: item["sha256"] for item in baseline["files"]}
    actual = {path.relative_to(package).as_posix(): sha(path)
              for path in package.rglob("*") if path.is_file()}
    checks.append({"kind": "entire_original_package", "file_count": len(actual),
                   "match": actual == expected})

    loop = {}
    for folder in ("09_unified_v1", "10_confirm_c1"):
        for path in sorted((candidate / "snapshots/results" / folder).glob("*.summary.json")):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = data["rows"]
            unique = len({row["inst"] for row in rows})
            means = {}
            for key, arm in (("R_vs_Mtrained", "R"), ("Mpre_vs_Mtrained", "Mpre")):
                observed = sum(row["Mtrained_J"] - row[arm + "_J"] for row in rows) / len(rows)
                match = math.isclose(observed, data[key + "_mean"], abs_tol=1e-12)
                checks.append({"kind": "paired_mean", "path": path.name, "contrast": key,
                               "mean_from_rows": observed, "reported_mean": data[key + "_mean"],
                               "match": match})
                means[key] = {"mean": observed, "archived_ci": data[key + "_ci"]}
            checks.append({"kind": "complete_unique_rows", "path": path.name,
                           "match": unique == len(rows) == data["n"] == 16})
            loop[path.name] = means

    c1 = {}
    for seed in (42, 43, 44):
        path = candidate / "snapshots/results/10_confirm_c1" / ("c1_s%d.analysis.json" % seed)
        c1[seed] = json.loads(path.read_text(encoding="utf-8-sig"))["accepted_paired"]
    a = all(c1[seed]["Mtrained_minus_R"]["ci_hi"] >= 0 for seed in c1)
    b = (all(c1[seed]["Mtrained_minus_Mpre"]["ci_hi"] < 0 for seed in (42, 43))
         and c1[44]["Mtrained_minus_Mpre"]["ci_lo"] <= 0 <= c1[44]["Mtrained_minus_Mpre"]["ci_hi"])
    c = all(loop["c1_step5_s%d.summary.json" % seed]["R_vs_Mtrained"]["archived_ci"][0] <= 0
            <= loop["c1_step5_s%d.summary.json" % seed]["R_vs_Mtrained"]["archived_ci"][1]
            and loop["c1_step5_s%d.summary.json" % seed]["Mpre_vs_Mtrained"]["archived_ci"][1] < 0
            for seed in (42, 43))
    checks.append({"kind": "original_c1_joint_failure", "match": a and not b and not c})
    return {"scope": "SHA, syntax AST and existing JSON row means only",
            "project_imports_executed": False, "models_loaded": False,
            "scientific_experiments_run": False, "ci_regenerated": False,
            "all_checks_pass": all(check["match"] for check in checks), "checks": checks,
            "c1_decision_from_archived_ci": {"a": a, "b": b, "c": c, "joint_pass": a and b and c},
            "supported": ["archived artifact identity", "paired means from existing rows", "C1 original decision", "syntax AST"],
            "unsupported": ["historical full dependency closure", "model inference", "from-zero training", "formal online replay"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = validate(args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
