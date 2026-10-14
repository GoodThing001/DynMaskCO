"""Build an immutable, artifact-only release candidate outside every original tree."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXT = Path(__file__).resolve().parents[2]
REPOSITORY = EXT.parent
PACKAGE = EXT / "negative_result_paper"
EXPECTED = EXT / "03_负结果线/发布候选/2026-10-09"
AUDIT = EXT / "03_负结果线/核查工件/论文封存与收口检查_2026-10-09.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_target(path: Path, occupied: bool | None = None) -> Path:
    resolved = path.resolve()
    if resolved != EXPECTED.resolve() or not resolved.is_relative_to(EXT.resolve()):
        raise ValueError("Output must be the exact new 03 line candidate directory")
    cursor = path.absolute()
    while cursor != EXT.resolve() and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("Candidate output cannot traverse a symbolic link")
        cursor = cursor.parent
    if resolved.exists() if occupied is None else occupied:
        raise FileExistsError("Existing candidate output cannot be overwritten")
    return resolved


def guard_checks() -> list[dict]:
    cases = [PACKAGE, REPOSITORY / "MASKCO_code", EXT / "results",
             EXPECTED.parent / "other", EXPECTED / "..", EXT.parent / "outside"]
    result = []
    for path in cases:
        try:
            validate_target(path, occupied=False)
        except ValueError:
            result.append({"case": str(path), "rejected": True})
        else:
            raise AssertionError("Unsafe output was accepted: " + str(path))
    try:
        validate_target(EXPECTED, occupied=True)
    except FileExistsError:
        result.append({"case": "already occupied exact candidate", "rejected": True})
    else:
        raise AssertionError("Existing output was accepted")
    validate_target(EXPECTED, occupied=False)
    return result


def imports(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names = set()
    relative = []
    dynamic = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.append({"level": node.level, "module": node.module})
            elif node.module:
                names.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
            dynamic.append(node.lineno)
    return {"modules": sorted(names), "relative_imports": relative, "dynamic_import_lines": dynamic}


def dependency_inventory(seeds: list[Path], pin: dict[str, Path], registered: dict[str, str]) -> dict:
    roots = [EXT / "scripts", REPOSITORY / "MASKCO_code"]
    query = subprocess.run(["rg", "--files", *(str(root) for root in roots)], check=True,
                           capture_output=True, text=True, encoding="utf-8")
    index = {}
    for line in query.stdout.splitlines():
        path = Path(line).resolve()
        if path.suffix != ".py":
            continue
        for root in roots:
            if path.is_relative_to(root):
                key = ".".join(path.relative_to(root).with_suffix("").parts)
                index.setdefault(key, []).append(path)
                index.setdefault(path.stem, []).append(path)
                break
    queue = list(seeds)
    visited = set()
    files = []
    external = set()
    unresolved = set()
    ambiguities = []
    while queue:
        path = queue.pop(0).resolve()
        if path in visited:
            continue
        visited.add(path)
        if len(visited) > 512:
            unresolved.add("static traversal limit reached")
            break
        analysis = imports(path)
        actual = sha(path)
        relative = path.relative_to(REPOSITORY).as_posix()
        expected = registered.get(path.relative_to(EXT).as_posix()) if path.is_relative_to(EXT) else None
        evidence = ("six-module archived pin" if path in pin.values() else
                    "s42 training identity SHA matched" if expected == actual else
                    "archived entry snapshot; complete original run identity not established" if path.is_relative_to(PACKAGE) else
                    "current available source only; no matched historical SHA evidence")
        files.append({"path": relative, "sha256": actual, "identity_evidence": evidence,
                      "registered_sha256": expected, "registered_sha_match": actual == expected if expected else None,
                      "ast": analysis})
        for name in analysis["modules"]:
            if name in pin:
                choices = [pin[name]]
            else:
                choices = list(dict.fromkeys(index.get(name, [])))
            if not choices:
                external.add(name)
            elif len(choices) > 1:
                ambiguities.append({"import": name, "choices": [str(choice) for choice in choices]})
            else:
                queue.extend(choices)
        if analysis["relative_imports"] or analysis["dynamic_import_lines"]:
            unresolved.add(relative + ": relative/dynamic imports need runtime origin validation")
    return {"method": "AST only; no project module imports or scientific calls", "files": files,
            "external_or_unresolved_import_names": sorted(external), "ambiguous_imports": ambiguities,
            "additional_gaps": sorted(unresolved), "fixed_state_full_closure_supported": False,
            "historical_closed_loop_supported": False,
            "boundary": "Inventory is not an import resolver: runtime sys.path edits and historical full module identities remain unverified. Current source is never substituted for an unrecorded historical dependency."}


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def build(output: Path) -> dict:
    guards = guard_checks()
    target = validate_target(output)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    baseline = audit["package_preservation_baseline"]
    source_map = {item["path"]: item["sha256"] for item in baseline["files"]}
    actual = {path.relative_to(PACKAGE).as_posix(): sha(path) for path in PACKAGE.rglob("*") if path.is_file()}
    if actual != source_map:
        raise ValueError("Original archived package changed; refusing to build")
    pins = {path.stem: path for path in sorted((PACKAGE / "release/v1_modules").glob("*.py"))}
    names = ("run_fixed_state_quality.py", "run_headroom_census.py", "run_step5_gate.py",
             "run_future_sampling_oracle.py", "make_figures.py")
    entries = [PACKAGE / "release/scripts" / name for name in names]
    identity = json.loads((EXT / "results/m0_scale/mpre_reinforce_s42_clean/training_identity.json").read_text(encoding="utf-8-sig"))
    analysis_entry = EXT / "scripts/evaluation/analyze_fixed_state.py"
    inventory = dependency_inventory(entries + list(pins.values()) + [analysis_entry], pins, identity["src_sha256"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    copied = []
    planned = [(path, "snapshots/entries/" + path.name) for path in entries]
    planned += [(path, "snapshots/v1_modules/" + path.name) for path in pins.values()]
    for folder in ("09_unified_v1", "10_confirm_c1"):
        planned += [(path, "snapshots/results/" + folder + "/" + path.name)
                    for path in sorted((PACKAGE / "release/results" / folder).glob("*.json"))]
    for source, destination in planned:
        relative = source.relative_to(PACKAGE).as_posix()
        if sha(source) != source_map[relative]:
            raise ValueError("Source changed after snapshot: " + relative)
        write_new(target / destination, source.read_bytes())
        copied.append({"source_package_relative": relative, "destination": destination,
                       "sha256": source_map[relative], "size": source.stat().st_size})
    manifest = {"created_date": "2026-10-09", "repository_root": str(REPOSITORY.resolve()),
                "original_package_relative": PACKAGE.relative_to(REPOSITORY).as_posix(),
                "original_package_baseline": baseline, "copied_files": copied,
                "assets_read_only": [{"repository_relative": item["path"], "sha256": item["expected_sha256"]}
                                     for item in audit["identity_list_checks"]],
                "supported_execution": "verify_static_artifacts.py only; SHA/AST and archived JSON arithmetic",
                "model_or_simulation_execution_supported": False,
                "guard_checks": guards}
    for name, value in (("candidate_manifest.json", manifest), ("dependency_inventory.json", inventory)):
        write_new(target / name, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    verifier = Path(__file__).with_name("verify_static_artifacts.py")
    write_new(target / verifier.name, verifier.read_bytes())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-X", "utf8", str(target / verifier.name), "--candidate", str(target)],
                         capture_output=True, text=True, encoding="utf-8", env=env)
    if run.returncode:
        raise RuntimeError("Artifact-only validation failed: " + run.stdout + run.stderr)
    result = json.loads(run.stdout)
    write_new(target / "validation_record.json", (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"candidate": str(target), "copied_files": len(copied),
            "copied_bytes": sum(item["size"] for item in copied), "ast_files": len(inventory["files"]),
            "all_artifact_checks_pass": result["all_checks_pass"], "guards_rejected": len(guards),
            "runtime_closure_supported": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPECTED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({"guard_checks": guard_checks()}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
