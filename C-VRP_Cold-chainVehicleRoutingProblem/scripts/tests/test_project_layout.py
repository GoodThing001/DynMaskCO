"""P0-M blocking checks for the reorganized workspace layout."""

from __future__ import annotations

import ast
import importlib.util
from importlib.machinery import PathFinder
import json
import os
import subprocess
import sys
import tempfile
import platform
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
SCRIPTS_BOOTSTRAP = TEST_DIR.parent
if str(SCRIPTS_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_BOOTSTRAP))

from project_paths import (  # noqa: E402
    EXTENSION_ROOT,
    MASKCO_ROOT,
    SCRIPTS_ROOT,
    WORKSPACE_ROOT,
    configure_imports,
    validate_layout,
)


RESULTS: list[dict[str, object]] = []


def _record(name: str, ok: bool, detail: object) -> None:
    RESULTS.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _module_origin(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.origin:
        return str(Path(spec.origin).resolve())
    locations = list(spec.submodule_search_locations or ())
    return str(Path(locations[0]).resolve()) if locations else None


def _origin_on_path(module_name: str, search_path: Path) -> str | None:
    """Resolve a module on one explicit path without importing its package."""

    spec = PathFinder.find_spec(module_name, [str(search_path)])
    if spec is None:
        return None
    if spec.origin:
        return str(Path(spec.origin).resolve())
    locations = list(spec.submodule_search_locations or ())
    return str(Path(locations[0]).resolve()) if locations else None


def test_layout_contract() -> None:
    roots = validate_layout()
    ok = (
        Path(roots["workspace_root"]) == WORKSPACE_ROOT
        and Path(roots["maskco_root"]) == MASKCO_ROOT
        and Path(roots["extension_root"]) == EXTENSION_ROOT
        and MASKCO_ROOT.name == "MASKCO_code"
    )
    _record("layout_contract", ok, roots)


def test_cwd_independence() -> None:
    probe = (
        "import json,sys;"
        f"sys.path.insert(0,{str(SCRIPTS_ROOT)!r});"
        "import project_paths as p;"
        "print(json.dumps(p.validate_layout(),sort_keys=True))"
    )
    expected = json.dumps(validate_layout(), sort_keys=True)
    directories = (WORKSPACE_ROOT, EXTENSION_ROOT, Path(tempfile.gettempdir()).resolve())
    outputs: dict[str, str] = {}
    for directory in directories:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
        )
        outputs[str(directory)] = completed.stdout.strip()
        if completed.returncode != 0:
            outputs[str(directory)] = f"ERROR: {completed.stderr.strip()}"
    _record("cwd_independence", all(value == expected for value in outputs.values()), outputs)


def test_upstream_resolution() -> None:
    origins = {
        "models.CVRPModel": _origin_on_path("CVRPModel", MASKCO_ROOT / "models"),
        "training.TrainConfig": _origin_on_path("TrainConfig", MASKCO_ROOT / "training"),
        "helpers": _origin_on_path("helpers", MASKCO_ROOT),
        "modules.functional": _origin_on_path("functional", MASKCO_ROOT / "modules"),
        "decoding.utils": _origin_on_path("utils", MASKCO_ROOT / "decoding"),
    }
    prefix = os.path.normcase(str(MASKCO_ROOT.resolve()))
    ok = all(
        origin is not None and os.path.normcase(origin).startswith(prefix)
        for origin in origins.values()
    )
    _record("upstream_resolution", ok, origins)


def test_extension_resolution() -> None:
    configure_imports(
        ("models", "simulation", "evaluation"),
        include_extension_root=True,
        include_upstream_models=True,
    )
    modules = (
        "DynamicColdChainModel",
        "strict_online_env",
        "action_contract",
        "counterfactual_teacher",
        "authoritative_evaluator",
    )
    origins = {name: _module_origin(name) for name in modules}
    prefix = os.path.normcase(str(EXTENSION_ROOT.resolve()))
    ok = all(
        origin is not None and os.path.normcase(origin).startswith(prefix)
        for origin in origins.values()
    )
    _record("extension_resolution", ok, origins)


def test_no_legacy_workspace_root_bootstrap() -> None:
    offenders: list[str] = []
    exact_fragments = (
        "_MASKCO = os.path.dirname(_CVRPTW)",
        "'..', '..', '..'",
        '"..", "..", ".."',
    )
    for path in SCRIPTS_ROOT.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if any(fragment in text for fragment in exact_fragments):
            offenders.append(str(path.relative_to(EXTENSION_ROOT)))
    _record("no_legacy_workspace_root_bootstrap", not offenders, offenders)


def test_representative_entrypoints() -> None:
    entries = {
        "evaluation": EXTENSION_ROOT / "scripts" / "evaluation" / "run_hfr_gate_a.py",
        "decoding": EXTENSION_ROOT / "scripts" / "decoding" / "run_r1_5_model_utility.py",
        "simulation": EXTENSION_ROOT / "scripts" / "simulation" / "rolling_horizon.py",
        "data": EXTENSION_ROOT / "scripts" / "data" / "generate_coldchain_data.py",
    }
    outcomes: dict[str, dict[str, object]] = {}
    for name, script in entries.items():
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=WORKSPACE_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        outcomes[name] = {
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip()[-500:],
        }

    training_script = EXTENSION_ROOT / "scripts" / "training" / "train_dynamic_cc.py"
    training_stubbed = importlib.util.find_spec("tensorboardX") is None
    if training_stubbed:
        launcher = (
            "import runpy,sys,types;"
            "m=types.ModuleType('tensorboardX');"
            "m.SummaryWriter=type('SummaryWriter',(),{});"
            "sys.modules['tensorboardX']=m;"
            f"script={str(training_script)!r};"
            "sys.argv=[script,'--help'];"
            "runpy.run_path(script,run_name='__main__')"
        )
        command = [sys.executable, "-c", launcher]
    else:
        command = [sys.executable, str(training_script), "--help"]
    completed = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    outcomes["training"] = {
        "returncode": completed.returncode,
        "tensorboardX_stubbed_for_path_smoke": training_stubbed,
        "stderr": completed.stderr.strip()[-500:],
    }
    ok = all(int(item["returncode"]) == 0 for item in outcomes.values())
    _record("representative_entrypoints", ok, outcomes)


def test_python_syntax() -> None:
    failures: dict[str, str] = {}
    checked = 0
    for path in SCRIPTS_ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            checked += 1
        except (SyntaxError, UnicodeError) as exc:
            failures[str(path.relative_to(EXTENSION_ROOT))] = str(exc)
    _record("python_syntax", not failures, {"checked": checked, "failures": failures})


def main() -> None:
    print("P0-M project layout checks")
    tests = (
        test_layout_contract,
        test_cwd_independence,
        test_upstream_resolution,
        test_extension_resolution,
        test_no_legacy_workspace_root_bootstrap,
        test_representative_entrypoints,
        test_python_syntax,
    )
    for test in tests:
        try:
            test()
        except Exception as exc:  # keep a complete machine-readable audit
            _record(test.__name__, False, f"{type(exc).__name__}: {exc}")

    output_dir = EXTENSION_ROOT / "results" / "p0m"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "project_layout_tests.json"
    summary = {
        "schema": "p0m-project-layout-v1",
        "python": sys.version,
        "environment": {
            "platform": platform.platform(),
            "executable": sys.executable,
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "path_migration_files": sorted(
            str(path.relative_to(EXTENSION_ROOT))
            for path in SCRIPTS_ROOT.rglob("*.py")
            if "from project_paths import" in path.read_text(encoding="utf-8")
        ),
        "results": RESULTS,
        "passed": sum(item["status"] == "PASS" for item in RESULTS),
        "total": len(RESULTS),
    }
    artifact.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArtifact: {artifact}")
    print(f"Summary: {summary['passed']}/{summary['total']} PASS")
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
