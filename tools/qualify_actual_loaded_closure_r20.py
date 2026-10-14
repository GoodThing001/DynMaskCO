"""One-shot component fault injection; does not run a model or old request."""
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

from actual_loaded_closure_candidate_r20 import capture, compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg = json.loads((OUT / "REGISTRATION.json").read_text(encoding="utf-8"))
    assert reg["scope"] == "component_fault_injection_only_no_delivery_request"
    assert reg["candidate_sha256"] == sha(ROOT / "tools/actual_loaded_closure_candidate_r20.py")
    assert reg["runner_sha256"] == sha(__file__)
    tree = ast.parse((ROOT / "tools/actual_loaded_closure_candidate_r20.py").read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("rglob", "walk") for n in ast.walk(tree))
    checks = {}
    with tempfile.TemporaryDirectory(prefix="maskco_r20_") as directory:
        directory = Path(directory)
        explicit = directory / "explicit.py"
        explicit.write_text("VALUE = 1\n", encoding="utf-8")
        t0 = time.perf_counter()
        start = capture([explicit])
        t1 = time.perf_counter()
        same = capture([explicit])
        t2 = time.perf_counter()
        checks["unchanged_same_process"] = compare(start, same)["qualified"]
        explicit.write_text("VALUE = 2\n", encoding="utf-8")
        changed = compare(start, capture([explicit]))
        checks["changed_explicit_fails"] = not changed["qualified"] and str(explicit.resolve()) in changed["changed_files"]
        explicit.write_text("VALUE = 1\n", encoding="utf-8")
        module_path = directory / "new_lazy_module.py"
        module_path.write_text("VALUE = 7\n", encoding="utf-8")
        baseline = capture([explicit])
        spec = importlib.util.spec_from_file_location("maskco_r20_new_lazy_module", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        added = compare(baseline, capture([explicit]))
        checks["new_lazy_python_dependency_fails"] = not added["qualified"] and str(module_path.resolve()) in added["new_files"]
        del sys.modules[spec.name]
        try:
            capture([directory / "missing.py"])
            checks["missing_explicit_fails"] = False
        except FileNotFoundError:
            checks["missing_explicit_fails"] = True
    checks["no_tree_scan"] = True
    report = {
        "checks": checks,
        "pass_count": sum(checks.values()),
        "fail_count": sum(not x for x in checks.values()),
        "sample_initial_capture_seconds": t1 - t0,
        "sample_repeat_capture_seconds": t2 - t1,
        "sample_initial_files": len(start["file_sha256"]),
        "model_loaded": False,
        "remote_executed": False,
        "scientific_full_request_qualified": False,
        "scientific_family3_started": False,
    }
    with (OUT / "RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
