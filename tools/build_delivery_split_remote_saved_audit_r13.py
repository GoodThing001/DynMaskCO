"""Create a distinct, source-bound remote saved-score audit package."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"
NEW = ROOT / "research/review_3080_20261010/delivery/split_final_saved_audit_root_v2"
SCRIPT = NEW / "audit_delivery_split_saved_remote_v2.py"
REG = NEW / "AUDIT_REGISTRATION.json"
EVIDENCE = r"C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\split_final_timed_execution_root_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    NEW.mkdir(exist_ok=False)
    source = (ROOT / "tools/audit_delivery_split_saved_r13.py").read_text(encoding="utf-8")
    source = source.replace('BASE = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"', 'BASE = Path(__file__).resolve().parent')
    source = source.replace('ARCHIVE = BASE / "downloaded"', 'ARCHIVE = None')
    source = source.replace('REG = BASE / "SAVED_SPLIT_AUDIT_REGISTRATION.json"', 'REG = BASE / "AUDIT_REGISTRATION.json"')
    source = source.replace('OUT = BASE / "INDEPENDENT_SAVED_SPLIT_AUDIT.json"', 'OUT = BASE / "INDEPENDENT_SAVED_SPLIT_AUDIT.json"')
    source = source.replace('    reg = json.loads(REG.read_text(encoding="utf-8"))',
                            '    global ARCHIVE\n    reg = json.loads(REG.read_text(encoding="utf-8"))\n    ARCHIVE = Path(reg["remote_evidence_root"])\n    for relative, expected in reg["input_sha256"].items():\n        if sha(ARCHIVE / relative) != expected:\n            raise RuntimeError("Registered remote evidence differs: " + relative)')
    source = source.replace('    assert reg["download_manifest_sha256"] == sha(BASE / "DOWNLOAD_MANIFEST.json")\n', '')
    source = source.replace('    assert reg["expected_records"] == 50', '    assert reg["expected_records"] == 50 and reg["runtime"] == "3080 actual JAX CPU"')
    SCRIPT.write_text(source, encoding="utf-8")
    local = OLD / "downloaded"
    inputs = {str(p.relative_to(local)).replace("\\", "/"): sha(p)
              for arm in ("pretrained_initial", "random888_initial")
              for p in sorted((local / f"{arm}_fault0_worker/request").glob("[0-2][0-9].npz"))}
    for relative in ("INDEPENDENT_TIMED_REQUEST_AUDIT.json", "pretrained_initial_fault0_worker/INITIAL_PARAMETERS.json"):
        inputs[relative] = sha(local / relative)
    old_reg = json.loads((OLD / "SAVED_SPLIT_AUDIT_REGISTRATION.json").read_text(encoding="utf-8"))
    reg = dict(old_reg)
    reg.pop("download_manifest_sha256")
    reg["script_sha256"] = sha(SCRIPT)
    reg["input_sha256"] = inputs
    reg["remote_evidence_root"] = EVIDENCE
    reg["runtime"] = "3080 actual JAX CPU"
    reg["local_attempt_failed_count"] = 50
    reg["no_request_rerun"] = True
    REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(json.dumps({"script_sha256": sha(SCRIPT), "registration_sha256": sha(REG), "input_files": len(inputs)}))


if __name__ == "__main__":
    main()
