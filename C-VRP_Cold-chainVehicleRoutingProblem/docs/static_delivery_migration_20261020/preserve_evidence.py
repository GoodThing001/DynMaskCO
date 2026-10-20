"""Copy selected local evidence into a verified archive; never move or delete it.

Administrative preservation only. Does not import models or run experiments.
Skips Windows reparse points, bytecode, shortcuts, and private/credential paths.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
OUT = EXT / "archive" / "transition_20261020"
TREES = [
    EXT / "01_取货线",
    EXT / "03_负结果线",
    EXT / "negative_result_paper",
    EXT / "results/i3_online_recovery_1009_v1",
    EXT / "results/i3_registered_1008_v5",
    ROOT / "research/review_3080_20261010/p0_i3_preservation_root_v1",
    EXT / "results/delivery_pilot_1008_v2",
    EXT / "results/delivery_pilot_1008_v1",
    EXT / "results/_q3_next_20261017",
    EXT / "results/q3_s3_float_outward_bound",
    EXT / "results/q3_s1_route_bracket",
    EXT / "results/q3_s2_online_cost",
    ROOT / "results/_q3_overnight",
]
FILES = [
    ROOT / name for name in (
        "AGENTS.md", "CLAUDE.md", "README.md", "工作区导航.md",
        "REPOSITORY_MAP.md", "EVIDENCE_MANIFEST.md",
    )
] + [EXT / name for name in (
    "项目当前状态.md", "00_项目导航.md", "README.md", "CLAUDE.md",
    "results/_m1b_q3/continuous_temperature_domain.json",
    "02_配送线/文档/结果与进度/配送路线重构pilot_正式裁决与失败机制_2026-10-08.md",
    "scripts/project_paths.py",
    "scripts/coldchain/coldchain_contract.py", "scripts/coldchain/coldchain_state.py",
    "scripts/delivery/delivery_contract.py", "scripts/delivery/delivery_evaluator.py",
    "scripts/delivery/delivery_baselines.py", "scripts/delivery/delivery_maskco.py",
    "scripts/delivery/run_delivery_pilot.py", "scripts/delivery/adjudicate_delivery_pilot.py",
)]


def reparse(path: Path) -> bool:
    return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def digest_stream(stream) -> str:
    h = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit("Preservation directory already exists; refuse overwrite")
    selected: set[Path] = set()
    excluded = []
    missing = []

    def consider(path: Path) -> None:
        relative = path.relative_to(ROOT).as_posix()
        reason = None
        if reparse(path):
            reason = "reparse_point"
        elif path.suffix.lower() in (".lnk", ".pyc", ".pyo"):
            reason = "shortcut_or_bytecode"
        elif any(part.lower() in ("private", ".git", "__pycache__") for part in path.parts):
            reason = "private_or_cache"
        elif any(term in path.name.lower() for term in ("secret", "credential", "token")):
            reason = "credential_name_excluded_without_reading"
        elif path.name == ".env" or path.suffix.lower() in (".pem", ".key", ".pfx"):
            reason = "credential_file_excluded_without_reading"
        if reason:
            excluded.append({"path": relative, "reason": reason})
        else:
            resolved = path.resolve()
            if not resolved.is_relative_to(ROOT.resolve()):
                raise RuntimeError("Archive source escaped workspace")
            selected.add(path)

    for tree in TREES:
        if not tree.exists():
            missing.append(tree.relative_to(ROOT).as_posix())
            continue
        if reparse(tree):
            raise RuntimeError("Selected root is a reparse point")
        for base, directories, filenames in os.walk(tree, followlinks=False):
            kept = []
            for name in directories:
                path = Path(base) / name
                if reparse(path) or name.lower() in ("private", ".git", "__pycache__"):
                    excluded.append({"path": path.relative_to(ROOT).as_posix(), "reason": "directory_excluded"})
                else:
                    kept.append(name)
            directories[:] = kept
            for name in filenames:
                consider(Path(base) / name)
    audit_docs = EXT / "docs/当前规划/决策与审计"
    for pattern in ("Q3*.md", "M1b_Q3*.md"):
        FILES.extend(audit_docs.glob(pattern))
    for path in FILES:
        if path.is_file():
            consider(path)
        else:
            missing.append(path.relative_to(ROOT).as_posix())

    OUT.mkdir(parents=True)
    archive = OUT / "pickup_negative_key_evidence.zip"
    rows = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(selected):
            with path.open("rb") as src:
                before = digest_stream(src)
            row = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": before}
            bundle.write(path, row["path"])
            with path.open("rb") as src:
                if digest_stream(src) != before:
                    raise RuntimeError("Source changed during preservation: " + row["path"])
            rows.append(row)
    with zipfile.ZipFile(archive) as bundle:
        for row in rows:
            with bundle.open(row["path"]) as item:
                if digest_stream(item) != row["sha256"]:
                    raise RuntimeError("Archive payload verification failed")
    with archive.open("rb") as src:
        archive_sha = digest_stream(src)
    manifest = {
        "created_local_date": "2026-10-20",
        "scope": "selected_key_evidence_not_full_training_or_workspace_backup",
        "sources_moved_or_deleted": False,
        "original_failure_verdicts_unchanged": True,
        "archive": archive.name, "archive_sha256": archive_sha,
        "archive_bytes": archive.stat().st_size,
        "files": rows, "excluded": excluded, "missing_requested_paths": missing,
        "validation": "all archived members SHA256 checked against unchanged source bytes",
        "external_assets": "Historical manifests may refer to data/weights outside this selection; not claimed self-contained.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "source_bytes": sum(r["bytes"] for r in rows),
                      "archive_bytes": archive.stat().st_size, "archive_sha256": archive_sha,
                      "excluded": len(excluded), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
