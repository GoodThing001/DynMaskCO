"""Deliver the local research deliverables into the Git index, explicitly.

Why this exists
---------------
`research/` is git-ignored by design: it holds 4,000+ MB of run outputs, per-cell
endpoints and archives that must not be pushed.  But the *documents* that state
the current project result (the project manual, the unified-comparison report,
the locked ORDER_SPLIT implementation) live inside that ignored tree, so they
never reach GitHub and the published repository drifts behind the manual.

This tool closes that gap with an explicit allowlist manifest instead of a
wholesale `git add`:

  * every published path comes from a `roots[].include` glob;
  * `EXCLUDE` wins over any include, so raw per-cell outputs, renders, logs,
    archives, weights and byte-code can never leak in by accident;
  * a file that would exceed `MAX_FILE_MB`, or whose path looks like a
    credential, aborts the run instead of being silently skipped;
  * the default mode is a dry run that only prints what it would do.

It never commits and never pushes.  Commit and push stay explicit, per
AGENTS.md rule 8/9.

Usage
-----
    python -B -X utf8 tools/git_deliver.py                # dry run, prints table
    python -B -X utf8 tools/git_deliver.py --apply         # git add -f the list
    python -B -X utf8 tools/git_deliver.py --check         # non-zero if drift
    python -B -X utf8 tools/git_deliver.py --print-manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAX_FILE_MB = 8.0
MAX_TOTAL_MB = 60.0
CHUNK = 200

# Paths that may never be published, no matter what an include glob says.
EXCLUDE = (
    "**/out/**", "**/outputs/**", "**/cells/**", "**/node_modules/**",
    "**/__pycache__/**", "**/manual_render/**", "**/script_render/**",
    "**/*.log", "**/*.zip", "**/*.npz", "**/*.npy", "**/*.pyc",
    "**/*.png.tmp", "**/originals.tar.gz",
)

# A path that matches any of these aborts the whole run.
SECRET = re.compile(
    r"(?i)(credential|secret|\.env$|token|password|passwd|"
    r"\.pem$|\.key$|\.pfx$|id_rsa|\.vscode/|sftp\.json|private_key)"
)

MANIFEST = {
    "groups": [
        {
            "name": "manual_20261121",
            "note": "完整模型与项目手册（2026-11-21）+ 汇报稿",
            "root": "research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261121",
            "include": ("README.md", "build/*.md", "build/*.json", "build/figures/*.png",
                        "figures/*.png", "output/*.md", "output/*.pdf", "output/*.docx"),
        },
        {
            "name": "manual_20261120",
            "note": "上一版项目手册，保留版本链",
            "root": "research/paper_preparation_20261023/bottleneck_plan/model_documentation_20261120",
            "include": ("README.md", "build/*.md", "build/figures/*.png",
                        "figures/*.png", "output/*.md", "output/*.pdf", "output/*.docx"),
        },
        {
            "name": "presentation_20261121",
            "note": "21 日汇报的取数脚本与证据快照",
            "root": "research/paper_preparation_20261023/bottleneck_plan/presentation_20261121",
            "include": ("build/*.py", "build/*.md", "build/*.json", "output/*.md", "output/*.pdf"),
        },
        {
            "name": "unified_external_20261120",
            "note": "最新统一 16 法比较：报告、端点表、配对表、界与脚本",
            "root": "research/paper_preparation_20261023/bottleneck_plan/unified_external_20261120",
            "include": ("*.py", "*.md", "*.json", "*.csv",
                        "**/RESULTS.md", "**/endpoints.csv", "**/paired*.csv",
                        "**/summary*.json", "**/independent_audit.json",
                        "**/*.py", "**/*.md", "**/*.csv"),
        },
        {
            "name": "order_split_impl",
            "note": "锁版 ORDER_SPLIT 实现、Split 求解器与锁版配置",
            "root": "research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119",
            "include": ("reserved_validation/*.py", "reserved_validation/*.json",
                        "reserved_validation/solver/*", "reserved_validation/harness/*",
                        "*.py", "*.md", "*.json",
                        "**/RESULTS.md", "**/locked_method.json", "**/endpoints.csv",
                        "**/paired*.csv", "**/summary*.json", "**/independent_audit.json",
                        "**/*.py"),
        },
        {
            "name": "followup_reports_20261119_21",
            "note": "同期的探针、校热图、跨数据集、距离插位、近五法扩展与补充同步报告",
            "root": "research/paper_preparation_20261023/bottleneck_plan",
            "include": ("maskco_followup_dev_20261120/*.py",
                        "maskco_followup_dev_20261120/*.md",
                        "maskco_followup_dev_20261120/**/RESULTS.md",
                        "maskco_followup_dev_20261120/**/analysis.json",
                        "maskco_followup_dev_20261120/**/paired.csv",
                        "supplement_sync_20261120/*.py",
                        "supplement_sync_20261120/*.md",
                        "supplement_sync_20261120/*.json",
                        "supplement_sync_20261120/**/endpoints.csv",
                        "supplement_sync_20261120/**/independent_audit.json",
                        "recent5_20261120/**/RESULTS.md",
                        "recent5_20261120/**/endpoints.csv",
                        "recent5_20261120/**/paired*.csv",
                        "maskco_cross_dataset_20261119/**/RESULTS.md",
                        "maskco_cross_dataset_20261119/**/*.csv",
                        "maskco_distance_slot_20261119/**/RESULTS.md",
                        "maskco_distance_slot_20261119/**/*.csv",
                        "maskco_heatmap_calibration_20261119/**/RESULTS.md",
                        "maskco_heatmap_calibration_20261119/**/*.csv"),
        },
    ],
}


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def excluded(rel: str) -> bool:
    return any(fnmatch(rel, pattern) for pattern in EXCLUDE)


def collect() -> tuple[dict[str, list[Path]], list[str]]:
    """Return {group: [paths]} plus a list of human-readable guard failures."""
    found: dict[str, list[Path]] = {}
    problems: list[str] = []
    seen: set[str] = set()
    for group in MANIFEST["groups"]:
        base = ROOT / group["root"]
        if not base.is_dir():
            problems.append(f"missing root: {group['root']}")
            continue
        paths: list[Path] = []
        for pattern in group["include"]:
            for path in sorted(base.glob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(ROOT).as_posix()
                if rel in seen or excluded(rel):
                    continue
                seen.add(rel)
                paths.append(path)
        found[group["name"]] = paths
    return found, problems


def report(found: dict[str, list[Path]], problems: list[str]) -> tuple[int, float, int]:
    total_bytes = 0
    total_files = 0
    for group in MANIFEST["groups"]:
        paths = found.get(group["name"], [])
        size = sum(p.stat().st_size for p in paths)
        total_bytes += size
        total_files += len(paths)
        print(f"{group['name']:<34} files={len(paths):<6} {size / 1024 / 1024:8.2f} MB  {group['note']}")
    print("-" * 96)
    print(f"{'TOTAL':<34} files={total_files:<6} {total_bytes / 1024 / 1024:8.2f} MB")
    return total_files, total_bytes / 1024 / 1024, 0


def guard(found: dict[str, list[Path]]) -> list[str]:
    problems: list[str] = []
    for group, paths in found.items():
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            if SECRET.search(rel):
                problems.append(f"secret-looking path in {group}: {rel}")
            size_mb = path.stat().st_size / 1024 / 1024
            if size_mb > MAX_FILE_MB:
                problems.append(f"oversize file in {group}: {rel} ({size_mb:.1f} MB > {MAX_FILE_MB})")
    return problems


def apply(found: dict[str, list[Path]]) -> None:
    paths = [p.relative_to(ROOT).as_posix() for group in found.values() for p in group]
    for start in range(0, len(paths), CHUNK):
        chunk = paths[start:start + CHUNK]
        git("add", "-f", "--", *chunk)
    print(f"staged {len(paths)} paths with git add -f (chunks of {CHUNK})")
    # Do not pass the whole path list back on one command line: Windows caps the
    # command line well below what 500+ paths need (WinError 206).
    staged = [line for line in git("diff", "--cached", "--name-only").splitlines() if line.strip()]
    print(f"index now holds {len(staged)} staged paths")
    for line in staged[:10]:
        print("  " + line)
    if len(staged) > 10:
        print(f"  ... and {len(staged) - 10} more")


def check(found: dict[str, list[Path]]) -> int:
    paths = [p.relative_to(ROOT).as_posix() for group in found.values() for p in group]
    missing: list[str] = []
    for start in range(0, len(paths), CHUNK):
        chunk = paths[start:start + CHUNK]
        out = subprocess.run(["git", "ls-files", "--error-unmatch", "--", *chunk],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            missing.extend(line for line in out.stderr.splitlines() if line.strip())
    if missing:
        print(f"DRIFT: {len(missing)} manifest paths are not tracked")
        for line in missing[:40]:
            print("  " + line)
        return 1
    print(f"IN SYNC: all {len(paths)} manifest paths are tracked")
    digests = {}
    for group in MANIFEST["groups"]:
        for path in found.get(group["name"], []):
            if path.suffix in (".md", ".json", ".py"):
                rel = path.relative_to(ROOT).as_posix()
                digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(json.dumps({"documents": digests}, ensure_ascii=False, indent=2, sort_keys=True)[:4000])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="git add -f the manifest")
    mode.add_argument("--check", action="store_true", help="fail if the manifest drifted from HEAD")
    mode.add_argument("--print-manifest", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args()

    if args.print_manifest:
        print(json.dumps(MANIFEST, ensure_ascii=False, indent=2))
        return 0

    found, problems = collect()
    files, total_mb, _ = report(found, problems)
    problems += guard(found)
    if total_mb > MAX_TOTAL_MB:
        problems.append(f"total {total_mb:.1f} MB exceeds MAX_TOTAL_MB={MAX_TOTAL_MB}")
    if problems:
        print("\nGUARD FAILURES")
        for line in problems:
            print("  " + line)
        return 2
    if files == 0:
        print("\nnothing to deliver")
        return 2
    if args.check:
        return check(found)
    if args.apply:
        apply(found)
        return 0
    print("\ndry run: pass --apply to stage, then commit and push explicitly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
