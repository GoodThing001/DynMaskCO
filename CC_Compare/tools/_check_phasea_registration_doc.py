"""Structural self-check for the P1-A Phase A attribution registration doc.

Read-only. Verifies:
  * every `§N` / `§N.M` cross-reference resolves to an existing heading;
  * every markdown table has a consistent column count;
  * required closure-situation strings are present and forbidden strings absent;
  * preserved sibling rulings are byte-identical to their recorded SHA-256.

Exit code 0 = clean.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

try:  # Windows consoles default to GBK; keep the checker's own output printable
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

DOC = pathlib.Path("CC_Compare/P1A_RoleC_方案1_PhaseA_归因登记_2026-10-14.md")

PRESERVED = {
    "CC_Compare/P1A_RoleC_方案1_PhaseA_归因登记_v1_reviewed_2026-10-14.md":
        "6ac795b223d9a3154a669a074a00697e533a506350db8537f868b198b2620db9",
    "CC_Compare/P1A_RoleC_收束裁定_2026-10-14.md":
        "5f3ac98c579759e442fc8f09eb3d83f297b3889edd708cc78caa2b60675f5b09",
    "CC_Compare/P1A_T2判据分层裁定_2026-10-14.md":
        "e053f5ed448139c9267fc34155df2490ab49ce0b7e11590622bbf7d9372d8e03",
}

REQUIRED = (
    "ENGINEERING_TIMING_QUALIFIED",
    "ONLINE_EXECUTABILITY_NOT_ESTABLISHED",
    "SCIENTIFIC_BATCH_NOT_RUN",
    "p1a-rolec-phasea-v1",
    "p1a-rolec-v1",
    "inconclusive_attribution_incomplete",
    "inconclusive_record_gap",
    "P9_measured",
    "P9_residual",
)

# forbidden: claims that were ruled out of bounds
FORBIDDEN = (
    "P1A_SCIENTIFIC_GATE_FAIL",
    "Role C proven worse",
)

# these appear only inside an explicit prohibition sentence
FORBIDDEN_UNLESS_NEGATED = (
    "不可通过计算优化解决",
    "必要工作量",
)


def _ncols(row: str) -> int:
    """Count table cell separators, ignoring escaped pipes (``\\|``)."""
    return len(re.findall(r"(?<!\\)\|", row))


# byte containers named with a raw sha256 in the sealed closure ruling
BYTE_CONTAINERS = {
    "CC_Compare/P1A_资源计时_报告_2026-10-14.json":
        "df91419855fedb1c2aab2b98d11474ed20915b9538b24aff653650f3c16f22cf",
    "CC_Compare/P1A_资源计时_监督器_2026-10-14.json":
        "bc29105b3607e0fd5eeb1f8be7c233a8498a0ad915b6bf6ec47c05d0b33ab668",
    "CC_Compare/P1A_资源门登记.json":
        "6d0e57c6d46bf46a61005b8f4d27edd99d8641dd47675e508c39c155f1a635bf",
}


def _byte_container_warnings() -> list[str]:
    """Warn when a registered byte container's HEAD blob != worktree bytes.

    The `.gitattributes` policy (see CC_Compare/.gitattributes header) is that
    ``blob == worktree bytes`` so the registered raw sha256 survives a fresh
    checkout on any platform. A mismatch means the registered value is only
    reproducible on one line-ending setting.
    """
    import subprocess

    out: list[str] = []
    for rel, expected in BYTE_CONTAINERS.items():
        p = pathlib.Path(rel)
        if not p.is_file():
            continue
        wt = hashlib.sha256(p.read_bytes()).hexdigest()
        try:
            blob = subprocess.run(
                ["git", "cat-file", "blob", f"HEAD:{rel}"],
                capture_output=True, check=True).stdout
        except Exception as exc:  # pragma: no cover
            out.append(f"could not read HEAD blob for {rel}: {exc}")
            continue
        bsha = hashlib.sha256(blob).hexdigest()
        if wt != expected:
            out.append(f"{rel}: worktree bytes != registered sha256 ({wt})")
        if bsha != wt:
            out.append(
                f"{rel}: HEAD blob != worktree bytes "
                f"(blob {bsha[:12]}..., worktree {wt[:12]}...) "
                f"-> missing .gitattributes entry; registered sha256 is not "
                f"reproducible from a fresh checkout"
            )
    return out


def main() -> int:
    problems: list[str] = []
    text = DOC.read_text(encoding="utf-8")

    headings = re.findall(r"^(##+)\s+(\d+(?:\.\d+)*|[0-9]+\.\d+)", text, re.M)
    tops = {h[1].split(".")[0] for h in headings}
    refs = sorted(set(re.findall(r"§(\d+(?:\.\d+)*)", text)))
    for r in refs:
        if r.split(".")[0] not in tops:
            problems.append(f"dangling section ref §{r}")

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        row = lines[i]
        if row.startswith("|") and i + 1 < len(lines):
            sep = re.sub(r"(?<!\\)\|", "", lines[i + 1]).replace(" ", "").replace("\t", "")
            if sep and set(sep) <= set("-:"):
                ncol = _ncols(row)
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    if _ncols(lines[j]) != ncol:
                        problems.append(
                            f"table col mismatch at line {j + 1}: "
                            f"{_ncols(lines[j])} != {ncol}"
                        )
                    j += 1
                i = j
                continue
        i += 1

    for s in REQUIRED:
        if s not in text:
            problems.append(f"missing required string: {s}")

    for s in FORBIDDEN:
        if s in text:
            problems.append(f"forbidden string present: {s}")

    for s in FORBIDDEN_UNLESS_NEGATED:
        for m in re.finditer(re.escape(s), text):
            ctx = text[max(0, m.start() - 160): m.end() + 160]
            negated = any(k in ctx for k in ("不得", "禁止", "废除", "REV1", "标签"))
            if not negated:
                problems.append(f"unnegated forbidden claim: {s}")

    for rel, expected in PRESERVED.items():
        p = pathlib.Path(rel)
        if not p.is_file():
            problems.append(f"preserved file missing: {rel}")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != expected:
            problems.append(f"SHA drift in {rel}: {got} != {expected}")

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    warn = _byte_container_warnings()
    for w in warn:
        print(f"WARN: {w}")
    print(f"OK: {DOC} ({len(text.encode('utf-8'))} bytes, "
          f"{len(lines)} lines, {len(refs)} section refs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
