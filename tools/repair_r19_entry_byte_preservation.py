"""Preserve pre-existing entry bytes after round-19 prefix/status insertion."""
import hashlib
import json
from pathlib import Path
from seal_goal_round_r19 import ROOT, OUT, FILES, PREFIXES


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    before = OUT / "entry_r19_format_before"
    before.mkdir(exist_ok=False)
    rows = json.loads((OUT / "UPDATE_IDENTITY.json").read_text(encoding="utf-8"))
    current = [p.read_bytes() for p in FILES]
    for i, b in enumerate(current):
        (before / f"{i}_{FILES[i].name}").write_bytes(b)
    for i, p in enumerate(FILES[:4]):
        old = (ROOT / rows[i]["backup"]).read_bytes()
        assert sha(old) == rows[i]["old_sha256"]
        p.write_bytes(PREFIXES[i].encode("utf-8") + old)
    old = (ROOT / rows[4]["backup"]).read_bytes()
    assert old.endswith(b"  }\r\n}\r\n") and sha(old) == rows[4]["old_sha256"]
    status = json.loads(current[4].decode("utf-8"))
    key = "latest_goal_round_20261014_r19_science_release"
    value = json.dumps(status[key], ensure_ascii=False, indent=2)
    value = ("  " + json.dumps(key, ensure_ascii=False) + ": " + value).replace("\n", "\r\n")
    FILES[4].write_bytes(old[:-8] + b"  },\r\n" + value.encode("utf-8") + b"\r\n}\r\n")
    assert json.loads(FILES[4].read_text(encoding="utf-8"))[key] == status[key]
    for row, p in zip(rows, FILES):
        row["new_sha256"] = sha(p.read_bytes())
    (OUT / "UPDATE_IDENTITY.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ENTRY_FORMAT_REPAIR.json").write_text(json.dumps({
        "reason": "retain exact previous entry bytes instead of normalizing historical mixed line endings",
        "prior_r19_files_backup": str(before.relative_to(ROOT)),
        "original_entry_bytes_preserved": True,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"entries": len(rows), "original_bytes_preserved": True}))


if __name__ == "__main__":
    main()
