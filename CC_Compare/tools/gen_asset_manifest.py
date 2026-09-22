"""Generate CC_Compare/ASSET_MANIFEST.json — source tree + asset identity for freeze.

Re-runnable: `python tools/gen_asset_manifest.py` from CC_Compare/.
Every file in each method dir is covered by exactly one of:
  - source_tree_sha256  (text/code files)
  - assets[]            (individual weight/checkpoint files, sha256 + size)
  - data_assets[]       (aggregated data/artifact dirs, tree sha256 + counts)
  - binary_artifacts    (aggregate of remaining binary files by extension)
"""
import hashlib
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

WEIGHT_EXTS = {".pt", ".pth", ".ckpt"}
BINARY_EXTS = {".pkl", ".npz", ".npy", ".zip", ".tar", ".tar.gz", ".pb", ".onnx", ".so"}

# method -> asset_dirs (individual weights), data_dirs (aggregated)
SPEC = {
    "RRNCO": {"asset_dirs": ["checkpoints"], "data_dirs": []},
    "CaDA": {"asset_dirs": ["50/result", "100/result"], "data_dirs": ["data"]},
    "RouteFinder": {"asset_dirs": ["checkpoints"], "data_dirs": []},
    "MVMoE": {"asset_dirs": ["pretrained"], "data_dirs": ["data"]},
    "POMO": {"asset_dirs": [], "data_dirs": [], "auto_weights": True},
    "AttentionModel": {"asset_dirs": ["pretrained"], "data_dirs": []},
    "Omni-VRP": {"asset_dirs": ["pretrained"], "data_dirs": ["data"]},
    "CO-enriched-ML": {"asset_dirs": ["training/models"],
                       "data_dirs": ["evaluation/results"]},
    "DeepACO": {"asset_dirs": ["pretrained"], "data_dirs": ["data"]},
    "SGBS": {"asset_dirs": ["CVRP/1_pre_trained_model", "TSP/1_pre_trained_model"],
             "data_dirs": [], "auto_weights": True},
    "Sym-NCO": {"asset_dirs": [], "data_dirs": ["Sym-NCO-AM/data", "Sym-NCO-POMO/data"],
                "auto_weights": True},
    "Learn-Improvement-Heuristics": {"asset_dirs": [], "data_dirs": [], "auto_weights": True},
    "Learning-to-Delegate": {"asset_dirs": [], "data_dirs": []},
    "PyVRP": {"asset_dirs": [], "data_dirs": []},
    "PIP-constraint": {"asset_dirs": [], "data_dirs": ["data"], "auto_weights": True},
    "MAPT": {"asset_dirs": [], "data_dirs": []},
}

SOURCES = {
    "RRNCO": "服务器已有副本（上游 ai4co/real-routing-nco，HF checkpoints）",
    "CaDA": "HF datasets/Goodyee/CaDA checkpoint.zip（经 hf-mirror，CRC 校验通过，解压后 zip 已删除；HF dataset sha 06bb906b8e7f3d71306a49d2929daaa1d35ed9a4）",
    "RouteFinder": "HF ai4co/routefinder（经 hf-mirror）",
    "MVMoE": "上游仓库自带",
    "POMO": "上游仓库自带",
    "AttentionModel": "上游仓库自带",
    "Omni-VRP": "上游仓库自带",
    "CO-enriched-ML": "上游仓库自带",
    "DeepACO": "上游仓库自带",
    "SGBS": "上游仓库自带",
    "Sym-NCO": "上游仓库自带",
    "Learn-Improvement-Heuristics": "上游仓库自带",
}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(entries):
    """entries: list of (relpath, sha256) -> combined tree sha256."""
    h = hashlib.sha256()
    for rel, fh in sorted(entries):
        h.update(rel.encode("utf-8", "surrogateescape"))
        h.update(b"\x00")
        h.update(fh.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


def norm(p):
    return p.replace("\\", "/")


def parse_pins():
    pins = {}
    with open(os.path.join(ROOT, "REPO_PINS.md"), encoding="utf-8") as f:
        for line in f:
            parts = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(parts) == 6 and parts[0] not in ("文件夹", "---", ""):
                if not parts[0].startswith("-"):
                    name, repo, commit, branch, lic, method = parts
                    if name not in ("文件夹",) and commit != "commit":
                        pins[name] = {"upstream_repo": repo, "upstream_commit": commit,
                                      "upstream_branch": branch, "license": lic,
                                      "download": method}
    return pins


def main():
    pins = parse_pins()
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
        "note": ("source_tree_sha256 在本地工作副本（Windows checkout）计算，"
                 "文本文件按工作树字节原样哈希；所有文件必被 source/assets/data_assets/"
                 "binary_artifacts 之一覆盖"),
        "global_exclusions": {"binary_extensions": sorted(BINARY_EXTS),
                              "weight_extensions": sorted(WEIGHT_EXTS)},
        "methods": [],
    }

    for name in sorted(SPEC):
        spec = SPEC[name]
        d = os.path.join(ROOT, name)
        if not os.path.isdir(d):
            continue
        asset_dirs = [norm(x) for x in spec["asset_dirs"]]
        data_dirs = [norm(x) for x in spec["data_dirs"]]
        auto = spec.get("auto_weights", False)

        source_entries = []
        assets = []
        data_asset_entries = {}
        bin_entries = []
        weight_paths = []

        for dirpath, dirnames, filenames in os.walk(d):
            rel_dir = norm(os.path.relpath(dirpath, ROOT))
            for fn in filenames:
                rel = norm(os.path.relpath(os.path.join(dirpath, fn), ROOT))
                ext = os.path.splitext(fn)[1].lower()
                if fn.lower().endswith(".tar.gz"):
                    ext = ".tar.gz"
                in_asset_dir = any(rel.startswith(f"{name}/{a}/") for a in asset_dirs)
                in_data_dir = any(rel.startswith(f"{name}/{x}/") for x in data_dirs)
                if in_asset_dir:
                    weight_paths.append(rel)
                elif in_data_dir:
                    data_asset_entries.setdefault(rel_dir, []).append((rel, file_sha256(os.path.join(dirpath, fn))))
                elif ext in WEIGHT_EXTS and auto:
                    weight_paths.append(rel)
                elif ext in BINARY_EXTS:
                    bin_entries.append((rel, file_sha256(os.path.join(dirpath, fn))))
                else:
                    source_entries.append((rel, file_sha256(os.path.join(dirpath, fn))))

        for rel in sorted(weight_paths):
            p = os.path.join(ROOT, rel)
            assets.append({"path": rel, "sha256": file_sha256(p),
                           "size_bytes": os.path.getsize(p),
                           "source": SOURCES.get(name, "上游仓库自带")})

        data_assets = []
        for rel_dir, entries in sorted(data_asset_entries.items()):
            total = sum(os.path.getsize(os.path.join(ROOT, r)) for r, _ in entries)
            data_assets.append({"path": rel_dir, "tree_sha256": tree_hash(entries),
                                "n_files": len(entries), "total_bytes": total,
                                "source": SOURCES.get(name, "上游仓库自带")})

        m = {
            "method": name,
            "upstream_repo": pins.get(name, {}).get("upstream_repo", "?"),
            "upstream_commit": pins.get(name, {}).get("upstream_commit", "?"),
            "upstream_branch": pins.get(name, {}).get("upstream_branch", "?"),
            "source_tree_sha256": tree_hash(source_entries),
            "source_n_files": len(source_entries),
            "assets": assets,
            "data_assets": data_assets,
            "binary_artifacts": {
                "tree_sha256": tree_hash(bin_entries),
                "n_files": len(bin_entries),
            } if bin_entries else None,
        }
        manifest["methods"].append(m)
        print(f"{name}: source={m['source_n_files']} files, "
              f"assets={len(assets)}, data_dirs={len(data_assets)}, "
              f"bin={len(bin_entries)}")

    out = os.path.join(ROOT, "ASSET_MANIFEST.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
