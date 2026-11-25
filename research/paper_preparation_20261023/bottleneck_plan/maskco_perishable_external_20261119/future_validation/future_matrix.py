"""Build the preregistered 18-method schedule from case identifiers only."""
import argparse
import json
from collections import Counter
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def build():
    base = load("comparison_design.json")
    extension = load("近期五法扩展预登记.json")
    catalog = load(base["panel"])
    cases = catalog["cases"]
    admission = load(base["admission"])
    if admission["status"] != "PASS" or set(admission["cases"]) != set(cases):
        raise ValueError("Registered inputs have not passed admission")
    if any(not (HERE / catalog["derived_instances"][case]["file"]).is_file() for case in cases):
        raise ValueError("An admitted projection is missing")
    for source, expected in base["sources"].items():
        groups = Counter(row["geometry"] for row in cases.values() if row["dataset"] == source)
        if len(groups) != expected["groups"] or sum(groups.values()) != expected["cases"]:
            raise ValueError(f"Registered source counts differ for {source}")
    binding = extension["candidate_binding"]
    primary = [binding.get(name, name) for name in base["methods"]]
    added = extension["added_methods"]
    methods = primary + added
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicated comparator")
    seeds = base["seeds"]
    case_seeds = [(case, seed) for case in sorted(cases) for seed in seeds]
    for names, key in ((primary, "original"), (added, "addition"), (methods, "combined")):
        cells = len(case_seeds) * len(names)
        if dict(methods=len(names), cells=cells, single_core_search_seconds=cells * base["seconds_per_method_per_block"]) != extension[key]:
            raise ValueError(f"Budget mismatch for {key}")
    rng = random.Random(20261120)
    rng.shuffle(case_seeds)
    order = methods.copy()
    rng.shuffle(order)
    blocks = []
    for index, (case, seed) in enumerate(case_seeds):
        offset = index % len(order)
        blocks.append(dict(case=case, seed=seed, methods=order[offset:] + order[:offset]))
    return dict(seconds=base["seconds_per_method_per_block"], methods=methods, original_methods=primary,
                added_methods=added, blocks=blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="write the case/seed/method schedule")
    args = parser.parse_args()
    schedule = build()
    if args.write is not None:
        args.write.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(schedule['blocks'])} blocks, {len(schedule['methods'])} methods, "
          f"{sum(len(block['methods']) for block in schedule['blocks'])} cells")
