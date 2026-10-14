"""Seal completed r18, preserving five exact prior status-entry byte copies."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
BASE = ROOT / "research/review_3080_20261010/goal_round_20261014_r18_pair_shard"
DEL = ROOT / "research/review_3080_20261010/delivery/source_hash_sharded_root_v1"
REPORT = EXT / "docs/当前规划/决策与审计/双线第二十轮_内侧掩码机会与四分片源门_2026-10-14.md"
EVIDENCE = ROOT / "research/review_3080_20261010/reproduction/GOAL_ROUND_R18_PAIR_SHARD_EVIDENCE_20261014.json"
UPDATE = BASE / "UPDATE_IDENTITY.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepend(path, value):
    path.write_bytes((value + "\n\n").encode("utf-8") + path.read_bytes())


def main():
    pickup = json.loads((BASE / "PICKUP_RESULT.json").read_text(encoding="utf-8"))
    audit = json.loads((BASE / "INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))
    delivery = json.loads((DEL / "RESULT.json").read_text(encoding="utf-8"))
    deltas = [[p["delta_c0_vs_original"] for p in x["pairs"]] for x in pickup["rows"]]
    if deltas != [[0.0] * 5] * 2 or audit["checks"] != 81 or audit["failures"] or delivery["all_source_files_verified"]:
        raise RuntimeError("Observed result differs from stage report")
    entries = [EXT / "项目当前状态.md",
               EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
               EXT / "01_取货线/README.md", EXT / "02_配送线/README.md",
               EXT / "results/research_20261010_continuation_v2/WORK_STATUS.json"]
    backup_dir = BASE / "current_entries_before_goal_round_r18"
    if backup_dir.exists() or EVIDENCE.exists() or UPDATE.exists():
        raise FileExistsError("r18 seal already started; inspect rather than rerun")
    backup_dir.mkdir()
    prior = {}
    for i, path in enumerate(entries):
        backup = backup_dir / f"{i}_{path.name}"
        shutil.copy2(path, backup)
        if sha(path) != sha(backup):
            raise RuntimeError("Backup mismatch")
        prior[path.relative_to(ROOT).as_posix()] = {"sha256": sha(path), "backup": backup.relative_to(ROOT).as_posix()}
    sources = [BASE / n for n in ["PICKUP_REGISTRATION.json", "PICKUP_RESULT.json", "INDEPENDENT_REGISTRATION.json", "INDEPENDENT_AUDIT.json"]]
    sources += [DEL / n for n in ["REGISTRATION.json", "RESOURCE_PREFLIGHT.json", "RUN_ATTEMPT.json", "UPLOAD_ACCEPTANCE.json", "RUN_RESPONSE.json", "COLLECT_ATTEMPT.json", "RESULT.json", "RESULT.json.receipt.json", "CPU_CHILD_RESULT.json", "CPU_CHILD_RESULT.json.receipt.json", "SHARD_0.json", "SHARD_0.json.receipt.json", "SHARD_1.json.missing.json", "SHARD_2.json.missing.json", "SHARD_3.json", "SHARD_3.json.receipt.json", "measure_source_shards_v1.py"]]
    sources += [ROOT / "tools" / n for n in ["register_pickup_interior_pair_c0_sweep_r18.py", "build_pickup_interior_pair_c0_sweep_r18.py", "register_delivery_source_shards_r18.py", "transport_delivery_source_shards_r18.py", "register_pair_shard_audit_r18.py", "audit_pair_shard_r18.py"]]
    new_json(EVIDENCE, {
        "report": REPORT.relative_to(EXT).as_posix(), "report_sha256": sha(REPORT),
        "sources_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "pickup_orders": [69, 77], "pickup_masks_per_state": 5, "pickup_all_final_c0_deltas": deltas,
        "pickup_learning_attribution_qualified": False,
        "delivery_shards_done_by_8s": [delivery["receipts"][f"hash_{i}"]["completed_by_8s"] for i in range(4)],
        "delivery_full_source_verified": False, "delivery_cpu_done_by_8s": delivery["receipts"]["cpu"]["completed_by_8s"],
        "delivery_parent_elapsed_seconds": delivery["parent_elapsed_seconds"],
        "delivery_all_owned_children_reaped": all(r["reaped"] for r in delivery["receipts"].values()),
        "delivery_actual_model_request": False, "independent_checks": 81, "independent_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    })
    name = REPORT.name
    prepend(entries[0], f"> 2026-10-14 双线第二十轮：取货两个相邻旧状态各5组固定内侧双单掩码，10个最终C0效用差全0；配送3080四分片源哈希仅2/4于8秒内完成，其余按界回收。独立81/0；仍无学习归因、等耗时迭代和真实强动态，稳定在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。")
    prepend(entries[1], f"> 2026-10-14 第二十轮双线工程止损：取货旧状态10组局部掩码C0差全0；配送四分片仅2/4按时完成，独立81/0。不得继续按旧状态/合成哈希追PASS；见[{name}]({name})。")
    prepend(entries[2], f"> 2026-10-14 固定两旧状态各5个新内侧双单掩码的最终C0差均0；局部教师无改进幅度，未训练模型、未建立增益。见[第二十轮](../docs/当前规划/决策与审计/{name})。")
    prepend(entries[3], f"> 2026-10-14 3080四分片旧源哈希压力下2/4分片在8秒内完成、其余取消回收；不是完整源证书或实际模型请求。见[第二十轮](../docs/当前规划/决策与审计/{name})。")
    status = json.loads(entries[4].read_text(encoding="utf-8"))
    status["updated"] = datetime.now(timezone.utc).isoformat()
    status["latest_goal_round_20261014_r18_pair_shard"] = {
        "report": REPORT.relative_to(EXT).as_posix(), "evidence": EVIDENCE.relative_to(ROOT).as_posix(),
        "pickup_masks_per_state": 5, "pickup_all_final_c0_deltas_zero": True,
        "pickup_learning_attribution_qualified": False,
        "delivery_hash_shards_done_by_8s": [True, False, False, True],
        "delivery_full_source_close_or_request_qualified": False,
        "independent_checks": 81, "independent_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    }
    entries[4].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    new_json(UPDATE, {"prior_entries": prior,
                      "updated_entries_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in entries},
                      "report_sha256": sha(REPORT), "evidence_sha256": sha(EVIDENCE)})
    print(json.dumps({"evidence": EVIDENCE.relative_to(ROOT).as_posix(), "checks": 81, "stable": False}))


if __name__ == "__main__":
    main()
