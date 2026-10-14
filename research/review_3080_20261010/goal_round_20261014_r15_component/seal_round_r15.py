from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TREE = ROUND.parent
WORKSPACE = TREE.parents[1]
EXT = WORKSPACE / "C-VRP_Cold-chainVehicleRoutingProblem"
DELIVERY = TREE / "delivery/source_manifest_timing_root_v1"
REPORT = EXT / "docs/当前规划/决策与审计/双线第十七轮_纯距离认证与源哈希耗时_2026-10-14.md"
EVIDENCE = TREE / "reproduction/GOAL_ROUND_R15_DISTANCE_C0_SOURCE_HASH_EVIDENCE_20261014.json"
STATUS = EXT / "results/research_20261010_continuation_v2/WORK_STATUS.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def put_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    audit = read(ROUND / "INDEPENDENT_R15_AUDIT_V2.json")
    pr = read(ROUND / "PICKUP_DISTANCE_REPAIR_RESULT.json")
    dr = read(DELIVERY / "RESULT.json")
    assert audit["checks"] == 28 and audit["failures"] == 0 and audit["saved_only"]
    assert read(ROUND / "INDEPENDENT_R15_V1_FAILURE.json")["exit_code"] == 1
    assert pr["pure_distance_candidate"] == {"0": [77, 40, 25, 32, 30, 69], "1": [], "2": []}
    assert dr["file_count"] == 8903 and dr["mismatch_count"] == 0
    assert dr["hash_elapsed_seconds"] > max(audit["prior_stage6_remaining_seconds"])
    sources = [
        ROUND / "PICKUP_DISTANCE_REPAIR_REGISTRATION.json", ROUND / "PICKUP_DISTANCE_REPAIR_RESULT.json",
        ROUND / "INDEPENDENT_R15_REGISTRATION.json", ROUND / "INDEPENDENT_R15_V1_FAILURE.json",
        ROUND / "INDEPENDENT_R15_V2_REGISTRATION.json", ROUND / "INDEPENDENT_R15_AUDIT_V2.json",
        DELIVERY / "REGISTRATION.json", DELIVERY / "RESOURCE_PREFLIGHT.json",
        DELIVERY / "UPLOAD_ACCEPTANCE.json", DELIVERY / "RUN_ATTEMPT.json",
        DELIVERY / "RUN_RESPONSE.json", DELIVERY / "RESULT.json",
        DELIVERY / "COLLECT_ATTEMPT.json", DELIVERY / "COLLECT_RECEIPT.json",
        DELIVERY / "measure_manifest_hash_time_v1.py",
        WORKSPACE / "tools/register_pickup_distance_repair_r15.py",
        WORKSPACE / "tools/run_pickup_distance_repair_r15.py",
        WORKSPACE / "tools/register_delivery_manifest_timing_r15.py",
        WORKSPACE / "tools/transport_delivery_manifest_timing_r15.py",
        WORKSPACE / "tools/register_dual_distance_hash_audit_r15.py",
        WORKSPACE / "tools/audit_dual_distance_hash_r15.py",
        WORKSPACE / "tools/register_dual_distance_hash_audit_r15_v2.py",
        WORKSPACE / "tools/audit_dual_distance_hash_r15_v2.py",
    ]
    assert REPORT.is_file() and all(x.is_file() for x in sources)
    put_new(EVIDENCE, {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "report_sha256": sha(REPORT),
        "sources_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in sources},
        "pickup_distance_candidate_equal_original": True,
        "pickup_model_minus_distance_utility": audit["pickup_model_minus_distance_utility"],
        "pickup_formal_equal_time_control": False, "pickup_learning_attribution_qualified": False,
        "delivery_hash_files": dr["file_count"], "delivery_hash_bytes": dr["bytes_hashed"],
        "delivery_hash_mismatches": dr["mismatch_count"],
        "delivery_isolated_hash_seconds": dr["hash_elapsed_seconds"],
        "delivery_prior_stage6_remaining_seconds": audit["prior_stage6_remaining_seconds"],
        "delivery_full_source_close_gate_measured": False,
        "delivery_full_request_budget_qualified": False,
        "original_independent_v1_exit_preserved": 1,
        "independent_saved_checks": audit["checks"], "independent_saved_failures": audit["failures"],
        "scientific_family3_started": False, "stable_online_gain": False,
    })
    entries = [EXT / "项目当前状态.md",
               EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
               EXT / "01_取货线/README.md", EXT / "02_配送线/README.md", STATUS]
    backup = ROUND / "current_entries_before_goal_round_r15"
    backup.mkdir(exist_ok=False)
    before = {}
    for i, entry in enumerate(entries):
        dest = backup / f"{i}_{entry.name}"
        shutil.copyfile(entry, dest)
        before[str(entry.relative_to(WORKSPACE)).replace("\\", "/")] = {
            "sha256": sha(entry), "backup": str(dest.relative_to(TREE)).replace("\\", "/")}
    name = REPORT.name
    notes = [
        f"> 2026-10-14 双线第十七轮：取货旧公开状态两单纯距离+C0修复回到原计划，模型候选效用低0.068202，学习归因仍缺；配送3080隔离哈希8903文件用5.618649秒，高于旧stage6余量，但完整源关闭与8+2请求未合格。独立恢复核查28/0，原核查程序失败保留；稳定核心在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。\n\n",
        f"> 2026-10-14 双线第十七轮：取货纯距离+C0旧状态反事实回原计划，配送源清单单次逐哈希5.618649秒；完整动态与预算门仍缺，科学第3族未启。见[记录](../决策与审计/{name})。\n\n",
        f"> 2026-10-14 旧订单77掩码25/30的纯距离C0修复逐字回原计划，效用289.861188；已训/随机同选模型候选效用289.792986，非正式等耗时或在线归因。独立恢复核查28/0。见[记录](../docs/当前规划/决策与审计/{name})。\n\n",
        f"> 2026-10-14 3080只读哈希8903旧源文件5.618649秒，超过旧两请求stage6余量；非完整发布门实测，不升级8+2资格。见[记录](../docs/当前规划/决策与审计/{name})。\n\n",
    ]
    for entry, note in zip(entries[:-1], notes):
        entry.write_text(note + entry.read_text(encoding="utf-8"), encoding="utf-8")
    status = read(STATUS)
    status["latest_goal_round_20261014_r15_distance_hash"] = {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "evidence": str(EVIDENCE.relative_to(WORKSPACE)).replace("\\", "/"),
        "pickup_model_minus_distance_utility": audit["pickup_model_minus_distance_utility"],
        "pickup_equal_time_and_learning_attribution_qualified": False,
        "delivery_isolated_hash_seconds": dr["hash_elapsed_seconds"],
        "delivery_full_source_close_gate_measured": False,
        "delivery_full_request_budget_qualified": False,
        "independent_v1_failure_preserved": True,
        "independent_saved_checks": 28, "independent_saved_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    put_new(ROUND / "UPDATE_IDENTITY.json", {
        "prior_entries": before,
        "updated_entries_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in entries},
        "report_sha256": sha(REPORT), "evidence_sha256": sha(EVIDENCE)})


if __name__ == "__main__":
    main()
