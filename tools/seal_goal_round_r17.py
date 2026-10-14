"""Seal the already executed r17 engineering stage; never modify prior evidence."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
ROUND = ROOT / "research/review_3080_20261010/goal_round_20261014_r17_teacher_dualcore"
REPORT = EXT / "docs/当前规划/决策与审计/双线第十九轮_双状态C0教师与双核源哈希边界_2026-10-14.md"
EVIDENCE = ROOT / "research/review_3080_20261010/reproduction/GOAL_ROUND_R17_TWO_STATE_C0_TEACHER_DUALCORE_EVIDENCE_20261014.json"
UPDATE = ROUND / "UPDATE_IDENTITY.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put_new(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepend(path, line):
    data = path.read_bytes()
    path.write_bytes((line + "\n\n").encode("utf-8") + data)


def main():
    entries = [
        EXT / "项目当前状态.md",
        EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
        EXT / "01_取货线/README.md",
        EXT / "02_配送线/README.md",
        EXT / "results/research_20261010_continuation_v2/WORK_STATUS.json",
    ]
    backups = ROUND / "current_entries_before_goal_round_r17"
    if backups.exists() or EVIDENCE.exists() or UPDATE.exists():
        raise FileExistsError("r17 seal already started; inspect before any retry")
    backups.mkdir(parents=True)
    prior = {}
    for index, path in enumerate(entries):
        backup = backups / f"{index}_{path.name}"
        shutil.copy2(path, backup)
        if sha(path) != sha(backup):
            raise RuntimeError("backup mismatch: " + str(path))
        prior[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "sha256": sha(path), "backup": str(backup.relative_to(ROOT)).replace("\\", "/")
        }

    pickup = json.loads((ROUND / "PICKUP_TEACHER_RESULT.json").read_text(encoding="utf-8"))
    audit = json.loads((ROUND / "INDEPENDENT_R17_AUDIT.json").read_text(encoding="utf-8"))
    delivery_dir = ROOT / "research/review_3080_20261010/delivery/source_overlap_dualcore_root_v1"
    delivery = json.loads((delivery_dir / "RESULT.json").read_text(encoding="utf-8"))
    if audit["checks"] != 37 or audit["failures"] != 0 or len(pickup["rows"]) != 2:
        raise RuntimeError("unexpected pickup/audit status")
    if delivery["receipts"]["hash"]["completed_by_8s"] or not delivery["receipts"]["hash"]["reaped"]:
        raise RuntimeError("unexpected delivery hash status")

    source_rel = [
        "research/review_3080_20261010/goal_round_20261014_r17_teacher_dualcore/" + name
        for name in ["PICKUP_TEACHER_REGISTRATION.json", "PICKUP_TEACHER_RESULT.json", "INDEPENDENT_R17_REGISTRATION.json", "INDEPENDENT_R17_AUDIT.json"]
    ]
    source_rel += [
        "research/review_3080_20261010/delivery/source_overlap_dualcore_root_v1/" + name
        for name in ["REGISTRATION.json", "RESOURCE_PREFLIGHT.json", "RUN_ATTEMPT.json", "UPLOAD_ACCEPTANCE.json", "RUN_RESPONSE.json", "COLLECT_ATTEMPT.json", "RESULT.json", "RESULT.json.receipt.json", "CPU_CHILD_RESULT.json", "CPU_CHILD_RESULT.json.receipt.json", "HASH_CHILD_RESULT.json.missing.json", "measure_overlap_dualcore_v1.py"]
    ]
    source_rel += ["tools/" + name for name in [
        "register_pickup_two_state_teacher_r17.py", "build_pickup_two_state_c0_teacher_r17.py",
        "register_dual_teacher_dualcore_audit_r17.py", "audit_dual_teacher_dualcore_r17.py",
        "register_delivery_overlap_dualcore_r17.py", "transport_delivery_overlap_dualcore_r17.py",
    ]]
    for rel in source_rel:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(rel)
    put_new(EVIDENCE, {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "report_sha256": sha(REPORT),
        "sources_sha256": {rel: sha(ROOT / rel) for rel in source_rel},
        "pickup_orders": [r["current_order"] for r in pickup["rows"]],
        "pickup_candidate_counts_each": [[len(s["candidates"]) for s in r["steps"]] for r in pickup["rows"]],
        "pickup_teacher_actions_each": [[s["selected_action"] for s in r["steps"]] for r in pickup["rows"]],
        "pickup_final_c0_utilities": [r["final_certificate"]["evaluation"]["utility"] for r in pickup["rows"]],
        "pickup_new_science_or_training": False,
        "delivery_hash_completed_by_8s": False,
        "delivery_cpu_completed_by_8s": True,
        "delivery_parent_elapsed_seconds": delivery["parent_elapsed_seconds"],
        "delivery_all_owned_children_reaped": True,
        "delivery_actual_model_request_run": False,
        "delivery_full_request_budget_qualified": False,
        "independent_checks": 37,
        "independent_failures": 0,
        "scientific_family3_started": False,
        "stable_online_gain": False,
    })

    name = REPORT.name
    prepend(entries[0], f"> 2026-10-14 双线第十九轮：取货旧订单69/77各自双掩码两步C0教师，8+5可行候选、同动作且回原计划；配送3080双核合成压力下源哈希8秒未完成、已回收。独立37/0；学习归因、等耗时迭代与真实强动态仍缺，稳定核心在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。")
    prepend(entries[1], f"> 2026-10-14 第十九轮双线工程阶段已收口：取货旧相邻双状态C0教师8+5动作/状态，配送双核8秒源哈希未完成，独立37/0；不能升级科学效应。见[{name}]({name})。")
    prepend(entries[2], f"> 2026-10-14 双状态双订单C0教师：旧订单69/77各8+5可行候选，两步皆(30,0,3)→(25,0,2)，最终各返原计划；非学习增益。见[第十九轮](../docs/当前规划/决策与审计/{name})。")
    prepend(entries[3], f"> 2026-10-14 3080双核固定合成负载下8903文件哈希8秒未完成，监督者8.0327044秒取消回收；非完整模型请求，源/参数发布前门仍缺。见[第十九轮](../docs/当前规划/决策与审计/{name})。")
    status = json.loads(entries[4].read_text(encoding="utf-8"))
    status["updated"] = datetime.now(timezone.utc).isoformat()
    status["latest_goal_round_20261014_r17_teacher_dualcore"] = {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "evidence": str(EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
        "pickup_orders": [69, 77], "pickup_two_step_c0_teacher_qualified_as_engineering": True,
        "pickup_learning_attribution_qualified": False,
        "delivery_hash_completed_by_8s_under_synthetic_dualcore_stress": False,
        "delivery_all_owned_children_reaped": True,
        "delivery_full_request_budget_qualified": False,
        "independent_checks": 37, "independent_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    }
    entries[4].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    put_new(UPDATE, {
        "prior_entries": prior,
        "updated_entries_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in entries},
        "report_sha256": sha(REPORT), "evidence_sha256": sha(EVIDENCE),
    })
    print(json.dumps({"report": str(REPORT), "evidence": str(EVIDENCE), "update": str(UPDATE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
