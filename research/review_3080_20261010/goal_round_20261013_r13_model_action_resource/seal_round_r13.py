from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
EXT = WORKSPACE / "C-VRP_Cold-chainVehicleRoutingProblem"
ROUND = ROOT / "goal_round_20261013_r13_model_action_resource"
DELIVERY = ROOT / "delivery/split_final_timed_execution_root_v1"
SAVED_AUDIT = ROOT / "delivery/split_final_saved_audit_root_v2"
REPORT = EXT / "docs/当前规划/决策与审计/双线第十五轮_取货动作收敛与配送实际分段头计时_2026-10-14.md"
EVIDENCE = ROOT / "reproduction/GOAL_ROUND_R13_ACTION_SPLIT_TIMED_EVIDENCE_20261014.json"
STATUS = EXT / "results/research_20261010_continuation_v2/WORK_STATUS.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_json(path, value):
    if path.exists():
        raise RuntimeError("Original output exists: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    sources = [
        ROUND / "PICKUP_MECHANISM_REGISTRATION.json", ROUND / "PICKUP_SAVED_ACTION_MECHANISM.json",
        ROUND / "PICKUP_MECHANISM_AUDIT_REGISTRATION.json", ROUND / "INDEPENDENT_PICKUP_ACTION_AUDIT.json",
        DELIVERY / "EXECUTION_REGISTRATION.json", DELIVERY / "LOCAL_EXECUTION_BINDING.json",
        DELIVERY / "RESOURCE_PREFLIGHT_20261013T214613400600Z.json",
        DELIVERY / "downloaded/RESULT.json", DELIVERY / "downloaded/RUN_EXIT.json",
        DELIVERY / "downloaded/AUDIT_RUN_EXIT.json", DELIVERY / "downloaded/CHAIN_EXIT.json",
        DELIVERY / "downloaded/INDEPENDENT_TIMED_REQUEST_AUDIT.json",
        DELIVERY / "SAVED_SPLIT_AUDIT_REGISTRATION.json", DELIVERY / "INDEPENDENT_SAVED_SPLIT_AUDIT.json",
        DELIVERY / "DOWNLOAD_MANIFEST.json",
        SAVED_AUDIT / "AUDIT_REGISTRATION.json", SAVED_AUDIT / "INDEPENDENT_SAVED_SPLIT_AUDIT.json",
        SAVED_AUDIT / "RUN_RESPONSE.json", SAVED_AUDIT / "COLLECT_RECEIPT.json",
        WORKSPACE / "tools/diagnose_pickup_saved_action_convergence_r13.py",
        WORKSPACE / "tools/audit_pickup_saved_action_r13.py",
        WORKSPACE / "tools/audit_delivery_split_saved_r13.py",
        WORKSPACE / "tools/build_delivery_split_remote_saved_audit_r13.py",
        WORKSPACE / "tools/transport_delivery_split_saved_audit_r13.py",
    ]
    # Select actual preflight by immutable filename rather than assuming when its clock occurred.
    sources = [p for p in sources if "RESOURCE_PREFLIGHT_20261013T214613400600Z" not in str(p)]
    sources.extend(sorted(DELIVERY.glob("RESOURCE_PREFLIGHT_*.json")))
    assert all(p.is_file() for p in sources) and REPORT.is_file()
    p = read(ROUND / "INDEPENDENT_PICKUP_ACTION_AUDIT.json")
    a = read(DELIVERY / "downloaded/INDEPENDENT_TIMED_REQUEST_AUDIT.json")
    local = read(DELIVERY / "INDEPENDENT_SAVED_SPLIT_AUDIT.json")
    remote = read(SAVED_AUDIT / "INDEPENDENT_SAVED_SPLIT_AUDIT.json")
    result = read(DELIVERY / "downloaded/RESULT.json")
    manifest = read(DELIVERY / "DOWNLOAD_MANIFEST.json")
    assert p["checks"] == 23 and p["failures"] == 0 and p["saved_only"]
    assert a["checks_count"] == 851 and a["failed"] == 28
    assert local["checks_count"] == 51 and local["failed"] == 50
    assert remote["checks_count"] == 51 and remote["failed"] == 0 and remote["max_abs_error"] == 0
    assert result["count"] == 2 and result["all_reaped"] and result["selected_count"] == 0
    assert all(r["actual_stage"] == 6 and not r["selected"] for r in result["receipts"])
    assert len(manifest) == 181 and sum(x["size"] for x in manifest) == 35976498
    for entry in manifest:
        artifact = DELIVERY / "downloaded" / entry["name"]
        assert artifact.stat().st_size == entry["size"] and sha(artifact) == entry["sha256"]
    assert read(DELIVERY / "downloaded/RUN_EXIT.json")["exit_code"] == 0
    assert read(DELIVERY / "downloaded/AUDIT_RUN_EXIT.json")["exit_code"] == 1
    assert read(DELIVERY / "downloaded/CHAIN_EXIT.json")["exit_code"] == 1
    new_json(EVIDENCE, {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"), "report_sha256": sha(REPORT),
        "sources_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in sources},
        "pickup_saved_mechanism_checks": 23, "pickup_saved_mechanism_failures": 0,
        "pickup_same_action_sequence": True, "pickup_learning_attribution_qualified": False,
        "delivery_actual_requests": 2, "delivery_actual_stage_each": 6,
        "delivery_actual_selected_count": 0, "delivery_workers_reaped": True,
        "delivery_original_independent_checks": 851, "delivery_original_independent_failures_retained": 28,
        "delivery_original_audit_exit_retained": 1, "delivery_original_chain_exit_retained": 1,
        "delivery_local_cross_runtime_saved_checks": 51, "delivery_local_cross_runtime_saved_failures_retained": 50,
        "delivery_actual_runtime_saved_checks": 51, "delivery_actual_runtime_saved_failures": 0,
        "delivery_actual_runtime_saved_max_abs_error": 0,
        "delivery_readonly_archived_files": 181, "delivery_readonly_archived_bytes": 35976498,
        "delivery_all_archived_original_hashes_match": True,
        "delivery_full_request_budget_qualified": False, "delivery_scientific_loaded_closure_qualified": False,
        "scientific_family3_started": False, "stable_online_gain": False,
    })
    entries = [EXT / "项目当前状态.md",
               EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
               EXT / "01_取货线/README.md", EXT / "02_配送线/README.md", STATUS]
    backup = ROUND / "current_entries_before_goal_round_r13"
    if backup.exists():
        raise RuntimeError("Original entry backup exists")
    backup.mkdir()
    old = {}
    for i, entry in enumerate(entries):
        copy = backup / f"{i}_{entry.name}"
        shutil.copyfile(entry, copy)
        old[str(entry.relative_to(WORKSPACE)).replace("\\", "/")] = {
            "sha256": sha(entry), "backup": str(copy.relative_to(ROOT)).replace("\\", "/")}
    name = REPORT.name
    general = (f"> 2026-10-14 双线第十五轮：取货订单77已存提议两模型参数/logits不同但两步动作相同，另进程23/0，学习归因仍缺。配送固定两真实8+2请求均到stage6并8秒取消回收、0发布；原审计851/28FAIL/chain1保留，实际3080 JAX保存评分窄核51/0，完整发布前门未到。稳定核心在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。\n\n")
    for entry in entries[:-1]:
        if entry == entries[1]:
            note = general.replace("docs/当前规划/决策与审计/", "")
        elif entry == entries[2]:
            note = f"> 2026-10-14 订单77已存提议两模型参数与logits不同，但两步动作与候选相同；影子选择来自共同外层评分，独立23/0，学习及在线收益未建立。见[记录](../docs/当前规划/决策与审计/{name})。\n\n"
        elif entry == entries[3]:
            note = f"> 2026-10-14 分段头固定两实际请求均完成25插入及C0至stage6，8秒取消回收且0发布；原审计851/28FAIL保留，3080实际JAX保存评分窄核51/0，完整8+2及源/参数发布前门仍缺。见[记录](../docs/当前规划/决策与审计/{name})。\n\n"
        else:
            note = general
        entry.write_text(note + entry.read_text(encoding="utf-8"), encoding="utf-8")
    status = read(STATUS)
    status["latest_goal_round_20261014_r13_action_split_timed"] = {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "evidence": str(EVIDENCE.relative_to(WORKSPACE)).replace("\\", "/"),
        "pickup_saved_independent_checks": 23, "pickup_learning_attribution_qualified": False,
        "delivery_actual_stage_each": 6, "delivery_selected_count": 0,
        "delivery_original_independent_checks": 851, "delivery_original_failures_preserved": 28,
        "delivery_actual_runtime_saved_checks": 51, "delivery_actual_runtime_saved_failures": 0,
        "delivery_full_request_budget_qualified": False,
        "scientific_family3_started": False, "stable_online_gain": False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    new_json(ROUND / "UPDATE_IDENTITY.json", {
        "prior_entries": old,
        "updated_entries_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in entries},
        "report_sha256": sha(REPORT), "evidence_sha256": sha(EVIDENCE),
    })


if __name__ == "__main__":
    main()
