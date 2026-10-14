"""Seal r16 evidence and append to the five local status entries."""
import hashlib
import json
import shutil
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TREE = ROUND.parent
WORKSPACE = TREE.parents[1]
EXT = WORKSPACE / "C-VRP_Cold-chainVehicleRoutingProblem"
DEL = TREE / "delivery/source_overlap_cpu15_root_v1"
REPORT = EXT / "docs/当前规划/决策与审计/双线第十八轮_C0动作排序与同核哈希重叠_2026-10-14.md"
EVIDENCE = TREE / "reproduction/GOAL_ROUND_R16_C0_RANK_CPU15_OVERLAP_EVIDENCE_20261014.json"
STATUS = EXT / "results/research_20261010_continuation_v2/WORK_STATUS.json"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def put(p, obj):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    p = read(ROUND / "PICKUP_C0_RANK_RESULT.json")
    d = read(DEL / "RESULT.json")
    a = read(ROUND / "INDEPENDENT_R16_AUDIT.json")
    assert a["checks"] == 34 and a["failures"] == 0
    assert all(row["phases"][0]["one_step_c0_regret"] > 1 for row in p["rows"])
    assert not d["receipts"]["hash"]["completed_by_8s"] and d["receipts"]["cpu"]["completed_by_8s"]
    assert all(x["reaped"] for x in d["receipts"].values())
    sources = [
        ROUND / "PICKUP_C0_RANK_REGISTRATION.json", ROUND / "PICKUP_C0_RANK_RESULT.json",
        ROUND / "INDEPENDENT_R16_REGISTRATION.json", ROUND / "INDEPENDENT_R16_AUDIT.json",
        DEL / "REGISTRATION.json", DEL / "RESOURCE_PREFLIGHT.json",
        DEL / "RUN_ATTEMPT.json", DEL / "UPLOAD_ACCEPTANCE.json",
        DEL / "RUN_RESPONSE.json", DEL / "COLLECT_ATTEMPT.json", DEL / "RESULT.json",
        DEL / "RESULT.json.receipt.json", DEL / "CPU_CHILD_RESULT.json",
        DEL / "CPU_CHILD_RESULT.json.receipt.json", DEL / "HASH_CHILD_RESULT.json.missing.json",
        DEL / "measure_overlap_cpu15_v1.py",
        WORKSPACE / "tools/register_dual_overlap_rank_r16.py",
        WORKSPACE / "tools/diagnose_pickup_saved_c0_rank_r16.py",
        WORKSPACE / "tools/register_delivery_overlap_cpu15_r16.py",
        WORKSPACE / "tools/transport_delivery_overlap_cpu15_r16.py",
        WORKSPACE / "tools/register_dual_overlap_rank_audit_r16.py",
        WORKSPACE / "tools/audit_dual_overlap_rank_r16.py",
    ]
    assert REPORT.is_file() and all(x.is_file() for x in sources)
    put(EVIDENCE, {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "report_sha256": sha(REPORT),
        "sources_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in sources},
        "pickup_first_step_c0_proxy_regret": p["rows"][0]["phases"][0]["one_step_c0_regret"],
        "pickup_pretrained_random_same_actions": True,
        "pickup_learned_attribution_qualified": False,
        "delivery_hash_completed_by_8s": False,
        "delivery_synthetic_cpu_completed_by_8s": True,
        "delivery_parent_elapsed_seconds": d["parent_elapsed_seconds"],
        "delivery_all_owned_children_reaped": True,
        "delivery_actual_model_request_run": False,
        "delivery_full_request_budget_qualified": False,
        "independent_saved_checks": 34, "independent_saved_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    })
    entries = [EXT / "项目当前状态.md",
               EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
               EXT / "01_取货线/README.md", EXT / "02_配送线/README.md", STATUS]
    backup = ROUND / "current_entries_before_goal_round_r16"
    backup.mkdir(exist_ok=False)
    prior = {}
    for i, entry in enumerate(entries):
        b = backup / f"{i}_{entry.name}"
        shutil.copyfile(entry, b)
        prior[str(entry.relative_to(WORKSPACE)).replace("\\", "/")] = {
            "sha256": sha(entry), "backup": str(b.relative_to(TREE)).replace("\\", "/")}
    name = REPORT.name
    notes = [
        f"> 2026-10-14 双线第十八轮：取货订单77已存首步模型排序对C0六候选有7/15与8/15逆序，最高动作落后该步C0最高1.09536，仅临时计划代理；配送CPU15同核合成4.9秒工作与旧源哈希并发，哈希8秒未完成且按界回收。独立34/0，完整请求和稳定在线增益仍未建立。见[记录](docs/当前规划/决策与审计/{name})。\n\n",
        f"> 2026-10-14 双线第十八轮：旧订单77首步模型/C0排序错位、配送同核哈希压力8秒未闭合；34项独立核查0失败，科学第3族未启。见[记录](../决策与审计/{name})。\n\n",
        f"> 2026-10-14 订单77固定旧动作首步模型所选临时C0效用比该步最高低1.09536；两模型最高动作同，非整链后悔或学习收益。独立34/0。见[记录](../docs/当前规划/决策与审计/{name})。\n\n",
        f"> 2026-10-14 3080 CPU15同核合成4.9秒CPU负载并发旧源哈希，8秒哈希未完成，8.015秒父端回收；非真实模型或完整发布门。见[记录](../docs/当前规划/决策与审计/{name})。\n\n",
    ]
    for entry, note in zip(entries[:-1], notes):
        entry.write_text(note + entry.read_text(encoding="utf-8"), encoding="utf-8")
    status = read(STATUS)
    status["latest_goal_round_20261014_r16_c0_rank_cpu15_overlap"] = {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "evidence": str(EVIDENCE.relative_to(WORKSPACE)).replace("\\", "/"),
        "pickup_first_step_c0_proxy_regret": p["rows"][0]["phases"][0]["one_step_c0_regret"],
        "pickup_learning_attribution_qualified": False,
        "delivery_hash_completed_by_8s_under_synthetic_same_cpu_stress": False,
        "delivery_all_owned_children_reaped": True,
        "delivery_full_request_budget_qualified": False,
        "independent_saved_checks": 34, "independent_saved_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    put(ROUND / "UPDATE_IDENTITY.json", {
        "prior_entries": prior,
        "updated_entries_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in entries},
        "report_sha256": sha(REPORT), "evidence_sha256": sha(EVIDENCE)})


if __name__ == "__main__":
    main()
