import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
EXT = WORKSPACE / "C-VRP_Cold-chainVehicleRoutingProblem"
ROUND = ROOT / "goal_round_20261014_r14_boundary"
REPORT = EXT / "docs/当前规划/决策与审计/双线第十六轮_取货评分职责与配送截止余量_2026-10-14.md"
EVIDENCE = ROOT / "reproduction/GOAL_ROUND_R14_RANK_AND_DEADLINE_EVIDENCE_20261014.json"
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
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    files = [
        ROUND / "PICKUP_RANK_REGISTRATION.json", ROUND / "PICKUP_RANK_V1_FAILURE.json",
        ROUND / "PICKUP_RANK_RECOVERY_REGISTRATION.json", ROUND / "PICKUP_RANK_V2_FAILURE.json",
        ROUND / "PICKUP_RANK_RECOVERY_R3_REGISTRATION.json", ROUND / "PICKUP_RANK_DIAGNOSIS.json",
        ROUND / "DELIVERY_TIMING_REGISTRATION.json", ROUND / "DELIVERY_POST_SELECTION_TIMING.json",
        ROUND / "INDEPENDENT_BOUNDARY_REGISTRATION.json", ROUND / "INDEPENDENT_BOUNDARY_AUDIT.json",
        WORKSPACE / "tools/register_dual_saved_boundary_r14.py",
        WORKSPACE / "tools/diagnose_pickup_saved_ranking_r14.py",
        WORKSPACE / "tools/diagnose_pickup_saved_ranking_r14_v2.py",
        WORKSPACE / "tools/diagnose_pickup_saved_ranking_r14_v3.py",
        WORKSPACE / "tools/diagnose_delivery_saved_post_selection_r14.py",
        WORKSPACE / "tools/audit_dual_saved_boundary_r14.py",
    ]
    assert REPORT.is_file() and all(x.is_file() for x in files)
    pickup = read(ROUND / "PICKUP_RANK_DIAGNOSIS.json")
    delivery = read(ROUND / "DELIVERY_POST_SELECTION_TIMING.json")
    audit = read(ROUND / "INDEPENDENT_BOUNDARY_AUDIT.json")
    assert audit["checks"] == 15 and audit["failures"] == 0
    assert [x["discordant_pairs"] for x in pickup["between_model_rankings"]] == [1, 0]
    assert all(x["phases"][0]["model_changes_distance_choice"] for x in pickup["rows"])
    assert all(not x["phases"][1]["model_changes_distance_choice"] for x in pickup["rows"])
    assert all(1.0 < x["stage6_to_deadline"]["seconds"] < 1.3 for x in delivery["rows"])
    assert not delivery["full_request_budget_qualified"]
    new_json(EVIDENCE, {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"), "report_sha256": sha(REPORT),
        "inputs_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in files},
        "pickup_first_action_model_changes_distance_only": True,
        "pickup_pretrained_random_top_action_equal": True,
        "pickup_rank_discordant_pairs_by_step": [1, 0],
        "pickup_original_v1_v2_harness_failures_preserved": True,
        "delivery_stage6_remaining_seconds": [x["stage6_to_deadline"]["seconds"] for x in delivery["rows"]],
        "delivery_post_selection_closure_duration_measured": False,
        "delivery_full_request_budget_qualified": False,
        "independent_saved_checks": 15, "independent_saved_failures": 0,
        "model_or_C0_or_old_request_rerun": False,
        "scientific_family3_started": False, "stable_online_gain": False,
    })
    entries = [EXT / "项目当前状态.md",
               EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
               EXT / "01_取货线/README.md", EXT / "02_配送线/README.md", STATUS]
    backup = ROUND / "current_entries_before_goal_round_r14"
    backup.mkdir(exist_ok=False)
    prior = {}
    for i, entry in enumerate(entries):
        copy = backup / f"{i}_{entry.name}"
        shutil.copyfile(entry, copy)
        prior[str(entry.relative_to(WORKSPACE)).replace("\\", "/")] = {
            "sha256": sha(entry), "backup": str(copy.relative_to(ROOT)).replace("\\", "/")}
    name = REPORT.name
    general = (f"> 2026-10-14 双线第十六轮：取货订单77已存候选分解显示两模型均在第一步改变纯距离动作，但两模型最高动作一致；排序不一致仅1/15、0/5对，独立15/0。配送stage6距8秒截止余1.221/1.164秒，后续源/参数门实际时长未得，原851/28FAIL保留。稳定核心在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。\n\n")
    for entry in entries[:-1]:
        if entry == entries[1]:
            note = general.replace("docs/当前规划/决策与审计/", "")
        elif entry == entries[2]:
            note = f"> 2026-10-14 订单77保存提议纯距离与两模型第一步动作不同，但预训练/随机仍同顶层动作；仅旧状态机制，学习归因与在线收益未建立。见[记录](../docs/当前规划/决策与审计/{name})。\n\n"
        elif entry == entries[3]:
            note = f"> 2026-10-14 两真实分段头请求stage6后至8秒仅余1.221/1.164秒，源/参数/发布后续未完成，原审计FAIL与0发布保留；已存时钟独立15项总审计0失败。见[记录](../docs/当前规划/决策与审计/{name})。\n\n"
        else:
            note = general
        entry.write_text(note + entry.read_text(encoding="utf-8"), encoding="utf-8")
    status = read(STATUS)
    status["latest_goal_round_20261014_r14_rank_and_deadline"] = {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "evidence": str(EVIDENCE.relative_to(WORKSPACE)).replace("\\", "/"),
        "pickup_model_changes_distance_first_action": True,
        "pickup_pretrained_random_same_top_actions": True,
        "pickup_learning_attribution_qualified": False,
        "delivery_stage6_remaining_seconds": [x["stage6_to_deadline"]["seconds"] for x in delivery["rows"]],
        "delivery_full_request_budget_qualified": False,
        "independent_saved_checks": 15, "independent_saved_failures": 0,
        "scientific_family3_started": False, "stable_online_gain": False,
    }
    STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    new_json(ROUND / "UPDATE_IDENTITY.json", {
        "prior_entries": prior,
        "updated_entries_sha256": {str(x.relative_to(WORKSPACE)).replace("\\", "/"): sha(x) for x in entries},
        "report_sha256": sha(REPORT), "evidence_sha256": sha(EVIDENCE),
    })


if __name__ == "__main__":
    main()
