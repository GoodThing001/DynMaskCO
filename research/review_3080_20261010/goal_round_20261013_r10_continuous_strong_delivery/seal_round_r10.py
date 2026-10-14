from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
EXT = WORKSPACE / "C-VRP_Cold-chainVehicleRoutingProblem"
ROUND = ROOT / "goal_round_20261013_r10_continuous_strong_delivery"
REPORT = EXT / "docs/当前规划/决策与审计/双线第十二轮_连续独立接拒与定形动作头严格门_2026-10-13.md"
EVIDENCE = ROOT / "reproduction/GOAL_ROUND_R10_CONTINUOUS_STRONG_PADDED_HEAD_EVIDENCE_20261013.json"
STATUS = EXT / "results/research_20261010_continuation_v2/WORK_STATUS.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, value: object) -> None:
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    pickup = ROOT / "pickup/strong_independent_owner_next_root_v2"
    inputs = [
        ROUND / "REGISTRATION.json",
        ROUND / "PICKUP_RECOVERY_REGISTRATION_V2.json",
        pickup / "RESULT.json",
        pickup / "INDEPENDENT_PRIVATE_OWNER.json",
        pickup / "INDEPENDENT_PENALTY_ASSUMPTION_RECOVERY_R3.json",
        ROUND / "DELIVERY_HEAD_SCALING_REGISTRATION.json",
        ROUND / "DELIVERY_HEAD_SCALING_DIAGNOSIS.json",
        ROUND / "INDEPENDENT_DELIVERY_HEAD_SCALING.json",
        ROUND / "DELIVERY_PADDED_HEAD_LOCAL_REGISTRATION.json",
        ROUND / "DELIVERY_PADDED_HEAD_LOCAL_RESULT.json",
        ROUND / "DELIVERY_PADDED_HEAD_AUDIT_REGISTRATION.json",
        ROUND / "INDEPENDENT_DELIVERY_PADDED_HEAD_LOCAL.json",
        ROUND / "SERVER_RESOURCE_OBSERVATION.json",
    ]
    assert all(x.is_file() for x in inputs)
    old_audit = json.loads((pickup / "INDEPENDENT_PRIVATE_OWNER.json").read_text(encoding="utf-8"))
    new_recovery = json.loads((pickup / "INDEPENDENT_PENALTY_ASSUMPTION_RECOVERY_R3.json").read_text(encoding="utf-8"))
    padded = json.loads((ROUND / "DELIVERY_PADDED_HEAD_LOCAL_RESULT.json").read_text(encoding="utf-8"))
    assert old_audit["checks_count"] == 626 and old_audit["failed"] == 1
    assert len(new_recovery["checks"]) == 7 and not new_recovery["failed"]
    assert padded["strict_equal_count"] == 513 and padded["strict_equal_total"] == 516
    evidence = {
        "scope": "r10 bounded pickup next-natural-event and delivery saved timing/fixed-shape head only",
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "report_sha256": digest(REPORT),
        "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in inputs},
        "pickup_v1_preproduction_failure_preserved": (ROOT / "pickup/strong_independent_owner_next_root_v1/FAILURE.json").is_file(),
        "pickup_production_checks": 7,
        "pickup_independent_original_checks": 626,
        "pickup_independent_original_failures_preserved": 1,
        "pickup_narrow_recovery_checks": 7,
        "pickup_formal_strong_dynamic_qualified": False,
        "delivery_saved_timing_independent_checks": 79,
        "delivery_padded_head_strict_equal": [513, 516],
        "delivery_padded_head_independent_checks": 9,
        "delivery_padded_head_qualified": False,
        "delivery_full_request_budget_qualified": False,
        "server_research_launched": False,
        "scientific_family3_started": False,
        "stable_online_gain": False,
    }
    write_new(EVIDENCE, evidence)

    entry_files = [
        EXT / "项目当前状态.md",
        EXT / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
        EXT / "01_取货线/README.md",
        EXT / "02_配送线/README.md",
        STATUS,
    ]
    assert all(p.is_file() for p in entry_files)
    backup = ROUND / "current_entries_before_goal_round_r10"
    assert not backup.exists()
    backup.mkdir()
    prior = {}
    for index, path in enumerate(entry_files):
        copied = backup / f"{index}_{path.name}"
        shutil.copyfile(path, copied)
        prior[str(path.relative_to(WORKSPACE)).replace("\\", "/")] = {"sha256": digest(path), "backup": str(copied.relative_to(ROOT)).replace("\\", "/")}
    summary = (
        "> 2026-10-13 双线第十二轮：取货从已选私有所有者自然继续到订单77，"
        "独立接拒与提交3.712秒、生产7/0；原独立626/1罚金假设FAIL保留，窄R3 7/0。"
        "配送旧17步计时/输入独立79/0，定形动作头516项仅513逐位相同、独立9/0，严格门FAIL未安装。"
        "3080只读资源合格但未派发，科学第3族未启。稳定MaskCO核心在线增益未建立。"
        "见[第十二轮记录](docs/当前规划/决策与审计/双线第十二轮_连续独立接拒与定形动作头严格门_2026-10-13.md)。\n\n"
    )
    for path in entry_files[:-1]:
        text = path.read_text(encoding="utf-8")
        if path == entry_files[1]:
            note = summary.replace("docs/当前规划/决策与审计/", "")
        elif path == entry_files[2]:
            note = "> 2026-10-13 下一自然订单77真实私有所有者独立接拒3.712秒、生产7/0；原审计626/1FAIL保留，窄R3 7/0。正式多日强动态与学习归因未合格。见[记录](../docs/当前规划/决策与审计/双线第十二轮_连续独立接拒与定形动作头严格门_2026-10-13.md)。\n\n"
        elif path == entry_files[3]:
            note = "> 2026-10-13 旧17步计时独立79/0；定形动作头516项仅513逐位同，严格FAIL未安装，独立9/0。完整8＋2秒及科学门未合格。见[记录](../docs/当前规划/决策与审计/双线第十二轮_连续独立接拒与定形动作头严格门_2026-10-13.md)。\n\n"
        else:
            note = summary
        path.write_text(note + text, encoding="utf-8")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    status["latest_goal_round_20261013_r10_continuous_strong_padded_head"] = {
        "report": str(REPORT.relative_to(EXT)).replace("\\", "/"),
        "evidence": str(EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
        "pickup_production_checks": 7,
        "pickup_original_independent_checks": 626,
        "pickup_original_independent_failures_preserved": 1,
        "pickup_narrow_recovery_checks": 7,
        "pickup_formal_strong_dynamic_qualified": False,
        "delivery_saved_timing_independent_checks": 79,
        "delivery_padded_head_strict_equal": "513/516",
        "delivery_padded_head_qualified": False,
        "delivery_full_request_budget_qualified": False,
        "server_research_launched": False,
        "scientific_family3_started": False,
        "stable_online_gain": False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_new(ROUND / "UPDATE_IDENTITY.json", {
        "prior_entries": prior,
        "updated_entries_sha256": {str(p.relative_to(WORKSPACE)).replace("\\", "/"): digest(p) for p in entry_files},
        "report_sha256": digest(REPORT),
        "evidence_sha256": digest(EVIDENCE),
    })


if __name__ == "__main__":
    main()
