"""Preserve five current entry files byte-for-byte before appending this review."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "research" / "review_3080_20261010"
EXT = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
HERE = Path(__file__).resolve().parent
BACKUP = TREE / "current_entries_before_goal_round_20261013_r3_adjudication"
REPORT_REL = "docs/当前规划/决策与审计/取货五日线索独立裁决与配送资源门_2026-10-13.md"
RECEIPT = TREE / "reproduction" / "GOAL_ROUND_R3_ADJUDICATION_EVIDENCE_20261013.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(data)


def main():
    review = json.loads((HERE / "SAVED_ONLY_ADJUDICATION.json").read_text(encoding="utf-8"))
    resource = json.loads((HERE / "SERVER_RESOURCE_OBSERVATION.json").read_text(encoding="utf-8"))
    assert review["checks_count"] == 31 and review["failed"] == 0
    assert resource["remote_exit_code"] == 0 and resource["research_launched"] is False
    assert resource["observation"]["cpu15_busy_percent"] > 20
    assert not (BACKUP / "UPDATE_IDENTITY.json").exists()
    assert not RECEIPT.exists()

    report_bytes = (EXT / REPORT_REL).read_bytes()
    evidence = {
        "stage": "goal_round_20261013_r3_adjudication",
        "saved_only_checks": 31,
        "saved_only_failures": 0,
        "original_audit_checks": 61,
        "original_audit_failures": 2,
        "same_day_mean_delta": review["observed"]["same_day_mean"],
        "recorded_explicit_mean_delta": review["observed"]["recorded_explicit_mean"],
        "empirical_bootstrap_interval": review["observed"]["exhaustive_empirical_bootstrap_95pct"],
        "initial_and_executed_preregistration_equivalent": False,
        "delivery_readonly_available_ram_gib": resource["observation"]["available_ram_gib"],
        "delivery_readonly_cpu15_busy_percent": resource["observation"]["cpu15_busy_percent"],
        "delivery_deployed": False,
        "pickup_40_day_stage_launched": False,
        "scientific_family3_started": False,
        "stable_online_gain": False,
        "report": REPORT_REL,
        "evidence_sha256": {name: sha((HERE / name).read_bytes()) for name in (
            "REVIEW_SCOPE.json", "review_saved_panel.py", "SAVED_ONLY_ADJUDICATION.json",
            "TRANSPORT_FAILURE_DEFAULT_SANDBOX.json", "inspect_3080_resources.py",
            "SERVER_RESOURCE_OBSERVATION.json")},
    }
    save_new(RECEIPT, (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    notes = {
        "项目当前状态.md": "> 2026-10-13 五日线索独立裁决：已存原件只读复算31/0，原生产29/0与独立审计61/2FAIL均保持。同期差+25.5357仅是非学习预算过滤器的开发线索；最初预登记的保留量规则/旧explicit_feat主对照与实际λ价格/同期SAA不同，经验自助区间[0,66.7049]含0，不可宣称最初规则验证或稳定MaskCO增益。配送3080只读RAM9.493GiB但CPU15忙90.6%且比较任务在跑，本轮不派发；既有服务器授权有效。详见[独立裁决与资源门](docs/当前规划/决策与审计/取货五日线索独立裁决与配送资源门_2026-10-13.md)。\n\n",
        "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md": "> 2026-10-13 五日线索再裁决：本次新只读31/0确认实际同期差，但最初预登记的规则、主对照、预算与执行版不等价；原61/2FAIL不能改PASS。40日扩展仅能作为另登记开发检验，0–4不重跑，新日/正式学习归因与强动态等门仍缺。3080实际CPU15忙90.6%、RAM9.493GiB，有比较任务，不启动配送。详见[裁决](取货五日线索独立裁决与配送资源门_2026-10-13.md)。\n\n",
        "01_取货线/README.md": "> 2026-10-13 五日正向线索复核：新保存原件31/0，同期均值+25.5357；最初预登记规则和主对照与执行版不同，原61/2FAIL保留，自助区间下界0。详见[裁决](../docs/当前规划/决策与审计/取货五日线索独立裁决与配送资源门_2026-10-13.md)。\n\n",
        "02_配送线/README.md": "> 2026-10-13 3080只读资源：RAM9.493GiB、CPU15瞬时忙90.6%且比较任务在跑，完整主干8+2请求当前资源门不合格；旧编译头/计时FAIL不重跑。既有服务器执行授权有效，下次重新核实际负载。详见[裁决](../docs/当前规划/决策与审计/取货五日线索独立裁决与配送资源门_2026-10-13.md)。\n\n",
    }
    paths = list(notes) + ["results/research_20261010_continuation_v2/WORK_STATUS.json"]
    old = {relative: (EXT / relative).read_bytes() for relative in paths}
    for relative, data in old.items():
        save_new(BACKUP / relative, data)
    new = {}
    for relative, note in notes.items():
        new[relative] = note.encode("utf-8") + old[relative]
    status_path = "results/research_20261010_continuation_v2/WORK_STATUS.json"
    status = json.loads(old[status_path])
    status["latest_goal_round_20261013_r3_adjudication"] = {
        "report": REPORT_REL,
        "evidence": "research/review_3080_20261010/reproduction/GOAL_ROUND_R3_ADJUDICATION_EVIDENCE_20261013.json",
        "saved_only_checks": 31,
        "saved_only_failures": 0,
        "original_audit_failures_preserved": 2,
        "pickup_development_lead_only": True,
        "delivery_blocked_by_observed_cpu15_busy": True,
        "server_permission_missing": False,
        "stable_online_gain": False,
    }
    new[status_path] = (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for relative in paths:
        if (EXT / relative).read_bytes() != old[relative]:
            raise RuntimeError("Concurrent entry change; preserved backup, no entry rewrite: " + relative)
    for relative in paths:
        path = EXT / relative
        temporary = path.with_name(path.name + ".goal_r3_new")
        with temporary.open("xb") as out:
            out.write(new[relative])
        os.replace(temporary, path)
    identity = {
        "files": [{"path": relative, "before_sha256": sha(old[relative]), "after_sha256": sha(new[relative])} for relative in paths],
        "report_sha256": sha(report_bytes),
        "receipt_sha256": sha(RECEIPT.read_bytes()),
    }
    save_new(BACKUP / "UPDATE_IDENTITY.json", (json.dumps(identity, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    assert all(sha((EXT / relative).read_bytes()) == sha(new[relative]) for relative in paths)
    print(json.dumps({"updated_entries": len(paths), "report_sha256": identity["report_sha256"], "receipt_sha256": identity["receipt_sha256"]}))


if __name__ == "__main__":
    main()
