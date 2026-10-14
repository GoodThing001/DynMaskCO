"""Back up and update the five unique local entry points for round 19."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r19_science_release"
REPORT = "双线第二十一轮_五日规则身份与完整请求科学放行_2026-10-14.md"
REL = "docs/当前规划/决策与审计/" + REPORT
FILES = [
    E / "项目当前状态.md",
    E / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
    E / "01_取货线/README.md",
    E / "02_配送线/README.md",
    E / "results/research_20261010_continuation_v2/WORK_STATUS.json",
]
PREFIXES = [
    f"> 2026-10-14 第二十一轮：五日接拒均值+25.5357仍是非学习探索线索；原预登记预留式与执行影子价格式不等价，旧40日不能自动确认。配送真实两臂stage6后余1.221/1.164秒、0发布。稳定增益未建立。见[记录]({REL})。\n\n",
    f"> 2026-10-14 第二十一轮科学放行核查：原接拒预留条件与实际影子价格条件不等价；配送真实分段头请求仍0发布。旧40日确认与第3族均不启动。见[{REPORT}]({REPORT})。\n\n",
    f"> 2026-10-14 五日规则身份独立核查：同期均值+25.5357为非学习探索线索，原预留式登记与执行影子价格式不等价；旧40日不自动确认。见[第二十一轮](../{REL})。\n\n",
    f"> 2026-10-14 真实分段头两臂stage6后余1.221/1.164秒、0合格发布；原独立审计FAIL，源/参数/动态闭包未入8+2整请求。见[第二十一轮](../{REL})。\n\n",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if not (OUT / "RELEASE_AUDIT.json").exists():
        raise RuntimeError("independent audit not complete")
    backup = OUT / "entry_before"
    backup.mkdir(exist_ok=False)
    prior = [p.read_bytes() for p in FILES]
    records = []
    for i, (p, data) in enumerate(zip(FILES, prior)):
        b = backup / f"{i}_{p.name}"
        b.write_bytes(data)
        assert b.read_bytes() == data
        records.append({"path": str(p.relative_to(ROOT)), "old_sha256": sha(data), "backup": str(b.relative_to(ROOT))})
    for p, data, prefix in zip(FILES[:4], prior[:4], PREFIXES):
        text = data.decode("utf-8-sig")
        if REPORT in text[:200]:
            raise RuntimeError("entry already contains this round")
        p.write_text(prefix + text, encoding="utf-8")
    status = json.loads(prior[4].decode("utf-8-sig"))
    key = "latest_goal_round_20261014_r19_science_release"
    if key in status:
        raise RuntimeError("status already contains this round")
    status[key] = {
        "report": REL,
        "evidence": "research/review_3080_20261010/goal_round_20261014_r19_science_release/RELEASE_AUDIT.json",
        "pickup_five_day_mean_delta": 25.53570745959878,
        "pickup_original_prereg_rule_matches_actual": False,
        "pickup_old_40_day_confirmation_released": False,
        "delivery_full_request_budget_qualified": False,
        "delivery_scientific_family3_started": False,
        "learning_attribution_qualified": False,
        "equal_time_iterative_reconstruction_qualified": False,
        "independent_strong_dynamic_accept_reject_qualified": False,
        "stable_online_gain": False,
    }
    FILES[4].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row, p in zip(records, FILES):
        row["new_sha256"] = sha(p.read_bytes())
        assert row["old_sha256"] == sha((ROOT / row["backup"]).read_bytes())
    (OUT / "UPDATE_IDENTITY.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"entries_updated": len(records), "backups_verified": len(records), "stable_online_gain": False}))


if __name__ == "__main__":
    main()
