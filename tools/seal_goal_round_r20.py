"""Seal round 20 by backing up original bytes before updating five entries."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem"
OUT = ROOT / "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure"
REPORT = "双线第二十二轮_TRAIN接拒活动与实际加载闭包候选_2026-10-14.md"
REL = "docs/当前规划/决策与审计/" + REPORT
FILES = [
    E / "项目当前状态.md",
    E / "docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md",
    E / "01_取货线/README.md",
    E / "02_配送线/README.md",
    E / "results/research_20261010_continuation_v2/WORK_STATUS.json",
]
PREFIXES = [
    f"> 2026-10-14 第二十二轮：取货 TRAIN 交叉诊断影子价格拒绝在200/200日活跃（独立0失败），仅算术机制；配送实际加载闭包候选本地故障注入5/0，未安装、未过全模型8+2秒请求。稳定在线增益未建立。见[记录]({REL})。\n\n",
    f"> 2026-10-14 第二十二轮：取货 TRAIN-only 接拒活动独立复算，配送未安装闭包候选本地5/0；科学新鲜日和第3族均未放行。见[{REPORT}]({REPORT})。\n\n",
    f"> 2026-10-14 TRAIN 200日对半标定诊断：200/200日价格筛选活跃、独立逐日0失败；无路线/冷链/效用，不作在线正向。见[第二十二轮](../{REL})。\n\n",
    f"> 2026-10-14 实际加载闭包候选本地故障注入5/0；未安装，完整模型与8+2秒整请求仍缺。旧FAIL保留。见[第二十二轮](../{REL})。\n\n",
]


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    assert (E / REL).exists()
    for name in ("PICKUP_REGISTRATION.json", "PICKUP_TRAIN_RESULT.json", "PICKUP_INDEPENDENT_AUDIT.json", "REGISTRATION.json", "RESULT.json"):
        assert (OUT / name).exists(), name
    assert json.loads((OUT / "PICKUP_INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))["days_failed"] == 0
    assert json.loads((OUT / "RESULT.json").read_text(encoding="utf-8"))["fail_count"] == 0
    originals = [p.read_bytes() for p in FILES]
    key = "latest_goal_round_20261014_r20_loaded_closure"
    old_status = json.loads(originals[4].decode("utf-8-sig"))
    assert key not in old_status
    for data in originals[:4]:
        assert REPORT.encode("utf-8") not in data[:1024]
    backup = OUT / "entry_before"
    backup.mkdir(exist_ok=False)
    index = []
    for i, (p, data) in enumerate(zip(FILES, originals)):
        saved = backup / f"{i}_{p.name}"
        saved.write_bytes(data)
        assert saved.read_bytes() == data
        index.append({"path": str(p.relative_to(ROOT)), "backup": str(saved.relative_to(ROOT)), "before_sha256": digest(data)})
    for p, data, prefix in zip(FILES[:4], originals[:4], PREFIXES):
        p.write_bytes(prefix.encode("utf-8") + data)
    entry = {
        "report": REL,
        "evidence": "research/review_3080_20261010/goal_round_20261014_r20_loaded_closure",
        "pickup_train_crossfit_days": 200,
        "pickup_train_days_with_price_rejections": 200,
        "pickup_independent_days_failed": 0,
        "pickup_online_utility_qualified": False,
        "pickup_original_40_day_confirmation_released": False,
        "delivery_loaded_closure_candidate_local_checks_passed": 5,
        "delivery_candidate_installed": False,
        "delivery_full_request_budget_qualified": False,
        "delivery_scientific_family3_started": False,
        "learning_attribution_qualified": False,
        "equal_time_iterative_reconstruction_qualified": False,
        "independent_strong_dynamic_accept_reject_qualified": False,
        "stable_online_gain": False,
    }
    status = originals[4]
    stripped = status.rstrip(b"\r\n \t")
    assert stripped.endswith(b"}")
    outer_close = len(stripped) - 1
    addition = (",\n  " + json.dumps(key, ensure_ascii=False) + ": " + json.dumps(entry, ensure_ascii=False, indent=2).replace("\n", "\n  ") + "\n}").encode("utf-8")
    new_status = stripped[:outer_close].rstrip(b"\r\n \t") + addition + status[len(stripped):]
    parsed = json.loads(new_status.decode("utf-8-sig"))
    assert parsed[key] == entry
    FILES[4].write_bytes(new_status)
    for row, p in zip(index, FILES):
        row["after_sha256"] = digest(p.read_bytes())
        assert row["before_sha256"] == digest((ROOT / row["backup"]).read_bytes())
    (OUT / "UPDATE_IDENTITY.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"updated": len(FILES), "backups_verified": len(FILES), "stable_online_gain": False}))


if __name__ == "__main__":
    main()
