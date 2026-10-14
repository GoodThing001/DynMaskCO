"""Read-only action-score vs C0 rank diagnosis on the fixed old order-77 state."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "research/review_3080_20261010"
BASE = TREE / "goal_round_20261014_r16_overlap_rank"
MODEL = TREE / "pickup/learned_after_strong_recovery_execution_root_v2/RESULT.json"
PRIOR = TREE / "goal_round_20261014_r14_boundary/PICKUP_RANK_DIAGNOSIS.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unpack(x):
    if isinstance(x, dict) and "__mapping__" in x:
        return {unpack(a): unpack(b) for a, b in x["__mapping__"]}
    if isinstance(x, dict) and ("__array__" in x or "__sequence__" in x):
        return unpack(x["values"])
    if isinstance(x, list):
        return [unpack(y) for y in x]
    return x


def main():
    reg = json.loads((BASE / "PICKUP_C0_RANK_REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(__file__) == reg["script_sha256"]
    assert sha(MODEL) == reg["model_sha256"] and sha(PRIOR) == reg["prior_rank_sha256"]
    model = unpack(json.loads(MODEL.read_text(encoding="utf-8")))
    old = json.loads(PRIOR.read_text(encoding="utf-8"))
    rows = []
    for m, record in enumerate(model["records"]):
        audits = record["actual"]["proposal"]["audit"]
        assert len(audits) == 10 and record["mode"] == old["rows"][m]["mode"]
        phases = []
        offset = 0
        for step, count in enumerate(reg["phase_counts"]):
            entries = audits[offset:offset + count]
            old_actions = [tuple(x["action"]) for x in old["rows"][m]["phases"][step]["candidates"]]
            assert [tuple(x[k] for k in ("order", "vehicle", "position")) for x in entries] == old_actions
            assert all(x["certificate"]["qualified"] and x["certificate"]["evaluation"]["hard_feasible"] for x in entries)
            by_model = sorted(entries, key=lambda x: (x["proposal_score"], x["order"], x["vehicle"], x["position"]))
            by_c0 = sorted(entries, key=lambda x: (-x["certificate"]["evaluation"]["utility"], x["order"], x["vehicle"], x["position"]))
            selected = old["rows"][m]["phases"][step]["selected"]
            assert tuple(selected) == tuple(by_model[0][k] for k in ("order", "vehicle", "position"))
            inv = 0
            n = 0
            for i, a in enumerate(entries):
                for b in entries[i + 1:]:
                    ds = a["proposal_score"] - b["proposal_score"]
                    du = a["certificate"]["evaluation"]["utility"] - b["certificate"]["evaluation"]["utility"]
                    if ds and du:
                        n += 1
                        inv += (ds * du > 0)  # low score should match high C0 utility
            selected_c0 = by_model[0]["certificate"]["evaluation"]["utility"]
            best_c0 = by_c0[0]["certificate"]["evaluation"]["utility"]
            phases.append({"step": step, "candidate_count": count,
                           "model_choice": list(selected),
                           "c0_best_action": [by_c0[0][k] for k in ("order", "vehicle", "position")],
                           "model_choice_c0_utility": selected_c0,
                           "c0_best_utility": best_c0,
                           "one_step_c0_regret": best_c0 - selected_c0,
                           "discordant_model_c0_pairs": inv, "comparable_pairs": n})
            offset += count
        rows.append({"mode": record["mode"], "phases": phases})
    out = {"registration_sha256": sha(BASE / "PICKUP_C0_RANK_REGISTRATION.json"),
           "model_input_sha256": sha(MODEL), "prior_rank_sha256": sha(PRIOR),
           "rows": rows, "saved_only": True, "no_model_or_C0_rerun": True,
           "not_new_day_or_equal_time_control": True, "learned_attribution_qualified": False}
    with (BASE / "PICKUP_C0_RANK_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
