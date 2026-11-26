from pathlib import Path
p = Path("scripts/update_state_docs.py")
t = p.read_text(encoding="utf-8")
t = t.replace("ROOT = Path(__file__).resolve().parents[5]", "ROOT = Path(__file__).resolve().parents[6]")
p.write_text(t, encoding="utf-8")
print("root fixed")
