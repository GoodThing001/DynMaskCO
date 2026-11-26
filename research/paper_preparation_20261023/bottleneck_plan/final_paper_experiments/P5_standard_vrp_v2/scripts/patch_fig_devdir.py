import ast
from pathlib import Path

p = Path("scripts/p5v2_figures.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''def dev_figures():
    d = load_json(V2 / "development_results/dev01/analysis_dev.json")''',
'''def dev_figures(dev_dir=None):
    root = Path(dev_dir) if dev_dir else V2 / "development_results/dev01_final"
    d = load_json(root / "analysis_dev.json")''')
t = t.replace('''def main():
    dev_figures()''',
'''def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev_figures(args[0] if args else None)''')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("figures accept dev dir")
