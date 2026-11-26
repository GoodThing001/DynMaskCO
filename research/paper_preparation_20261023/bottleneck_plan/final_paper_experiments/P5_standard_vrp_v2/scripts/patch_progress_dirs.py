from pathlib import Path
p = Path("scripts/p5v2_progress.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''            diag="/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01")''',
              '''            dev="/tmp/p5v2_dev/dev01")''')
p.write_text(t, encoding="utf-8")
print("progress script now tracks dev instead of diag")
