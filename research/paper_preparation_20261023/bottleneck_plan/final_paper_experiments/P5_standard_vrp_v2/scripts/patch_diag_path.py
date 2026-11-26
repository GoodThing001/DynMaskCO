from pathlib import Path
p = Path("scripts/p5v2_diag_deploy.py")
t = p.read_text(encoding="utf-8")
t = t.replace('P5 = "/home/hzeng/project/MASKCO-final_PAPER_PLACEHOLDER"',
              'P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"')
t = t.replace('"cp /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/stage01/unified_runner.py . "',
              '"cp /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/stage01/unified_runner.py . "')
p.write_text(t, encoding="utf-8")
print("diag deploy path fixed")
