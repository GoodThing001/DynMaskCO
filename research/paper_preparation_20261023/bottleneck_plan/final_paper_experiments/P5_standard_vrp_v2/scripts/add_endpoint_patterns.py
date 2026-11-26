from pathlib import Path
q = Path(r"D:\PyCharm_\MASKCO-Main\tools\git_deliver.py")
t = q.read_text(encoding="utf-8")
old = '"development_results/*/out/*.json",'
new = ('"development_results/*/endpoints/*.json", "development_results/*/out/*.json",')
t = t.replace(old, new, 1)
old2 = '"formal_results/*/out/*.json",'
new2 = '"formal_results/*/endpoints/*.json", "formal_results/*/out/*.json",'
t = t.replace(old2, new2, 1)
old3 = '"diagnostics/*/status.json", "diagnostics/*/out/*.json",'
new3 = '"diagnostics/*/status.json", "diagnostics/*/endpoints/*.json", "diagnostics/*/out/*.json",'
t = t.replace(old3, new3, 1)
old4 = '"external_results/*/run*/*.json", "external_results/*/run*/**/*.json",'
new4 = ('"external_results/*/run*/*.json", "external_results/*/run*/**/*.json",\n'
        '                        "external_results/*/endpoints/*.json",')
t = t.replace(old4, new4, 1)
q.write_text(t, encoding="utf-8")
import ast; ast.parse(t)
print("manifest patterns updated for endpoints/")
