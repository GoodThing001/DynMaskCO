from pathlib import Path
p = Path("scripts/add_deliver_group.py")  # 仅为留下修改痕迹：实际改 git_deliver
q = Path(r"D:\PyCharm_\MASKCO-Main\tools\git_deliver.py")
t = q.read_text(encoding="utf-8")
old = '"external_results/*/run*/**/*.json",'
new = '"external_results/*/run*/*.json", "external_results/*/run*/**/*.json",'
assert old in t
q.write_text(t.replace(old, new, 1), encoding="utf-8")
print("external pattern fixed")
