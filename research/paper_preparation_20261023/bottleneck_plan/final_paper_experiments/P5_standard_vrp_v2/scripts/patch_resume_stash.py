import ast
from pathlib import Path

p = Path("scripts/p5v2_worker.py")
t = p.read_text(encoding="utf-8")
t = t.replace('''    dest = HERE / "out" / f"{idx}_{version}.json"
    if dest.exists():
        try:
            done = json.loads(dest.read_text(encoding="utf-8"))
        except Exception:
            dest.unlink()
        else:
            if done.get("case") == case and done.get("seed") == seed:
                return      # 断点续跑：保留既有端点（runner 会再次核验）''',
'''    dest = HERE / "out" / f"{idx}_{version}.json"
    stash = HERE.parent / "dev01_stash" / f"{idx}_{version}.json"
    if not dest.exists() and stash.exists():
        # 断点续跑：该端点在上一轮（同一代码路径）已产出，直接搬入本轮 out/
        try:
            done = json.loads(stash.read_text(encoding="utf-8"))
        except Exception:
            done = None
        if done and done.get("case") == case and done.get("seed") == seed:
            (HERE / "out").mkdir(exist_ok=True)
            dest.write_bytes(stash.read_bytes())
            return
    if dest.exists():
        try:
            done = json.loads(dest.read_text(encoding="utf-8"))
        except Exception:
            dest.unlink()
        else:
            if done.get("case") == case and done.get("seed") == seed:
                return''')
p.write_text(t, encoding="utf-8")
ast.parse(t)

q = Path("scripts/p5v2_dev_resume.py")
t2 = q.read_text(encoding="utf-8")
t2 = t2.replace('''        cmd = ("cd " + RUN + " && "
               "rm -f out/*.log status.json runner.log && "
               "for p in $(pgrep -f 'dev01/unified_worker[.]py'); do kill $p 2>/dev/null; done; "
               "for p in $(pgrep -f 'dev01/unified_runner[.]py'); do kill $p 2>/dev/null; done; "
               "sleep 2; echo \\"[kept] $(ls out/*.json 2>/dev/null | wc -l)\\"; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 4; pgrep -fc 'dev01/unified_runner[.]py' || true")''',
'''        cmd = ("cd " + RUN + " && "
               "for p in $(pgrep -f 'dev01/unified_worker[.]py'); do kill $p 2>/dev/null; done; "
               "for p in $(pgrep -f 'unified_runner[.]py'); do kill $p 2>/dev/null; done; "
               "sleep 2; mkdir -p ../dev01_stash && cp -n out/*.json ../dev01_stash/ 2>/dev/null; "
               "echo \\"[stashed] $(ls ../dev01_stash/*.json 2>/dev/null | wc -l)\\"; "
               "rm -rf out status.json runner.log; "
               "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
               "sleep 5; ps -eo pid,etime,args | grep 'unified_runner' | grep -v grep | head -2")''')
q.write_text(t2, encoding="utf-8")
ast.parse(t2)
print("worker stash + resume script patched")
