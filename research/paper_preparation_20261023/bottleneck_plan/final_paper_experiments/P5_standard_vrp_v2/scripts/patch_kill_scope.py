from pathlib import Path
p = Path("scripts/p5v2_kill_dev.py")
t = p.read_text(encoding="utf-8")
t = t.replace("\"for p in $(pgrep -f 'unified_worker[.]py'); do kill $p 2>/dev/null; done\",",
              "\"for p in $(pgrep -f 'dev01/unified_worker[.]py'); do kill $p 2>/dev/null; done\",")
t = t.replace("\"for p in $(pgrep -f 'unified_runner[.]py'); do kill $p 2>/dev/null; done\",",
              "\"for p in $(pgrep -f 'dev01/unified_runner[.]py'); do kill $p 2>/dev/null; done\",")
t = t.replace("echo \\\"[workers] $(pgrep -fc 'unified_worker[.]py' || true)\\\"",
              "echo \\\"[workers] $(pgrep -fc 'dev01/unified_worker[.]py' || true)\\\"")
t = t.replace("echo \\\"[runners] $(pgrep -fc 'unified_runner[.]py' || true)\\\"",
              "echo \\\"[runners] $(pgrep -fc 'dev01/unified_runner[.]py' || true)\\\"")
p.write_text(t, encoding="utf-8")
print("kill script scoped to dev01")
