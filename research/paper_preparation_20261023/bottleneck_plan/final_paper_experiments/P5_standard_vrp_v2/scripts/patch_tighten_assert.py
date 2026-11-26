from pathlib import Path
p = Path("standard_vrp/tests/test_kernels_v2.py")
t = p.read_text(encoding="utf-8")
t = t.replace("        assert elapsed <= 4.0 + 1.0, (variant, elapsed)",
              "        assert elapsed <= 4.0 + 0.3, (variant, elapsed)")
t = t.replace("        assert elapsed <= 3.0 + 1.2, (variant, elapsed)",
              "        assert elapsed <= 3.0 + 0.3, (variant, elapsed)")
p.write_text(t, encoding="utf-8")
print("deadline asserts tightened")
