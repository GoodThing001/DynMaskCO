#!/bin/bash
# 全量语法自检：把本流水线会用到的 shell 脚本都 `bash -n` 一遍。
# 起因：2026-10-13 我用 Python 文本模式重写 _formal_v2_coenrich.sh / _formal_v2_mapt.sh 时，
# Windows 换行翻译把它们变成了 CRLF，bash 立刻报 `$'do\r'` / `unexpected EOF` ——
# 若没及时发现，CO-enriched 会「启动即死」并让 xf2_after 白等 40 h。
# ⇒ 批量改过 shell 文件之后，必须跑一次这个（只报 OK/FAIL，不做任何修改）。
cd /home/hzeng/project/MASKCO-Main/CC_Compare || exit 1
fail=0
for f in _formal_v2_*.sh _gpu_leg_1003_fallback.sh tools/_batch_status.sh; do
  [ -f "$f" ] || continue
  if bash -n "$f" 2>/tmp/_bn_err; then
    echo "OK   $f"
  else
    echo "FAIL $f  ->  $(head -1 /tmp/_bn_err)"
    fail=$((fail + 1))
  fi
  if grep -q $'\r' "$f" 2>/dev/null; then
    echo "     ** 含 CRLF（bash 会报 \$'do\\r'）** $f"
    fail=$((fail + 1))
  fi
done
echo "语法自检：失败 $fail 个"
