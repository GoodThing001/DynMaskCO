# 本地「守候 + 自动收口」：等服务器流水线打印「收口完成」后，
# 自动回传数据、重出 Markdown/Excel/图、做跨格式与冻结对账，最后打印 FINALIZED。
# 用法（仓库根）：pwsh -File CC_Compare/tools/_local_finalize.ps1
#                 pwsh -File CC_Compare/tools/_local_finalize.ps1 -SkipWait   # 事后补跑
#
# `-SkipWait`（2026-10-13 新增，**为恢复场景而加**）：
#   正常路径是「先守着、等 收口完成 出现再干活」。但**新鲜度守卫**要求 after.log 的 mtime
#   **大于本次 watcher 的启动时刻** ⇒ **在流水线已经收口之后**才启动 watcher，
#   它会把早已写好的 收口完成 判为「陈旧」并**空等满 72 h**，等于**无法补跑**。
#   而本地收口是唯一产出 **Excel / 配图 / 6 道门禁汇总 / 冻结对账 / README §2 文本** 的环节
#   （服务器侧只出 Markdown + dataset.json），一旦本地守护进程随会话消失，就必须能补跑。
#   ⇒ 事后手动补跑请加 `-SkipWait`：跳过等待，直接执行收口链（其余步骤完全不变）。
param([switch]$SkipWait)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$TBL = 'CC_Compare/results/a1_compare_table'
$deadline = (Get-Date).AddHours(72)
$ok = $false
# 新鲜度守卫：只认「本 watcher 启动**之后**被写过的」after.log 里的 收口完成。
# 起因（2026-10-13 实测踩到）：after.log 在流水线重启时被 `>` 截断重写，若 watcher 恰好在
# 截断前读到**上一轮遗留**的 收口完成，就会立刻误判「流水线已完成」，进而生成一份内容全为
# FAIL 的陈旧 `V2批次结果.md`（17:27 那份即如此：当时四条腿 18:25 才启动）。
$t0 = [int](Get-Date -UFormat %s)

Write-Output "WATCHER_START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($SkipWait) {
    # 人工确认流水线已收口（或明确要求现在就重出产物）⇒ 跳过**新鲜度**要求，直接干活。
    # 但**仍然要求** after.log 里确实出现过 收口完成 —— 否则就成了「在流水线没跑完时
    # 强行收口」，会生成一份看起来权威、实则陈旧的摘要（正是 17:27 那份 STALE 文件的缺陷类）。
    $chk = python tools\_srv.py "bash -lc 'grep -c 收口完成 /tmp/xformal_v2/after.log || true'" 120 2>&1
    $found = (($chk | Out-String) -match '(?m)^\s*[1-9]\d*\s*$')
    if (-not $found) {
        Write-Output 'SKIP_WAIT_REFUSED: after.log 里没有「收口完成」⇒ 拒绝收口（避免生成陈旧/过早摘要）'
        Write-Output '  （若确实只想要当前状态的重出，请先确认流水线已完成；本开关只跳过「新鲜度」判定。）'
        exit 2
    }
    $ok = $true
    Write-Output 'SKIP_WAIT: after.log 已出现「收口完成」；跳过新鲜度判定，直接执行收口链'
}
while ((-not $SkipWait) -and ((Get-Date) -lt $deadline)) {
    $out = python tools\_srv.py "bash -lc 'stat -c %Y /tmp/xformal_v2/after.log 2>/dev/null || echo 0; tail -3 /tmp/xformal_v2/after.log'" 120 2>&1 |
            Select-String -NotMatch 'Deprecation|cipher|algorithms'
    $lines = @($out | ForEach-Object { $_.ToString() })
    $mt = 0
    if ($lines.Count -gt 0) { [void][int]::TryParse((($lines[0] -replace '\D', '')), [ref]$mt) }
    $s = ($lines | Out-String)
    if (($s -match '收口完成') -and ($mt -gt $t0)) { $ok = $true; break }
    if ($s -match '收口完成') {
        Write-Output ("[{0}] 忽略陈旧的 收口完成（log mtime={1} <= watcher start={2}）" -f (Get-Date -Format 'HH:mm:ss'), $mt, $t0)
    } else {
        Write-Output ("[{0}] waiting..." -f (Get-Date -Format 'HH:mm:ss'))
    }
    Start-Sleep -Seconds 300
}

if (-not $ok) { Write-Output 'WATCHER_TIMEOUT'; exit 2 }
Write-Output "PIPELINE_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ---- 收口前必查：3080 GPU 腿的 8 个 gate 是否已回传到 1003 ----
# 为什么需要：`_formal_v2_after.sh` 只等 1003 侧的 gpu/cpuA/cpuB/cpuC 四个 marker，而
# `gpu.done` 是**空操作**（GPU 腿已搬到 3080）。3080 的 gate 由本地
# `_gpu3080_orchestrate.ps1` 独立回传。若服务端 [5/5] 采集早于回传，则
# mvmoe/routefinder/cada/rrnco 会静默回落到 v1 臂（GROUP_RANK 关系），总表混入旧口径。
# 所以这里先等齐 8 个 gate，再**重跑一次服务端采集**，最后才下载 dataset.json。
Write-Output '== 等 3080 GPU 腿 8 个 gate 回传到 1003 =='
$GPU_ARMS = @('mvmoe_0', 'mvmoe_51015', 'routefinder_0', 'routefinder_51015',
              'cada_0', 'cada_51015', 'rrnco_0', 'rrnco_51015')
$gdeadline = (Get-Date).AddHours(24)
$allGpu = $false
while ((Get-Date) -lt $gdeadline) {
    $missing = @()
    foreach ($a in $GPU_ARMS) {
        $r = python tools\_srv.py "bash -lc 'test -f /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_ext_formal2/$a/gate.json && echo YES || echo NO'" 120 2>&1
        if (($r | Out-String) -notmatch 'YES') { $missing += $a }
    }
    if ($missing.Count -eq 0) { $allGpu = $true; break }
    Write-Output ("[{0}] 等 GPU 腿回传，尚缺 {1}: {2}" -f (Get-Date -Format 'HH:mm:ss'), $missing.Count, ($missing -join ','))
    Start-Sleep -Seconds 300
}
if ($allGpu) { Write-Output 'GPU_GATES_READY' } else { Write-Output 'GPU_GATES_TIMEOUT' }

Write-Output '== 重跑服务端采集（把 GPU 腿纳入 dataset.json）=='
python tools\_srv.py "bash -lc 'cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && /home/hzeng/envs/cc_compare/bin/python ../CC_Compare/tools/collect_compare_table.py --json ../CC_Compare/results/a1_compare_table/dataset.json'" 1800 2>&1 |
    Select-Object -Last 6

Write-Output '== 记录服务端结束冻结快照 =='
python tools\_srv.py "bash -lc 'cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && /home/hzeng/envs/cc_compare/bin/python ../CC_Compare/tools/record_freeze_v2.py --json ../CC_Compare/results/a1_compare_table/freeze_v2_end.json'" 600 2>&1 |
    Select-String -NotMatch 'Deprecation|cipher|algorithms'

Write-Output '== 回传数据 =='
python tools\_sftp_get.py "$TBL/dataset.json" "$TBL/freeze_v2_end.json" 2>&1 |
    Select-String -NotMatch 'Deprecation|cipher|algorithms'

Write-Output '== 冻结对账（启动 vs 结束；按**键的并集**分类，不静默）=='
# 2026-10-13 修复：原先 `d=[k for k in a if a[k]!=b.get(k)]` 只遍历 start 的键 ⇒
# 只在 end 的文件（批次中途往 record_freeze_v2.py::FILES 加的新文件）会被**完全静默忽略**。
# 这正是 README §12 R9（两个 v2-only provider 不在冻结清单里）不能靠事后补清单解决的原因。
python -c @"
import json
a=json.load(open(r'$TBL/freeze_v2_start.json',encoding='utf-8'))
b=json.load(open(r'$TBL/freeze_v2_end.json',encoding='utf-8'))
keys=sorted(set(a)|set(b))
chg=[k for k in keys if k in a and k in b and a[k]!=b.get(k)]
add=[k for k in keys if k not in a]
rem=[k for k in keys if k not in b]
print('冻结文件 start %d / end %d / 并集 %d' % (len(a), len(b), len(keys)))
print('  CHANGED %d 个' % len(chg))
for k in chg: print('    CHANGED:', k)
print('  只在 end %d 个（start 无基线 ⇒ 无法判定是否被改过）' % len(add))
for k in add: print('    ADDED_ONLY_IN_END:', k)
print('  只在 start %d 个' % len(rem))
for k in rem: print('    REMOVED:', k)
"@

Write-Output '== 重出表格与配图 =='
python CC_Compare\tools\render_compare_table.py --data "$TBL/dataset.json" --outdir $TBL 2>&1 | Select-Object -Last 4
python CC_Compare\tools\render_compare_figure.py --data "$TBL/dataset.json" --outdir $TBL 2>&1 | Select-Object -Last 2

# ---- 把「README §2 该写什么」落成产物（2026-10-13 新增）------------------------- #
# 动机：验收 §3 第 5 项（文档同步）是全链**唯一必须有人在场**的环节。表格/配图/摘要/门禁
# 都由本脚本自动产出，但 §2 的正文一直是「人工从数据推」。现在直接把可粘贴文本写成文件，
# 即使会话提前结束，用户也只需复制粘贴，不必重新推导（尤其避免我们已踩过的两个口径坑：
# 「最好外部」要区分学习型 vs 经典求解器；「Ours − C1」要用 gate 内配对差而非 C1 基线）。
Write-Output '== 产出 README §2 可粘贴文本（供文档同步）=='
$S2 = Join-Path $TBL 'README_SECTION2_paste.md'
python CC_Compare\tools\print_readme_section2.py --data "$TBL/dataset.json" 2>&1 |
    Out-File -FilePath $S2 -Encoding utf8
if (Test-Path $S2) {
    Write-Output ("  已写 %s（%d B）—— 直接复制其中的表体与派生量到 README §2" -f $S2, (Get-Item $S2).Length)
} else {
    Write-Output '  ⚠️ 未能写出 §2 文本（print_readme_section2.py 失败？）'
}

$GATES = [ordered]@{}
# ---- 耗时扫描落盘（2026-10-13 新增）------------------------------------------- #
# 动机：`超时/天` 与 `最大单决策` 是「必须随每个数字一起报告」的量，而
# **数字对账不能替代耗时扫描** —— 本轮实测：`lih_0`(超时 25.35/天, max 54.35 s) 与
# `symnco_0`(0.075/天, max 17.40 s) 都通过了 v1↔v2 数值对账，却明显被 CPU 争用推高了耗时
# （只有 `symnco_51015` 因偏离 7.9% 被门禁抓出）。故收口时把扫描结果落成产物，
# 让「哪些行的数字必须带超时限定」一眼可见。扫描必须在 1003 上跑（那里才有全部臂的 gate）。
Write-Output '== 耗时扫描（超时/天、最大单决策）→ TIMEOUT_SCAN.md =='
$TS = Join-Path $TBL 'TIMEOUT_SCAN.md'
python tools\_srv.py "bash -lc 'cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && /home/hzeng/envs/cc_compare/bin/python ../CC_Compare/tools/_scan_timeouts.py'" 900 2>&1 |
    Select-String -NotMatch 'Deprecation|cipher|algorithms' |
    Out-File -FilePath $TS -Encoding utf8
if (Test-Path $TS) {
    Write-Output ("  已写 %s（%d B）" -f $TS, (Get-Item $TS).Length)
    Get-Content $TS | Select-String -Pattern '需要带超时' | Select-Object -Last 1
}
python CC_Compare\tools\check_table_consistency.py --md "$TBL/A-v1对比方法总表.md" --xlsx "$TBL/A-v1对比方法总表.xlsx" 2>&1 | Select-Object -Last 3
$GATES['table_consistency'] = $LASTEXITCODE
python CC_Compare\tools\check_readme_numbers.py 2>&1 | Select-Object -Last 3
$GATES['readme_numbers'] = $LASTEXITCODE

Write-Output '== 臂来源批次门禁（防 v2 未生效 → 总表混入 v1 CPU 设备口径）=='
python CC_Compare\tools\check_arm_provenance.py 2>&1 | Select-Object -Last 8
$GATES['arm_provenance'] = $LASTEXITCODE

Write-Output '== GPU 腿设备口径门禁（8 个臂的实际 device 键必须都是 cuda）=='
python CC_Compare\tools\check_gpu_device.py 2>&1 | Select-Object -Last 12
$GATES['gpu_device'] = $LASTEXITCODE

Write-Output '== 方法覆盖门禁（无静默丢弃 / 无自相矛盾 / 19 key 齐 / 依据文档在位）=='
python CC_Compare\tools\check_method_coverage.py 2>&1 | Select-Object -Last 12
$GATES['method_coverage'] = $LASTEXITCODE

Write-Output '== v1↔v2 逐臂对账门禁（rc=0/无超时/身份合格 仍可能是空臂）=='
python tools\_srv.py "bash -lc 'cd /home/hzeng/project/MASKCO-Main && /home/hzeng/envs/cc_compare/bin/python CC_Compare/tools/check_v1_v2_agreement.py'" 900 2>&1 |
    Select-Object -Last 18
$GATES['v1_v2_agreement'] = $LASTEXITCODE
# 已知例外（2026-10-13 定位）：`symnco_51015` 因**本轮 CPU 争用**（96 核上 load≈151，
# 每 worker 吃 5–9 核）把单次 order() 推高 14× ⇒ 9 天超时 ⇒ 效用偏离 7.9%。
# 该臂仍可裁决（hfr=1.0、fail_counts={}），只是数字必须带超时限定。
# 这里在汇总处直接给出解释，避免「看到 GATES_FAIL 就以为批次坏了」。
if ($GATES['v1_v2_agreement'] -ne 0) {
    Write-Output '  ▸ 已知例外：`symnco_51015` 的偏离已定位为 CPU 争用（非方法/权重差异）——'
    Write-Output '    详见 `读表注意与已知限定.md` §2.8 与 README §12 R10；引用该行须同时报 `超时/天`。'
}

Write-Output '== 生成 v2 结果摘要（供无人值守交付）=='
python CC_Compare\tools\render_v2_summary.py --data "$TBL/dataset.json" --outdir $TBL 2>&1 | Select-Object -Last 2

Write-Output '== 各臂身份/裁决与新臂清单 =='
# 2026-10-13 修复：原先打印 `repaired_vs_baseline.cond_hist.formal_adjudication` ——
# 而 `_pair()` **从不产出** `formal_adjudication`，于是该列**恒为 None**，读起来像
# 「没有正式裁决」，与事实（v2 各臂落盘 `formal_adjudication=true`）**相反**。
# 现改读采集器新采的 `gate_adjudication`（gate 内 `<arm>_minus_cond_hist` 的真值），
# 并把两者分开打印，避免再次混淆。
python -c @"
import json
d=json.load(open(r'$TBL/dataset.json',encoding='utf-8'))
v2=[r for r in d['arms'] if r['group']=='a1_ext_formal2']
print('a1_ext_formal2 臂数:', len(v2))
for r in sorted(v2, key=lambda x:(x['method'],x['p_c'])):
    rep=r.get('reported') or {}
    pid=r.get('pairing_identity_recorded') or {}
    ga=r.get('gate_adjudication') or {}
    pd=(r.get('repaired_vs_baseline') or {}).get('cond_hist') or {}
    print('  %-22s %-16s util=%-9s to=%-6s identity_verified=%-5s gate_formal=%-5s re_paired_adjudicable=%s' % (
        r['arm_dir'], r['p_c'],
        ('%.1f'%rep['utility']['mean']) if (rep.get('utility') or {}).get('mean') is not None else 'n/a',
        rep.get('mean_timeouts'), pid.get('identity_verified'),
        ga.get('formal_adjudication'), pd.get('adjudicable')))
"@

# ---- 汇总门禁退出码（2026-10-13 新增）---------------------------------------- #
# 动机：此前每道门禁只把输出 `Select-Object -Last N` 打出来，**完全忽略 exit code** ——
# 无人值守交付时，一个 FAIL 很容易被淹没在上百行日志里（本批已经出现过「判据永远不可能
# 通过」和「列恒为 None」这类**看起来正常其实报错**的输出）。这里把 6 道门禁的 rc 汇总，
# 并给出**一句话总判定**，与 `_rehearsal_prefinalize.ps1` 的 RH_SUMMARY 同一口径。
Write-Output ''
Write-Output '================ 门禁汇总（收口总判定）================'
foreach ($k in $GATES.Keys) {
    Write-Output ("  {0,-20} rc={1}  {2}" -f $k, $GATES[$k], $(if ($GATES[$k] -eq 0) { 'PASS' } else { '**FAIL**' }))
}
$nBad = @($GATES.Values | Where-Object { $_ -ne 0 }).Count
if ($nBad -eq 0) {
    Write-Output '  ⇒ 全部门禁通过（GATES_ALL_PASS）'
} else {
    Write-Output ("  ⇒ 有 {0} 道门禁未通过（GATES_FAIL: {1}）—— 见上文各门禁明细，勿只看最后几行" -f `
        $nBad, (($GATES.Keys | Where-Object { $GATES[$_] -ne 0 }) -join ','))
}

Write-Output 'FINALIZED'
