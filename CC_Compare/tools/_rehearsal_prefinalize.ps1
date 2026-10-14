# 收口前排演（pre-finalize rehearsal）：用**真实数据**把 collect -> render -> 门禁全链跑一遍，
# 全程写临时位置，**不碰交付物**。
#
# 为什么需要这个脚本（2026-10-13 的教训）：
#   「判据永远不可能通过」这类缺陷（缺陷 J：两个判定器要求全部正式臂都有配对身份，
#   而 6 个纯规则臂按设计没有身份块）**只有用真实数据完整跑一遍才会暴露**。
#   如果只在收口时跑第一次，就等于拿交付物做实验 —— 最可能的结果是用户拿到一份写满 FAIL
#   的 v2 摘要，并误以为十几个小时的批次白跑了。
#
# 做四件事：
#   1) 服务端 collect_compare_table.py -> CC_Compare/results/a1_compare_table/_rehearsal_dataset.json
#      （临时文件，脚本结束时本地+远端都删除；产物名带 `_rehearsal` 前缀，绝不用交付文件名）
#   2) 回传本地 -> 渲染 MD/Excel 到 $env:TEMP
#   3) 跑 4 道门禁：表一致性(MD vs Excel)、臂来源批次、方法覆盖、GPU 设备口径
#   4) 打印 RH_PASS / RH_FAIL 汇总
#
# 不包含 `check_readme_numbers.py`：它把 README 数字与**交付目录的** dataset.json 对比，
# 只有在交付物已刷新之后才有意义（排演阶段比的是旧 dataset，结论无意义）。
#
# 用法：pwsh -NoProfile -File CC_Compare/tools/_rehearsal_prefinalize.ps1
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$TBL = 'CC_Compare/results/a1_compare_table'
$TMPREL = "$TBL/_rehearsal_dataset.json"
$work = Join-Path $env:TEMP ('rehearsal_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$EXT = 'C-VRP_Cold-chainVehicleRoutingProblem'

Write-Output "REHEARSAL_START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
New-Item -ItemType Directory -Force -Path $work | Out-Null

$rc = @{}
try {
    Write-Output '== [1/5] 服务端采集（写临时文件，不覆盖 dataset.json）=='
    python tools\_srv.py "bash -lc 'cd /home/hzeng/project/MASKCO-Main/$EXT && /home/hzeng/envs/cc_compare/bin/python ../CC_Compare/tools/collect_compare_table.py --json ../$TMPREL 2>&1 | tail -3'" 1800 2>&1 |
        Select-String -NotMatch 'Deprecation|cipher|algorithms'

    Write-Output '== [2/5] 回传 + 渲染到临时目录 =='
    python tools\_sftp_get.py $TMPREL 2>&1 | Select-String -Pattern 'downloaded|FAIL'
    if (-not (Test-Path ($TMPREL -replace '/', '\'))) { Write-Output 'RH_FAIL: 回传失败'; return }
    python CC_Compare\tools\render_compare_table.py --data $TMPREL --outdir $work 2>&1 |
        Select-Object -Last 2

    Write-Output '== [3/5] 复现收口链的**其余渲染步骤**（收口会按这个顺序跑，任一步崩溃都会中断）=='
    # 2026-10-13 补：收口链是 table -> figure -> ... -> summary。此前排演只跑 table，
    # 若 figure/summary 在新数据集形状（19 方法 / 新增参数量列 / 新 arm key）上崩溃，
    # 收口会**在门禁之前**中断，交付物里既没有表也没有图。故排演必须覆盖它们。
    Write-Output '--- 配图（render_compare_figure.py）---'
    python CC_Compare\tools\render_compare_figure.py --data $TMPREL --outdir $work 2>&1 | Select-Object -Last 3
    $rc['figure'] = $LASTEXITCODE
    $fig = Get-ChildItem $work -Filter '*.png' -ErrorAction SilentlyContinue
    Write-Output ('    PNG 产出: ' + $(if ($fig) { ($fig | ForEach-Object { $_.Name + ' (' + $_.Length + ' B)' }) -join ', ' } else { '**无**' }))

    Write-Output '--- v2 结果摘要（render_v2_summary.py）---'
    python CC_Compare\tools\render_v2_summary.py --data $TMPREL --outdir $work 2>&1 | Select-Object -Last 2
    $rc['v2_summary'] = $LASTEXITCODE

    Write-Output '== [4/5] 门禁 =='
    $md = Join-Path $work 'A-v1对比方法总表.md'
    $xl = Join-Path $work 'A-v1对比方法总表.xlsx'

    Write-Output '--- 表一致性（MD vs Excel 逐列）---'
    python CC_Compare\tools\check_table_consistency.py --md $md --xlsx $xl 2>&1 | Select-Object -Last 4
    $rc['table_consistency'] = $LASTEXITCODE

    Write-Output '--- 臂来源批次（防静默回落 v1）---'
    python CC_Compare\tools\check_arm_provenance.py --data $TMPREL 2>&1 | Select-Object -Last 10
    $rc['arm_provenance'] = $LASTEXITCODE

    Write-Output '--- 方法覆盖（静默丢弃 / 自相矛盾 / 19 key / 身份缺口）---'
    python CC_Compare\tools\check_method_coverage.py --data $TMPREL 2>&1 | Select-Object -Last 10
    $rc['method_coverage'] = $LASTEXITCODE

    Write-Output '--- GPU 设备口径（读交付目录已回传的 gate）---'
    python CC_Compare\tools\check_gpu_device.py 2>&1 | Select-Object -Last 4
    $rc['gpu_device'] = $LASTEXITCODE

    Write-Output '== [5/5] 汇总 =='
    foreach ($k in $rc.Keys) {
        Write-Output ("  {0,-18} rc={1} {2}" -f $k, $rc[$k], $(if ($rc[$k] -eq 0) { 'OK' } else { 'CHECK' }))
    }
    Write-Output ('RH_SUMMARY ok={0}/{1}' -f (@($rc.Values | Where-Object { $_ -eq 0 }).Count), $rc.Count)
    Write-Output '（注：rc!=0 不一定是缺陷 —— 批次未跑完时 arm_provenance/method_coverage/gpu_device 本就会报缺，属预期。）'
}
finally {
    Write-Output '== 清理临时文件（本地 + 远端）=='
    python tools\_srv.py "bash -lc 'rm -f /home/hzeng/project/MASKCO-Main/$TMPREL'" 300 2>&1 | Out-Null
    Remove-Item -Force ($TMPREL -replace '/', '\') -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    Write-Output ('  local  rehearsal_dataset 残留: ' + (Test-Path ($TMPREL -replace '/', '\')))
    Write-Output ('  local  render dir 残留: ' + (Test-Path $work))
}
Write-Output "REHEARSAL_END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
