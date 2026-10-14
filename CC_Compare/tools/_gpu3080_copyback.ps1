# Copy the 8 GPU-leg gate.json files from the 3080 to 1003 through the local
# workspace bridge, and VERIFY each one actually landed.
#
# Why this exists (2026-10-13, found by review BEFORE the copy step ever ran):
#   `_gpu3080_orchestrate.ps1` step 4 did
#       Copy-Item -Force <local staging> <local $EXT/results/a1_ext_formal2/$a/gate.json>
#   but `Copy-Item` does NOT create missing parent directories, and the local
#   `results/a1_ext_formal2/` did not exist. So every arm failed at Copy-Item,
#   then `_sftp_put.py` failed too (local file missing) -- and because
#   $ErrorActionPreference='Continue' swallowed both, the 8 gates would silently
#   never reach 1003. Downstream `_local_finalize.ps1` would wait 24 h, time out,
#   and the 4 GPU methods would fall back to v1 arms (GROUP_RANK): the table would
#   mix batch provenance for exactly those rows, caught only a day later by
#   `check_arm_provenance.py`.
#
# Design:
#   * INCREMENTAL: every pass copies whatever gates already exist on the 3080, so
#     the 6 non-rrnco gates move to 1003 immediately instead of idling until the
#     last method finishes. Re-runs are idempotent (verified arms are skipped).
#   * VERIFIED: after upload it reads the size back FROM 1003 and requires an
#     exact match, so "uploaded" cannot be mistaken for "landed".
#   * FAIL-LOUD: prints COPYBACK_PASS / COPYBACK_FAIL(name,size) instead of
#     swallowing errors.
#
# Usage: pwsh -NoProfile -File CC_Compare/tools/_gpu3080_copyback.ps1
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$CFG3080 = '.vscode/sftp.3080.json'
$EXT = 'C-VRP_Cold-chainVehicleRoutingProblem'
$TAGS = @('mvmoe', 'routefinder', 'cada', 'rrnco')
$ARMS = @()
foreach ($t in $TAGS) { $ARMS += "${t}_0"; $ARMS += "${t}_51015" }
$STAGE_REL = 'results/a1_ext_formal2_3080'   # 相对扩展根（3080 侧远端路径用这个）
$STAGE = "$EXT/$STAGE_REL"                   # 相对仓库根（本地暂存 / SFTP 路径用这个）
$DEST = "$EXT/results/a1_ext_formal2"
$RROOT = '/home/hzeng/project/MASKCO-Main'

function Srv([string]$cmd, [int]$timeout = 300) {
    python tools\_srv.py $cmd $timeout 2>&1 | Select-String -NotMatch 'Deprecation|cipher|algorithms'
}
function Srv3080([string]$cmd, [int]$timeout = 300) {
    python tools\_srv.py $cmd $timeout $CFG3080 2>&1 | Select-String -NotMatch 'Deprecation|cipher|algorithms'
}

Write-Output "COPYBACK_START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$verified = @{}
$deadline = (Get-Date).AddHours(30)

while ((Get-Date) -lt $deadline) {
    $pending = @($ARMS | Where-Object { -not $verified.ContainsKey($_) })
    if ($pending.Count -eq 0) { break }

    foreach ($a in $pending) {
        $rel = "$STAGE/$a/gate.json"
        # 注意：3080 侧远端路径 = 盘符根 + 扩展目录 + STAGE_REL（**不能**再用 $STAGE，它已含 $EXT）
        $remote3080 = 'C:\Users\zh\project\MASKCO-Main\' + $EXT + '/' + $STAGE_REL + "/$a/gate.json"
        $srcLocal = Join-Path $root ($rel -replace '/', '\')
        $dstRel = "$DEST/$a/gate.json"
        $dstLocal = Join-Path $root ($dstRel -replace '/', '\')

        $r = Srv3080 ('Test-Path "' + $remote3080 + '"') 300
        if (($r | Out-String) -notmatch 'True') { continue }   # 还没跑完，下一轮再看

        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $srcLocal) | Out-Null
        python tools\_sftp_get.py $rel --config $CFG3080 2>&1 | Select-String -Pattern 'downloaded' | Out-Null
        if (-not (Test-Path $srcLocal)) { Write-Output "  FAIL $a : download 未落地"; continue }

        # 关键修复：目标目录必须先建（Copy-Item 不会创建父目录）
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dstLocal) | Out-Null
        Copy-Item -Force $srcLocal $dstLocal
        if (-not (Test-Path $dstLocal)) { Write-Output "  FAIL $a : copy 未落地"; continue }

        $sz = (Get-Item $dstLocal).Length
        python tools\_sftp_put.py $dstRel 2>&1 | Select-String -Pattern 'uploaded' | Out-Null

        # 从 1003 读回真实大小，必须逐字节一致（否则等于没传上去）
        $rr = Srv ("bash -lc 'stat -c %s $RROOT/$dstRel 2>/dev/null || echo 0'") 300
        $rsz = 0
        [void][int64]::TryParse((($rr | Out-String) -replace '\D', ''), [ref]$rsz)
        if ($rsz -eq $sz -and $sz -gt 0) {
            Write-Output ("  OK   {0}  {1} bytes（3080->local->1003 逐字节一致）" -f $a, $sz)
            $verified[$a] = $sz
        } else {
            Write-Output ("  FAIL {0}  local={1} remote1003={2}" -f $a, $sz, $rsz)
        }
    }

    $left = @($ARMS | Where-Object { -not $verified.ContainsKey($_) })
    if ($left.Count -eq 0) { break }
    Write-Output ("[{0}] 已回传 {1}/8，待回传: {2}" -f (Get-Date -Format 'HH:mm:ss'), $verified.Count, ($left -join ','))
    Start-Sleep -Seconds 300
}

$left = @($ARMS | Where-Object { -not $verified.ContainsKey($_) })
Write-Output ("COPYBACK_SUMMARY verified={0}/8 pending={1}" -f $verified.Count, $left.Count)
if ($left.Count -eq 0) {
    Write-Output 'COPYBACK_PASS'
} else {
    Write-Output ('COPYBACK_FAIL: ' + ($left -join ','))
}
Write-Output "COPYBACK_END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
