# 3080 GPU 腿编排（本地后台运行，全程无人值守）：
#   1) 等 1003 新基线 gate 落盘 → 回传本地 → 上传到 3080
#   2) 在 3080 上用 WMI 起 4 个脱离会话的 per-method 进程（彼此并行，各自跑两档）
#   3) 等 4 个 <tag>.done
#   4) 把 8 个 gate.json 从 3080 回传到 1003 的 results/a1_ext_formal2/<臂>/
#   5) 打印摘要
# 用法：pwsh -NoProfile -File CC_Compare/tools/_gpu3080_orchestrate.ps1
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$CFG3080 = '.vscode/sftp.3080.json'
$BASE_REL = 'C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1fresh_reveal_40/gate.json'
$EXT = 'C-VRP_Cold-chainVehicleRoutingProblem'
$TAGS = @('mvmoe', 'routefinder', 'cada', 'rrnco')
$ARMS = @()
foreach ($t in $TAGS) { $ARMS += "${t}_0"; $ARMS += "${t}_51015" }
$REMOTE_OUT = 'results/a1_ext_formal2_3080'

function Srv([string]$cmd, [int]$timeout = 300) {
    python tools\_srv.py $cmd $timeout 2>&1 | Select-String -NotMatch 'Deprecation|cipher|algorithms'
}
function Srv3080([string]$cmd, [int]$timeout = 300) {
    python tools\_srv.py $cmd $timeout $CFG3080 2>&1 | Select-String -NotMatch 'Deprecation|cipher|algorithms'
}

Write-Output "GPU3080_ORCH_START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ---- 1) 等新基线 ----
$deadline = (Get-Date).AddHours(6)
$ready = $false
while ((Get-Date) -lt $deadline) {
    $r = Srv "bash -lc 'test -f /home/hzeng/project/MASKCO-Main/$BASE_REL && echo YES || echo NO'"
    if (($r | Out-String) -match 'YES') { $ready = $true; break }
    Start-Sleep -Seconds 180
}
if (-not $ready) { Write-Output 'BASE_TIMEOUT'; exit 2 }
Write-Output "BASE_READY $(Get-Date -Format 'HH:mm:ss')"

Write-Output '== 回传新基线并上传 3080 =='
python tools\_sftp_get.py $BASE_REL 2>&1 | Select-String -Pattern 'downloaded'
python tools\_sftp_put.py $BASE_REL --config $CFG3080 2>&1 | Select-String -Pattern 'uploaded'

# ---- 1b) 等 3080 的 cc_compare env 就绪 ----
# routefinder/cada/rrnco 的 provider 需要 rl4co + torchrl，必须跑在 cc_compare
# （torch 2.11.0+cu128，对齐 1003 的 /home/hzeng/envs/cc_compare）。
Write-Output '== 等 3080 cc_compare env =='
$deadline = (Get-Date).AddHours(2)
$envReady = $false
while ((Get-Date) -lt $deadline) {
    # 功能性检查（不看 .done 标记，因为建环境脚本无论如何都会写它）：
    # ENVOK 只有在 torch+CUDA+rl4co+torchrl+tensordict 都能 import 时才打印。
    $r = Srv3080 'C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_check_cc_compare.cmd' 300
    if (($r | Out-String) -match 'ENVOK') { $envReady = $true; break }
    Start-Sleep -Seconds 120
}
if (-not $envReady) { Write-Output 'CC_COMPARE_ENV_TIMEOUT'; exit 3 }
Write-Output 'CC_COMPARE_ENV_READY'
Srv3080 'powershell -NoProfile -Command "Select-String -Path C:\Users\zh\gpu3080_logs\cc_compare_setup.log -Pattern VERIFY,CUDA_AVAIL | Select-Object -Last 3"' 300

# ---- 2) 起 4 个方法进程 ----
Write-Output '== 清理旧 .done 标记 =='
foreach ($t in $TAGS) { Srv3080 ('Remove-Item -Force -ErrorAction SilentlyContinue C:\Users\zh\gpu3080_logs\' + $t + '.done') | Out-Null }
Write-Output '== 在 3080 上启动 4 个 per-method 进程（WMI 脱离会话）=='
foreach ($t in $TAGS) {
    $cmdline = 'cmd /c C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_gpu3080\_gpu3080_' + $t + '.cmd'
    $ps = 'Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="' + $cmdline + '"} | Select-Object -ExpandProperty ProcessId'
    Write-Output ("  launch {0}: {1}" -f $t, ((Srv3080 $ps) | Out-String).Trim())
}
Start-Sleep -Seconds 240
Srv3080 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv' 300

# ---- 3) 等 4 个 done ----
Write-Output '== 等 4 个方法完成 =='
$deadline = (Get-Date).AddHours(30)
while ((Get-Date) -lt $deadline) {
    $c = 0
    foreach ($t in $TAGS) {
        $r = Srv3080 ('Test-Path C:\Users\zh\gpu3080_logs\' + $t + '.done')
        if (($r | Out-String) -match 'True') { $c++ }
    }
    Write-Output ("[{0}] done {1}/4" -f (Get-Date -Format 'HH:mm:ss'), $c)
    if ($c -ge 4) { break }
    Start-Sleep -Seconds 300
}

# ---- 4) 回传 8 个 gate.json 到 1003 ----
# 2026-10-13 修复（**在拷贝步骤首次真正执行之前**由代码审查发现）：
#   原实现直接 `Copy-Item -Force <本地暂存> <本地 $EXT/results/a1_ext_formal2/<臂>/gate.json>`，
#   但 ①该目标目录在本地并不存在 ②`Copy-Item` **不会创建父目录** ③随后的
#   `_sftp_put.py` 又因本地文件不存在而失败 ④`$ErrorActionPreference='Continue'` 把
#   两次失败都吞掉 ⇒ 8 个 gate **永远不会到达 1003**，下游 `_local_finalize.ps1` 白等 24 h
#   超时，那 4 个 GPU 方法静默回落到 v1 臂（GROUP_RANK），只有 `check_arm_provenance.py`
#   会在一天后才报出来。
#   现统一委托给 `_gpu3080_copyback.ps1`：建父目录 + 下载/拷贝/上传 + **从 1003 读回大小逐字节校验**，
#   并打印明确的 COPYBACK_PASS / COPYBACK_FAIL。
Write-Output '== 回传各臂 gate.json 到 1003（委托 _gpu3080_copyback.ps1）=='
& pwsh -NoProfile -File CC_Compare\tools\_gpu3080_copyback.ps1

Write-Output '== 3080 GPU 腿各臂摘要（1003 上核对）=='
$py = @'
import glob, json, os
for p in sorted(glob.glob('results/a1_ext_formal2/*/gate.json')):
    name = os.path.basename(os.path.dirname(p))
    if not any(name.startswith(t) for t in ('mvmoe','routefinder','cada','rrnco')):
        continue
    d = json.load(open(p, encoding='utf-8'))
    for yr, yb in (d.get('results') or {}).items():
        pid = yb.get('_pairing_identity') or {}
        for pname, blk in yb.items():
            if pname.startswith('_') or not isinstance(blk, dict): continue
            for an, ab in (blk.get('arms') or {}).items():
                if an in ('cond_hist','uncond_hist','explicit_feat'): continue
                u = ab.get('utility_all_days') or {}
                print('  %-18s %-16s util=%-9s to=%-6s hfr=%s paired=%s' % (
                    name, pname, ('%.1f'%u['mean']) if u.get('mean') is not None else 'n/a',
                    ab.get('mean_timeouts'), ab.get('hard_feasible_rate'),
                    pid.get('identity_verified')))
'@
$py | Out-File -Encoding UTF8 "_tmp_3080_summary.py"
python tools\_srv.py "bash -lc 'cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem && /home/hzeng/envs/a1_ext/bin/python ../_tmp_3080_summary.py'" 600 2>&1 |
    Select-String -NotMatch 'Deprecation|cipher|algorithms'

Write-Output 'GPU3080_ORCH_DONE'
