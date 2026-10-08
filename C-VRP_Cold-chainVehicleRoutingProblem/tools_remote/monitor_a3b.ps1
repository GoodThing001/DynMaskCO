# 第三监控 v2：A3 在线比较 9 批（B/C/D × 三种子）——全部落盘（run.exit 且 gate.json）即结束
$log = 'C-VRP_Cold-chainVehicleRoutingProblem\results\monitor_a3b.log'
$base = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results'
$dirs = @()
foreach ($arm in @('B', 'C', 'D')) {
    foreach ($seed in @('20260926', '20260930', '20261001')) {
        $dirs += ('a1_a3_online_density_' + $arm + '_' + $seed + '_40_1003_040454')
    }
}
function Log($msg) {
    $line = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}
for ($i = 1; $i -le 480; $i++) {
    $remote = ''
    foreach ($d in $dirs) {
        $remote += 'f=' + $base + '/' + $d + '/run.exit; test -f $f && echo EXIT_' + $d + ' $(cat $f | tr "\n" " "); '
        $remote += 'g=' + $base + '/' + $d + '/gate.json; test -f $g && echo GATE_' + $d + '_READY; '
    }
    $remote += 'echo LOAD $(uptime | sed ' + "'s/.*average/average/')"
    $out = python tools/_srv.py $remote 2>$null
    $lines = @($out | Select-String -Pattern 'EXIT_|GATE_|LOAD ' | ForEach-Object { $_.Line.Trim() })
    Log ('POLL ' + $i + ' ' + (($lines | Where-Object { $_ -like 'LOAD *' }) -join ' '))
    $lines | Where-Object { $_ -notlike 'LOAD *' } | ForEach-Object { Log ('  ' + $_) }
    $nExit = ($lines | Where-Object { $_ -like 'EXIT_*' }).Count
    $nGate = ($lines | Where-Object { $_ -like 'GATE_*_READY' }).Count
    if ($nExit -ge 9 -and $nGate -ge 9) {
        Log 'ALL_9_A3_BATCHES_LANDED'
        break
    }
    Start-Sleep -Seconds 300
}
Log 'MONITOR3B_END'
