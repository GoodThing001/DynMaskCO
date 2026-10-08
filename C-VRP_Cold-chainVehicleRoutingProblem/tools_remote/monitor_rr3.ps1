# rr3 落盘监控（文件日志版）：每 5 分钟探测 run.exit/adjudication.json，
# 结果同时写 monitor_rr3.log 与标准输出；双批齐备即告警退出。
$log = 'C-VRP_Cold-chainVehicleRoutingProblem\results\monitor_rr3.log'
$base = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results'
$remote = 'for d in reveal density; do f=' + $base + '/a1_rr3_${d}_40_1001_234935/run.exit; test -f $f && echo EXIT_${d} $(cat $f); done; ' +
          'for d in reveal density; do g=' + $base + '/a1_rr3_${d}_40_1001_234935/adjudication.json; test -f $g && echo ADJ_${d}_READY; done; ' +
          'for d in reveal density; do g=' + $base + '/a1_rr3_${d}_40_1001_234935/gate.json; test -f $g && echo GATE_${d}_READY; done; ' +
          'echo WORKERS $(pgrep -c -P 2364137 2>/dev/null || echo 0)/$(pgrep -c -P 2364142 2>/dev/null || echo 0); ' +
          'echo LOAD $(uptime | sed ' + "'s/.*average/average/')"
function Log($msg) {
    $line = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}
for ($i = 1; $i -le 144; $i++) {
    $out = python tools/_srv.py $remote 2>$null
    $load   = @($out | Select-String -Pattern 'LOAD '  | ForEach-Object { $_.Line.Trim() })
    $exits  = @($out | Select-String -Pattern 'EXIT_'  | ForEach-Object { $_.Line.Trim() })
    $adjs   = @($out | Select-String -Pattern 'ADJ_'   | ForEach-Object { $_.Line.Trim() })
    $gates  = @($out | Select-String -Pattern 'GATE_'  | ForEach-Object { $_.Line.Trim() })
    $wkr    = @($out | Select-String -Pattern 'WORKERS ' | ForEach-Object { $_.Line.Trim() })
    Log ('POLL ' + $i + ' probe=' + ($load -join ' | ') + ' ' + ($wkr -join ' '))
    if ($exits.Count -gt 0) { $exits | ForEach-Object { Log ('  ' + $_) } }
    else { Log '  no run.exit yet' }
    if ($adjs.Count -gt 0) { $adjs | ForEach-Object { Log ('  ' + $_) } }
    if ($gates.Count -gt 0) { $gates | ForEach-Object { Log ('  ' + $_) } }
    if (($wkr -join ' ' -match 'WORKERS 0/9|WORKERS 9/0') -or
        (($wkr -join ' ' -match 'WORKERS 0/0') -and $exits.Count -eq 0)) {
        Log '  ALERT: worker 数异常（可能静默死亡），需人工核查'
    }
    $doneReveal  = ($exits -join ' ') -match 'EXIT_reveal'
    $doneDensity = ($exits -join ' ') -match 'EXIT_density'
    $adjReveal   = ($adjs  -join ' ') -match 'ADJ_reveal_READY'
    $adjDensity  = ($adjs  -join ' ') -match 'ADJ_density_READY'
    if ($doneReveal -and $doneDensity -and $adjReveal -and $adjDensity) {
        Log 'BOTH_ARMS_DONE_AND_ADJUDICATED'
        break
    }
    Start-Sleep -Seconds 300
}
Log 'MONITOR_END'
