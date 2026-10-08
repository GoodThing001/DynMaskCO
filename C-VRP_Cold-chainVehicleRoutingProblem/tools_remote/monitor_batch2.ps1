# 第二监控：S3-3 全量 ×2 + density 复验 ×2（每 5 分钟，文件日志 + 存活守卫）
$log = 'C-VRP_Cold-chainVehicleRoutingProblem\results\monitor_batch2.log'
$base = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results'
$dirs = 'a1_s3_3_op5_n40_full_1002_063636 a1_s3_3_op10_n40_full_1002_063636 a1_rr3_robust_20260930_density_40_1002_145359 a1_rr3_robust_20261001_density_40_1002_145359' -split ' '
function Log($msg) {
    $line = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}
for ($i = 1; $i -le 288; $i++) {
    $remote = ''
    foreach ($d in $dirs) {
        $name = $d
        $remote += 'f=' + $base + '/' + $d + '/run.exit; test -f $f && echo EXIT_' + $name + ' $(cat $f | tr "\n" " "); '
        $remote += 'g=' + $base + '/' + $d + '/adjudication.json; test -f $g && echo ADJ_' + $name + '_READY; '
    }
    $remote += 'for n in s3_3_op5 s3_3_op10; do c=' + $base + '/a1_' + '${n}' + '_n40_full_1002_063636/progress_p_c=\(10,20,30\)_cond_hist.txt; test -f $c && echo PROG_' + '${n}' + ' $(wc -l < $c)/40; done; '
    $remote += 'echo LOAD $(uptime | sed ' + "'s/.*average/average/')"
    $out = python tools/_srv.py $remote 2>$null
    $lines = @($out | Select-String -Pattern 'EXIT_|ADJ_|PROG_|LOAD ' | ForEach-Object { $_.Line.Trim() })
    Log ('POLL ' + $i)
    $lines | ForEach-Object { Log ('  ' + $_) }
    $done = (($lines -join ' ') -match 'EXIT_a1_s3_3_op5' -and ($lines -join ' ') -match 'EXIT_a1_s3_3_op10' -and
             ($lines -join ' ') -match 'EXIT_a1_rr3_robust_20260930' -and ($lines -join ' ') -match 'EXIT_a1_rr3_robust_20261001')
    if ($done) {
        Log 'ALL_BATCH2_DONE'
        break
    }
    Start-Sleep -Seconds 300
}
Log 'MONITOR2_END'
