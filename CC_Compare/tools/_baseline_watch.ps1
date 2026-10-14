# 守候新基线落盘 → 立刻做「同协议复现 + 超时污染」判定（不等四条腿）。
# 判定内容：
#   1) 新基线 gate 是否存在 identity / source_stable；
#   2) 用 compare_two_gates.py 对齐 c1rr2：逐日效用差 + 双方 mean_timeouts；
#      —— 这是判定「负载事故是否污染本批」的直接依据（c1rr2 的 3.5 次/天为参照）。
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$deadline = (Get-Date).AddHours(14)
$gate = 'C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1fresh_reveal_40/gate.json'

Write-Output "BASELINE_WATCH_START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$found = $false
while ((Get-Date) -lt $deadline) {
    $r = python tools\_srv.py "bash -lc 'test -f /home/hzeng/project/MASKCO-Main/$gate && echo YES || echo NO'" 120 2>&1 |
         Select-String -Pattern 'YES'
    if ($r) { $found = $true; break }
    Start-Sleep -Seconds 180
}
if (-not $found) { Write-Output 'BASELINE_WATCH_TIMEOUT'; exit 2 }
Write-Output "BASELINE_READY $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Output '== 回传新基线 gate =='
python tools\_sftp_get.py "$gate" 2>&1 | Select-String -NotMatch 'Deprecation|cipher|algorithms'

Write-Output '== 与新基线 identity 状态 =='
python -c @"
import json
p = r'C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1fresh_reveal_40/gate.json'
d = json.load(open(p, encoding='utf-8'))
i = d.get('identity') or {}
print('identity keys:', len(i), '| source_stable:', i.get('source_stable'))
print('contract:', str(i.get('contract_sha256'))[:16], '| data:', str(i.get('data_sha256'))[:16])
print('budget B:', d.get('budget', {}).get('B'))
for pname, blk in d['gate'].items():
    for an, ab in blk['arms'].items():
        u = ab.get('utility_all_days') or {}
        print('  %-16s %-14s util=%8s to/day=%6.2f hfr=%s' % (
            pname, an,
            ('%.1f' % u['mean']) if u.get('mean') is not None else 'n/a',
            ab.get('mean_timeouts') or 0, ab.get('hard_feasible_rate')))
"@

Write-Output '== 与 c1rr2 逐日对账（判定污染）=='
python CC_Compare\tools\compare_two_gates.py `
    "C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1fresh_reveal_40/gate.json" `
    "C-VRP_Cold-chainVehicleRoutingProblem/results/a1_c1rr2_reveal_40/gate.json" 2>&1 |
    Select-Object -First 60

Write-Output 'BASELINE_CHECK_DONE'
