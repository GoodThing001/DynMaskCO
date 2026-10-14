# 3080 side watcher: wait for rrnco.done, then run the RRNCO native-face C2
# (weight-causality) probe, closing the known gap registered in
# CC_Compare/results/a1_compare_table/模型参与证据.md.
#
# 2026-10-13 两次修复记录（都是实测踩到才发现的）：
#  1) **SIGALRM**：`verify_*.py` 的墙钟熔断用了 `signal.SIGALRM`，而 Windows 没有该属性
#     ⇒ 探针在 3080 上**两次尝试都秒退**、零产物（`.done` 里只留 `rc=1 ok=False`）。
#     已在两个 py 里按平台能力启用（无 SIGALRM 时打印说明并跳过）。
#  2) **`conda run` 在本机不可靠**：第二次重跑时 `conda.exe` 自己抛
#     `NativeCommandError` + ERROR REPORT（`conda\exceptions.py`），探针根本没启动。
#     改用**环境直连解释器**（实测 `D:\anaconda\envs\cc_compare\python.exe` =
#     py3.11.17 / torch 2.11.0+cu128，`rl4co/tensordict/torchrl` 均可 import）。
#  3) 日志编码：PS 5.1 的 `*>>` 写 UTF-16，读起来字字带空格 ⇒ 改为 `Out-File -Encoding utf8`。
$ErrorActionPreference = 'Continue'
$Root = 'C:\Users\zh\project\MASKCO-Main'
$Logs = 'C:\Users\zh\gpu3080_logs'
if (-not (Test-Path $Logs)) { New-Item -ItemType Directory -Path $Logs | Out-Null }
$done = Join-Path $Logs 'rrnco.done'
$log  = Join-Path $Logs 'rrnco_c2.log'
$mark = Join-Path $Logs 'rrnco_c2.done'
$json = Join-Path $Root 'CC_Compare\results\a1_compare_table\participation\rrnco.json'
if (Test-Path $mark) { Remove-Item $mark -Force }
"== RRNCO_C2 WAIT START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" |
    Out-File -FilePath $log -Encoding utf8

$n = 0
while (-not (Test-Path $done)) {
  Start-Sleep -Seconds 60
  $n++
  if ($n -ge 480) {
    "== TIMEOUT after $n min: rrnco.done never appeared ==" |
        Out-File -Append -FilePath $log -Encoding utf8
    "RRNCO_C2_TIMEOUT $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $mark -Encoding utf8
    exit 4
  }
}
"== rrnco.done seen after $n min; $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==" |
    Out-File -Append -FilePath $log -Encoding utf8

Set-Location $Root
$out = Join-Path $Root 'CC_Compare\results\a1_compare_table\participation'

# 优先用环境直连解释器；不存在再退回 conda run（本机实测 conda run 会崩，仅供兜底）
$PY = 'D:\anaconda\envs\cc_compare\python.exe'
$useConda = -not (Test-Path $PY)
"---- interpreter: $(if ($useConda) { 'conda run (fallback)' } else { $PY }) ----" |
    Out-File -Append -FilePath $log -Encoding utf8

function Get-C2Ok {
  if (-not (Test-Path $json)) { return $false }
  try {
    $j = Get-Content $json -Raw | ConvertFrom-Json
    if ($j.metrics -and ($null -ne $j.metrics.order_changed_by_weights)) { return $true }
  } catch { }
  return $false
}
function Invoke-Probe([string]$dev) {
  "---- PASS: --device $dev ----" | Out-File -Append -FilePath $log -Encoding utf8
  if ($useConda) {
    & conda run -n cc_compare python CC_Compare\tools\verify_rrnco_c2.py --device $dev --outdir $out 2>&1 |
        Out-File -Append -FilePath $log -Encoding utf8
  } else {
    & $PY CC_Compare\tools\verify_rrnco_c2.py --device $dev --outdir $out 2>&1 |
        Out-File -Append -FilePath $log -Encoding utf8
  }
  "-- rc=$LASTEXITCODE --" | Out-File -Append -FilePath $log -Encoding utf8
  return $LASTEXITCODE
}

$rc = Invoke-Probe 'cpu'          # 先按与其它 11 个 provider 相同的口径（CPU 单线程）
$device = 'cpu'
$ok = Get-C2Ok
if (-not $ok) {
  "---- cpu pass gave no order_changed_by_weights; retry with --device cuda (idle GPU) ----" |
      Out-File -Append -FilePath $log -Encoding utf8
  $rc = Invoke-Probe 'cuda'
  $device = 'cuda'
  $ok = Get-C2Ok
}

"-- rrnco_c2 rc=$rc ok=$ok device=$device --" | Out-File -Append -FilePath $log -Encoding utf8
"RRNCO_C2_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') rc=$rc ok=$ok device=$device" |
    Out-File -FilePath $mark -Encoding utf8
exit 0
