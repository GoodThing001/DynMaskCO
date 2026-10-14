# 收口后自动跑「文档同步 + 覆盖度 + 数字」三道本地门，并把结论落盘。
#
# 为什么需要：`_local_finalize.ps1`（本地收口）产出 Excel/配图/§2 文本/门禁汇总，
# 但**不含**「散文里的个数与极值是否还跟数据一致」这一项 —— v2 让 coenrich/mapt 进表后，
# README §2/§8/§10 的「13 个外部方法」会变成 15 或 16（取决于 CO-enriched 两模式并成一行
# 还是各占一行），**这是唯一必须人工判断的环节**。
# 本脚本只做「自动检测 + 落盘结论」，**不改任何文档**，避免机器误改散文。
#
# 用法（仓库根）：
#   pwsh -NoProfile -File CC_Compare/tools/_post_finalize_check.ps1 [-Hours 30]
param([double]$Hours = 30, [switch]$SkipWait)

$ErrorActionPreference = 'Continue'
# ⚠️ 必须在这里设：中文 Windows 的默认控制台编码是 GBK，而本文件要跑的 Python 门禁会打印
# 「✅ / ⇒」等字符并读取中文 JSON ⇒ 不设就会**误报 FAIL**：子进程解码报
# `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbb`、
# 子进程输出报 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'`。
# 2026-10-14 实测踩到：`_doc_prose_check.py` 与 `_check_doc_refs.py` 明明 PASS 却双双被记成 rc=1。
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$tableDir = Join-Path $root 'CC_Compare\results\a1_compare_table'
$paste    = Join-Path $tableDir 'README_SECTION2_paste.md'
$outFile  = Join-Path $tableDir '_post_finalize_doccheck.txt'
$t0 = Get-Date
$deadline = $t0.AddHours($Hours)

# ⚠️ 必须用 Write-Host（而非 Tee-Object / Write-Output）输出：后者会把对象塞进**成功流**，
# 于是 Run-Check 里夹带的每一次日志都会成为返回值的一部分，$rcs[...] 拿到的就不是退出码了。
function W([string]$line) {
    Add-Content -Path $outFile -Value $line -Encoding utf8
    Write-Host $line
}

"=== _post_finalize_check 启动 $($t0.ToString('yyyy-MM-dd HH:mm:ss'))（等 $Hours h）===" | Out-File -FilePath $outFile -Encoding utf8

$ready = $false
if ($SkipWait) {
    # 与 `_local_finalize.ps1 -SkipWait` 同一约定：事后启动的 watcher 会把**早已写好的**
    # 收口产物当成陈旧而空等 30 h。但仍要求产物**确实存在**，否则拒绝执行（exit 2），
    # 以免在「还没收口」的树上生成一份看似通过的摘要。
    if (Test-Path $paste) {
        $ready = $true
        W "SkipWait：直接对现有收口产物跑门禁（$paste）"
    } else {
        W "REFUSE：-SkipWait 但 $paste 不存在 ⇒ 可能尚未收口，拒绝生成摘要"
        exit 2
    }
}
while (-not $ready -and (Get-Date) -lt $deadline) {
    if (Test-Path $paste) {
        $m = (Get-Item $paste).LastWriteTime
        if ($m -gt $t0) { $ready = $true; break }
    }
    Start-Sleep -Seconds 300
}

if (-not $ready) {
    W "TIMEOUT：$Hours h 内未见到新写的 README_SECTION2_paste.md（收口可能未完成或本地守护已随会话消失）"
    W "恢复：确认服务器 after.log 出现「收口完成」后，手动执行"
    W "  pwsh -NoProfile -File CC_Compare/tools/_local_finalize.ps1 -SkipWait"
    exit 3
}

W "检测到新收口产物：$paste ($((Get-Item $paste).LastWriteTime))"
Start-Sleep -Seconds 20

function Run-Check([string]$label, [string[]]$argv) {
    W ""
    W "=== $label ==="
    try {
        $o = & python @argv 2>&1 | Out-String
        W $o.Trim()
        W "rc=$LASTEXITCODE"
        return $LASTEXITCODE
    } catch {
        W "EXCEPTION $_"
        return 99
    }
}

$rcs = @{}
$rcs['coverage']    = Run-Check '方法覆盖度'      @('CC_Compare\tools\check_method_coverage.py')
$rcs['readme']      = Run-Check 'README 数字'      @('CC_Compare\tools\check_readme_numbers.py')
$rcs['docprose']    = Run-Check '散文/个数口径'    @('CC_Compare\tools\_doc_prose_check.py')
$rcs['docrefs']     = Run-Check '文档路径引用'     @('CC_Compare\tools\_check_doc_refs.py')
# 2026-10-14 用户要求新增：配对身份（A−B=D 恒等式 + 同源 + 跨协议混用）
$rcs['pairing']     = Run-Check '配对身份恒等式'   @('CC_Compare\tools\check_pairing_identity.py')

W ""
W "=== 汇总 ==="
$bad = @()
foreach ($k in $rcs.Keys | Sort-Object) {
    W ("  {0,-12} rc={1}" -f $k, $rcs[$k])
    if ($rcs[$k] -ne 0) { $bad += $k }
}
if ($bad.Count -eq 0) {
    W "POST_FINALIZE_CHECKS_ALL_PASS"
} else {
    W ("POST_FINALIZE_CHECKS_FAIL: " + ($bad -join ', '))
    W "注：_doc_prose_check 的 FAIL 多数是「个数 13 → 15/16」未改散文，按它打印的应有值改 README §2/§8/§10 后重跑即可。"
}
