@echo off
REM ============================================================================
REM 3080 GPU leg, SEQUENTIAL driver (one method at a time).
REM
REM Why not 4 concurrent (the original design):
REM   2026-10-13 实测把 3080 跑到 **commit-limit 崩溃**：PowerShell 起不来
REM   (HRESULT 0x800705AF = ERROR_COMMITMENT_LIMIT)、SSH 被 reset，
REM   `wmic` 显示 TotalVisibleMemorySize≈31.8 GB 而 FreePhysicalMemory 仅 ≈418 MB。
REM   根因：重起前的那次 kill 只杀掉了 1 个进程（我的进程匹配写得太窄），
REM   于是**旧臂 + 新臂同时跑** ⇒ 4 臂 × 4 workers 变成 ~24-32 个 torch 进程 ⇒ 内存耗尽。
REM   （所以这不只是「3080 内存小」，更是「重启没清干净」。）
REM
REM 现在的策略：**串行**跑 4 个方法（每个方法内部仍是 --workers 4）。
REM   峰值内存 ≈ 单臂 4-5 个 worker ≈ 6-8 GB，留足余量；
REM   GPU 本来就是 94% 利用率（瓶颈在 GPU），串行对总时长影响有限。
REM   每个方法各自写自己的 <tag>.done，编排的等待逻辑不变。
REM ============================================================================
set D=C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_gpu3080
set LOGS=C:\Users\zh\gpu3080_logs
echo == SEQUENTIAL START %DATE% %TIME% == > %LOGS%\sequential.log
for %%T in (mvmoe routefinder cada rrnco) do (
  echo ---- start %%T %TIME% ---- >> %LOGS%\sequential.log
  call "%D%\_gpu3080_%%T.cmd"
  echo ---- end %%T %TIME% ---- >> %LOGS%\sequential.log
)
echo == SEQUENTIAL END %DATE% %TIME% == >> %LOGS%\sequential.log
echo SEQUENTIAL_ALLDONE %DATE% %TIME% > %LOGS%\sequential.done
