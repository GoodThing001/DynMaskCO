@echo off
REM ============================================================================
REM 3080: RRNCO 原生 solver 面 C2（权重因果）对照 —— 补 模型参与证据.md 的已知缺口。
REM
REM 为什么放在 3080 而不是 1003：
REM   1003 正在跑 v2 正式批次（cpuA/cpuB 各 9 workers，每决策 10s 硬时限），
REM   台账纪律明令「正式运行期间不加 CPU 负载」。3080 是 16 核，且 rrnco 是
REM   串行驱动的**最后一个**方法 —— rrnco.done 一出现，3080 就空了。
REM
REM 口径与其它 11 个 provider **完全一致**：--device cpu、单线程、
REM   n_pool=40、不可见节点 60、seed=7、每次真解码前清缓存。
REM   本机执行只换机器，不换口径（机器差异已在 GPU 腿的 v1↔v2 逐位复现中对账过）。
REM
REM 为什么先等 rrnco.done：RRNCO policy 在 12 GB 卡上 40 起点解码，
REM   与在跑的正式臂抢显存没有收益；等它收工再跑，零干扰。
REM ============================================================================
set ROOT=C:\Users\zh\project\MASKCO-Main
set LOGS=C:\Users\zh\gpu3080_logs
if not exist %LOGS% mkdir %LOGS%
del /q %LOGS%\rrnco_c2.done 2>nul
echo == RRNCO_C2 WAIT START %DATE% %TIME% == > %LOGS%\rrnco_c2.log

set /a N=0
:WAIT
if exist %LOGS%\rrnco.done goto GO
set /a N+=1
if %N% GEQ 480 goto TIMEOUT
timeout /t 60 /nobreak >nul
goto WAIT

:TIMEOUT
echo == TIMEOUT: rrnco.done 未出现（等了 %N% 分钟）== >> %LOGS%\rrnco_c2.log
echo RRNCO_C2_TIMEOUT %DATE% %TIME% > %LOGS%\rrnco_c2.done
goto :EOF

:GO
echo == rrnco.done seen after %N% min; %DATE% %TIME% == >> %LOGS%\rrnco_c2.log
cd /d %ROOT%
conda run -n cc_compare python CC_Compare\tools\verify_rrnco_c2.py --device cpu --outdir CC_Compare\results\a1_compare_table\participation >> %LOGS%\rrnco_c2.log 2>&1
set RC=%ERRORLEVEL%
echo -- rrnco_c2 rc=%RC% -- >> %LOGS%\rrnco_c2.log
echo RRNCO_C2_DONE %DATE% %TIME% rc=%RC% > %LOGS%\rrnco_c2.done
