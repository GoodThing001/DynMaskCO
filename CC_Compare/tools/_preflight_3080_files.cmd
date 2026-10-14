@echo off
REM Pre-flight: do the 3080 sources/checkpoints needed by the 4 arms exist?
cd /d C:\Users\zh\project\MASKCO-Main
echo ===RRNCO_NATIVE===
if exist CC_Compare\RRNCO\dcc_rh_v4\rrnco_backend.py (echo OK rrnco_backend.py) else (echo MISSING rrnco_backend.py)
if exist CC_Compare\RRNCO\rrnco\models\policy.py (echo OK rrnco package) else (echo MISSING rrnco package)
if exist CC_Compare\RRNCO\dcc_vrp\run_dcc.py (echo OK run_dcc.py) else (echo MISSING run_dcc.py)
if exist CC_Compare\RRNCO\a1_accept_v1\run_formal_rr.sh (echo OK run_formal_rr.sh) else (echo MISSING run_formal_rr.sh)
echo ===CHECKPOINTS===
for %%F in (CC_Compare\RRNCO\checkpoints\rcvrptw\epoch_199.ckpt CC_Compare\RouteFinder\checkpoints\100\rf-transformer.ckpt CC_Compare\CaDA\100\result\2024-1121-1355\checkpoint-300.pt CC_Compare\MVMoE\pretrained\pomo_vrptw_n100\epoch-5000.pt) do call :sz %%F
echo ===PROVIDERS===
for %%D in (CC_Compare\RouteFinder\a1_accept_v1 CC_Compare\CaDA\a1_accept_v1 CC_Compare\MVMoE\a1_accept_v1 CC_Compare\CO-enriched-ML\a1_accept_v1 CC_Compare\MAPT_code\a1_accept_v1) do call :dir %%D
echo ===END===
goto :eof

:sz
if exist %1 (for %%A in (%1) do echo OK %~nx1 %%~zA bytes) else (echo MISSING %1)
goto :eof

:dir
if exist %1 (echo OK %1) else (echo MISSING %1)
goto :eof
