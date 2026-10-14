@echo off
REM 3080 上跑 MAPT 臂的最小烟测（1 天，6 workers），验证驱动+provider+env 链路
set PYTHONIOENCODING=utf-8
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == MAPT 1-day smoke on 3080 ==
conda run -n pytorch python scripts\evaluation\run_a1_external_accept.py --solver ordering ^
  --provider mapt ^
  --provider-ckpt ..\CC_Compare\MAPT_code\a1_accept_v1\mapt_a1.pt ^
  --provider-device cpu ^
  --baseline-from results\a1_c1rr2_reveal_40\gate.json ^
  --dev-instances 1 --workers 6 --penalty "p_c=0" ^
  --out results\a1_ext_formal2_3080\smoke_mapt_1d
echo rc=%ERRORLEVEL%
if exist results\a1_ext_formal2_3080\smoke_mapt_1d\gate.json (
  echo GATE_EXISTS
) else (
  echo NO_GATE
)
