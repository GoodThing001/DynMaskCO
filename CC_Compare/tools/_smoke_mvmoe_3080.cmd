@echo off
REM 3080：MVMoE 1 天烟测（纯 torch，用现有 pytorch env）——验证迁移链路
set PYTHONIOENCODING=utf-8
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == MVMoE 1-day smoke on 3080 ==
conda run -n pytorch python scripts\evaluation\run_a1_external_accept.py --solver ordering ^
  --provider mvmoe ^
  --provider-ckpt ..\CC_Compare\MVMoE\pretrained\pomo_vrptw_n100\epoch-5000.pt ^
  --provider-device cpu ^
  --baseline-from results\a1_c1rr2_reveal_40\gate.json ^
  --dev-instances 1 --workers 6 --penalty "p_c=0" ^
  --out results\a1_ext_formal2_3080\smoke_mvmoe_1d
echo rc=%ERRORLEVEL%
if exist results\a1_ext_formal2_3080\smoke_mvmoe_1d\gate.json (echo GATE_EXISTS) else (echo NO_GATE)
