@echo off
REM Pre-flight part 2: files imported by the rrnco/ordering solver paths.
cd /d C:\Users\zh\project\MASKCO-Main
echo ===RRNCO_PATH===
if exist C-VRP_Cold-chainVehicleRoutingProblem\scripts\evaluation\rrnco_accept_replanner.py (echo OK rrnco_accept_replanner.py) else (echo MISSING rrnco_accept_replanner.py)
if exist CC_Compare\RRNCO\dcc_rh_v4\rrnco_guided_adapter.py (echo OK rrnco_guided_adapter.py) else (echo MISSING rrnco_guided_adapter.py)
if exist C-VRP_Cold-chainVehicleRoutingProblem\scripts\evaluation\ordering_providers.py (echo OK ordering_providers.py) else (echo MISSING ordering_providers.py)
echo ===REGISTRY_ON_3080===
findstr /C:"'mvmoe'" /C:"'routefinder'" /C:"'cada'" C-VRP_Cold-chainVehicleRoutingProblem\scripts\evaluation\ordering_providers.py
echo ===END===
