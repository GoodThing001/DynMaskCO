@echo off
echo CWD_BEFORE=%CD%
cd /d C:\Users\zh\project\MASKCO-Main
echo CWD_AFTER=%CD%
where conda
conda --version
dir /b C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\verify_rrnco_c2.py
dir /b C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\verify_provider_participation.py
conda run -n cc_compare python CC_Compare\tools\verify_rrnco_c2.py --help
echo PREFLIGHT_RC=%ERRORLEVEL%
