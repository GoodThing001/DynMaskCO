@echo off
REM Launch the cc_compare env build detached (WMI), so an SSH drop cannot kill it.
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c C:\Users\zh\project\MASKCO-Main\CC_Compare\tools\_setup_cc_compare_3080.cmd'} | Select-Object -ExpandProperty ProcessId"
