@echo off
REM Terminal state of the cc_compare env build (setup5).
if exist C:\Users\zh\gpu3080_logs\cc_compare_setup5.done echo DONE_MARKER
if exist C:\Users\zh\gpu3080_logs\cc_compare_setup5.failed echo FAILED_MARKER
echo ===KEY_LINES===
findstr /C:"VERIFY" /C:"torch rc=" /C:"torchvision rc=" /C:"rl4co rc=" /C:"extras rc=" /C:"verify rc=" /C:"CUDA_AVAIL" /C:"Successfully installed" C:\Users\zh\gpu3080_logs\cc_compare_setup5.log
echo ===TAIL===
powershell -NoProfile -Command "Get-Content C:\Users\zh\gpu3080_logs\cc_compare_setup5.log -Tail 3"
echo ===END===
