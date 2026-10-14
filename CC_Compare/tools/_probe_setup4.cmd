@echo off
REM Progress of the Aliyun wheel download + setup4 log tail.
echo ===WHEEL===
if exist D:\anaconda\envs\cc_compare\_wheels\*.whl (
  for %%A in (D:\anaconda\envs\cc_compare\_wheels\*.whl) do echo %%~nxA  %%~zA bytes  modified %%~tA
) else (
  echo no wheel file yet
)
echo ===LOG===
powershell -NoProfile -Command "Get-Content C:\Users\zh\gpu3080_logs\cc_compare_setup4.log -Tail 4"
echo ===MARKERS===
if exist C:\Users\zh\gpu3080_logs\cc_compare_setup4.done echo DONE_MARKER
if exist C:\Users\zh\gpu3080_logs\cc_compare_setup4.failed echo FAILED_MARKER
echo ===END===
