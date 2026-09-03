@echo off
setlocal
chcp 65001 >nul

echo ==============================================
echo   Cursor cleanup (run AFTER closing Cursor)
echo ==============================================
echo.

REM ---- Guard: abort if Cursor is still running ----
tasklist /FI "IMAGENAME eq Cursor.exe" 2>nul | find /I "Cursor.exe" >nul
if not errorlevel 1 (
    echo [WARN] Cursor is still running. Close it completely, then re-run.
    echo.
    pause
    exit /b 1
)

set "GS=%APPDATA%\Cursor\User\globalStorage"
set "R=%APPDATA%\Cursor"

REM ---- 1. Rename bloated state.vscdb (recoverable) ----
echo [1/2] Moving state.vscdb -^> state.vscdb.old ...
if exist "%GS%\state.vscdb" (
    if exist "%GS%\state.vscdb.old" del /q "%GS%\state.vscdb.old"
    move /y "%GS%\state.vscdb" "%GS%\state.vscdb.old" >nul
    del /q "%GS%\state.vscdb-shm" 2>nul
    del /q "%GS%\state.vscdb-wal" 2>nul
    echo       done.
) else (
    echo       state.vscdb not found, skipped.
)

REM ---- 2. Clear caches ----
echo [2/2] Clearing caches ...
for %%D in (Cache GPUCache CachedData "Code Cache" DawnGraphiteCache DawnWebGPUCache logs WebStorage CachedExtensionVSIXs Crashpad) do (
    if exist "%R%\%%~D" (
        rmdir /s /q "%R%\%%~D" 2>nul
    )
)
echo       done.

echo.
echo ==============================================
echo   Finished. Reopen Cursor and verify.
echo.
echo   If everything is normal, you can safely delete:
echo     %GS%\state.vscdb.old
echo.
echo   (claude-dev/checkpoints was left untouched - it is
echo    your edit-revert history. Ask Claude to clean it if
echo    you want that removed too.)
echo ==============================================
echo.
pause
endlocal
